"""Read-only workspace tools and validated, in-memory builder proposals."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml
from jinja2.sandbox import SandboxedEnvironment

from ultrarag.mcp_logging import redact_config

from . import pipeline_manager as pm

MAX_TEXT = 48_000
FORBIDDEN_KEYS = {"__proto__", "prototype", "constructor"}


def parse_document(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or len(text) > MAX_TEXT:
        raise ValueError("YAML must be text of at most 48000 characters")
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise ValueError(f"Invalid YAML{location}") from exc
    if not isinstance(value, dict):
        raise ValueError("Pipeline YAML must be a mapping")
    # Reject recursive/excessive aliases before serializing or redacting.
    nodes = 0

    def visit(item: Any, ancestors: set[int], depth: int = 0) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > 10_000 or depth > 30:
            raise ValueError("YAML nesting or alias expansion is too large")
        if isinstance(item, (dict, list)):
            if id(item) in ancestors:
                raise ValueError("Recursive YAML aliases are unsupported")
            nested = ancestors | {id(item)}
            if isinstance(item, dict):
                if any(
                    not isinstance(key, str) or key in FORBIDDEN_KEYS for key in item
                ):
                    raise ValueError("YAML keys must be safe strings")
                children = item.values()
            else:
                children = item
            for child in children:
                visit(child, nested, depth + 1)

    visit(value, set())
    json.dumps(
        value, allow_nan=False
    )  # Reject non-JSON YAML values such as dates and sets.
    return value


def _schema(
    name: str,
    description: str,
    properties: dict | None = None,
    required: list | None = None,
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


STRING = {"type": "string"}
TOOLS = [
    _schema("list_servers", "List available workspace servers and their tool counts."),
    _schema(
        "inspect_server",
        "Read actual tool/prompt I/O contracts and redacted defaults for one server.",
        {"name": STRING},
        ["name"],
    ),
    _schema("list_pipelines", "List saved pipelines, including unbuilt drafts."),
    _schema(
        "read_pipeline",
        "Read a saved pipeline or the current unsaved draft (omit name).",
        {"name": STRING},
    ),
    _schema(
        "read_parameters",
        "Read redacted parameters; unbuilt drafts use server defaults.",
        {"name": STRING},
    ),
    _schema("list_prompts", "List workspace Jinja templates."),
    _schema(
        "read_prompt",
        "Read a Jinja template under prompt/.",
        {"filename": STRING},
        ["filename"],
    ),
    _schema(
        "validate_pipeline",
        "Statically check YAML, real server/tool names and I/O remapping. Does not build or run.",
        {"yaml": STRING},
        ["yaml"],
    ),
    _schema(
        "propose_pipeline",
        "Validate and stage a complete YAML proposal for the user to review and apply. Does not save.",
        {"yaml": STRING},
        ["yaml"],
    ),
    _schema(
        "propose_parameter",
        "Stage a non-secret parameter change using its dotted path; does not save.",
        {"path": STRING, "value": {}},
        ["path", "value"],
    ),
    _schema(
        "propose_prompt",
        "Stage a Jinja template proposal under prompt/ for review; does not save.",
        {"filename": STRING, "content": STRING},
        ["filename", "content"],
    ),
]


class ToolWorkspace:
    def __init__(self, context: dict[str, Any]) -> None:
        self.name = str(context.get("selectedPipeline") or "")
        self.mode = str(context.get("currentMode") or "pipeline")
        self.draft = str(context.get("pipelineYaml") or "")
        self.base_yaml_hash = (
            hashlib.sha256(self.draft.encode()).hexdigest() if self.draft else None
        )
        self.parameters = redact_config(copy.deepcopy(context.get("parameters")))
        self.prompt_name = str(context.get("currentPromptFile") or "")
        self.prompt_content = str(context.get("promptContent") or "")
        self._actions: dict[tuple[str, str], dict] = {}

    @property
    def actions(self) -> list[dict]:
        return list(self._actions.values())

    def _server(self, name: str) -> tuple[Path, dict]:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,127}", name):
            raise ValueError("Invalid server name")
        root = pm.SERVERS_DIR.resolve()
        folder = (root / name).resolve()
        if not folder.is_relative_to(root) or not folder.is_dir():
            raise ValueError(f"Unknown workspace server: {name}")
        return folder, pm._ensure_server_yaml(folder)

    def _prompt(self, filename: str) -> Path:
        root = (pm.PROJECT_ROOT / "prompt").resolve()
        if filename.startswith("prompt/"):
            filename = filename[7:]
        candidate = (root / filename).resolve()
        if not candidate.is_relative_to(root) or candidate.suffix not in {
            ".jinja",
            ".jinja2",
            ".j2",
        }:
            raise ValueError("Prompt must be a Jinja template under prompt/")
        return candidate

    def _pipeline(self, name: str = "") -> dict:
        if not name and self.draft:
            return parse_document(self.draft)
        document = pm.load_pipeline(pm._sanitize_pipeline_name(name or self.name))
        document.pop("_raw_yaml", None)
        return document

    def validate(self, text: str) -> dict:
        errors: list[str] = []
        try:
            document = parse_document(text)
            if redact_config(document) != document:
                raise ValueError(
                    "Keep credentials in parameters or model settings, not pipeline proposals"
                )
            servers = document.get("servers")
            if not isinstance(servers, dict) or not servers:
                raise ValueError("Declare at least one workspace server")
            catalog = {}
            for alias, folder in servers.items():
                if not isinstance(folder, str) or not re.fullmatch(
                    r"servers/[A-Za-z_][A-Za-z0-9_-]*", folder
                ):
                    raise ValueError(
                        f"Server {alias} must use a known servers/<name> directory"
                    )
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,127}", alias):
                    raise ValueError("Server aliases must be simple names")
                _, catalog[alias] = self._server(folder.split("/")[1])

            def steps(items: Any, location: str) -> None:
                if not isinstance(items, list):
                    errors.append(f"{location}: steps must be a list")
                    return
                for index, step in enumerate(items):
                    here = f"{location}[{index}]"
                    if isinstance(step, dict) and set(step) == {"loop"}:
                        block = step["loop"]
                        if (
                            not isinstance(block, dict)
                            or type(block.get("times")) is not int
                            or not 1 <= block["times"] <= 1000
                        ):
                            errors.append(
                                f"{here}: loop times must be an integer from 1 to 1000"
                            )
                        else:
                            steps(block.get("steps"), here + ".loop")
                        continue
                    if isinstance(step, dict) and set(step) == {"branch"}:
                        block = step["branch"]
                        if not isinstance(block, dict) or not isinstance(
                            block.get("branches"), dict
                        ):
                            errors.append(
                                f"{here}: branch requires router and named branches"
                            )
                        else:
                            steps(block.get("router"), here + ".router")
                            for label, branch in block["branches"].items():
                                steps(branch, here + "." + label)
                        continue
                    if isinstance(step, str):
                        name, remapping = step, {}
                    elif isinstance(step, dict) and len(step) == 1:
                        name, remapping = next(iter(step.items()))
                    else:
                        errors.append(f"{here}: expected one server.tool step")
                        continue
                    if not isinstance(name, str) or name.count(".") != 1:
                        errors.append(f"{here}: expected server.tool")
                        continue
                    alias, tool = name.split(".")
                    server = catalog.get(alias, {})
                    spec = {
                        **(server.get("tools") or {}),
                        **(server.get("prompts") or {}),
                    }.get(tool)
                    if not isinstance(spec, dict):
                        errors.append(f"{here}: unknown tool or prompt {name}")
                        continue
                    if remapping is None:
                        remapping = {}
                    if not isinstance(remapping, dict):
                        errors.append(f"{here}: remapping must be an object")
                        continue
                    if set(remapping) - {"input", "output"}:
                        errors.append(f"{here}: use input/output remapping keys")
                    for direction in ("input", "output"):
                        mapping = remapping.get(direction, {})
                        declared = spec.get(direction) or {}
                        if not isinstance(mapping, dict) or any(
                            key not in declared
                            or not isinstance(value, str)
                            or not value
                            for key, value in mapping.items()
                        ):
                            errors.append(
                                f"{here}: invalid {direction} remapping for {name}"
                            )

            if not document.get("pipeline"):
                errors.append("Add at least one pipeline step")
            else:
                steps(document["pipeline"], "pipeline")
        except (ValueError, TypeError, OSError, pm.PipelineManagerError) as exc:
            errors.append(str(exc))
        return {
            "valid": not errors,
            "errors": errors,
            "scope": "Static checks only; Build and a test run are still required.",
        }

    def call(self, name: str, arguments: dict) -> dict:
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
        schema = next(
            (
                tool["function"]["parameters"]
                for tool in TOOLS
                if tool["function"]["name"] == name
            ),
            None,
        )
        if schema is None:
            raise ValueError(f"Unknown assistant tool: {name}")
        if set(arguments) - set(schema["properties"]) or set(schema["required"]) - set(
            arguments
        ):
            raise ValueError(f"Invalid arguments for {name}")
        for key, value in arguments.items():
            if key != "value" and not isinstance(value, str):
                raise ValueError(f"{key} must be a string")

        if name == "list_servers":
            return {"servers": sorted(pm.list_servers())}
        if name == "inspect_server":
            folder, metadata = self._server(arguments["name"])
            defaults_path = folder / "parameter.yaml"
            defaults = (
                yaml.safe_load(defaults_path.read_text())
                if defaults_path.is_file()
                else {}
            )
            return {
                "server": folder.name,
                "metadata": redact_config(metadata),
                "defaults": redact_config(defaults),
            }
        if name == "list_pipelines":
            return {
                "pipelines": [
                    {"name": item["name"], "is_ready": item["is_ready"]}
                    for item in pm.list_pipelines()
                ]
            }
        if name == "read_pipeline":
            return {
                "pipeline": redact_config(self._pipeline(arguments.get("name", "")))
            }
        if name == "read_parameters":
            selected = arguments.get("name") or self.name
            if selected == self.name and isinstance(self.parameters, dict):
                return {"parameters": self.parameters}
            try:
                parameters = pm.load_parameters(selected)
            except pm.PipelineManagerError:
                parameters = {}
                for alias, folder in (
                    self._pipeline(arguments.get("name", "")).get("servers", {}).items()
                ):
                    directory, _ = self._server(str(folder).removeprefix("servers/"))
                    parameter = directory / "parameter.yaml"
                    parameters[alias] = (
                        yaml.safe_load(parameter.read_text())
                        if parameter.exists()
                        else {}
                    )
            return {
                "parameters": redact_config(parameters),
                "source": "saved parameters or server defaults",
            }
        if name == "list_prompts":
            root = pm.PROJECT_ROOT / "prompt"
            return {
                "prompts": [
                    path.relative_to(root).as_posix()
                    for path in sorted(root.rglob("*"))
                    if path.is_file()
                    and path.suffix in {".jinja", ".jinja2", ".j2"}
                    and path.resolve().is_relative_to(root.resolve())
                ][:1000]
            }
        if name == "read_prompt":
            path = self._prompt(arguments["filename"])
            if path.stat().st_size > MAX_TEXT:
                raise ValueError("Template is too large")
            content = (
                self.prompt_content
                if arguments["filename"] == self.prompt_name
                else path.read_text(encoding="utf-8")
            )
            return {
                "filename": path.relative_to(pm.PROJECT_ROOT / "prompt").as_posix(),
                "content": content,
            }
        if name == "validate_pipeline":
            return self.validate(arguments["yaml"])
        if name == "propose_pipeline":
            self._actions.pop(("pipeline", self.name), None)
            report = self.validate(arguments["yaml"])
            if report["valid"]:
                self.draft = arguments["yaml"]
                self._actions[("pipeline", self.name)] = {
                    "type": "modify_pipeline",
                    "content": self.draft,
                    "preview": self.draft,
                    "targetPipeline": self.name,
                    "baseYamlHash": self.base_yaml_hash,
                }
            return {**report, "staged": report["valid"], "saved": False}
        if name == "propose_parameter":
            path, value = arguments["path"], arguments["value"]
            parse_document(json.dumps({"value": value}, allow_nan=False))
            parts = path.split(".")
            if not path or any(not part or part in FORBIDDEN_KEYS for part in parts):
                raise ValueError("Invalid parameter path")
            if (
                any(
                    redact_config({part: "marker"})[part] == "[REDACTED]"
                    for part in parts
                )
                or redact_config(value) != value
                or "[REDACTED]" in json.dumps(value)
            ):
                raise ValueError(
                    "Configure credentials in model settings; do not propose or overwrite secrets"
                )
            if parts[0] not in self._pipeline().get("servers", {}):
                raise ValueError("Parameter must belong to a declared pipeline server")
            current = self.call("read_parameters", {})["parameters"]
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    raise ValueError(
                        "Unknown parameter path; inspect actual parameters first"
                    )
                current = current[part]
            json.dumps(value, allow_nan=False)
            self._actions[("parameter", path)] = {
                "type": "modify_parameter",
                "path": path,
                "value": value,
                "preview": f"{path} = {json.dumps(value, ensure_ascii=False)}",
                "targetPipeline": self.name,
                "baseYamlHash": self.base_yaml_hash,
            }
            return {"staged": True, "saved": False, "path": path}
        if name == "propose_prompt":
            path = self._prompt(arguments["filename"])
            content = arguments["content"]
            if len(content) > MAX_TEXT:
                raise ValueError("Template is too large")
            SandboxedEnvironment().parse(content)
            filename = path.relative_to(pm.PROJECT_ROOT / "prompt").as_posix()
            self._actions[("prompt", filename)] = {
                "type": "modify_prompt",
                "filename": filename,
                "content": content,
                "preview": content,
            }
            return {"staged": True, "saved": False, "filename": filename}
        raise ValueError(f"Unsupported tool: {name}")
