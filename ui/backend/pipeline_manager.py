from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import time
import queue
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from types import ModuleType
import yaml
import ast
import os

try:
    from pymilvus import MilvusClient
except ImportError:
    MilvusClient = None

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

BASE_DIR = Path(__file__).resolve().parent.parent
SERVERS_DIR = PROJECT_ROOT / "servers"
PIPELINES_DIR = PROJECT_ROOT / "examples"
LEGACY_PIPELINES_DIR = BASE_DIR / "pipelines"
CHAT_DATASET_DIR = PROJECT_ROOT / "data" / "chat_sessions"
OUTPUT_DIR = PROJECT_ROOT / "output"

KB_ROOT = PROJECT_ROOT / "data" / "knowledge_base"
KB_RAW_DIR = KB_ROOT / "raw"       
KB_CORPUS_DIR = KB_ROOT / "corpus" 
KB_CHUNKS_DIR = KB_ROOT / "chunks" 
KB_INDEX_DIR = KB_ROOT / "index"
KB_CONFIG_PATH = KB_ROOT / "kb_config.json"

for d in [LEGACY_PIPELINES_DIR, CHAT_DATASET_DIR, OUTPUT_DIR, KB_RAW_DIR, KB_CORPUS_DIR, KB_CHUNKS_DIR, KB_INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)



class PipelineManagerError(RuntimeError):
    pass

STUBBED_MODULES: set[str] = set()

# Client Function Loader (Lazy Import)
_client_funcs = {}

def _ensure_client_funcs():
    if _client_funcs:
        return _client_funcs
    
    try:
        from ultrarag.client import (
            build, 
            load_pipeline_context,
            create_mcp_client,
            execute_pipeline
        )
        _client_funcs["build"] = build
        _client_funcs["load_ctx"] = load_pipeline_context
        _client_funcs["create_client"] = create_mcp_client
        _client_funcs["exec_pipe"] = execute_pipeline
        return _client_funcs
    except ModuleNotFoundError as exc:
        msg = f"Missing dependency: {exc.name}"
        LOGGER.error(msg)
        raise PipelineManagerError(msg) from exc

# Session Management (Multi-Client Support)
class DemoSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._client = None
        self._context = None
        self._active = False
        self._thread.start()
        self._current_future = None

    def _bg_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def start(self, name: str):
        if self._active: return
        
        funcs = _ensure_client_funcs()
        config_file = _find_pipeline_file(name)
        if not config_file: raise PipelineManagerError(f"Pipeline {name} not found")
        
        # Phase 1: Load Context
        param_path = _resolve_parameter_path(name, for_write=False)
        self._context = funcs["load_ctx"](str(config_file), str(param_path))
        
        # Phase 2: Create Client
        self._client = funcs["create_client"](self._context["mcp_cfg"])
        
        # Phase 3: Connect (Enter Context)
        future = asyncio.run_coroutine_threadsafe(self._client.__aenter__(), self._loop)
        try:
            future.result(timeout=15)
            self._active = True
            LOGGER.info(f"Session {self.session_id} connected for pipeline {name}")
        except Exception as e:
            LOGGER.error(f"Session start failed: {e}")
            raise PipelineManagerError(f"Connection failed: {e}")

    def stop(self):
        if self._active and self._client:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._client.__aexit__(None, None, None), self._loop
                ).result(timeout=5)
            except Exception as e:
                LOGGER.warning(f"Error during disconnect: {e}")
        
        self._active = False
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1)
        LOGGER.info(f"Session {self.session_id} stopped")

    def run_chat(self, callback, dynamic_params: Dict[str, Any] = None):
        if not self._active: raise PipelineManagerError("Session not active")
        
        funcs = _ensure_client_funcs()
        self._current_future = asyncio.run_coroutine_threadsafe(
            funcs["exec_pipe"](
                self._client, 
                self._context, 
                is_demo=True, 
                return_all=True,
                stream_callback=callback,
                override_params=dynamic_params
            ),
            self._loop
        )
        
        try:
            return self._current_future.result()
        except asyncio.CancelledError:
            LOGGER.info(f"Session {self.session_id} task cancelled by user.")
            raise # 继续抛出，让外层捕获
        finally:
            self._current_future = None # 任务结束，清空引用
    def interrupt_task(self):
        if self._current_future and not self._current_future.done():
            self._current_future.cancel()
            LOGGER.info(f"Interrupt signal sent to session {self.session_id}")
            return True
        return False

class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, DemoSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> DemoSession:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = DemoSession(session_id)
            return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[DemoSession]:
        with self._lock: return self._sessions.get(session_id)

    def remove(self, session_id: str):
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].stop()
                del self._sessions[session_id]

SESSION_MANAGER = SessionManager()

# API Logic (Wrappers)

def start_demo_session(name: str, session_id: str):
    session = SESSION_MANAGER.get_or_create(session_id)
    try:
        session.start(name)
        return {"status": "started", "session_id": session_id}
    except Exception as e:
        SESSION_MANAGER.remove(session_id)
        raise e

def stop_demo_session(session_id: str):
    SESSION_MANAGER.remove(session_id)
    return {"status": "stopped", "session_id": session_id}

# Chat Logic (Streaming + Context)

def _prepare_chat_context(name: str, question: str):
    # This prepares the temporary JSONL file for Benchmark/Retriever
    param_path = _resolve_parameter_path(name, for_write=False)
    original_text = param_path.read_text(encoding="utf-8")
    params = load_parameters(name)
    
    dataset_path = None
    if "benchmark" in params and "benchmark" in params["benchmark"]:
        d_dir = CHAT_DATASET_DIR
        d_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = d_dir / f"{name}_{int(time.time()*1000)}.jsonl"
        record = {"id": 0, "question": question, "golden_answers": [], "meta_data": {}}
        dataset_path.write_text(json.dumps(record, ensure_ascii=False)+"\n", encoding="utf-8")
        
        params["benchmark"]["benchmark"]["path"] = _as_project_relative(dataset_path)
        save_parameters(name, params)
    
    return param_path, original_text, dataset_path

def chat_demo_stream(name: str, question: str, session_id: str, dynamic_params: Dict[str, Any] = None):
    """Generator function for SSE streaming"""
    session = SESSION_MANAGER.get(session_id)
    if not session:
        yield f"data: {json.dumps({'type': 'error', 'message': 'Session expired'})}\n\n"
        return

    # 1. Prepare (Write JSONL)
    param_path, original_text, dataset_path = _prepare_chat_context(name, question)
    before_memory = {str(path) for path in sorted(OUTPUT_DIR.glob(f"memory_*_{name}_*.json"))}
    
    # 2. Queue for async -> sync bridge
    token_queue = queue.Queue()

    async def token_callback(event_data):
        if isinstance(event_data, dict):
            # client.py 发来的结构化事件 (step_start, sources, step_end, token字典)
            token_queue.put(event_data)
        else:
            # 兼容旧逻辑：如果是纯字符串，视为普通 token
            token_queue.put({"type": "token", "content": str(event_data)})

    # 3. Background Task
    def run_bg():
        try:
            # Execute
            res = session.run_chat(token_callback, dynamic_params)
            
            # Cleanup & Resolve
            param_path.write_text(original_text, encoding="utf-8")
            
            # Resolve Answer
            final_ans = _extract_result(res)
            mem_ans, mem_file = _find_memory_answer(name, before_memory)
            
            final_data = {
                "status": "succeeded",
                "answer": final_ans or mem_ans or "No answer generated",
                "dataset_path": _as_project_relative(dataset_path) if dataset_path else None,
                "memory_path": _as_project_relative(mem_file) if mem_file else None
            }
            token_queue.put({"type": "final", "data": final_data})

        except asyncio.CancelledError:
            # [新增] 捕获 CancelledError，视为用户主动停止
            # 不发送 error 消息，或者发送一个特定的 info
            # 实际上由于前端已经 Abort 了，这里发什么前端都不收了，主要是为了后端日志干净
            LOGGER.info("Chat background task cancelled.")
            token_queue.put(None) # 结束队列
            return   
        except Exception as e:
            LOGGER.error(f"Chat error: {e}")
            token_queue.put({"type": "error", "message": str(e)})
        finally:
            token_queue.put(None) # Sentinel

    threading.Thread(target=run_bg, daemon=True).start()

    # 4. Yield Stream
    while True:
        item = token_queue.get()
        if item is None: break
        yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

# Helpers (Config & Files) 

def _extract_result(res: Any) -> Optional[str]:
    if not res: return None
    
    # 1. 如果 res 是 execute_pipeline 返回的包装字典 (return_all=True)
    if isinstance(res, dict) and "final_result" in res:
        # 取出真正的 result 数据
        target = res["final_result"]
    else:
        target = res

    # 2. 尝试解析目标数据
    try:
        # 情况 A: target 是 JSON 字符串 (MockResult.data)
        if isinstance(target, str):
            try:
                parsed = json.loads(target)
                if isinstance(parsed, dict) and "ans_ls" in parsed:
                    return parsed["ans_ls"][0]
            except:
                return target # 解析失败就返回原始字符串

        # 情况 B: target 已经是 Dict (如果 MockResult 存的是 Dict)
        if isinstance(target, dict) and "ans_ls" in target:
            return target["ans_ls"][0]
            
        # 情况 C: 仍然没找到，尝试从 content 结构里找 (兜底)
        if hasattr(target, 'content') and isinstance(target.content, list):
            text = target.content[0].text
            return json.loads(text).get("ans_ls", [""])[0]
            
    except Exception as e: 
        LOGGER.warning(f"Failed to extract result: {e}")
        
    return str(target) if target else None

def _find_memory_answer(name: str, before: set[str]):
    files = sorted(OUTPUT_DIR.glob(f"memory_*_{name}_*.json"))
    candidates = [p for p in files if str(p) not in before]
    if not candidates: return None, None
    target = max(candidates, key=lambda p: p.stat().st_mtime)
    
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if data:
            last_mem = data[-1].get("memory", {})
            # Simple heuristic
            for k in ["ans_ls", "pred_ls", "result", "answer"]:
                if k in last_mem: 
                    val = last_mem[k]
                    return str(val[0]) if isinstance(val, list) and val else str(val), target
    except Exception: pass
    return None, target

def _as_project_relative(path: Path) -> str:
    try: return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError: return path.as_posix()

def pipeline_path(name: str) -> Path: 
    return PIPELINES_DIR / f"{name.replace('..','_')}.yaml"

def _find_pipeline_file(name: str) -> Path | None:
    p = pipeline_path(name)
    if p.exists(): return p
    l = LEGACY_PIPELINES_DIR / f"{name}.yaml"
    return l if l.exists() else None

def _parameter_candidates(config_file: Path) -> List[Path]:
    base = config_file.stem
    return [
        config_file.parent / "parameter" / f"{base}_parameter.yaml",
        PIPELINES_DIR / "parameter" / f"{base}_parameter.yaml"
    ]

def _resolve_parameter_path(name: str, *, for_write: bool = False) -> Path:
    cfg = _find_pipeline_file(name)
    if not cfg: raise PipelineManagerError("Pipeline not found")
    cands = _parameter_candidates(cfg)
    if not for_write:
        for c in cands: 
            if c.exists(): return c
    for c in cands:
        if c.parent.exists() or for_write: return c
    return cands[0]

# CRUD Operations

def list_servers() -> Dict:
    res = {}
    if SERVERS_DIR.exists():
        for d in SERVERS_DIR.iterdir():
            if d.is_dir(): res[d.name] = _ensure_server_yaml(d)
    return res

def list_server_tools() -> List[ServerTool]:
    res = []
    for srv, cfg in list_servers().items():
        for t, m in (cfg.get("tools") or {}).items():
            res.append(ServerTool(srv, t, "tool", m.get("input",{}), m.get("output",[])))
        for t, m in (cfg.get("prompts") or {}).items():
            res.append(ServerTool(srv, t, "prompt", m.get("input",{}), m.get("output",[])))
    return res

def list_pipelines() -> List[Dict[str, Any]]:
    res = []
    for d in [PIPELINES_DIR, LEGACY_PIPELINES_DIR]:
        if d.exists():
            for f in d.glob("*.yaml"):
                if not any(r["name"] == f.stem for r in res):
        
                    param_path = d / "parameter" / f"{f.stem}_parameter.yaml"
                    if not param_path.exists():
                        param_path = f.parent / "parameter" / f"{f.stem}_parameter.yaml"
                    
                    is_ready = param_path.exists()

                    res.append({
                        "name": f.stem, 
                        "config": yaml.safe_load(f.read_text(encoding="utf-8")) or {},
                        "is_ready": is_ready 
                    })
    return res

def load_pipeline(name: str) -> Dict:
    p = _find_pipeline_file(name)
    if not p: raise PipelineManagerError(f"Pipeline {name} not found")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def save_pipeline(payload: Dict) -> Dict:
    name = payload.get("name")
    if not name: raise PipelineManagerError("Name required")
    
    steps = payload.get("pipeline", [])
    servers = set()

    def _scan(s):
        if isinstance(s, str) and "." in s:
            servers.add(s.split(".")[0])
        elif isinstance(s, list):
            for item in s:
                _scan(item)
        elif isinstance(s, dict):
            for v in s.values():
                _scan(v)

    _scan(steps)
    
    yaml_data = {
        "servers": {s: f"servers/{s}" for s in sorted(servers)},
        "pipeline": steps
    }
    
    with pipeline_path(name).open("w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)
    
    return {"name": name, **yaml_data}

def delete_pipeline(name: str):
    p = _find_pipeline_file(name)
    if p: p.unlink()

def load_parameters(name: str) -> Dict:
    p = _resolve_parameter_path(name, for_write=False)
    if not p.exists(): raise PipelineManagerError("Parameters not found. Build first.")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def save_parameters(name: str, params: Dict):
    p = _resolve_parameter_path(name, for_write=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False, allow_unicode=True)

def build(name: str) -> Dict:
    funcs = _ensure_client_funcs()
    p = _find_pipeline_file(name)
    if not p: raise PipelineManagerError(f"Not found: {name}")
    
    try:
        # Build is fast, run synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(funcs["build"](str(p)))
        loop.close()
    except Exception as e:
        raise PipelineManagerError(f"Build failed: {e}")
    return {"status": "ok"}

# Internal Utils (Stubs & Parsing)
@dataclass
class ServerTool:
    server: str; tool: str; kind: str; input_spec: Dict; output_spec: List
    @property
    def identifier(self): return f"{self.server}.{self.tool}"

def _flatten_param_keys(data: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(data, dict):
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            keys.add(path)
            keys |= _flatten_param_keys(value, path)
    return keys

def _generate_server_stub(
    server_dir: Path, module_path: Path, parameter_path: Path
) -> Dict[str, Any]:
    """
    静态分析策略：
    1. 扫描全文，记录所有函数名及其参数 (definitions)。
    2. 扫描全文，记录所有工具注册行为 (registrations)。
    3. 最后统一匹配，忽略代码定义的先后顺序。
    """
    try:
        source = module_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}

    tree = ast.parse(source, filename=str(module_path))

    # === 数据容器 ===
    # 存储所有函数定义: { "func_name": ["arg1", "arg2"] }
    definitions: Dict[str, List[str]] = {}
    # 存储所有注册意图: [ {"name": "func_name", "output": "...", "type": "tool"} ]
    registrations: List[Dict[str, Any]] = []

    # === 辅助函数 ===
    def parse_output_spec(keywords):
        for kw in keywords:
            if kw.arg == "output" and isinstance(kw.value, ast.Constant):
                val = kw.value.value
                if not val: return []
                out_part = val.split("->", 1)[-1]
                return [x.strip() for x in out_part.split(",") if x.strip().lower() != "none"]
        return []

    # === AST 访问器 ===
    class Collector(ast.NodeVisitor):
        def visit_FunctionDef(self, node): self._handle_func(node)
        def visit_AsyncFunctionDef(self, node): self._handle_func(node)

        def _handle_func(self, node):
            # 1. [收集定义] 记录函数名和参数 (忽略 self/cls)
            args = [a.arg for a in node.args.args if a.arg not in ('self', 'cls')]
            definitions[node.name] = args

            # 2. [收集装饰器] @app.tool
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
                    if dec.func.attr in {'tool', 'prompt'}:
                        registrations.append({
                            "name": node.name,
                            "output": parse_output_spec(dec.keywords),
                            "type": dec.func.attr # 'tool' or 'prompt'
                        })
            
            # 继续遍历函数体 (虽然一般不会在函数里定义函数，但为了保险)
            self.generic_visit(node)

        def visit_Call(self, node):
            # 3. [收集显式调用] mcp.tool(self.func)
            if isinstance(node.func, ast.Attribute) and node.func.attr in {'tool', 'prompt'}:
                tool_name = None
                # 解析参数: mcp.tool(self.my_func) -> my_func
                if node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Attribute): tool_name = first.attr 
                    elif isinstance(first, ast.Name): tool_name = first.id
                
                if tool_name:
                    registrations.append({
                        "name": tool_name,
                        "output": parse_output_spec(node.keywords),
                        "type": node.func.attr
                    })
            
            self.generic_visit(node)

    # 执行扫描
    Collector().visit(tree)

    # === 匹配与构建 ===
    
    # 准备参数文件里的 keys 用于 $ 匹配
    param_keys: set[str] = set()
    if parameter_path.exists():
        try:
            param_data = yaml.safe_load(parameter_path.read_text(encoding="utf-8")) or {}
            param_keys = {k.split(".")[0] for k in _flatten_param_keys(param_data)}
        except: pass

    tools_map = {}
    prompts_map = {}

    for reg in registrations:
        name = reg["name"]
        kind = reg["type"]
        outputs = reg["output"]
        
        # 从定义池中查找参数
        # 注意：这里假设文件名内没有重名的函数（或者最后定义的覆盖前面的）
        # 在 MCP Server 文件中，工具函数名通常是唯一的
        func_args = definitions.get(name, [])
        
        # 构建输入映射
        input_mapping = {}
        for arg in func_args:
            if arg in param_keys: input_mapping[arg] = f"${arg}"
            else: input_mapping[arg] = arg

        entry = {"input": input_mapping}
        if outputs: entry["output"] = outputs

        if kind == "prompt":
            prompts_map[name] = entry
        else:
            tools_map[name] = entry

    return {
        "path": str(module_path),
        "parameter": str(parameter_path),
        "tools": tools_map,
        "prompts": prompts_map,
    }

def _ensure_server_yaml(server_dir: Path) -> Dict[str, Any]:
    server_name = server_dir.name
    yaml_path = server_dir / "server.yaml"
    parameter_path = server_dir / "parameter.yaml"
    
    # 1. 优先读取现成的 server.yaml (Build 过的情况)
    if yaml_path.exists():
        try: 
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            # 只要文件格式正确就返回，即使内容为空
            if isinstance(data, dict): 
                return data
        except Exception as e: 
            LOGGER.warning(f"Failed to load existing server.yaml for {server_name}: {e}")
    
    # 2. 如果没有 yaml，尝试从源码解析 (Stub)
    # 策略 A: 标准路径 src/{server_name}.py
    module_path = server_dir / "src" / f"{server_name}.py"
    
    # 策略 B: 如果标准路径不存在，扫描 src 下的任何 .py 文件
    if not module_path.exists():
        src_dir = server_dir / "src"
        if src_dir.exists():
            # 排除 __init__.py，找其他 .py 文件
            candidates = [p for p in src_dir.glob("*.py") if p.name != "__init__.py"]
            if candidates:
                # 优先找包含 'server' 或 'app' 字样的文件，否则取第一个
                module_path = next((p for p in candidates if "server" in p.name or "app" in p.name), candidates[0])
                LOGGER.info(f"Inferred entry point for {server_name}: {module_path.name}")

    # 3. 执行解析
    if module_path.exists():
        try:
            # 动态生成配置，让前端能看到工具列表
            stub = _generate_server_stub(server_dir, module_path, parameter_path)
            if not stub.get("tools") and not stub.get("prompts"):
                LOGGER.warning(f"Server {server_name} parsed but no tools/prompts found in {module_path.name}")
            return stub
        except Exception as e:
            LOGGER.warning(f"Failed to generate stub for {server_name}: {e}")

    LOGGER.warning(f"Skipping {server_name}: No valid entry point or config found.")
    return {}

# Stub dependencies for imports
def _ensure_stub_module(name): 
    if name not in sys.modules:
        m = ModuleType(name)
        sys.modules[name] = m

def interrupt_chat(session_id: str):
    session = SESSION_MANAGER.get(session_id)
    if session:
        success = session.interrupt_task()
        return {"status": "interrupted", "active_task_cancelled": success}
    return {"status": "session_not_found"}

# Knowledge Base Management   
def load_kb_config() -> Dict[str, Any]:
    default_config = {
        "milvus": {
            "uri": str(KB_INDEX_DIR / "default.db"),
            "token": "",
            "id_field_name": "id",
            "vector_field_name": "vector",
            "text_field_name": "contents",
            "id_max_length": 64,
            "text_max_length": 60000,
            "metric_type": "IP",
            "index_params": {
                "index_type": "AUTOINDEX",
                "metric_type": "IP"
            },
            "search_params": {
                "metric_type": "IP",
                "params": {}
            },
            "index_chunk_size": 1000
        }
    }
    
    if not KB_CONFIG_PATH.exists():
        return default_config
        
    try:
        saved = json.loads(KB_CONFIG_PATH.read_text(encoding="utf-8"))
        if "milvus" not in saved: 
            return default_config
        
        full_cfg = default_config["milvus"].copy()
        full_cfg.update(saved["milvus"])
        return {"milvus": full_cfg}
        
    except Exception:
        return default_config
    
def save_kb_config(config: Dict[str, str]):
    KB_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

def _get_milvus_client():
    cfg = load_kb_config()
    try:
        milvus_cfg = cfg.get("milvus", {})
        uri = milvus_cfg.get("uri", "")
        
        if not uri: raise ValueError("URI is empty")

        if not uri.startswith("http") and not Path(uri).is_absolute():
            pass 
            
        return MilvusClient(uri=uri, token=milvus_cfg.get("token", ""))
    except Exception as e:
        LOGGER.error(f"Failed to connect to Milvus: {e}")
        raise e

def list_kb_files() -> Dict[str, List[Dict[str, Any]]]:
    def _scan_files(d: Path, category: str):
        return [
            {
                "name": f.name, "path": _as_project_relative(f), 
                "size": f.stat().st_size, "mtime": f.stat().st_mtime, 
                "category": category
            }
            for f in sorted(d.glob("*")) if f.is_file() and not f.name.startswith(".")
        ]
    
    collections = []
    db_status = "unknown"
    try:
        client = _get_milvus_client()
        names = client.list_collections()
        for name in names:
            res = client.get_collection_stats(name)
            count = res.get("row_count", 0)
            collections.append({
                "name": name,
                "count": count,
                "category": "collection"
            })
        client.close()
        db_status = "connected"
    except Exception as e:
        LOGGER.warning(f"Milvus connection failed: {e}")
        db_status = "error"

    return {
        "raw": _scan_files(KB_RAW_DIR, "raw"),
        "corpus": _scan_files(KB_CORPUS_DIR, "corpus"),
        "chunks": _scan_files(KB_CHUNKS_DIR, "chunks"),
        "index": collections,
        "db_status": db_status, 
        "db_config": load_kb_config() 
    }

def upload_kb_file(file_obj) -> Dict[str, Any]:
    from werkzeug.utils import secure_filename
    filename = secure_filename(file_obj.filename)
    save_path = KB_RAW_DIR / filename
    file_obj.save(save_path)
    LOGGER.info(f"File uploaded: {save_path}")
    return {
        "name": filename,
        "path": _as_project_relative(save_path),
        "size": save_path.stat().st_size
    }

def delete_kb_file(category: str, filename: str) -> Dict[str, str]:
    if category == "collection" or category == "index":
        try:
            client = _get_milvus_client()
            if client.has_collection(filename):
                client.drop_collection(filename)
            client.close()
            LOGGER.info(f"Collection dropped: {filename}")
            return {"status": "dropped"}
        except Exception as e:
            raise PipelineManagerError(f"Failed to drop collection: {e}")

    target_dir = {"raw": KB_RAW_DIR, "corpus": KB_CORPUS_DIR, "chunks": KB_CHUNKS_DIR}.get(category)
    if target_dir:
        file_path = target_dir / Path(filename).name
        if file_path.exists(): file_path.unlink()
        return {"status": "deleted"}
    
    raise PipelineManagerError("Invalid category")

def run_kb_pipeline_tool(
    pipeline_name: str, 
    target_file_path: str,
    output_dir: str,
    collection_name: Optional[str] = None,
    index_mode: str = "append", # 'append' (追加), 'overwrite' (覆盖)
) -> Dict[str, Any]:

    pipeline_cfg = load_pipeline(pipeline_name)
    if not pipeline_cfg:
        raise PipelineManagerError(f"Pipeline '{pipeline_name}' not found. Please create it in Builder.")
        
    base_params = load_parameters(pipeline_name) 
    if not base_params:
        raise PipelineManagerError(f"Pipeline '{pipeline_name}' parameters not found. Please Build & Save first.")
    
    target_file = Path(target_file_path)
    stem = target_file.stem
    override_params = {}
    
    if pipeline_name == "build_text_corpus":
        out_path = os.path.join(output_dir, f"{stem}.jsonl")
        override_params = {
            "corpus": {
                "parse_file_path": str(target_file),
                "text_corpus_save_path": out_path,
            }
        }

    elif pipeline_name == "corpus_chunk":
        out_path = os.path.join(output_dir, f"{stem}.jsonl")
        override_params = {
            "corpus": {
                "raw_chunk_path": str(target_file),
                "chunk_path": out_path
            }
        }
        
    elif pipeline_name == "milvus_index":
        final_collection_name = collection_name if collection_name else stem

        is_overwrite = (index_mode == "overwrite")

        full_kb_cfg = load_kb_config()
        milvus_config_dict = full_kb_cfg["milvus"] 
        
        # 确保父目录存在
        KB_INDEX_DIR.mkdir(parents=True, exist_ok=True)

        override_params = {
            "retriever": {
                "corpus_path": str(target_file),
                "collection_name": final_collection_name,
                "overwrite": is_overwrite,
                "index_backend": "milvus",
                "index_backend_configs": {
                    "milvus": milvus_config_dict
                }
            }
        }

    else:
        raise PipelineManagerError(f"Unsupported KB Pipeline: {pipeline_name}")

    funcs = _ensure_client_funcs()
    config_file = str(_find_pipeline_file(pipeline_name))
    param_file = str(_resolve_parameter_path(pipeline_name))

    async def _async_task():
        context = funcs["load_ctx"](config_file, param_file)
        
        client = funcs["create_client"](context["mcp_cfg"])
        
        async with client:
            return await funcs["exec_pipe"](
                client, 
                context, 
                is_demo=True, 
                return_all=True,
                override_params=override_params 
            )
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            raw_result = loop.run_until_complete(_async_task())
        finally:
            loop.close()
            
        return {"status": "success", "result": _extract_result(raw_result)}
        
    except Exception as e:
        LOGGER.error(f"KB Task Failed: {e}")
        raise e