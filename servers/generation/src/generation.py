import asyncio
import logging
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Union, Optional

from openai import AsyncOpenAI, AuthenticationError
from openai._utils._logs import httpx_logger
from tqdm import tqdm
import base64
import mimetypes

from fastmcp.exceptions import ToolError
from ultrarag.server import UltraRAG_MCP_Server

app = UltraRAG_MCP_Server("generation")
httpx_logger.setLevel(logging.WARNING)


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output."""
    stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(stdout_fd)

    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stdout_fd)
        os.close(devnull)
        yield
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.close(saved_stdout_fd)


def _suppress_vllm_logging():
    """Suppress verbose logging from vLLM and related libraries."""
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
    os.environ.setdefault("RAY_DEDUP_LOGS", "0")
    os.environ.setdefault("GLOO_LOG_LEVEL", "ERROR")
    os.environ.setdefault("NCCL_DEBUG", "ERROR")

    for name in ["vllm", "ray", "torch", "transformers"]:
        logging.getLogger(name).setLevel(logging.ERROR)


class Generation:
    def __init__(self, mcp_inst: UltraRAG_MCP_Server):
        mcp_inst.tool(
            self.generation_init,
            output="backend_configs,sampling_params,extra_params,backend->None",
        )
        mcp_inst.tool(
            self.generate,
            output="prompt_ls,system_prompt->ans_ls",
        )
        mcp_inst.tool(
            self.multimodal_generate,
            output="multimodal_path,prompt_ls,system_prompt,image_tag->ans_ls",
        )
        mcp_inst.tool(
            self.multiturn_generate,
            output="messages,system_prompt->ans_ls",
        )
        mcp_inst.tool(
            self.vllm_shutdown,
            output="->None",
        )

    def _drop_keys(self, d: Dict[str, Any], banned: List[str]) -> Dict[str, Any]:
        """Remove banned keys and None values from dictionary.

        Args:
            d: Dictionary to filter
            banned: List of keys to remove

        Returns:
            Filtered dictionary
        """
        return {k: v for k, v in (d or {}).items() if k not in banned and v is not None}

    def _extract_text_prompts(
        self, prompt_ls: List[Union[str, Dict[str, Any]]]
    ) -> List[str]:
        """Extract text content from various prompt formats.

        Args:
            prompt_ls: List of prompts in various formats (str, dict, etc.)

        Returns:
            List of text strings

        Raises:
            ValueError: If prompt format is not supported
        """
        prompts = []
        for m in prompt_ls:
            if hasattr(m, "content") and hasattr(m.content, "text"):
                prompts.append(m.content.text)
            elif isinstance(m, dict):
                if (
                    "content" in m
                    and isinstance(m["content"], dict)
                    and "text" in m["content"]
                ):
                    prompts.append(m["content"]["text"])
                elif "content" in m and isinstance(m["content"], str):
                    prompts.append(m["content"])
                elif "text" in m:
                    prompts.append(m["text"])
                else:
                    warn_msg = f"Unsupported dict prompt format: {m}"
                    app.logger.warning(warn_msg)
                    raise ValueError(warn_msg)
            elif isinstance(m, str):
                prompts.append(m)
            else:
                err_msg = f"Unsupported message format: {m}"
                app.logger.error(err_msg)
                raise ValueError(err_msg)
        return prompts

    def _to_data_url(self, path_or_url: str) -> str:
        """Convert image path to data URL.

        Args:
            path_or_url: Image file path or URL

        Returns:
            Data URL string (data:image/...;base64,...)

        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        s = str(path_or_url).strip()

        if s.startswith(("http://", "https://", "data:image/")):
            return s

        if not os.path.isfile(s):
            err_msg = f"image not found: {s}"
            app.logger.error(err_msg)
            raise FileNotFoundError(err_msg)
        mime, _ = mimetypes.guess_type(s)
        mime = mime or "image/jpeg"
        with open(s, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def generation_init(
        self,
        backend_configs: Dict[str, Any],
        sampling_params: Dict[str, Any],
        extra_params: Optional[Dict[str, Any]] = None,
        backend: str = "vllm",
    ) -> None:
        """Initialize generation backend (vllm, openai, or hf).

        Args:
            backend_configs: Dictionary of backend-specific configurations
            sampling_params: Sampling parameters for generation
            extra_params: Optional extra parameters (e.g., chat_template_kwargs)
            backend: Backend name ("vllm", "openai", or "hf")

        Raises:
            ImportError: If required dependencies are not installed
            ValueError: If backend is unsupported or required config is missing
        """
        self.backend = backend.lower()
        self.backend_configs = backend_configs or {}
        cfg: Dict[str, Any] = (self.backend_configs.get(self.backend) or {}).copy()

        if self.backend == "vllm":
            _suppress_vllm_logging()

            try:
                from vllm import LLM, SamplingParams
            except ImportError:
                err_msg = (
                    "vllm is not installed. Please install it with `pip install vllm`."
                )
                app.logger.error(err_msg)
                raise ImportError(err_msg)

            gpu_ids = str(cfg.get("gpu_ids"))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

            model_name_or_path = cfg.get("model_name_or_path")

            vllm_pass_cfg = self._drop_keys(
                cfg,
                banned=["gpu_ids", "model_name_or_path"],
            )

            vllm_pass_cfg["tensor_parallel_size"] = len(gpu_ids.split(","))

            if extra_params:
                self.chat_template_kwargs = extra_params.get("chat_template_kwargs", {})
            else:
                self.chat_template_kwargs = {}

            with suppress_stdout():
                self.model = LLM(model=model_name_or_path, **vllm_pass_cfg)
            self.sampling_params = SamplingParams(**sampling_params)

        elif self.backend == "openai":
            self.model_name = cfg.get("model_name")
            if not self.model_name:
                error_msg = "model_name is required for openai backend"
                app.logger.error(error_msg)
                raise ValueError(error_msg)

            base_url = cfg.get("base_url")
            if not base_url:
                base_url = "https://api.openai.com/v1"
                warn_msg = f"base_url is not set, default to {base_url}"
                app.logger.warning(warn_msg)

            api_key = cfg.get("api_key") or os.environ.get("LLM_API_KEY")
            if not api_key:
                api_key = "None"
                warn_msg = "api_key is not set, default to None"
                app.logger.warning(warn_msg)

            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

            if extra_params:
                sampling_params["extra_body"] = extra_params
            self.sampling_params = sampling_params

            self._max_concurrency = int(cfg.get("concurrency", 1))
            self._retries = int(cfg.get("retries", 3))
            self._base_delay = float(cfg.get("base_delay", 1.0))

        elif self.backend == "hf":
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
            except ImportError:
                err_msg = "transformers is not installed. Please install it with `pip install transformers`."
                app.logger.error(err_msg)
                raise ImportError(err_msg)

            gpu_ids = str(cfg.get("gpu_ids"))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

            model_name_or_path = cfg.get("model_name_or_path")
            hf_pass_cfg = self._drop_keys(
                cfg,
                banned=["gpu_ids", "model_name_or_path", "batch_size"],
            )
            if extra_params:
                self.chat_template_kwargs = extra_params.get("chat_template_kwargs", {})
            else:
                self.chat_template_kwargs = {}
            self.sampling_params = sampling_params
            self.batch_size = int(cfg.get("batch_size", 1))

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                **hf_pass_cfg,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                padding_side="left",
            )
            added_tokens = 0
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    added_tokens = self.tokenizer.add_special_tokens(
                        {"pad_token": "[PAD]"}
                    )

            if added_tokens > 0 and hasattr(self.model, "resize_token_embeddings"):
                self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            err_msg = f"Unsupported backend: {self.backend}"
            app.logger.error(err_msg)
            raise ValueError(err_msg)

    async def _generate(
        self,
        msg_ls: List[List[Dict[str, str]]],
    ) -> List[str]:
        """Internal method to generate responses from message lists.

        Args:
            msg_ls: List of message lists (one per request)

        Returns:
            List of generated text responses

        Raises:
            ValueError: If backend is unsupported
        """

        if self.backend == "vllm":
            with suppress_stdout():
                outputs = self.model.chat(
                    msg_ls,
                    self.sampling_params,
                    chat_template_kwargs=self.chat_template_kwargs,
                )
            ret = [o.outputs[0].text for o in outputs]

        elif self.backend == "openai":
            sem = asyncio.Semaphore(self._max_concurrency)

            async def call_with_retry(
                idx,
                msg,
                client,
                model_name,
                sampling_params,
                retries: int,
                base_delay: float,
            ):

                import random
                from openai import RateLimitError, APIStatusError

                delay = base_delay
                for attempt in range(retries):
                    try:
                        async with sem:
                            resp = await client.chat.completions.create(
                                model=model_name,
                                messages=msg,
                                **sampling_params,
                            )
                        return idx, (resp.choices[0].message.content or "")
                    except AuthenticationError as e:
                        error_msg = (
                            f"[{e.status_code}] Unauthorized: Access denied at {getattr(client, 'base_url', 'unknown')}."
                            " Invalid or missing LLM_API_KEY."
                        )
                        app.logger.error(error_msg)
                        raise ToolError(error_msg)
                    except RateLimitError as e:
                        warn_msg = f"[{e.status_code}] API Rate limited (idx={idx}, attempt={attempt+1}): {e}"
                        app.logger.warning(warn_msg)
                        raise ToolError(warn_msg)
                    except APIStatusError as e:
                        if e.status_code >= 500:
                            warn_msg = f"[{e.status_code}] Server error (idx={idx}, attempt={attempt+1}): {e}"
                            app.logger.warning(warn_msg)
                        else:
                            error_msg = f"[{e.status_code}] API error (idx={idx}, attempt={attempt+1}): {e}"
                            app.logger.error(error_msg)
                            raise ToolError(error_msg)
                    except Exception as e:
                        error_msg = f"[Retry {attempt+1}] Failed (idx={idx}): {e}"
                        app.logger.error(error_msg)
                        raise ToolError(error_msg)

                    await asyncio.sleep(delay + random.random() * 0.25)
                    delay *= 2

                return idx, "<error>"

            tasks = [
                asyncio.create_task(
                    call_with_retry(
                        idx,
                        msg,
                        self.client,
                        self.model_name,
                        self.sampling_params,
                        retries=getattr(self, "_retries", 3),
                        base_delay=getattr(self, "_base_delay", 1.0),
                    )
                )
                for idx, msg in enumerate(msg_ls)
            ]
            ret = [None] * len(msg_ls)

            for coro in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="OpenAI Generating: ",
            ):
                idx, ans = await coro
                ret[idx] = ans

        elif self.backend == "hf":
            prompt_txt_ls: List[str] = []
            for msg in msg_ls:
                prompt_txt = self.tokenizer.apply_chat_template(
                    msg,
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
                prompt_txt_ls.append(prompt_txt)

            device = self.model.device
            bs = self.batch_size

            ret: List[str] = []
            for i in tqdm(
                range(0, len(prompt_txt_ls), bs),
                desc="HF Generating",
            ):
                batch_prompts = prompt_txt_ls[i : i + bs]
                enc = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                generated = self.model.generate(
                    **enc,
                    use_cache=False,
                    **self.sampling_params,
                )

                input_lens = enc["attention_mask"].sum(dim=1).tolist()
                for row_idx, in_len in enumerate(input_lens):
                    out_ids = generated[row_idx, int(in_len) :].tolist()
                    text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
                    ret.append(text)

        else:
            err_msg = f"Unsupported backend: {self.backend}"
            app.logger.error(err_msg)
            raise ValueError(err_msg)

        return ret

    async def generate(
        self,
        prompt_ls: List[Union[str, Dict[str, Any]]],
        system_prompt: str = "",
    ) -> Dict[str, List[str]]:
        """Generate responses for a list of prompts.

        Args:
            prompt_ls: List of prompts (strings or dicts)
            system_prompt: Optional system prompt to prepend

        Returns:
            Dictionary with 'ans_ls' containing generated responses
        """
        system_prompt = str(system_prompt or "").strip()
        add_system = bool(system_prompt)
        prompts = [str(p).strip() for p in self._extract_text_prompts(prompt_ls)]
        if not prompts:
            info_msg = (
                "empty prompt list; return empty ans_ls."
                f"system_prompt={system_prompt}"
            )
            app.logger.info(info_msg)
            return {"ans_ls": []}

        system_msgs = (
            [{"role": "system", "content": system_prompt}] if add_system else []
        )
        msg_ls = [system_msgs + [{"role": "user", "content": p}] for p in prompts]
        ret = await self._generate(msg_ls)
        return {"ans_ls": ret}

    async def multiturn_generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
    ) -> Dict[str, List[str]]:
        """Generate response for multi-turn conversation.

        Args:
            messages: Conversation history list, each message format: {"role": "user"|"assistant", "content": "..."}
            system_prompt: Optional system prompt

        Returns:
            Dictionary with 'ans_ls' containing assistant response
        """
        system_prompt = str(system_prompt or "").strip()

        if not messages:
            app.logger.info("empty messages; return empty ans_ls.")
            return {"ans_ls": []}

        # Build complete message list
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant", "system") and content:
                full_messages.append({"role": role, "content": str(content)})

        if not full_messages or all(m["role"] == "system" for m in full_messages):
            app.logger.info("no valid user/assistant messages; return empty ans_ls.")
            return {"ans_ls": []}

        # Call generation (pass single message list)
        ret = await self._generate([full_messages])
        return {"ans_ls": ret}

    async def multimodal_generate(
        self,
        multimodal_path: List[List[str]],
        prompt_ls: List[Union[str, Dict[str, Any]]],
        system_prompt: str = "",
        image_tag: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """Generate responses with multimodal inputs (text + images).

        Args:
            multimodal_path: List of image path lists (one per prompt)
            prompt_ls: List of text prompts
            system_prompt: Optional system prompt
            image_tag: Optional tag to split prompt and insert images (e.g., "<image>")

        Returns:
            Dictionary with 'ans_ls' containing generated responses

        Raises:
            ValueError: If image tag count doesn't match image path count
        """
        system_prompt = str(system_prompt or "").strip()
        add_system = bool(system_prompt)
        prompts = [str(p).strip() for p in self._extract_text_prompts(prompt_ls)]
        if not prompts:
            info_msg = (
                "empty prompt list; return empty ans_ls."
                f"system_prompt={system_prompt}"
            )
            app.logger.info(info_msg)
            return {"ans_ls": []}

        paths: List[List[str]] = []
        mm_len = len(multimodal_path or [])
        if mm_len < len(prompts):
            warn_msg = (
                f"multimodal_path shorter than prompts: {mm_len} < {len(prompts)}; "
                "missing entries will be treated as empty."
            )
            app.logger.warning(warn_msg)
        elif mm_len > len(prompts):
            warn_msg = (
                f"multimodal_path longer than prompts: {mm_len} > {len(prompts)}; "
                "extra entries will be ignored."
            )
            app.logger.warning(warn_msg)

        for i in range(len(prompts)):
            entry = multimodal_path[i] if i < mm_len else []
            if isinstance(entry, (str, bytes)):
                entry = [str(entry)]
            elif not isinstance(entry, list):
                warn_msg = (
                    f"idx={i} path entry not list/str; "
                    "got {type(entry)}; "
                    "fallback to empty list."
                )
                app.logger.warning(warn_msg)
                entry = []
            paths.append([str(pth).strip() for pth in entry])

        msg_ls: List[List[Dict[str, Any]]] = []
        for i, (p, pths) in enumerate(zip(prompts, paths)):
            msgs: List[Dict[str, Any]] = []
            if add_system:
                msgs.append({"role": "system", "content": system_prompt})

            content: List[Dict[str, Any]] = []

            use_tag_mode = bool(image_tag) and bool(str(image_tag).strip())
            tag = str(image_tag).strip() if use_tag_mode else None

            if use_tag_mode:
                prompt_image_num = p.count(tag)
                actual_image_num = len(pths)
                if prompt_image_num != actual_image_num:
                    raise ValueError(
                        f"Number of ({tag}) image tag: ({prompt_image_num}) "
                        f"does not match number of image paths: ({actual_image_num})"
                    )

                parts = p.split(tag)
                for j, part in enumerate(parts):
                    if part.strip():
                        content.append({"type": "text", "text": part})
                    if j < actual_image_num:
                        mp = pths[j]
                        if not mp:
                            continue
                        try:
                            content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": self._to_data_url(mp)},
                                }
                            )
                        except Exception as e:
                            app.logger.warning(
                                f"[Image skip] idx={j}, path={mp}, err={e}"
                            )
            else:
                for mp in pths:
                    if not mp:
                        continue
                    try:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": self._to_data_url(mp)},
                            }
                        )
                    except Exception as e:
                        app.logger.warning(f"[Image skip] idx={i}, path={mp}, err={e}")
                content.append({"type": "text", "text": p})

            msgs.append({"role": "user", "content": content})
            msg_ls.append(msgs)

        ret = await self._generate(msg_ls)
        return {"ans_ls": ret}

    def vllm_shutdown(self) -> None:
        """Shutdown vLLM model and clean up resources.

        This method attempts to properly shutdown the vLLM model and free GPU memory.
        """
        try:
            if getattr(self, "backend", None) != "vllm":
                app.logger.info("[vllm_shutdown] backend is not 'vllm'; skip.")
                return

            if getattr(self, "model", None) is None:
                app.logger.info("[vllm_shutdown] model is None; nothing to do.")
                return

            with suppress_stdout():
                fn = getattr(self.model, "shutdown", None)
                if callable(fn):
                    app.logger.info("[vllm_shutdown] calling self.model.shutdown()")
                    fn()
                else:
                    for path in ("llm_engine", "engine"):
                        eng = getattr(self.model, path, None)
                        if eng:
                            for attr in ("shutdown", "close", "terminate"):
                                f = getattr(eng, attr, None)
                                if callable(f):
                                    app.logger.info(
                                        f"[vllm_shutdown] calling self.model.{path}.{attr}()"
                                    )
                                    f()
                                    break

                self.model = None
                import gc, torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            app.logger.info("[vllm_shutdown] complete")

        except Exception as e:
            app.logger.warning(f"[vllm_shutdown] cleanup warning: {e}")


if __name__ == "__main__":
    Generation(app)
    app.run(transport="stdio")
