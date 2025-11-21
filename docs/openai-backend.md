# OpenAI Backend Usage

This document describes how to use UltraRAG's **OpenAI** backend for generation (and, by analogy, for OpenAI embeddings in the retriever).

The goal is to make OpenAI models work out of the box, including newer reasoning models that restrict which sampling parameters are allowed.

## 1. Prerequisites

1. Create and activate the UltraRAG environment (conda or uv), then install UltraRAG:

```bash
conda create -n ultrarag python=3.11
conda activate ultrarag
pip install -e .
```

2. Set your OpenAI API key via environment variables (preferably using a `.env` / secrets file that is **not** committed to git):

```bash
export LLM_API_KEY="sk-..."        # used by the generation server
export RETRIEVER_API_KEY="sk-..."  # optional, for OpenAI embeddings
```

UltraRAG's OpenAI generation backend reads:

```python
api_key = cfg.get("api_key") or os.environ.get("LLM_API_KEY")
```

so leaving `api_key` empty in YAML forces it to use `LLM_API_KEY`.

## 2. Configuring the generation server for OpenAI

Edit `servers/generation/parameter.yaml` and set the backend to `openai`:

```yaml
backend: openai  # options: vllm, openai, hf

backend_configs:
  openai:
    model_name: gpt-4.1-mini        # or o3-mini, gpt-4o, etc.
    base_url: https://api.openai.com/v1
    api_key: ""                    # leave empty to read LLM_API_KEY
    concurrency: 8                  # maximum in-flight requests
    retries: 3                      # retry count on transient errors
    base_delay: 1.0                 # base backoff (seconds)

sampling_params:
  temperature: 1
  max_completion_tokens: 2048
  # Note: top_p is intentionally omitted; see section 3 below.

system_prompt: ""
image_tag: null
```

With this configuration, UltraRAG will send Chat Completions requests to the OpenAI API using the chosen `model_name`.

## 3. Parameter compatibility and automatic pruning

Different OpenAI models support different subsets of sampling parameters. In particular, some newer reasoning models (for example, **`o3-mini`**) do **not** support `top_p` or other fine-grained sampling controls, and will return errors like:

```text
Unsupported parameter: 'top_p' is not supported with this model.
```

To avoid breaking pipelines for these models, UltraRAG's OpenAI backend does two things:

1. **Never sends `chat_template_kwargs` to OpenAI**  
   `chat_template_kwargs` is only meaningful for local backends such as `vllm` / `hf`. For `backend == "openai"`, UltraRAG sets:

   ```python
   self.chat_template_kwargs = {}
   ```

   and drops `chat_template_kwargs` from the sampling parameters entirely.

2. **Drops certain sampling parameters that are known to trigger `unsupported_parameter` errors**  
   For OpenAI backends, UltraRAG currently filters out `top_p` and `top_k` before calling `client.chat.completions.create`, because models like `o3-mini` reject them. This means:

   - You can keep `top_p` / `top_k` in your YAML `sampling_params` without the pipeline failing for models that do not support them.
   - For models that *do* support these parameters, the current behavior is conservative: they are not sent. If you need full control over such parameters for specific models, you may need to adjust the code or run a custom fork.

Additionally, for backward compatibility, if a config still uses `max_tokens`, UltraRAG will log a warning and internally map it to `max_completion_tokens` before calling the OpenAI API.

### Example of a 400 error avoided by this behavior

Without the filtering, a call to a reasoning model like `o3-mini` with `top_p` set could yield:

```text
BadRequestError: Error code: 400 - {'error': {'message': "Unsupported parameter: 'top_p' is not supported with this model.",
                                             'type': 'invalid_request_error',
                                             'param': 'top_p',
                                             'code': 'unsupported_parameter'}}
```

By dropping `top_p` for the OpenAI backend, UltraRAG prevents this error and allows the pipeline to complete.

## 4. Using OpenAI embeddings in the retriever (optional)

UltraRAG can also use OpenAI embeddings as a retriever backend. A typical configuration in `servers/retriever/parameter.yaml` looks like:

```yaml
backend: openai  # options: infinity, sentence_transformers, openai, bm25

backend_configs:
  openai:
    model_name: text-embedding-3-small
    base_url: https://api.openai.com/v1
    api_key: ""  # leave empty to use RETRIEVER_API_KEY
```

Set `RETRIEVER_API_KEY` in your environment (or reuse `LLM_API_KEY`) and rebuild the pipeline:

```bash
ultrarag build examples/rag.yaml
ultrarag run examples/rag.yaml
```

This offloads embedding computation to OpenAI instead of using a local embedding model.

## 5. Quick checklist for OpenAI-only setups

1. **Environment**

   ```bash
   conda activate ultrarag
   export LLM_API_KEY="sk-..."        # required for OpenAI generation
   export RETRIEVER_API_KEY="sk-..."  # only if using OpenAI embeddings
   ```

2. **Configure backends**

   - `servers/generation/parameter.yaml`: `backend: openai` and a valid `model_name` (e.g., `gpt-4.1-mini`, `o3-mini`).
   - Optionally `servers/retriever/parameter.yaml`: `backend: openai` with `text-embedding-3-small`.

3. **Build and run a pipeline**

   ```bash
   ultrarag build examples/rag.yaml
   ultrarag run examples/rag.yaml
   ```

4. **If you see 400 errors from the OpenAI API**

   - Check that your API key and base URL are correct.
   - Ensure you are not adding custom unsupported parameters via other code.
   - For reasoning models like `o3-mini`, remember that UltraRAG already strips `top_p` and `top_k` on your behalf; if you reintroduce them via custom code, you may see `unsupported_parameter` errors again.