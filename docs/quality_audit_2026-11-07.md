# Architecture and quality audit — 2026-11-07

Baseline: `a9c82a1` on `main`. Implementation: `codex/core-ui-quality`, PR #567.

## Assessment

The orchestration-first architecture is appropriate: YAML describes data flow, registrations define tool/prompt contracts, and execution clients own transport lifecycles. Direct execution should preserve those contracts without inheriting the process isolation assumptions of MCP subprocesses.

The main weaknesses found were shared state leaking across those boundaries, duplicated configuration/startup handling, failure paths reporting success, unowned concurrent requests, and frontend state that outlived its chat turn. The fixes below introduce small, testable ownership boundaries while preserving public CLI commands, YAML, generated configuration formats, and MCP/local execution parity.

| Boundary | Owner after this change | Responsibility |
| --- | --- | --- |
| YAML documents | `configuration.py` | Mapping validation, optional parameter loading, atomic writes |
| Pipeline data and step ordering | `client.py` / `UltraData` | Remapping, branches, loops, snapshots |
| Transport and registration lifetimes | `execution.py` | Local imports, mixed MCP routing, nested contexts, lifespan cleanup |
| Registration metadata | `registration.py`, local/MCP servers | Shared I/O rules and generated metadata |
| Diagnostics | `mcp_logging.py` | Application verbosity, inherited server loggers, credential redaction |
| Concurrent provider requests | `async_utils.py` | Cancel and drain siblings owned by a failed/cancelled step |
| Foreground chat ownership | `ChatTurnController.ts` | Reject stale events; keep a stopped turn owned until server stop settles |
| Rendering and preferences | Markdown components / storage helpers | Lazy, memoized rendering; optional browser persistence |

`client.py`, `pipeline_manager.py`, and the page components still coordinate substantial domain behavior. Their size is a maintainability observation, not evidence that the orchestration model is incorrect. This change extracts the boundaries that caused demonstrated defects rather than mechanically moving intertwined branches into new files.

## Defects and changes

| Area | Reproduced behavior | Result |
| --- | --- | --- |
| Logging | Constructing a local server reset shared verbosity to `warn`; CLI verbosity also has to propagate to MCP subprocesses | Default server construction preserves application/handler levels; explicit server filtering uses a child logger; the historical `log_level` environment key reaches child processes |
| Diagnostics | Debug configuration/arguments and dynamic overrides could expose credentials or the entire inherited environment | Credential-bearing fields are redacted; startup logs list server names rather than environment dictionaries |
| Configuration | Empty YAML returned `None`; scalar/list YAML failed later with unrelated errors | Empty documents become `{}`; other non-mappings fail with a file-specific error |
| Generated files | Interrupted serialization could leave truncated YAML | Temporary files are serialized and atomically replaced; failed serialization preserves the previous file |
| Startup | Build/run duplicated server validation; missing generated servers were silently omitted; library code exited the process for missing Node | Shared validation, actionable rebuild hints, recoverable typed errors |
| Local lifecycle | Nested client contexts reloaded and prematurely discarded registrations; partial startup cleanup lost the triggering exception | Nested contexts retain registrations; entered lifespans receive startup failures |
| Loop state | Nested execution could overwrite its caller's `ContextVar` state | Every pipeline restores the previous context, including on failure |
| Generation streaming | A stream exception could be logged and converted into a successful partial answer | Failure propagates; cleanup still runs |
| API batches | Failed generation/embedding requests left siblings active; exhausted retries returned `"<error>"` as answer text | Siblings are cancelled/drained, exhaustion raises a tool error, no unnecessary final retry sleep |
| Provider initialization | Generation initialization mutated the caller's sampling dictionary; zero concurrency could hang | Sampling configuration is copied; concurrency/attempt counts are positive |
| Subprocesses / CLI | `prctl` was attempted on macOS; `preexec_fn` could deadlock in threaded hosts; large logos wrapped on narrow terminals | Linux setup runs in a fresh interpreter; other platforms use their supported path; narrow banners stay readable |
| SSE | Parsing depended on packet boundaries and LF-only separators | Incremental UTF-8 and line parsing supports CR/LF/CRLF, mixed endings, multiline data, malformed-frame recovery, final frames, and reader cleanup |
| Chat state | An old turn could write into a newly opened chat; cancellation could display an error or race the next turn | Turn ownership gates updates; new/open/clear/delete/logout paths stop active work; stale history responses are ignored |
| Request recovery | A stalled request could hold the UI indefinitely; `Headers` arguments were not merged correctly | JSON requests have bounded timeouts and caller cancellation; stop requests time out after 15 seconds and discard an uncertain cached engine |
| Settings | Clearing model settings left old values visible; rejected actions could create unhandled promises | Successful clearing resets visible fields, keys are masked, and async action errors are handled |
| Responsive navigation | Sidebar hid at 992px while its launcher disappeared at 768px | One consistent breakpoint; keyboard focus/escape handling; hidden sidebars and panels are inert |
| Builder | Mobile canvas widths/heights clipped editors and controls | Stacked, scrollable canvas/YAML/parameter/prompt panels; bounded assistant drawer; labelled controls and accessible contrast |
| Rendering | Every page and all highlighter grammars loaded at startup; completed answers were reparsed during unrelated changes | Routes and rich Markdown load on demand, common grammars replace the full bundle, completed Markdown is memoized, duplicate CSS/fonts are removed |
| Static delivery | Bundled assets were uncompressed and hashed assets were forced to bypass cache | Negotiated gzip and immutable hashed assets; entry/theme files revalidate; API/SSE/range responses retain their behavior |

## Verification

### Before/after and contract tests

- Baseline checkout: 96 existing Python tests passed, 1 skipped with FAISS available. The new logging regressions fail against the original implementation and pass after the fix.
- Updated checkout: 144 Python tests passed, 1 skipped with FAISS available. The suite includes all four CLI logging levels in both transports, real stdio/local parity, remapping, loops, snapshots, independent sessions/imports, shutdown, partial startup, stream failure, failed API batches, configuration writes, static negotiation, and subprocess portability.
- Core Ruff lint and formatting pass. All 79 repository Python source files were parsed successfully.
- Frontend lint, TypeScript, production build, and 15 behavioral tests pass. Frontend tests cover SSE framing, UTF-8, cancellation, turn races, stop failures, blocked storage, HTTP deadlines, cancellation reasons, headers, and error status preservation.
- CI now runs those frontend checks and the generation/retriever regression suites, alongside the existing Python 3.11/3.12, CPU image, dependency-resolution, and CodeQL checks.

### Live API parity

The opt-in `script/quality_api_smoke.py` reads credentials only from environment variables. It creates a temporary three-passage corpus, embeds it, builds a CPU FAISS index, retrieves two capital-city questions, and generates answers. It writes no credential-bearing parameter files.

| Path | Generation / embedding | Verified result |
| --- | --- | --- |
| Original main, stdio MCP and local | `qwen3.8-27b` / `qwen3-embedding-0.6b` | Correct rankings; `Paris`, `Beijing`; finite `3 × 1024` embeddings |
| Updated code and security-updated dependencies, stdio MCP and local | Same models through the OpenAI backend | Rankings, answers, embedding shapes, and finite-value checks match the baseline |
| Updated code, stdio MCP and local | Same endpoint/models through LiteLLM | Correct rankings, answers, and embedding shapes |

The first code-only before/after run also matched raw embedding hashes within each transport. Later security-update and LiteLLM runs returned different hashes, including within the same transport; exact numerical parity is therefore not established. The artifacts retain these hashes rather than suppressing the variation. Retrieval rankings, answers, finite-value checks, and embedding shapes agree across every recorded run and tested mode.

Example, after setting the two environment variables without saving credentials to source:

```bash
uv run --extra mcp python script/quality_api_smoke.py \
  --model YOUR_GENERATION_MODEL --embedding-model YOUR_EMBEDDING_MODEL \
  --output /tmp/openai-parity.json

uv run --extra mcp --extra litellm python script/quality_api_smoke.py \
  --backend litellm --model YOUR_GENERATION_MODEL \
  --embedding-model YOUR_EMBEDDING_MODEL --output /tmp/litellm-parity.json
```

The live audit additionally needs the optional CPU FAISS dependency. Temporary test model settings were cleared after verification.

### Browser and device checks

Both pages were checked at `320×568`, `390×844`, `768×1024`, `820×1180`, `992×900`, `1280×800`, `1920×1080`, and landscape `844×390`. Page width matched viewport width; editor/input/send/toolbar bounds remained reachable.

Observed interactions include sidebar open/close and focus restoration, Escape, language switching and document language, multiline composition, pipeline selection, all builder panels and the assistant drawer, rich tables/code/math, real streaming, starting a new chat during generation, subsequent generation, and clearing model settings. Blocking a lazy builder asset produced the recovery screen; unblocking and reloading restored the builder. Browser emulation/blocking was reset afterward.

### Lighthouse

Lighthouse 13.6.0, local production bundle served by Flask, default simulated mobile (`412×823`) and desktop presets, fresh report runs. These are lab page-load measurements, not server concurrency or sustained-load claims.

| Page / device | Baseline P / A / BP / SEO | Updated P / A / BP / SEO |
| --- | --- | --- |
| Chat / mobile | 55 / 96 / 100 / 91 | 90 / 100 / 100 / 100 |
| Chat / desktop | 74 / 96 / 100 / 91 | 100 / 100 / 100 / 100 |
| Builder / mobile | Not recorded as an initial baseline | 90 / 100 / 100 / 100 |
| Builder / desktop | Not recorded as an initial baseline | 100 / 100 / 100 / 100 |

The table uses the final release runs. An earlier updated chat/mobile run scored 100; the final run scored 90 (LCP 3.6 seconds), so the best run is not presented as a guaranteed result. Both sets of reports are retained for comparison.

The initial monolithic JavaScript asset was about 1.85 MB (578 KB gzip). The new bootstrap is about 259 KB (82 KB gzip), with separate chat, builder, shared, and Markdown chunks. Chat's initial application chunks total approximately 466 KB before gzip; rich Markdown is deferred until used. The original theme and visual treatment are preserved.

Raw HTML/JSON reports, screenshots, and API parity results are retained as local review artifacts rather than committed runtime data. Implementation references: [React lazy loading](https://react.dev/reference/react/lazy), [SSE framing](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events), and [Lighthouse scoring](https://developer.chrome.com/docs/lighthouse/performance/performance-scoring).

## Dependency review and remaining limits

The initial GitHub alert snapshot contained 189 open entries, including 8 critical entries. Compatible updates include DOMPurify 3.4.17, Pillow 13, Authlib 1.9.5, PyJWT 2.15.1, current HTTP/notebook/PDF dependencies, and FileLock 4.2.3. A scoped minimatch 11 override fixes the remaining TypeScript ESLint development dependency advisory; frontend checks pass with it.

- `npm audit` reports zero vulnerabilities for the updated frontend dependency tree.
- Comparing the original Python advisory ranges to the updated lockfile leaves zero critical matches and 46 matches in optional packages: vLLM (36), Transformers (5), setuptools (2), Torch (1), Accelerate (1), and DiskCache (1).
- Those remaining affected packages are absent from the tested core/API/MCP environment. They belong to the GPU/local-model or optional parsing stack; the project still retains its existing vLLM 0.22.1 / CUDA 12.9 / Torch 2.11 compatibility selection. This is not a claim that the complete all-extras environment is vulnerability-free.
- A migration of that coupled GPU stack requires GPU/model validation. It was not replaced with an untested major backend upgrade during this API/UI change. CUDA inference, large-corpus quality, multi-user concurrency, physical-device keyboards, and long-duration load were not validated here.

The security source snapshot is distinct from GitHub's post-merge rescan; no alert is claimed closed solely because a version changed in this branch. Relevant upstream release notes: [DOMPurify](https://github.com/cure53/DOMPurify/releases/tag/3.4.17), [Pillow 13](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html), and [Authlib](https://docs.authlib.org/en/stable/upgrades/changelog.html).
