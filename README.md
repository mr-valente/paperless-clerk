# Paperless Clerk

Paperless Clerk is a focused local-AI sidecar for
[Paperless-ngx](https://docs.paperless-ngx.com/). It reads documents page by
page, retains existing OCR as a Paperless file-version backup, publishes Clerk
OCR as the default version, and classifies each document against the metadata
system already living in Paperless.

Its filing rule is simple:

> **reuse → normalize → extend only when necessary**

Clerk supports OpenAI-compatible local endpoints only. OCR/vision and metadata
share one endpoint while retaining independent model selection, so a vision
model can transcribe pages while a smaller text model handles classification.

## What Clerk does

- OCRs every page independently with bounded concurrency and persisted page
  results.
- Writes complete OCR into Paperless only after every page succeeds.
- When meaningful OCR already exists, preserves its complete file and text as a
  `Pre-Clerk OCR backup` version and creates a latest `Paperless Clerk OCR`
  version containing Clerk's text.
- Persists and polls Paperless's asynchronous version task so retries resume the
  same upload instead of creating another version.
- Classifies correspondents, document types, tags, titles, intrinsic dates, and
  existing custom-field values with structured model outputs.
- Retrieves the live Paperless vocabulary before every metadata run, validates
  all reused IDs, normalizes aliases and near-duplicates, and permits carefully
  justified vocabulary growth.
- Uses page-aware map/reduce metadata analysis, so a 100-page PDF is never sent
  to a model as one request.
- Keeps durable SQLite jobs, retries, conflicts, decision history, and bounded
  model-validation diagnostics.
- Offers a responsive local UI for status, intervention, decisions, connection
  settings, and manual processing.

Clerk does not replace Paperless storage, rewrite source files, provide chat or
RAG, connect to hosted AI services, or mirror the Paperless document library.
The version workflow requires Paperless-ngx 3.0 or newer.

## Quick start with Docker Compose

1. Create a Paperless API token for the user Clerk should act as.
2. Find the Docker network containing the Paperless webserver (commonly
   `<compose-project>_default`).
3. Copy the examples and set the token and model names:

   ```bash
   cp .env.example .env
   cp compose.example.yml compose.yml
   docker compose up -d --build
   ```

4. Open `http://localhost:8080`, test all three connections in **Settings**, and
   process a document ID manually.
5. Enable automatic discovery only after the first reviewed run. For an
   explicit opt-in queue, create a Paperless tag such as `clerk` and configure
   it as Clerk's queue/watch tag.

If Clerk is added directly to the existing Paperless Compose file, use the
Paperless service name in `PAPERLESS_URL` and remove the external-network block.
On Linux, `extra_hosts: [host.docker.internal:host-gateway]` lets the container
reach a model server on the Docker host. The model server must listen on an
address accessible to Docker, not only `127.0.0.1`.

Only `/app/data` is persistent. It contains `clerk.db` with job state, page OCR
needed for resume/conflict review, settings, and decision history. Source PDFs
and rendered page images use temporary storage and are removed after each run.
The container runs as root so it can use ordinary named volumes and bind mounts
without additional UID/GID configuration.

## First-run behavior

Automatic processing is off by default. Connecting Clerk therefore does not
silently rewrite an existing archive.

For a manual or discovered document:

1. Clerk streams the archived Paperless file to a temporary path and hashes it.
2. It renders one page at a time and holds at most the configured number of page
   images while local OCR calls run.
3. Completed pages are committed immediately to SQLite. A process restart or
   retry skips those pages if the source has not changed.
4. If Paperless has no meaningful content, Clerk patches the complete OCR output
   on the current version.
5. If content exists, Clerk uploads the unchanged current file as a new version,
   waits for Paperless consumption to complete, labels the prior version as a
   backup when it had no label, and patches Clerk OCR onto the explicit new
   version. That version is latest, so Paperless search and content use it by
   default while the original reading remains available in version history.
6. Metadata text is split at page/paragraph boundaries. Map calls extract
   compact candidates; hierarchical reduce calls operate on those candidates,
   never on the whole source document.
7. If an otherwise untagged document has no usable general-pass tag, Clerk runs
   one bounded, tag-only second review against the canonical candidates. If it
   returns empty while vocabulary growth is enabled, Clerk challenges that
   abstention once with a stricter form-versus-subject audit. A genuine
   abstention is then recorded explicitly as **No tags** rather than hidden or
   replaced with a speculative label.
8. Clerk fetches the current vocabulary, validates every ID and proposed value,
   applies one conservative Paperless patch, and records what was reused,
   created, rejected, or withheld.
9. If the document carried the configured queue/watch tag, Clerk removes it in
   that successful metadata patch.

The default metadata policy is **preserve existing values**. Tags are additive;
single-value metadata is filled when absent; a non-generic title and an
intrinsic date that differs from the ingestion date are retained. The UI can
enable confident replacement when desired.

## Automatic discovery and the queue tag

**Automatically process Paperless documents** enables Clerk's poller. It runs
after the setting is enabled and then at `CLERK_AUTOMATION_INTERVAL_SECONDS`.
Each poll reads a bounded page ordered by Paperless's most recently modified
documents and enqueues a full OCR-plus-metadata job when a document is new to
Clerk or its Paperless `modified` timestamp has changed. Active jobs are
deduplicated. Polls alternate between the newest page and later backlog pages,
so “recent” describes prioritization—not a permanent age cutoff.

The queue/watch tag changes which documents are eligible:

- With `CLERK_AUTOMATION_TAG` blank, all documents are eligible. Clerk will
  gradually work through the existing library while continuing to prioritize
  new and changed documents.
- With a tag configured, only documents carrying the exact Paperless tag are
  eligible. Create the tag in Paperless first, then add it to a document to
  place that document in Clerk's queue.
- The tag is workflow state, not document metadata. Clerk removes it from the
  current tag IDs and controlled vocabulary shown to the LLM, and rejects any
  model attempt to propose it as a new tag.
- Clerk keeps the tag while work is queued, running, retrying, failed, or
  waiting on an OCR conflict. It is removed only after metadata processing and
  its Paperless patch succeed. Resolving an OCR conflict queues that final
  metadata stage, which consumes the tag on success.
- A successful manual full or metadata-only run also consumes the configured
  tag. An OCR-only run leaves it in place because filing is not yet complete.

This makes the tag a dependable inbox marker: disappearance means Clerk
finished the filing decision, while a retained tag remains visible for retry or
intervention.

## Model endpoint contract

The shared endpoint must implement `POST /chat/completions` in the OpenAI chat
shape. Clerk sends vision pages as base64 `image_url` content and requests JSON
Schema output for metadata. If a local server rejects `json_schema`, Clerk
falls back to `json_object`; the result is still validated locally before any
Paperless write.

### OCR request profiles

A general vision-language model accepts Clerk's normal system instruction and
detailed transcription prompt. A specialist OCR model does not: it is trained
against one short task command, and its serving stack may need request fields
of its own. Every profile is defined in one place,
[`ocr_profiles.py`](src/paperless_clerk/ocr_profiles.py), and selected under
**Settings → Vision OCR**:

| Profile | Prompt | Extra request fields | |
| --- | --- | --- | --- |
| `generic` | Clerk's full transcription instruction, with a system prompt | none | offered |
| `deepseek_ocr_llamacpp` (GGUF) | `Free OCR.` | `top_k: 1` | offered |
| `deepseek_ocr` (vLLM) | `Free OCR.` | `skip_special_tokens: false` + `vllm_xargs` for the n-gram repetition guard | held back |
| `glm_ocr` (vLLM) | `Text Recognition:` | none | held back |

Every profile sends one image-first user message at temperature 0. Specialist
profiles send no system prompt and no page-number prefix. Clerk never sends a
chat template — the server owns that.

Response scaffolding is removed before publishing: reasoning blocks, code
fences, DeepSeek's `<|ref|>`/`<|det|>` layout annotations, and any remaining
control tokens. Transcribed text itself is never rewritten.

#### The vLLM profiles are held back

Both vLLM profiles pass **Test OCR model** and then fail on real pages: they
transcribe correctly for a while, fall into a decoder loop, and run to the
output limit. It reproduces on a one-page letter with both models, on
DeepSeek-OCR-2 even with the n-gram logits processor registered and ~3,000
tokens of context to spare. That is a serving bug in vLLM's OCR paths, not a
request Clerk can reshape, so neither profile appears in **Settings → Vision
OCR**.

Set `CLERK_ENABLE_VLLM_PROFILES=1` to put them back in the list and retest once
vLLM ships a fix. A database that still names a held-back profile keeps loading;
the run falls back to `generic` and logs a warning rather than failing to start.

**Use `deepseek_ocr_llamacpp` for specialist OCR** — the DeepSeek-OCR-2 GGUF
build under llama.cpp is the known-good path and needs no server flags. For
everything else use `generic` with a capable general vision model.

The profile is part of the OCR fingerprint, so a job retried after any request
change re-runs its pages instead of mixing two contracts. Clerk records the
effective model, profile, prompt, extra request fields, and render settings in
each job's event history, which makes a working configuration recoverable
without digging through container logs.

**Test OCR model** renders a portrait page and sends it through the real
production request, so it verifies the vision path and the selected profile
rather than plain text chat. It returns the transcription for you to judge, and
fails only when none of the page's known text comes back.

#### The `generic` profile

This is the default and, with the vLLM paths held back, the profile most setups
will use. It targets an instruction-following vision model rather than a
specialist OCR checkpoint — anything in the Qwen3-VL, InternVL, or Gemma vision
families, at 7B and up. Smaller models transcribe short pages acceptably but
start paraphrasing dense ones.

The prompt is written against the specific ways a chat model spoils a document
pipeline, and each instruction is load-bearing:

- **No preamble, no code fence.** "Here is the transcription of the document:"
  becomes the first line of your Paperless content field.
- **Never redact.** Vision models routinely mask account numbers, addresses, and
  dates of birth on financial and medical scans. A redacted transcription is
  worse than none, because it looks like a successful read.
- **Transcribe, don't describe.** Without this, a page with a logo or a chart
  comes back narrated instead of read.
- **`[blank page]` for an empty page.** An empty reply is treated as a failed
  page and takes the whole document with it.
- **`[x]`/`[ ]` for checkboxes, and keep the line breaks** of addresses and
  labelled fields, so the metadata stage can still see the page's structure.

The profile sends no server-specific fields at all, so it works against any
OpenAI-compatible endpoint. It sends `temperature: 0` and nothing else:
repetition penalties are deliberately absent, because they damage the legitimate
repetition in tables, dot leaders, and repeated form labels.

#### Lemonade recipes

Clerk talks to [Lemonade](https://lemonade-server.ai/docs/) over its
OpenAI-compatible endpoint. It does rewrite parts of the request — `max_tokens`
is clamped to fit `ctx_size` — and whether a profile's `vllm_xargs` survive the
hop to vLLM is unconfirmed. Model names below are Lemonade registration IDs, not
Hugging Face IDs; confirm with `lemonade list`.

Per-model server flags live in Lemonade's `recipe_options.json` (in its cache
directory). Write them with `lemonade load … --vllm-args "…" --save-options`,
which keeps each model's flags to that model and away from your metadata model.

The next two recipes drive the held-back vLLM profiles, and are kept here for
when they can be retested. Both need `CLERK_ENABLE_VLLM_PROFILES=1`.

**DeepSeek-OCR-2 via vLLM.** vLLM's DeepSeek-OCR recipe is a *matched pair*: the
server must register the n-gram logits processor and each request must supply
its window. Clerk sends the request half; the server half is this `--vllm-args`
string:

```sh
lemonade backends install vllm:rocm
lemonade pull user.DeepSeek-OCR-2 \
  --checkpoint main deepseek-ai/DeepSeek-OCR-2 \
  --recipe vllm \
  --label vision
lemonade load user.DeepSeek-OCR-2 --vllm rocm --ctx-size 8192 --save-options \
  --vllm-args "--limit-mm-per-prompt.image 1 --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor --mm-processor-cache-gb 0"
```

which is stored as:

```json
"user.DeepSeek-OCR-2": {
  "ctx_size": 8192,
  "vllm_args": "--limit-mm-per-prompt.image 1 --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor --mm-processor-cache-gb 0",
  "vllm_backend": "rocm"
}
```

```env
CLERK_OPENAI_BASE_URL=http://host.docker.internal:13305/v1
CLERK_OCR_MODEL=user.DeepSeek-OCR-2
CLERK_OCR_PROFILE=deepseek_ocr
CLERK_OCR_MAX_OUTPUT_TOKENS=4096
CLERK_ENABLE_VLLM_PROFILES=1
```

This needs vLLM 0.12.0 or newer. Appending `--no-enable-prefix-caching` is an
optional throughput tweak: page images are never repeated, so the prefix hashing
Lemonade enables by default earns nothing here.

Missing the server half guarantees a loop, but supplying it does not prevent
one: this recipe still loops on a real page with both halves in place. Whether
Lemonade forwards the request half's `vllm_xargs` to vLLM at all is the open
question, and posting the same request straight to the `backend_url` reported by
`GET /v1/health` is the way to settle it.

**DeepSeek-OCR-2 GGUF via llama.cpp** is the known-good path and needs no server
flags. Use `CLERK_OCR_PROFILE=deepseek_ocr_llamacpp` with the
`sabafallah/DeepSeek-OCR-2-GGUF` build. GLM-OCR has no working llama.cpp path;
serve it through vLLM.

**GLM-OCR via vLLM** needs no logits processor and no extra request fields:

```sh
lemonade pull user.GLM-OCR \
  --checkpoint main zai-org/GLM-OCR \
  --recipe vllm \
  --label vision
lemonade load user.GLM-OCR --vllm rocm --ctx-size 16384 --save-options \
  --vllm-args "--limit-mm-per-prompt.image 1 --limit-mm-per-prompt.video 0"
```

```env
CLERK_OPENAI_BASE_URL=http://host.docker.internal:13305/v1
CLERK_OCR_MODEL=user.GLM-OCR
CLERK_OCR_PROFILE=glm_ocr
CLERK_OCR_MAX_OUTPUT_TOKENS=8192
CLERK_RENDER_DPI=200
CLERK_ENABLE_VLLM_PROFILES=1
```

GLM-OCR needs a recent vLLM and Transformers. If loading reports an unsupported
architecture, update Lemonade and reinstall `vllm:rocm` — older bundles predate
native `DeepseekOCR2ForCausalLM` and `GlmOcrForConditionalGeneration` support.

GLM-OCR has no logits processor to lean on, and its repetition loop under vLLM
is reported upstream. Adding `--max-num-batched-tokens 32768` to `--vllm-args`
is the circulating workaround — it is not one of Lemonade's protected flags, so
it is settable — but it reportedly trades the loop for truncated responses.

#### Sizing the context

Only two numbers matter, and Clerk owns just one of them. Lemonade's `ctx_size`
sets the server's total budget — page image plus output. Clerk's
`CLERK_OCR_MAX_OUTPUT_TOKENS` is the per-page output cap it requests. Clerk has
no OCR context setting, because one image plus a short command is the entire
request; the server's context is the only real limit, and Lemonade already
clamps an oversized output request to fit it.

Budget: `ctx_size ≥ image tokens + output tokens`, with the two models sizing
their images very differently.

**DeepSeek-OCR-2** has a hard 8,192-token context (`max_position_embeddings`)
and a *fixed* internal view — it resizes any page into one 1024×1024 global view
plus up to six 768×768 crops, which is `6 × 144 + 256 = 1,120` visual tokens at
most, regardless of the DPI Clerk renders at. So `ctx_size` 8192 is both the
maximum and the right value, roughly 7,000 tokens are free for output, and 4096
is a comfortable per-page cap. Raising render DPI past the point of legibility
buys nothing here: the model cannot see more than those views hold.

**GLM-OCR** is the opposite. Its 131,072-token context is never the constraint,
but it tokenizes the page dynamically at 28×28 pixels per visual token, so the
image cost scales with render DPI. For US Letter:

| `CLERK_RENDER_DPI` | Image tokens | + 8,192 output | Fits `ctx_size` 16384? |
| --- | --- | --- | --- |
| 160 | ~3,050 | ~11,250 | yes |
| 200 | ~4,770 | ~12,960 | yes |
| 250 | ~7,450 | ~15,650 | barely |
| 300 | ~10,730 | ~18,930 | **no** |

Hence 200 DPI with `ctx_size` 16384 and an 8,192-token output cap. Raise
`ctx_size` before raising DPI, or Lemonade will silently clamp the output cap
and pages will start hitting the token limit. Note `CLERK_RENDER_DPI` is global,
so it applies to whichever profile is active.

#### When a page hits the output limit

Raising `CLERK_OCR_MAX_OUTPUT_TOKENS` is almost never the fix. A dense page of
prose transcribes to well under 1,000 tokens, so a page that produces 4,096 has
almost certainly fallen into a decoder loop — repeating a line or a run of dot
leaders until it runs out of budget. Raising the cap just buys a longer loop.

Clerk distinguishes the two causes from the response's own token counts rather
than guessing. Output stopping *short* of what Clerk requested means the server
shortened it to fit the context, and the message names the context size to
clear; output arriving at exactly the requested number means the model ran away,
and the message says so. Either way Clerk keeps the transcription from before
the cut instead of discarding the page, trimming a repeated tail block if one is
present, and logs a warning with the counts. Only a page with nothing usable
left after trimming fails.

For DeepSeek-OCR-2 the loop guard is the server-side n-gram logits processor,
which does nothing unless the request also carries `vllm_xargs` — the
`deepseek_ocr` profile sends exactly the pair the vLLM recipe documents. GLM-OCR
has no equivalent guard, so a persistent loop there is best addressed at the
server.

Lemonade's vLLM backend is Linux-only and requires its documented ROCm/CWSR
kernel setup on Strix Halo. Keep Clerk's per-page output limit below the
server's context budget.

Recommended endpoint characteristics:

- vision input support for the OCR model;
- deterministic or low-temperature inference;
- JSON output support for the metadata model;
- enough OCR output tokens for a dense single page;
- no model-side request queue timeout shorter than Clerk's configured timeout.

API keys are optional because many local servers do not require one. When set,
Clerk sends `Authorization: Bearer <key>`. Paperless uses its native
`Authorization: Token <token>` authentication.

## Configuration

Environment values seed the SQLite settings on first start and any explicitly
supplied environment variable remains authoritative at runtime and after
restarts. Those fields are visibly read-only in the UI; remove the variable and
restart to manage one there. Values not supplied by the environment use the
UI-persisted configuration. Blank secret fields retain the saved secret; a
separate checkbox clears it intentionally. Settings responses expose only
`*_configured` booleans and names of managed fields—not secret values.

| Variable | Default | Purpose |
| --- | --- | --- |
| `PAPERLESS_URL` | `http://paperless-webserver:8000` | Paperless root URL (a trailing `/api` is accepted) |
| `PAPERLESS_TOKEN` | empty | Paperless API token |
| `PAPERLESS_VERIFY_SSL` | `true` | Verify Paperless TLS certificates |
| `CLERK_OPENAI_BASE_URL` | `http://host.docker.internal:11434/v1` | Shared local endpoint for OCR and metadata models |
| `CLERK_OPENAI_API_KEY` | empty | Optional bearer token shared by both model clients |
| `CLERK_OCR_MODEL` | `qwen2.5vl:7b` | Vision model name |
| `CLERK_OCR_PROFILE` | `generic` | OCR request contract: `generic` or `deepseek_ocr_llamacpp` (plus `deepseek_ocr` and `glm_ocr` when unhidden below) |
| `CLERK_ENABLE_VLLM_PROFILES` | unset | Offer the held-back `deepseek_ocr` and `glm_ocr` profiles again |
| `CLERK_OCR_MAX_OUTPUT_TOKENS` | `4096` | Per-page OCR output cap; must fit the serving context alongside the page image |
| `CLERK_METADATA_MODEL` | `qwen2.5:14b` | Metadata model name |
| `CLERK_METADATA_CONTEXT_TOKENS` | `16384` | Metadata context budget |
| `CLERK_METADATA_MAX_OUTPUT_TOKENS` | `4096` | Structured output cap |
| `CLERK_REQUEST_TIMEOUT_SECONDS` | `300` | Paperless/model request timeout |
| `CLERK_MODEL_MAX_RETRIES` | `3` | Retryable HTTP attempts per request |
| `CLERK_JOB_MAX_ATTEMPTS` | `3` | Persisted whole-job attempts |
| `CLERK_JOB_WORKERS` | `2` | Documents processed concurrently so one large file cannot stop the queue |
| `CLERK_PAGE_CONCURRENCY` | `1` | OCR requests within one document |
| `CLERK_METADATA_CONCURRENCY` | `1` | Metadata map calls within one document |
| `CLERK_RENDER_DPI` | `160` | Initial page rendering DPI |
| `CLERK_MAX_IMAGE_PIXELS` | `16000000` | Per-page pixel ceiling; DPI scales down to fit |
| `CLERK_JPEG_QUALITY` | `86` | Rendered page JPEG quality |
| `CLERK_OCR_MIN_CHARS` | `24` | Minimum meaningful existing OCR size |
| `CLERK_METADATA_CHUNK_CHARS` | `12000` | Maximum map input text size |
| `CLERK_METADATA_CANDIDATE_LIMIT` | `80` | Bounded entities of each type shown to a model |
| `CLERK_METADATA_MIN_CONFIDENCE` | `0.68` | Minimum confidence accepted for an assignment |
| `CLERK_METADATA_APPLY_MODE` | `missing_only` | `missing_only` or `overwrite` |
| `CLERK_ALLOW_NEW_TAGS` | `true` | Permit validated broad tags when the first document in a new reusable subject arrives |
| `CLERK_ALLOW_NEW_CORRESPONDENTS` | `true` | Permit validated distinct correspondents |
| `CLERK_ALLOW_NEW_DOCUMENT_TYPES` | `true` | Permit validated stable types |
| `CLERK_ALLOW_NEW_CUSTOM_FIELDS` | `false` | Permit new field definitions |
| `CLERK_AUTOMATION_ENABLED` | `false` | Poll and enqueue new or changed eligible documents |
| `CLERK_AUTOMATION_INTERVAL_SECONDS` | `120` | Discovery interval |
| `CLERK_AUTOMATION_PAGE_SIZE` | `25` | Documents inspected on each newest/backlog poll |
| `CLERK_AUTOMATION_TAG` | empty | Optional workflow-only queue tag; blank makes all documents eligible |
| `CLERK_NOTIFICATIONS_ENABLED` | `false` | Send ntfy alerts for terminal failures and intervention states |
| `CLERK_NTFY_URL` | `https://ntfy.sh` | ntfy server root URL |
| `CLERK_NTFY_TOPIC` | empty | ntfy topic; required when notifications are enabled |
| `CLERK_NTFY_TOKEN` | empty | Optional bearer token for a protected ntfy topic |
| `CLERK_DATA_DIR` | `./data` (`/app/data` in image) | SQLite data directory |
| `CLERK_HOST` / `CLERK_PORT` | `0.0.0.0` / `8080` | HTTP listener |
| `CLERK_LOG_LEVEL` | `INFO` | Application log level |

The UI exposes the settings that materially affect local deployment and model
behavior, including container log detail under **Limits & reliability**.
Internal prompts are application-owned rather than a general prompt playground.
Appearance preferences—system/light/dark theme, comfortable/compact density,
and system/full/reduced motion—are stored in Clerk through the UI and cached in
the browser to avoid a theme flash during startup.

### ntfy notifications

Enable notifications in **Settings → Notifications**, then enter a topic and
optionally an access token. The **Send test** button publishes a normal-priority
test message. Operational alerts use high priority and are limited to events
that need attention:

- a job reaches its final failed state after bounded retries;
- an OCR conflict or another review-required state is created;
- automatic Paperless discovery fails (one alert per continuous outage).

Successful jobs and intermediate retry attempts do not send notifications.
ntfy delivery is auxiliary: a timeout, authentication error, or unavailable
ntfy server is written to the container log and the job event history without
changing the processing outcome.

Topics on the public `ntfy.sh` service should be long and difficult to guess;
without access controls, the topic name acts as the subscription secret. A
notification contains the Paperless document title and ID plus a bounded error
or review message, so use a protected topic or a self-hosted ntfy server if that
metadata is sensitive.

Clerk normalizes existing string, long-text, boolean, integer, float, monetary,
date, URL, and select custom fields. It deliberately omits document-link values
because the model is not given an authoritative bounded set of document IDs;
assign those links in Paperless. Creating field definitions is disabled by
default and supports only safely typed definitions.

## Versions, corrections, and legacy conflicts

The prior reading remains selectable in Paperless's document version history,
so correction normally means choosing or editing a version in Paperless rather
than blocking Clerk's job. Clerk never publishes partial, empty, or rejected
model output; those remain genuine failures/interventions.

Databases upgraded from the older comparison workflow may still contain open OCR
conflicts. The Intervention screen and conflict-resolution API remain available
to finish those records, but new OCR jobs do not create comparison conflicts.

Recent metadata decisions show the exact patch, canonical entities reused, new
entities created, near-duplicates normalized, and candidates withheld. The
document opens directly in Paperless for correction, and metadata can be rerun
without redoing OCR. **View diagnostic log** expands the underlying proposal,
bounded candidate vocabulary with IDs and names, validation rejections,
focused tag review, and bounded invalid-output previews for that decision. It
deliberately excludes full OCR text and model request prompts.

## Reliability and privacy

- Active jobs are unique per document and claimed with expiring worker leases.
- Automatic discovery alternates the newest bounded Paperless result page with
  later backlog pages, so new arrivals stay timely and a processed first page
  cannot permanently hide older documents.
- SQLite WAL mode and short `BEGIN IMMEDIATE` claims make duplicate work
  resistant without another service.
- Retry counts, due times, errors, page status, and decisions survive restarts.
- Retryable HTTP failures use bounded exponential backoff.
- ntfy alerts are attempted only for actionable or terminal states; delivery
  failures are observable but never fail or retry an otherwise completed job.
- Verbose model rationale fields are normalized to their schema limit, so a
  valid metadata or tag choice is not discarded only because its explanation
  ran long.
- Page failures do not cancel successful sibling pages and never publish a
  partial document.
- Paperless OCR is not replaced after a model parse error or partial page run.
  The prior version is retained before a complete Clerk result becomes latest.
- At the default `INFO` level, container logs include job lifecycle, metadata
  outcomes, tag-review counts, retries, and concise validation failures. They
  never include document contents or model prompts. Clerk installs its own
  stderr handler so these records do not depend on Uvicorn's root logger.
- Bounded previews of invalid structured model output are retained only in an
  explicitly opened Decision diagnostic log. These previews may contain
  extracted metadata or short evidence and therefore remain private Clerk data.
- Full OCR text is returned by Clerk's API only for an explicitly opened
  conflict detail.

Clerk has no built-in multi-user authentication because it targets a
single-user homelab. Do not expose port 8080 directly to the public internet.
Put it behind the same authenticated reverse proxy or private network used for
Paperless. The SQLite volume contains sensitive OCR and API credentials and
must be protected accordingly.

## Development

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev
CLERK_DATA_DIR=./data uv run paperless-clerk
```

The UI is plain HTML, CSS, and JavaScript served by FastAPI, so it has no Node
build step. Validation commands:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
docker build -t paperless-clerk .
```

See [`docs/architecture.md`](docs/architecture.md) for component boundaries,
state flow, reference-project findings, and the exact Paperless API operations.

## API

Interactive OpenAPI documentation is served at `/api/docs`. The UI uses:

- `/api/jobs` for enqueue, status, retry, and cancellation;
- `/api/conflicts` for comparison detail and resolution;
- `/api/decisions` for metadata rationale;
- `/api/interventions` for the review desk;
- `/api/settings` and `/api/settings/test/{target}` for configuration and
  Paperless, model, or ntfy test requests;
- `/api/health` for container health checks.

## License

MIT
