# Paperless Clerk

Paperless Clerk is a focused local-AI sidecar for
[Paperless-ngx](https://docs.paperless-ngx.com/). It reads documents page by
page, verifies rather than blindly replaces existing OCR, and classifies each
document against the metadata system already living in Paperless.

Its filing rule is simple:

> **reuse → normalize → extend only when necessary**

Clerk supports OpenAI-compatible local endpoints only. OCR/vision and metadata
share one endpoint while retaining independent model selection, so a vision
model can transcribe pages while a smaller text model handles classification.

## What Clerk does

- OCRs every page independently with bounded concurrency and persisted page
  results.
- Writes complete OCR into Paperless only after every page succeeds.
- Compares Clerk OCR with existing Paperless OCR using token, vocabulary,
  ordered-shingle, length, and numeric agreement.
- Selects either Clerk or the existing Paperless OCR after a trusted match,
  according to the configured preference. Profiles that require
  content-changing decoding guards are review-only and never replace Paperless
  OCR automatically.
- Adds an `ocr-conflict` Paperless tag and retains the available readings when
  they disagree or guarded OCR has no Paperless baseline.
- Lets the user resolve a conflict side by side and resumes metadata processing
  only after that decision.
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

Clerk does not replace Paperless storage, upload rewritten PDFs, provide chat or
RAG, connect to hosted AI services, or mirror the Paperless document library.

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
4. If Paperless has no meaningful content, Clerk patches ordinary OCR output.
   A guarded DeepSeek result instead creates a review item because Paperless has
   no existing OCR baseline to retain.
5. If content exists, Clerk compares the two complete readings. A high score
   selects the configured preferred source (Clerk by default); a low score
   creates an intervention without replacing either reading automatically.
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

Most general vision-language models accept Clerk's normal system instruction
and detailed transcription prompt. Specialist OCR models often do not: their
training expects one image and one exact, short task command. Choose the serving
stack under **Settings → Vision OCR**:

- **Generic vision model** keeps the normal request for Qwen and other
  instruction-following vision models.
- **DeepSeek OCR vLLM profile** targets DeepSeek-OCR and DeepSeek-OCR-2 on
  vLLM. It renders pages losslessly, then sends one image-first user message
  containing DeepSeek's document task,
  `<|grounding|>Convert the document to markdown.`, along with
  `skip_special_tokens: false` and the model author's 20-token/90-token-window
  n-gram guard.
  Clerk removes DeepSeek's paired layout-class and coordinate annotations while
  retaining the recognized Markdown text for comparison and review. Because
  the required guard changes decoding, this profile never publishes OCR
  automatically: existing Paperless OCR remains canonical, or a
  review item is created when no baseline exists.
- **DeepSeek OCR-2 GGUF via llama.cpp** reproduces Clerk's earlier request for
  `sabafallah/DeepSeek-OCR-2-GGUF`: JPEG at the configured render resolution,
  one image-first user message containing `Free OCR.`, temperature 0, and
  `top_k: 1`. It deliberately sends no vLLM-only fields, but remains an
  experimental A/B path rather than a fidelity guarantee.

Selecting the profile changes the OCR processing fingerprint, so a retried job
will discard page results produced with a different request contract. The
**Test OCR model** action renders and
sends a portrait document containing known header, body, and footer text; it
therefore verifies the vision projector and selected production profile instead
of merely testing text chat. The check succeeds only when markers from all
three regions are present, so reading the header while dropping the footer is a
failure.

#### vLLM serving recipe

Use a recent upstream vLLM release with native `DeepseekOCR2ForCausalLM`
support:

```sh
vllm serve deepseek-ai/DeepSeek-OCR-2 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --logits_processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"image": 1}'
```

Then configure the endpoint or model router used by Clerk:

```env
CLERK_OPENAI_BASE_URL=http://your-vllm-host:8000/v1
CLERK_OCR_MODEL=deepseek-ai/DeepSeek-OCR-2
CLERK_OCR_PROFILE=deepseek_ocr
```

Current vLLM releases provide a built-in DeepSeek-OCR fallback chat template,
which formats Clerk's image-first message as the model's native
`<image>`-plus-document-task prompt. Clerk does not send a request-level
`chat_template`, so neither `--chat-template` nor
`--trust-request-chat-template` is needed. Leave request-template trust disabled.

DeepSeek-OCR-2 does not reliably terminate under unguarded greedy decoding.
Loading the vLLM processor is not enough: Clerk must also send its per-request
`vllm_xargs`. Clerk uses DeepSeek's single-image `ngram_size: 20`,
`window_size: 90`, and table-token whitelist. This is more aggressive than
vLLM's generic online example because that example's `30/90` setting was not
enough for the observed financial-letter loop. It is still not a lossless
setting: the processor forbids a previously seen continuation and can therefore
alter a legitimate exact repeated span. Clerk exposes that limitation instead
of hiding it—DeepSeek output is advisory, token-limited responses are rejected,
and only a human can promote guarded output when no existing Paperless OCR exists.

Clerk records the effective OCR model, profile, prompt, render settings, and
completion policy in each job's event history. This makes an older successful
configuration recoverable without relying on container logs or memory.

#### Completeness limits

DeepSeek-OCR-2 internally resizes a full page into a fixed global view and a
bounded set of local crops. Raising Clerk's DPI cannot add model-side visual
tokens once those views are populated. Full-page footer omissions can therefore
be model failures even when the source render is sharp. When Paperless already
has meaningful OCR, Clerk now measures directional coverage as well as overall
similarity and retains Paperless's text if the new result omits material
existing words, especially a trailing block. The job records the coverage and
suffix counts. This safeguard prevents data loss; it cannot recover omitted
text when no prior OCR exists.

For documents where independent OCR completeness is mandatory, use a
layout-aware pipeline that detects regions, crops each region, recognizes them
separately, and merges reading order. The GLM-OCR SDK provides that architecture
and can call an external OpenAI-compatible GLM model while its layout detector
runs on CPU. Its stock configuration intentionally discards headers, footers,
footnotes, page numbers, and several marginal-text classes, so an archival
configuration must map those labels to the text task rather than accepting the
defaults. This requires a separate SDK service and real hardware validation; a
single full-page chat request is not an equivalent substitute.

The original DeepSeek repository's pinned vLLM wheel and example scripts drive
an in-process `AsyncLLMEngine`; they are not an OpenAI-compatible HTTP serving
recipe. Clerk should point at `vllm serve` (or a compatible model router), not
at those scripts. A bare vLLM process normally serves one base model. Because
Clerk intentionally uses one shared OpenAI-compatible URL for OCR and metadata,
that URL must also route the configured metadata model if it is hosted by a
different vLLM worker.

The profile cannot compensate for an older vLLM release that does not support
the model architecture. Keep Clerk's per-page output limit below the server's
available context budget. A token-limit error on a short page is evidence of a
generation loop, not a reason to raise the limit; a larger limit only makes that
failure slower.

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
| `CLERK_OCR_PROFILE` | `generic` | OCR request contract: `generic`, vLLM-oriented `deepseek_ocr`, or `deepseek_ocr_llamacpp` for the earlier GGUF stack |
| `CLERK_PREFER_CLERK_OCR` | `true` | After a trusted OCR match, publish ordinary Clerk OCR; guarded DeepSeek vLLM output is always review-only |
| `CLERK_OCR_CONTEXT_TOKENS` | `8192` | Declared OCR context limit |
| `CLERK_OCR_MAX_OUTPUT_TOKENS` | `4096` | Per-page OCR output cap |
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
| `CLERK_JPEG_QUALITY` | `86` | Rendered JPEG quality for generic and llama.cpp OCR; the DeepSeek vLLM profile uses lossless PNG |
| `CLERK_OCR_MIN_CHARS` | `24` | Minimum meaningful existing OCR size |
| `CLERK_OCR_SIMILARITY_THRESHOLD` | `0.82` | Score required to choose either OCR source automatically |
| `CLERK_CONFLICT_TAG` | `ocr-conflict` | Paperless review tag |
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

## Conflict and correction workflow

An open OCR conflict retains the OCR snapshot available when the review was
created, Clerk's generated text, and comparison metrics. The Paperless document
receives the conflict tag but its OCR is not changed. Resolution always
re-fetches the live document. In **Intervention**:

- **Keep Paperless OCR** requires meaningful current Paperless OCR and closes
  the conflict without changing `content`.
- **Use Clerk OCR** replaces `content` with the assembled page text only when
  Paperless OCR has not changed since the review was created.

Only a successful action removes Clerk's conflict tag and enqueues metadata-only
analysis. An empty keep choice or stale Clerk choice leaves the review open.
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
- Paperless OCR is not replaced after a model parse error, partial page run, or
  low-confidence comparison. The preferred-source option applies only after a
  trusted match.
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
