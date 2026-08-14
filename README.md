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
  according to the configured preference. Clerk OCR is preferred by default.
- Adds an `ocr-conflict` Paperless tag and retains both complete readings when
  they disagree.
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
4. If Paperless has no meaningful content, Clerk patches the assembled text.
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

### DeepSeek OCR request profile

Most general vision-language models accept Clerk's normal system instruction
and detailed transcription prompt. Specialist OCR models often do not: their
training expects one image and one exact, short task command. Clerk provides a
single specialist checkbox directly below the OCR model name in **Settings →
Vision OCR**:

- **Unchecked (generic)** keeps the normal request for Qwen and other
  instruction-following vision models.
- **DeepSeek OCR / OCR 2 profile** sends one image-first user message containing
  only `Free OCR.`. It supports both DeepSeek-OCR generations and strips any
  accidental DeepSeek reference/coordinate scaffolding before comparison or
  publication. Clerk deliberately uses plain OCR rather than grounded Markdown
  because Paperless's OCR field is the canonical readable text, not a layout
  annotation store.

Selecting the profile changes the OCR processing fingerprint, so a retried job
will discard page results produced with a different request contract. The
**Test OCR model** action renders and
sends a real image containing known text; it therefore verifies the vision
projector and selected profile instead of merely testing text chat.

This profile adapts Clerk's API request, but it cannot compensate for a
missing or mismatched multimodal projector or an older inference server that
does not support the model architecture. DeepSeek-OCR-2 in particular requires
a recent llama.cpp build with its matching OCR-2 `mmproj` GGUF.

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
| `CLERK_OCR_PROFILE` | `generic` | OCR request contract: `generic` or `deepseek_ocr` (covers DeepSeek OCR and OCR 2) |
| `CLERK_PREFER_CLERK_OCR` | `true` | After a trusted OCR match, publish Clerk OCR instead of retaining existing Paperless OCR |
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
| `CLERK_JPEG_QUALITY` | `86` | Rendered page JPEG quality |
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
| `CLERK_DATA_DIR` | `./data` (`/app/data` in image) | SQLite data directory |
| `CLERK_HOST` / `CLERK_PORT` | `0.0.0.0` / `8080` | HTTP listener |
| `CLERK_LOG_LEVEL` | `INFO` | Application log level |

The UI exposes the settings that materially affect local deployment and model
behavior, including container log detail under **Limits & reliability**.
Internal prompts are application-owned rather than a general prompt playground.
Appearance preferences—system/light/dark theme, comfortable/compact density,
and system/full/reduced motion—are stored in Clerk through the UI and cached in
the browser to avoid a theme flash during startup.

Clerk normalizes existing string, long-text, boolean, integer, float, monetary,
date, URL, and select custom fields. It deliberately omits document-link values
because the model is not given an authoritative bounded set of document IDs;
assign those links in Paperless. Creating field definitions is disabled by
default and supports only safely typed definitions.

## Conflict and correction workflow

An open OCR conflict retains both full text versions and comparison metrics in
Clerk. The Paperless document receives the conflict tag but its OCR is not
changed. In **Intervention**:

- **Keep Paperless OCR** closes the conflict without changing `content`.
- **Use Clerk OCR** replaces `content` with the complete assembled page text.

Both actions remove Clerk's conflict tag and enqueue metadata-only analysis.
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
- `/api/settings` and `/api/settings/test/{target}` for configuration;
- `/api/health` for container health checks.

## License

MIT
