# Paperless Clerk architecture

Paperless Clerk is a deliberately small sidecar. Paperless-ngx remains the
system of record; Clerk keeps only the state required to make model work
durable, reviewable, and safe to retry.

## Reference review

The reference applications demonstrate useful Paperless API conventions and a
good manual-review workflow, but also show the costs of broad provider support,
configuration-heavy prompt editing, exact-string taxonomy matching, in-memory
job state, and whole-document model requests. Clerk retains the useful parts:
token authentication, paginated vocabulary reads, tag-driven/manual processing,
separate OCR and metadata models, per-document progress, and an explicit review
surface. It does not inherit either application's data model or provider matrix.

Paperless-ngx's current API supports the operations Clerk needs:

- `GET /api/documents/{id}/` returns effective OCR content and metadata.
- `GET /api/documents/{id}/download/` streams the archived document (or the
  original with `?original=true`).
- `PATCH /api/documents/{id}/` accepts `content`, canonical metadata IDs,
  `created`, and custom-field instances shaped as `{field, value}`.
- `POST /api/documents/{id}/update_version/` consumes an uploaded file as a new
  version and returns a Paperless task UUID.
- `GET /api/tasks/?task_id={uuid}` exposes completion and the created version ID;
  `PATCH /api/documents/{id}/?version={version_id}` targets that version's text.
- `PATCH /api/documents/{id}/versions/{version_id}/` labels a retained version.
- Tags, correspondents, document types, and custom fields are paginated API
  resources and can be created independently.

Clerk updates OCR text through the `content` field. With
`keep_original_version` enabled, a baseline causes it to upload the unchanged
current file and target the new version's content; it never deletes or rewrites
the prior version. With retention disabled, it targets the current version
directly. The optional version contract requires Paperless-ngx 3.0 or newer.

## Components

```text
Browser UI
    |
FastAPI JSON API ---- SQLite (jobs, page results, conflicts, decisions, settings)
    |
Durable worker pool
    +---- Paperless client (documents and controlled vocabulary)
    +---- PDF page renderer (PyMuPDF, temporary files)
    +---- OCR client (OpenAI-compatible chat completions with images)
    +---- Metadata client (OpenAI-compatible structured chat completions)
    +---- ntfy client (terminal failure and intervention alerts)
```

There is no Redis or external task service. SQLite uses WAL mode, short
transactions, a partial unique index for active document jobs, and leases so a
crashed worker can resume work after restart.

## Job and OCR flow

1. A manual request or optional poller enqueues one active job per document.
2. A worker claims the job with a lease and fetches the current Paperless
   document.
3. The source file is streamed to a temporary file and hashed.
4. Pages are rendered one at a time. At most `page_concurrency` encoded page
   images are retained while OCR requests run. Each page is stored immediately,
   including its attempts and error state.
5. Page retries use bounded exponential backoff. A failed page does not cancel
   sibling pages. A job with exhausted pages becomes an intervention rather
   than publishing incomplete OCR.
6. Successful page text is assembled with stable page markers.
7. Re-fetch the Paperless document before publishing. If Paperless changed
   during inference, recheck the source hash; a changed source retries from its
   new pages. If there is still no meaningful content, Clerk patches the
   complete assembled OCR onto the current version.
8. If meaningful content exists and `keep_original_version` is disabled, patch
   the complete Clerk text directly onto the current latest version without an
   upload. If retention is enabled, upload the unchanged current file through
   the version endpoint with label `Paperless Clerk OCR`. Record an upload-started
   checkpoint before the POST, persist the returned task UUID before polling it,
   then persist the created version ID. A timeout or restart resumes that same
   Paperless task rather than uploading again. If the POST response is lost
   before its task UUID can be recorded, Clerk will reuse a completed
   Clerk-labeled version if it appears but will not issue a blind duplicate
   upload while the outcome remains ambiguous.
9. If the prior version had no label, label it `Pre-Clerk OCR backup`. Patch the
   complete Clerk text with an explicit `?version={version_id}` target and verify
   that the created version is still latest. Paperless therefore uses Clerk OCR
   for search and content by default while the earlier file/text remains
   selectable in version history.
10. A later run recognizes the latest `Paperless Clerk OCR` label and updates
   that version rather than creating a new backup. A genuinely newer user/file
   version causes a source retry and receives its own backup on the next pass.

Reclaimed jobs reuse successful page rows from the same run. Paperless writes
occur only after all required model work and validation has succeeded.

## Metadata map/reduce flow

1. Fetch the complete current Paperless vocabulary before classification. If a
   queue/watch tag is configured, remove its ID from both the document's current
   tags and the candidate vocabulary supplied to the model; the planner also
   rejects attempts to recreate its name.
2. Build bounded candidate lists using normalized words, acronyms, current
   assignments, and usage counts. This keeps large libraries inside local-model
   context limits without trusting a model-invented existing ID.
3. Split OCR at page/paragraph boundaries according to the configured context
   budget. Each map call returns compact facts and candidates with source page,
   confidence, evidence, and an explicit `existing_id` versus `new_name` choice.
4. Send only compact map results and candidate vocabulary to a reduce call. The
   reducer returns a single structured proposal.
5. If an otherwise untagged document has no usable tag in that proposal, run
   one bounded tag-only review using canonical candidates and a representative
   cross-document text sample. When growth is enabled and that review abstains,
   audit the abstention once with explicit form-versus-subject guidance. This
   allows Invoice plus a broad Veterinary tag while rejecting the composite
   Veterinary Invoice. Preserve an explicit abstention assessment when no tag
   is justified after the audit.
6. Serialize the short validation/write section across document workers and
   re-fetch the vocabulary inside that lock, preventing concurrent alias
   proposals from creating near-duplicates. Validate every existing ID against
   the freshly fetched resource of the right type. Canonicalize every proposed
   new name using case/punctuation folding, singularization, corporate-suffix
   removal where appropriate, acronym matching, and token/name similarity.
   Convert near-duplicates to existing IDs or omit them.
7. Normalize custom-field values to the Paperless field type. New custom-field
   definitions are rejected unless the explicit option is enabled.
8. Re-fetch the document immediately before writing. If its OCR changed during
   inference, retry analysis; otherwise merge against its current metadata so a
   user's intervening tags or field corrections are not overwritten. Apply
   according to the conservative metadata policy: tags are additive; missing
   single-value fields are filled; replacing populated correspondent, type,
   date, or a non-generic title requires the configured overwrite policy.
9. In the same successful metadata patch, remove the configured queue/watch tag
   if the document has it. Failures and OCR conflicts retain the tag, so it
   remains a truthful marker for unfinished work.
10. Persist the proposal, applied/withheld changes, candidate duplicates,
   confidence, reasons, source chunks, focused tag assessment, and bounded
   structured-output repair diagnostics without logging full document text.

The governing classifier rule is `reuse -> normalize -> extend only when
necessary`. Reuse and creation can occur in the same tag proposal.

## Failure boundaries

- HTTP and model retries are bounded and classify retryable status codes.
- Whole-job retries have persisted attempt counts and due times.
- Stale worker leases are reclaimed on startup and during polling.
- An active-job uniqueness constraint prevents duplicate manual/poller jobs.
- Paperless version task UUIDs and created version IDs survive Clerk restarts;
  terminal Paperless failures clear the checkpoint so an explicit retry can
  start a new upload, while timeouts retain it for safe resumption. A lost
  upload response remains a deliberate intervention because Paperless provides
  no task ID with which Clerk could safely deduplicate another POST.
- OCR conflict resolutions use an atomic claim, preventing opposing concurrent
  choices from both modifying Paperless. They also validate the live OCR before
  writing, so an empty keep choice or stale Clerk replacement leaves the review
  open.
- A pathological document occupies one worker only; other workers continue.
- ntfy delivery is outside the job transaction. Failures are logged and added
  to the job timeline without changing the document-processing result.
- Page text and conflicts are private API details and never appear in list or
  log payloads.
- Container logs expose lifecycle, counts, and concise validation errors. A
  full Decision detail can reveal bounded invalid-output previews on demand,
  but never retains the source OCR or request prompt in that diagnostic log.
- No Paperless OCR is published on partial OCR or model failure. Only after the
  complete Clerk result passes local validation is existing OCR either moved
  behind a new latest version or replaced on the current version, according to
  `keep_original_version`.

## UI information architecture

The build-free web client has five focused views: overview, jobs, intervention,
decisions, and settings. The intervention queue unifies OCR conflicts and
exhausted jobs while retaining distinct actions. Conflict detail is loaded only
on demand and presents both texts, scores, and mismatch excerpts. Decision
detail includes an on-demand diagnostic log containing the bounded model and
validation trace. Manual processing is always available from the overview and
jobs views. The HTML is revalidated on every visit and references JavaScript,
CSS, and favicon assets with a shared content-derived fingerprint. Changed
assets therefore receive new URLs, while matching fingerprinted assets can be
cached immutably.
