const content = document.querySelector("#content");
const drawer = document.querySelector("#drawer");
const drawerContent = document.querySelector("#drawer-content");
const scrim = document.querySelector("#scrim");
const processDialog = document.querySelector("#process-dialog");
const processForm = document.querySelector("#process-form");
const state = { route: "overview", dashboard: null, jobs: [], interventions: null, decisions: [], settings: null, jobFilter: "all", poll: null };

const pageMeta = {
  overview: ["Your document desk", "Overview"],
  jobs: ["Durable and resumable", "Processing"],
  review: ["Human judgment", "Intervention"],
  decisions: ["Transparent classification", "Recent decisions"],
  settings: ["Useful controls only", "Settings"],
};

const appearanceStorageKey = "paperless-clerk-appearance";
const darkModeQuery = window.matchMedia("(prefers-color-scheme: dark)");
const reducedMotionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");

function applyAppearance(settings = {}, persist = true) {
  const appearance = {
    theme: settings.appearance_theme || settings.theme || "system",
    density: settings.appearance_density || settings.density || "comfortable",
    motion: settings.appearance_motion || settings.motion || "system",
  };
  document.documentElement.dataset.theme = appearance.theme;
  document.documentElement.dataset.density = appearance.density;
  document.documentElement.dataset.motion = appearance.motion;
  const dark = appearance.theme === "dark" || (appearance.theme === "system" && darkModeQuery.matches);
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", dark ? "#101715" : "#153f3b");
  if (persist) {
    try { localStorage.setItem(appearanceStorageKey, JSON.stringify(appearance)); }
    catch { /* appearance still applies for this page */ }
  }
}

try { applyAppearance(JSON.parse(localStorage.getItem(appearanceStorageKey) || "null") || {}, false); }
catch { applyAppearance({}, false); }

darkModeQuery.addEventListener("change", () => {
  if (document.documentElement.dataset.theme === "system") applyAppearance({ theme: "system", density: document.documentElement.dataset.density, motion: document.documentElement.dataset.motion }, false);
});

function escapeHtml(value = "") {
  return String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}

function titleCase(value = "") {
  return String(value)
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase())
    .replace(/\bOcr\b/g, "OCR");
}

function relativeTime(value) {
  if (!value) return "—";
  const seconds = Math.round((new Date(value).getTime() - Date.now()) / 1000);
  const formatter = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });
  const units = [["year", 31536000], ["month", 2592000], ["day", 86400], ["hour", 3600], ["minute", 60]];
  for (const [unit, size] of units) if (Math.abs(seconds) >= size) return formatter.format(Math.round(seconds / size), unit);
  return formatter.format(seconds, "second");
}

function fullTime(value) {
  return value ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value)) : "—";
}

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json", ...(options.headers || {}) }, ...options });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { const body = await response.json(); detail = body.detail || body.error || detail; } catch { /* response was not JSON */ }
    throw new Error(detail);
  }
  if (response.status === 204) return null;
  return response.json();
}

function toast(title, message = "", type = "success") {
  const item = document.createElement("div");
  item.className = `toast ${type}`;
  item.innerHTML = `<strong>${escapeHtml(title)}</strong>${message ? `<p>${escapeHtml(message)}</p>` : ""}`;
  document.querySelector("#toasts").append(item);
  setTimeout(() => item.remove(), 4600);
}

function statusChip(status) { return `<span class="status-chip ${escapeHtml(status)}">${escapeHtml(titleCase(status))}</span>`; }
function emptyState(icon, title, text) { return `<div class="empty-state"><div><span class="empty-icon">${icon}</span><h3>${escapeHtml(title)}</h3><p>${escapeHtml(text)}</p></div></div>`; }
function docThumb(id) { return `<span class="doc-thumb"><img src="/api/documents/${id}/thumbnail" alt="" loading="lazy" onerror="this.remove();this.parentElement.textContent='${id}'" /></span>`; }

function jobRow(job, actions = true) {
  const active = ["running", "queued", "retry_wait"].includes(job.status);
  const retry = ["failed", "needs_review", "cancelled"].includes(job.status) && job.phase !== "ocr_conflict";
  return `<article class="job-row clickable" data-action="job-detail" data-id="${job.id}">
    ${docThumb(job.document_id)}
    <div class="row-title"><strong>${escapeHtml(job.document_title || `Document ${job.document_id}`)}</strong><small>#${job.document_id} · ${escapeHtml(titleCase(job.mode))}</small>${active && job.progress_total ? `<div class="progress-track"><i style="width:${job.progress_percent}%"></i></div>` : ""}</div>
    <div class="row-meta"><strong>${escapeHtml(titleCase(job.phase))}</strong><br /><span>${job.progress_total ? `${job.progress_current} / ${job.progress_total}` : `Attempt ${job.attempt}/${job.max_attempts}`}</span></div>
    <div>${statusChip(job.status)}</div>
    ${actions ? `<div class="row-actions">${retry ? `<button class="button ghost small" data-action="retry-job" data-id="${job.id}">Retry</button>` : ""}${["queued", "retry_wait"].includes(job.status) ? `<button class="button ghost small" data-action="cancel-job" data-id="${job.id}">Cancel</button>` : ""}</div>` : ""}
  </article>`;
}

function conflictRow(item) {
  return `<article class="conflict-row clickable" data-action="conflict-detail" data-id="${item.id}">
    ${docThumb(item.document_id)}
    <div class="row-title"><strong>${escapeHtml(item.document_title || `Document ${item.document_id}`)}</strong><small>#${item.document_id} · OCR verification</small></div>
    <div class="row-meta">Compared ${relativeTime(item.created_at)}</div>
    <div><span class="score">${Math.round(item.score * 100)}% match</span></div>
    <div class="row-actions"><button class="button secondary small" data-action="conflict-detail" data-id="${item.id}">Review</button></div>
  </article>`;
}

function decisionChanges(decision) {
  const names = (decision.applied?.patch_fields || Object.keys(decision.applied?.patch || {})).map(titleCase);
  const created = decision.applied?.created_count ?? decision.applied?.created?.length ?? 0;
  const reused = decision.applied?.reused_count ?? decision.applied?.reused?.length ?? 0;
  const removed = decision.applied?.removed_count ?? decision.applied?.removed?.length ?? 0;
  const tagNote = decision.status === "no_tags" ? " · no tags selected" : "";
  if (!names.length && !created && !reused && !removed) return decision.status === "no_tags" ? "No tags selected" : "No library changes";
  return `${names.slice(0, 3).join(", ") || "Metadata"}${names.length > 3 ? ` +${names.length - 3}` : ""} · ${reused} reused${created ? ` · ${created} new` : ""}${removed ? ` · ${removed} workflow tag removed` : ""}${tagNote}`;
}

function decisionDiagnosticLog(item) {
  return {
    decision: {
      id: item.id,
      job_id: item.job_id,
      paperless_document_id: item.document_id,
      created_at: item.created_at,
      status: item.status,
    },
    pipeline: {
      source_chunks: item.rationale?.source_chunks || [],
      candidate_counts: item.rationale?.candidate_counts || {},
      candidate_ids: item.rationale?.candidate_ids || {},
      candidate_vocabulary: item.rationale?.candidate_vocabulary || {},
      model_diagnostics: item.rationale?.model_diagnostics || [],
    },
    focused_tag_review: item.rationale?.tag_review || null,
    model_proposal: item.proposal || {},
    validation: {
      rejected_candidates: item.rationale?.rejected || [],
      candidate_duplicates: item.rationale?.candidate_duplicates || [],
    },
    paperless_application: {
      before: item.before || {},
      result: item.applied || {},
    },
  };
}

function decisionRow(item) {
  return `<article class="decision-row clickable" data-action="decision-detail" data-id="${item.id}">
    <span class="decision-mark">✓</span>
    <div class="row-title"><strong>${escapeHtml(item.document_title || `Document ${item.document_id}`)}</strong><small>#${item.document_id} · ${escapeHtml(decisionChanges(item))}</small></div>
    <div class="row-meta">${relativeTime(item.created_at)}</div>
    <div>${statusChip(item.status)}</div>
  </article>`;
}

function updateChrome() {
  const [eyebrow, title] = pageMeta[state.route] || pageMeta.overview;
  document.querySelector("#page-eyebrow").textContent = eyebrow;
  document.querySelector("#page-title").textContent = title;
  document.querySelectorAll("[data-route]").forEach((item) => item.classList.toggle("active", item.dataset.route === state.route));
  if (state.dashboard) {
    const counts = state.dashboard.counts;
    setBadge("nav-active-count", counts.active);
    setBadge("nav-review-count", counts.open_conflicts + counts.failed + counts.needs_review);
    const pill = document.querySelector("#automation-pill");
    pill.classList.toggle("off", !state.dashboard.automation_enabled);
    pill.querySelector("span").textContent = state.dashboard.automation_enabled
      ? (state.dashboard.automation_tag ? `Watching ${state.dashboard.automation_tag}` : "Watching recent documents")
      : "Automation off";
  }
}

function setBadge(id, count) {
  const item = document.querySelector(`#${id}`);
  item.textContent = count || "";
  item.classList.toggle("visible", Boolean(count));
}

async function renderRoute({ quiet = false } = {}) {
  if (!quiet) content.innerHTML = `<div class="initial-loader"><span class="loader-mark"></span><p>Checking the desk…</p></div>`;
  try {
    if (state.route === "overview") await renderOverview();
    else await refreshDashboard();
    if (state.route === "jobs") await renderJobs();
    if (state.route === "review") await renderReview();
    if (state.route === "decisions") await renderDecisions();
    if (state.route === "settings") await renderSettings();
  } catch (error) {
    if (!quiet) content.innerHTML = `<div class="panel">${emptyState("!", "The desk could not be loaded", error.message)}</div>`;
  }
  updateChrome();
}

async function refreshDashboard() {
  state.dashboard = await api("/api/dashboard");
  updateChrome();
}

async function renderOverview() {
  state.dashboard = await api("/api/dashboard");
  const { counts, jobs, conflicts, decisions } = state.dashboard;
  content.innerHTML = `
    <section class="grid metrics">
      <article class="metric-card"><span class="metric-icon">↻</span><span class="metric-label">In progress</span><div class="metric-value">${counts.active}</div><span class="metric-note">${counts.queued} waiting in the durable queue</span></article>
      <article class="metric-card warning"><span class="metric-icon">◇</span><span class="metric-label">Needs attention</span><div class="metric-value">${counts.open_conflicts + counts.needs_review}</div><span class="metric-note">${counts.open_conflicts} OCR conflicts to inspect</span></article>
      <article class="metric-card"><span class="metric-icon">✓</span><span class="metric-label">Decisions today</span><div class="metric-value">${counts.decisions_today}</div><span class="metric-note">Concise rationale retained</span></article>
      <article class="metric-card ${counts.failed ? "error" : ""}"><span class="metric-icon">!</span><span class="metric-label">Exhausted retries</span><div class="metric-value">${counts.failed}</div><span class="metric-note">Nothing fails silently</span></article>
    </section>
    <section class="grid overview-grid">
      <div>
        <article class="panel desk-note">
          <p class="eyebrow">The next filing run</p><h2>Turn scanned pages into trustworthy, organized documents.</h2>
          <p>Clerk reads page by page, verifies existing OCR, then fits each document into the metadata vocabulary you already use.</p>
          <div class="desk-actions"><button class="button primary" data-action="open-process">Process documents</button><a class="button ghost" href="#settings">Check connections</a></div>
        </article>
        <article class="panel">
          <header class="panel-head"><div><h2>Processing desk</h2><p>Live status from the durable queue</p></div><a href="#jobs" class="panel-link">View all</a></header>
          <div class="status-list">${jobs.length ? jobs.map((job) => jobRow(job, false)).join("") : emptyState("↻", "The queue is clear", "Run documents manually or enable automatic discovery in Settings.")}</div>
        </article>
      </div>
      <div>
        <article class="panel">
          <header class="panel-head"><div><h2>On your desk</h2><p>Items that need human judgment</p></div>${conflicts.length ? `<a href="#review" class="panel-link">Review all</a>` : ""}</header>
          <div class="status-list">${conflicts.length ? conflicts.map(conflictRow).join("") : emptyState("✓", "Nothing needs review", "OCR disagreements and exhausted jobs will appear here.")}</div>
        </article>
        <article class="panel">
          <header class="panel-head"><div><h2>Recent filing decisions</h2><p>What Clerk changed, reused, or withheld</p></div><a href="#decisions" class="panel-link">History</a></header>
          <div class="status-list">${decisions.length ? decisions.slice(0, 4).map(decisionRow).join("") : emptyState("⌁", "No decisions yet", "Metadata decisions appear after the first completed analysis.")}</div>
        </article>
      </div>
    </section>`;
}

async function renderJobs() {
  state.jobs = await api("/api/jobs?limit=250");
  const filters = ["all", "active", "completed", "failed", "needs_review"];
  const filtered = state.jobs.filter((job) => {
    if (state.jobFilter === "all") return true;
    if (state.jobFilter === "active") return ["running", "queued", "retry_wait"].includes(job.status);
    return job.status === state.jobFilter;
  });
  content.innerHTML = `<section class="page-intro"><div><h2>Every run has a durable paper trail</h2><p>Page progress, retries, and model or Paperless errors survive restarts. One difficult document never blocks the rest of the queue.</p></div><div class="actions"><button class="button primary" data-action="open-process">＋ Process documents</button></div></section>
    <section class="panel"><div class="toolbar"><div class="filter-tabs">${filters.map((filter) => `<button class="${filter === state.jobFilter ? "active" : ""}" data-action="job-filter" data-filter="${filter}">${titleCase(filter)}</button>`).join("")}</div><span class="spacer"></span><input class="search-input" id="job-search" type="search" placeholder="Filter by document or ID" /></div><div class="status-list" id="job-list">${filtered.length ? filtered.map(jobRow).join("") : emptyState("↻", "No matching jobs", "Try another filter or start a processing run.")}</div></section>`;
}

async function renderReview() {
  state.interventions = await api("/api/interventions");
  const { conflicts, failed_jobs: failed, review_jobs: review } = state.interventions;
  const nonConflictReview = review.filter((job) => job.phase !== "ocr_conflict");
  const count = conflicts.length + failed.length + nonConflictReview.length;
  content.innerHTML = `<section class="page-intro"><div><h2>${count ? `${count} item${count === 1 ? "" : "s"} waiting for you` : "The intervention desk is clear"}</h2><p>Clerk pauses whenever choosing automatically could destroy trustworthy text or hide a persistent failure.</p></div></section>
    ${conflicts.length ? `<div class="alert-banner"><span>◇</span><div><strong>OCR conflicts are intentionally unresolved</strong><p>Compare both readings and choose; metadata analysis will resume afterward.</p></div></div>` : ""}
    <section class="grid review-grid">
      <article class="panel"><header class="panel-head"><div><h2>OCR conflicts</h2><p>Existing Paperless text versus Clerk vision OCR</p></div>${statusChip(conflicts.length ? "open" : "completed")}</header><div class="status-list">${conflicts.length ? conflicts.map(conflictRow).join("") : emptyState("✓", "No OCR conflicts", "Formatting-only differences are accepted automatically.")}</div></article>
      <article class="panel"><header class="panel-head"><div><h2>Errors & retries</h2><p>Runs that exhausted bounded retries</p></div>${failed.length || nonConflictReview.length ? statusChip("failed") : statusChip("completed")}</header><div class="status-list">${failed.length || nonConflictReview.length ? [...failed, ...nonConflictReview].map(jobRow).join("") : emptyState("✓", "No exhausted jobs", "Transient failures retry automatically before appearing here.")}</div></article>
    </section>`;
}

async function renderDecisions() {
  state.decisions = await api("/api/decisions?limit=250");
  content.innerHTML = `<section class="page-intro"><div><h2>Classification you can account for</h2><p>Each entry separates canonical reuse, genuinely new vocabulary, withheld changes, confidence filtering, and near-duplicates considered.</p></div></section><section class="panel"><div class="toolbar"><span class="muted">${state.decisions.length} retained decision${state.decisions.length === 1 ? "" : "s"}</span><span class="spacer"></span><input class="search-input" id="decision-search" type="search" placeholder="Filter document decisions" /></div><div class="status-list" id="decision-list">${state.decisions.length ? state.decisions.map(decisionRow).join("") : emptyState("⌁", "No metadata decisions yet", "Complete a full or metadata-only run to populate this history.")}</div></section>`;
}

function settingInput(name, label, value, options = {}) {
  const type = options.type || "text";
  const configured = options.configured;
  const locked = Boolean(state.settings?.environment_overrides?.includes(name));
  const inputId = `setting-${name}`;
  const noteText = locked ? "Managed by an environment variable; remove it and restart to edit here." : options.note;
  const note = noteText ? `<small class="${locked ? "environment-note" : ""}">${escapeHtml(noteText)}</small>` : "";
  if (type === "select") return `<div class="field ${options.full ? "full" : ""} ${locked ? "locked" : ""}"><label for="${inputId}">${escapeHtml(label)}</label><select id="${inputId}" name="${name}" ${locked ? "disabled" : ""}>${options.choices.map(([key, text]) => `<option value="${key}" ${value === key ? "selected" : ""}>${escapeHtml(text)}</option>`).join("")}</select>${note}</div>`;
  const clearSecret = type === "password" && configured && !locked ? `<label class="clear-secret" for="clear-${name}"><input id="clear-${name}" type="checkbox" name="clear_${name}" /> Clear saved secret</label>` : "";
  return `<div class="field ${options.full ? "full" : ""} ${locked ? "locked" : ""}"><label for="${inputId}">${escapeHtml(label)}</label><div class="${configured !== undefined ? "input-with-status" : ""}"><input id="${inputId}" name="${name}" type="${type}" value="${type === "password" ? "" : escapeHtml(value ?? "")}" ${options.min !== undefined ? `min="${options.min}"` : ""} ${options.max !== undefined ? `max="${options.max}"` : ""} ${type === "number" ? `step="${options.step || "any"}"` : ""} ${type === "password" ? `placeholder="${configured ? "Leave blank to keep saved secret" : "Enter secret if required"}" autocomplete="new-password"` : ""} ${locked ? "disabled" : ""} />${configured !== undefined ? `<span>${configured ? "configured" : "not set"}</span>` : ""}</div>${clearSecret}${note}</div>`;
}

function settingCheck(name, title, description, checked) {
  const locked = Boolean(state.settings?.environment_overrides?.includes(name));
  const note = locked ? "Managed by an environment variable; remove it and restart to edit here." : description;
  return `<label class="check-row ${locked ? "locked" : ""}"><input type="checkbox" name="${name}" ${checked ? "checked" : ""} ${locked ? "disabled" : ""} /><span><strong>${escapeHtml(title)}</strong><small class="${locked ? "environment-note" : ""}">${escapeHtml(note)}</small></span></label>`;
}

function ocrProfileSetting(profile) {
  const selected = profile === "deepseek_ocr" ? profile : "generic";
  const locked = Boolean(state.settings?.environment_overrides?.includes("ocr_profile"));
  return `<fieldset class="profile-setting full ${locked ? "locked" : ""}">
    <legend>Specialist request profile</legend>
    <input id="setting-ocr_profile" type="hidden" name="ocr_profile" value="${selected}" ${locked ? "disabled" : ""} />
    <div class="profile-options"><label class="profile-choice"><input type="checkbox" data-ocr-profile="deepseek_ocr" ${selected === "deepseek_ocr" ? "checked" : ""} ${locked ? "disabled" : ""} /><span><strong>DeepSeek OCR / OCR 2 profile</strong><small>Uses DeepSeek's plain OCR mode without chat or grounded-layout scaffolding.</small></span></label></div>
    <small class="profile-help ${locked ? "environment-note" : ""}">${locked ? "Managed by an environment variable; remove it and restart to edit here." : "Leave unchecked for general vision models such as Qwen."}</small>
  </fieldset>`;
}

async function renderSettings() {
  state.settings = await api("/api/settings");
  const s = state.settings;
  applyAppearance(s);
  content.innerHTML = `<section class="page-intro"><div><h2>Connect Clerk to your local stack</h2><p>OCR and metadata use one OpenAI-compatible endpoint with independently selected models. Secrets stay server-side and are never returned to this browser. Save changed values before testing a connection.</p></div></section>
    <div class="settings-layout">
      <nav class="settings-nav"><a href="#settings-paperless">Paperless</a><a href="#settings-ai">Local AI API</a><a href="#settings-ocr">Vision OCR</a><a href="#settings-metadata">Metadata model</a><a href="#settings-workflow">Automation & filing</a><a href="#settings-appearance">Appearance</a><a href="#settings-advanced">Limits & reliability</a></nav>
      <form class="settings-form" id="settings-form">
        <section class="panel settings-section" id="settings-paperless"><header class="panel-head"><div><h3>Paperless-ngx</h3><p class="section-description">System of record and the only external application Clerk modifies.</p></div><button class="button ghost small" type="button" data-action="test-connection" data-target="paperless">Test connection</button></header><div class="panel-body"><div class="form-grid">${settingInput("paperless_url", "Paperless URL", s.paperless_url, { full: true, note: "Root URL or a URL ending in /api." })}${settingInput("paperless_token", "API token", "", { type: "password", configured: s.paperless_token_configured })}</div>${settingCheck("paperless_verify_ssl", "Verify TLS certificates", "Disable only for a trusted local endpoint with a self-signed certificate.", s.paperless_verify_ssl)}</div></section>
        <section class="panel settings-section" id="settings-ai"><header class="panel-head"><div><h3>OpenAI-compatible API</h3><p class="section-description">One local endpoint and credential shared by both Clerk models.</p></div></header><div class="panel-body"><div class="form-grid">${settingInput("openai_base_url", "Base URL", s.openai_base_url, { full: true, note: "Usually ends in /v1; Clerk appends /chat/completions." })}${settingInput("openai_api_key", "API key", "", { type: "password", configured: s.openai_api_key_configured, full: true })}</div></div></section>
        <section class="panel settings-section" id="settings-ocr"><header class="panel-head"><div><h3>Vision OCR model</h3><p class="section-description">Receives one rendered page image per request through the shared API.</p></div><button class="button ghost small" type="button" data-action="test-connection" data-target="ocr">Test OCR model</button></header><div class="panel-body"><div class="form-grid">${settingInput("ocr_model", "Model name", s.ocr_model, { full: true })}${ocrProfileSetting(s.ocr_profile)}<div class="ocr-preference">${settingCheck("prefer_clerk_ocr", "Prefer Clerk OCR after a trusted match", "Enabled by default. When the OCR versions meet the trust threshold, publish Clerk's result; otherwise retain Paperless's existing text. A low match always goes to manual review.", s.prefer_clerk_ocr)}</div>${settingInput("ocr_context_tokens", "Context limit", s.ocr_context_tokens, { type: "number", min: 1024 })}${settingInput("ocr_max_output_tokens", "Maximum output tokens", s.ocr_max_output_tokens, { type: "number", min: 256 })}</div></div></section>
        <section class="panel settings-section" id="settings-metadata"><header class="panel-head"><div><h3>Metadata model</h3><p class="section-description">Receives bounded OCR chunks and returns structured classification decisions.</p></div><button class="button ghost small" type="button" data-action="test-connection" data-target="metadata">Test metadata model</button></header><div class="panel-body"><div class="form-grid">${settingInput("metadata_model", "Model name", s.metadata_model, { full: true })}${settingInput("metadata_context_tokens", "Context limit", s.metadata_context_tokens, { type: "number", min: 2048 })}${settingInput("metadata_max_output_tokens", "Maximum output tokens", s.metadata_max_output_tokens, { type: "number", min: 256 })}</div></div></section>
        <section class="panel settings-section" id="settings-workflow"><header class="panel-head"><div><h3>Automation & filing</h3><p class="section-description">Control discovery and conservative vocabulary growth.</p></div></header><div class="panel-body"><div class="check-grid">${settingCheck("automation_enabled", "Automatically process Paperless documents", "Polls Paperless for unseen or subsequently modified documents and queues full OCR plus metadata jobs. Add a queue tag below to make automation opt-in per document.", s.automation_enabled)}${settingCheck("allow_new_tags", "Allow distinct new tags", "Creates broad reusable subjects when the first document of a genuinely new category arrives; near-duplicates still normalize to existing tags.", s.allow_new_tags)}${settingCheck("allow_new_correspondents", "Allow distinct correspondents", "Aliases and abbreviations are aggressively canonicalized.", s.allow_new_correspondents)}${settingCheck("allow_new_document_types", "Allow stable document types", "Specific or synonymous variants are rejected.", s.allow_new_document_types)}${settingCheck("allow_new_custom_fields", "Allow new custom-field definitions", "Experimental and intentionally disabled by default.", s.allow_new_custom_fields)}</div><div class="form-grid">${settingInput("automation_tag", "Paperless queue tag (optional)", s.automation_tag, { full: true, note: "When set, only documents carrying this tag are discovered. Clerk hides it from the model, keeps it on failures or OCR conflicts, and removes it after metadata succeeds." })}${settingInput("automation_interval_seconds", "Discovery interval (seconds)", s.automation_interval_seconds, { type: "number", min: 15 })}${settingInput("metadata_apply_mode", "Existing metadata policy", s.metadata_apply_mode, { type: "select", choices: [["missing_only", "Preserve existing values"], ["overwrite", "Allow confident replacement"]], note: "Tags remain additive in both modes." })}</div></div></section>
        <section class="panel settings-section" id="settings-appearance"><header class="panel-head"><div><h3>Appearance</h3><p class="section-description">Personalize this browser without changing Paperless or processing behavior.</p></div></header><div class="panel-body"><div class="form-grid">${settingInput("appearance_theme", "Color theme", s.appearance_theme, { type: "select", choices: [["system", "Follow system"], ["light", "Light"], ["dark", "Dark"]], note: "Previewed immediately and saved for all Clerk sessions." })}${settingInput("appearance_density", "Interface density", s.appearance_density, { type: "select", choices: [["comfortable", "Comfortable"], ["compact", "Compact"]] })}${settingInput("appearance_motion", "Animation and motion", s.appearance_motion, { type: "select", choices: [["system", "Follow system"], ["full", "Full motion"], ["reduced", "Reduced motion"]], full: true, note: "Reduced motion minimizes transitions, loaders, and drawer animation." })}</div></div></section>
        <section class="panel settings-section" id="settings-advanced"><header class="panel-head"><div><h3>Limits & reliability</h3><p class="section-description">Bound memory and inference without imposing a document page limit.</p></div></header><div class="panel-body"><div class="form-grid">${settingInput("page_concurrency", "Concurrent OCR pages", s.page_concurrency, { type: "number", min: 1, max: 16 })}${settingInput("metadata_concurrency", "Concurrent metadata chunks", s.metadata_concurrency, { type: "number", min: 1, max: 8 })}${settingInput("job_workers", "Document workers", s.job_workers, { type: "number", min: 1, max: 8, note: "Changing this requires a restart." })}${settingInput("job_max_attempts", "Whole-job attempts", s.job_max_attempts, { type: "number", min: 1, max: 10 })}${settingInput("model_max_retries", "Request retries", s.model_max_retries, { type: "number", min: 0, max: 10 })}${settingInput("request_timeout_seconds", "Request timeout (seconds)", s.request_timeout_seconds, { type: "number", min: 10 })}${settingInput("render_dpi", "Page render DPI", s.render_dpi, { type: "number", min: 72, max: 400 })}${settingInput("max_image_pixels", "Maximum page pixels", s.max_image_pixels, { type: "number", min: 1000000 })}${settingInput("ocr_similarity_threshold", "OCR trust threshold", s.ocr_similarity_threshold, { type: "number", min: 0.5, max: 0.99 })}${settingInput("ocr_min_chars", "Meaningful OCR characters", s.ocr_min_chars, { type: "number", min: 1 })}${settingInput("metadata_chunk_chars", "Metadata chunk characters", s.metadata_chunk_chars, { type: "number", min: 2000 })}${settingInput("metadata_candidate_limit", "Vocabulary candidates per type", s.metadata_candidate_limit, { type: "number", min: 10, max: 500 })}${settingInput("metadata_min_confidence", "Minimum metadata confidence", s.metadata_min_confidence, { type: "number", min: 0, max: 1 })}${settingInput("log_level", "Container log detail", s.log_level, { type: "select", choices: [["DEBUG", "Debug"], ["INFO", "Info (recommended)"], ["WARNING", "Warnings only"], ["ERROR", "Errors only"]], full: true, note: "Clerk lifecycle and decision counts appear at Info. Changing this requires a restart." })}</div></div></section>
        <div class="settings-save"><p>Appearance previews immediately. Processing changes apply to newly claimed jobs.</p><button class="button primary" type="submit">Save settings</button></div>
      </form>
    </div>`;
}

function openDrawer(html) {
  drawerContent.innerHTML = html;
  drawer.classList.add("open"); scrim.classList.add("visible"); drawer.setAttribute("aria-hidden", "false");
}
function closeDrawer() { drawer.classList.remove("open"); scrim.classList.remove("visible"); drawer.setAttribute("aria-hidden", "true"); }
function drawerHeader(eyebrow, title) { return `<header class="drawer-head"><div><p class="eyebrow">${escapeHtml(eyebrow)}</p><h2>${escapeHtml(title)}</h2></div><button class="icon-button" data-action="close-drawer" aria-label="Close">×</button></header>`; }

async function showJob(id) {
  openDrawer(`${drawerHeader("Processing detail", "Loading job…")}<div class="drawer-body"><div class="skeleton"></div></div>`);
  try {
    const job = await api(`/api/jobs/${id}`);
    openDrawer(`${drawerHeader(`Paperless document #${job.document_id}`, job.document_title || `Document ${job.document_id}`)}<div class="drawer-body">
      <section class="detail-section"><div class="detail-grid"><div class="detail-stat"><span>Status</span><strong>${titleCase(job.status)}</strong></div><div class="detail-stat"><span>Scope</span><strong>${titleCase(job.mode)}</strong></div><div class="detail-stat"><span>Attempt</span><strong>${job.attempt} / ${job.max_attempts}</strong></div><div class="detail-stat"><span>Phase</span><strong>${titleCase(job.phase)}</strong></div><div class="detail-stat"><span>Progress</span><strong>${job.progress_total ? `${job.progress_current} / ${job.progress_total}` : "—"}</strong></div><div class="detail-stat"><span>Started</span><strong>${relativeTime(job.started_at)}</strong></div></div></section>
      ${job.error_message ? `<section class="detail-section"><h3>Last error</h3><div class="resolution-box"><p class="danger-text">${escapeHtml(job.error_message)}</p></div></section>` : ""}
      ${job.page_failures?.length ? `<section class="detail-section"><h3>Page failures</h3><div class="change-list">${job.page_failures.map((page) => `<div class="change"><i>!</i><div><strong>Page ${page.page_number} · ${escapeHtml(titleCase(page.status))}</strong><small>${escapeHtml(page.error || "No page error detail was returned.")} · ${page.attempts} attempt${page.attempts === 1 ? "" : "s"}</small></div></div>`).join("")}</div></section>` : ""}
      <section class="detail-section"><h3>Run events</h3><div class="event-list">${job.events.length ? job.events.map((event) => `<div class="event ${event.level}"><strong>${escapeHtml(titleCase(event.event_type))}</strong><span>${fullTime(event.created_at)}</span><p>${escapeHtml(event.message)}</p></div>`).join("") : `<p class="muted">No events retained.</p>`}</div></section>
      <div class="resolution-actions"><a class="button ghost" href="${escapeHtml((state.dashboard?.paperless_url || "").replace(/\/$/, ""))}/documents/${job.document_id}/details" target="_blank" rel="noreferrer">Open in Paperless ↗</a>${["failed", "needs_review", "cancelled"].includes(job.status) && job.phase !== "ocr_conflict" ? `<button class="button secondary" data-action="retry-job" data-id="${job.id}">Retry job</button>` : ""}</div>
    </div>`);
  } catch (error) { toast("Could not load job", error.message, "error"); closeDrawer(); }
}

async function showConflict(id) {
  openDrawer(`${drawerHeader("OCR conflict", "Loading comparison…")}<div class="drawer-body"><div class="skeleton"></div></div>`);
  try {
    const item = await api(`/api/conflicts/${id}`);
    const metrics = [["Overall match", item.score], ["Token overlap", item.metrics.token_overlap], ["Reading order", item.metrics.ordered_shingle_overlap], ["Number agreement", item.metrics.numeric_overlap], ["Length agreement", item.metrics.length_agreement]];
    openDrawer(`${drawerHeader(`Paperless document #${item.document_id}`, item.document_title || `Document ${item.document_id}`)}<div class="drawer-body">
      <section class="detail-section"><h3>Comparison signals</h3><div class="detail-grid">${metrics.map(([name, value]) => `<div class="detail-stat"><span>${name}</span><strong>${Math.round(Number(value) * 100)}%</strong><div class="metric-bar"><i style="width:${Math.round(Number(value) * 100)}%"></i></div></div>`).join("")}</div></section>
      <section class="detail-section"><h3>Both complete OCR versions</h3><div class="text-compare"><div class="text-pane"><header>Existing Paperless OCR</header><pre>${escapeHtml(item.existing_text)}</pre></div><div class="text-pane"><header>Clerk vision OCR</header><pre>${escapeHtml(item.generated_text)}</pre></div></div></section>
      ${item.diff?.length ? `<section class="detail-section"><h3>Sample mismatches</h3><div class="diff-list">${item.diff.map((diff) => `<div class="diff-item"><b>${escapeHtml(diff.operation)}</b><p><strong>Paperless:</strong> ${escapeHtml(diff.existing)}</p><p><strong>Clerk:</strong> ${escapeHtml(diff.generated)}</p></div>`).join("")}</div></section>` : ""}
      <section class="resolution-box"><h3>Choose the trustworthy reading</h3><p>Keeping Paperless leaves its OCR byte-for-byte unchanged. Choosing Clerk replaces only the OCR content field. Either choice removes the conflict tag and queues metadata analysis.</p><div class="resolution-actions"><button class="button ghost" data-action="resolve-conflict" data-id="${item.id}" data-resolution="keep_existing">Keep Paperless OCR</button><button class="button primary" data-action="resolve-conflict" data-id="${item.id}" data-resolution="use_clerk">Use Clerk OCR</button><a class="button ghost" href="${escapeHtml((state.dashboard?.paperless_url || "").replace(/\/$/, ""))}/documents/${item.document_id}/details" target="_blank" rel="noreferrer">Open document ↗</a></div></section>
    </div>`);
  } catch (error) { toast("Could not load conflict", error.message, "error"); closeDrawer(); }
}

async function showDecision(id) {
  openDrawer(`${drawerHeader("Metadata decision", "Loading rationale…")}<div class="drawer-body"><div class="skeleton"></div></div>`);
  try {
    const item = await api(`/api/decisions/${id}`);
    const reused = item.applied?.reused || [], created = item.applied?.created || [], removed = item.applied?.removed || [], assignments = item.applied?.assignments || [], withheld = item.applied?.withheld || [], duplicates = item.rationale?.candidate_duplicates || [], rejected = item.rationale?.rejected || [];
    const tagReview = item.rationale?.tag_review;
    const diagnosticLog = JSON.stringify(decisionDiagnosticLog(item), null, 2);
    const rationaleNote = (entry, fallback) => `${fallback}${entry.confidence !== undefined ? ` · ${Math.round(entry.confidence * 100)}% confidence` : ""}${entry.reason ? ` · ${entry.reason}` : ""}${entry.source_pages?.length ? ` · page${entry.source_pages.length === 1 ? "" : "s"} ${entry.source_pages.join(", ")}` : ""}${entry.evidence ? ` · “${String(entry.evidence).slice(0, 140)}”` : ""}`;
    const changes = [...reused.map((entry) => ({ icon: "↺", title: `Reused ${entry.name}`, note: rationaleNote(entry, titleCase(entry.resource)) })), ...created.map((entry) => ({ icon: "+", title: `Created ${entry.name}`, note: rationaleNote(entry, titleCase(entry.resource)) })), ...removed.map((entry) => ({ icon: "−", title: `Removed ${entry.name}`, note: entry.reason })), ...assignments.map((entry) => ({ icon: "✓", title: `Set ${entry.field_name || titleCase(entry.field)}`, note: rationaleNote(entry, String(entry.value)) })), ...withheld.map((entry) => ({ icon: "—", title: `Preserved ${titleCase(entry.field)}`, note: entry.reason }))];
    openDrawer(`${drawerHeader(`Paperless document #${item.document_id}`, item.document_title || `Document ${item.document_id}`)}<div class="drawer-body">
      <section class="detail-section"><div class="detail-grid"><div class="detail-stat"><span>Outcome</span><strong>${titleCase(item.status)}</strong></div><div class="detail-stat"><span>Vocabulary reused</span><strong>${reused.length}</strong></div><div class="detail-stat"><span>Vocabulary created</span><strong>${created.length}</strong></div></div></section>
      <section class="detail-section"><h3>Decision summary</h3><p class="muted">${escapeHtml(item.rationale?.summary || "No model summary was provided.")}</p><div class="change-list">${changes.length ? changes.map((change) => `<div class="change"><i>${change.icon}</i><div><strong>${escapeHtml(change.title)}</strong><small>${escapeHtml(change.note)}</small></div></div>`).join("") : `<p class="muted">No Paperless fields needed to change.</p>`}</div></section>
      ${tagReview ? `<section class="detail-section"><h3>Tag assessment</h3><div class="change-list"><div class="change"><i>${tagReview.accepted_count ? "✓" : "?"}</i><div><strong>${tagReview.accepted_count ? `${tagReview.accepted_count} tag${tagReview.accepted_count === 1 ? "" : "s"} accepted` : "No applicable tag selected"}</strong><small>${escapeHtml(tagReview.assessment || "No assessment was recorded.")}${tagReview.performed ? " · focused second pass performed" : ""}${tagReview.abstention_audit_performed ? " · empty result challenged once" : ""}</small></div></div></div></section>` : ""}
      ${duplicates.length ? `<section class="detail-section"><h3>Vocabulary overlap checks</h3><div class="change-list">${duplicates.map((entry) => `<div class="change"><i>≃</i><div><strong>${escapeHtml(entry.outcome === "created_as_explicit_distinction" ? `${entry.proposed} kept distinct from ${entry.matched_name}` : `${entry.proposed} → ${entry.matched_name}`)}</strong><small>${escapeHtml(entry.reason)} · ${Math.round(entry.score * 100)}% similarity${entry.justification ? ` · ${escapeHtml(entry.justification)}` : ""}</small></div></div>`).join("")}</div></section>` : ""}
      ${rejected.length ? `<section class="detail-section"><h3>Omitted candidates</h3><div class="change-list">${rejected.slice(0, 20).map((entry) => `<div class="change"><i>×</i><div><strong>${escapeHtml(titleCase(entry.kind))}</strong><small>${escapeHtml(entry.reason)}</small></div></div>`).join("")}</div></section>` : ""}
      <section class="detail-section"><h3>Applied Paperless patch</h3><pre class="code-block">${escapeHtml(JSON.stringify(item.applied?.patch || {}, null, 2))}</pre></section>
      <section class="detail-section diagnostic-log" id="decision-diagnostic-log" hidden><h3>Decision diagnostic log</h3><p class="diagnostic-note">Stored only in Clerk's private decision history. It includes bounded model-validation previews and metadata evidence, but never the full OCR text or model request prompt.</p><pre class="code-block" id="decision-diagnostic-output">${escapeHtml(diagnosticLog)}</pre></section>
      <div class="resolution-actions"><a class="button primary" href="${escapeHtml((state.dashboard?.paperless_url || "").replace(/\/$/, ""))}/documents/${item.document_id}/details" target="_blank" rel="noreferrer">Review or correct in Paperless ↗</a><button class="button ghost" data-action="reprocess-metadata" data-document-id="${item.document_id}">Re-run metadata</button><button class="button ghost" data-action="toggle-decision-log">View diagnostic log</button></div>
    </div>`);
  } catch (error) { toast("Could not load decision", error.message, "error"); closeDrawer(); }
}

async function retryJob(id) {
  try { await api(`/api/jobs/${id}/retry`, { method: "POST" }); toast("Job queued", "Saved page results will be reused where possible."); closeDrawer(); await renderRoute({ quiet: true }); }
  catch (error) { toast("Retry failed", error.message, "error"); }
}

async function resolveConflict(id, resolution) {
  if (resolution === "use_clerk" && !window.confirm("Replace Paperless OCR with Clerk's complete OCR result?")) return;
  try { await api(`/api/conflicts/${id}/resolve`, { method: "POST", body: JSON.stringify({ resolution }) }); toast("Conflict resolved", "Metadata analysis has been queued."); closeDrawer(); await renderRoute({ quiet: true }); }
  catch (error) { toast("Resolution failed", error.message, "error"); }
}

function parseIds(value) { return [...new Set(value.split(/[^0-9]+/).filter(Boolean).map(Number).filter((id) => id > 0))]; }
async function enqueue(documentIds, mode = "full") {
  const result = await api("/api/jobs", { method: "POST", body: JSON.stringify({ document_ids: documentIds, mode }) });
  const created = result.jobs.filter((item) => item.created).length;
  toast("Documents queued", `${created} new job${created === 1 ? "" : "s"}; ${result.jobs.length - created} already active.`);
}

document.addEventListener("click", async (event) => {
  const target = event.target.closest("[data-action]"); if (!target) return;
  const action = target.dataset.action;
  if (action === "open-process") processDialog.showModal();
  if (action === "close-process") processDialog.close();
  if (action === "close-drawer") closeDrawer();
  if (action === "job-detail") showJob(target.dataset.id);
  if (action === "conflict-detail") showConflict(target.dataset.id);
  if (action === "decision-detail") showDecision(target.dataset.id);
  if (action === "toggle-decision-log") {
    const logPanel = drawerContent.querySelector("#decision-diagnostic-log");
    if (logPanel) {
      logPanel.hidden = !logPanel.hidden;
      target.textContent = logPanel.hidden ? "View diagnostic log" : "Hide diagnostic log";
      if (!logPanel.hidden) logPanel.scrollIntoView({ block: "start" });
    }
  }
  if (action === "retry-job") { event.stopPropagation(); retryJob(target.dataset.id); }
  if (action === "cancel-job") { event.stopPropagation(); try { await api(`/api/jobs/${target.dataset.id}/cancel`, { method: "POST" }); toast("Job cancelled"); await renderRoute({ quiet: true }); } catch (error) { toast("Cancel failed", error.message, "error"); } }
  if (action === "resolve-conflict") resolveConflict(target.dataset.id, target.dataset.resolution);
  if (action === "job-filter") { state.jobFilter = target.dataset.filter; renderJobs(); }
  if (action === "test-connection") { target.disabled = true; const old = target.textContent; target.textContent = "Testing…"; try { const result = await api(`/api/settings/test/${target.dataset.target}`, { method: "POST" }); toast("Connection succeeded", result.model ? `${result.model}: ${result.response}` : `${result.documents} Paperless documents visible`); } catch (error) { toast("Connection failed", error.message, "error"); } finally { target.disabled = false; target.textContent = old; } }
  if (action === "reprocess-metadata") { try { await enqueue([Number(target.dataset.documentId)], "metadata"); closeDrawer(); } catch (error) { toast("Could not queue metadata", error.message, "error"); } }
});

scrim.addEventListener("click", closeDrawer);
document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeDrawer(); });
document.querySelector("#mobile-menu").addEventListener("click", () => document.querySelector(".sidebar").classList.toggle("mobile-open"));
document.querySelectorAll(".nav-list a").forEach((item) => item.addEventListener("click", () => document.querySelector(".sidebar").classList.remove("mobile-open")));

processForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(processForm); const ids = parseIds(String(form.get("document_ids") || ""));
  if (!ids.length) return toast("Enter at least one document ID", "", "error");
  const button = processForm.querySelector('[type="submit"]'); button.disabled = true;
  try { await enqueue(ids, String(form.get("mode"))); processDialog.close(); processForm.reset(); if (state.route !== "jobs") location.hash = "jobs"; else await renderRoute({ quiet: true }); }
  catch (error) { toast("Could not queue documents", error.message, "error"); }
  finally { button.disabled = false; }
});

content.addEventListener("submit", async (event) => {
  if (event.target.id !== "settings-form") return;
  event.preventDefault(); const form = event.target; const data = new FormData(form); const values = {};
  const locked = new Set(state.settings?.environment_overrides || []);
  const numeric = new Set(["ocr_context_tokens", "ocr_max_output_tokens", "metadata_context_tokens", "metadata_max_output_tokens", "automation_interval_seconds", "page_concurrency", "metadata_concurrency", "job_workers", "job_max_attempts", "model_max_retries", "request_timeout_seconds", "render_dpi", "max_image_pixels", "ocr_min_chars", "metadata_chunk_chars", "metadata_candidate_limit"]);
  const decimal = new Set(["ocr_similarity_threshold", "metadata_min_confidence"]);
  const checks = ["paperless_verify_ssl", "prefer_clerk_ocr", "automation_enabled", "allow_new_tags", "allow_new_correspondents", "allow_new_document_types", "allow_new_custom_fields"];
  for (const [key, value] of data.entries()) {
    if (key.startsWith("clear_")) continue;
    values[key] = numeric.has(key) ? Number.parseInt(value, 10) : decimal.has(key) ? Number.parseFloat(value) : value;
  }
  for (const key of checks) if (!locked.has(key)) values[key] = data.get(key) === "on";
  for (const key of ["paperless_token", "openai_api_key"]) {
    if (data.get(`clear_${key}`) === "on") values[key] = "";
    else if (!values[key]) delete values[key];
  }
  const button = form.querySelector('[type="submit"]'); button.disabled = true;
  try { const result = await api("/api/settings", { method: "PATCH", body: JSON.stringify({ values }) }); toast("Settings saved", result.restart_required.length ? `Restart required for: ${result.restart_required.join(", ")}` : "New jobs will use these values."); state.settings = result.settings; applyAppearance(result.settings); const scrollPosition = window.scrollY; await refreshDashboard(); await renderSettings(); requestAnimationFrame(() => window.scrollTo({ top: scrollPosition })); }
  catch (error) { toast("Settings were not saved", error.message, "error"); }
  finally { button.disabled = false; }
});

content.addEventListener("input", (event) => {
  if (event.target.id === "job-search") {
    const query = event.target.value.casefold ? event.target.value.casefold() : event.target.value.toLowerCase();
    document.querySelectorAll("#job-list .job-row").forEach((row) => row.hidden = !row.textContent.toLowerCase().includes(query));
  }
  if (event.target.id === "decision-search") {
    const query = event.target.value.toLowerCase();
    document.querySelectorAll("#decision-list .decision-row").forEach((row) => row.hidden = !row.textContent.toLowerCase().includes(query));
  }
});

content.addEventListener("change", (event) => {
  if (event.target.form?.id !== "settings-form") return;
  if (event.target.matches("[data-ocr-profile]")) {
    const hidden = event.target.form.querySelector('[name="ocr_profile"]');
    if (!hidden) return;
    const selected = event.target.checked ? event.target.dataset.ocrProfile : "generic";
    hidden.value = selected;
    event.target.form.querySelectorAll("[data-ocr-profile]").forEach((choice) => {
      choice.checked = selected !== "generic" && choice.dataset.ocrProfile === selected;
    });
    return;
  }
  if (!event.target.name?.startsWith("appearance_")) return;
  const form = new FormData(event.target.form);
  applyAppearance({
    appearance_theme: form.get("appearance_theme"),
    appearance_density: form.get("appearance_density"),
    appearance_motion: form.get("appearance_motion"),
  }, false);
});

async function navigateFromHash() {
  const anchor = location.hash.slice(1);
  const requested = anchor.split("-")[0] || "overview";
  state.route = pageMeta[requested] ? requested : "overview";
  await renderRoute();
  if (anchor.startsWith("settings-")) requestAnimationFrame(() => document.getElementById(anchor)?.scrollIntoView({ behavior: document.documentElement.dataset.motion === "reduced" || (document.documentElement.dataset.motion === "system" && reducedMotionQuery.matches) ? "auto" : "smooth", block: "start" }));
}

window.addEventListener("hashchange", navigateFromHash);

async function initialize() {
  try { state.settings = await api("/api/settings"); applyAppearance(state.settings); }
  catch { /* cached or system appearance remains active */ }
  try { const health = await api("/api/health"); document.querySelector("#app-version").textContent = `Paperless Clerk ${health.version}`; }
  catch { /* main route will show the error */ }
  await navigateFromHash();
  state.poll = setInterval(async () => { if (document.hidden || state.route === "settings" || drawer.classList.contains("open")) return; try { await renderRoute({ quiet: true }); } catch { /* retain last good screen */ } }, 5000);
}

initialize();
