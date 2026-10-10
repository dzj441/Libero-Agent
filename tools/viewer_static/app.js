"use strict";

const state = {
  runs: [],
  detail: null,
  selectedRun: null,
  selectedStep: 0,
};

const elements = {
  loading: document.querySelector("#loading"),
  error: document.querySelector("#error"),
  content: document.querySelector("#content"),
  runSelect: document.querySelector("#run-select"),
  refresh: document.querySelector("#refresh-button"),
  status: document.querySelector("#run-status"),
  taskTitle: document.querySelector("#task-title"),
  subtitle: document.querySelector("#run-subtitle"),
  summaryGrid: document.querySelector("#summary-grid"),
  video: document.querySelector("#run-video"),
  noVideo: document.querySelector("#no-video"),
  sessionOverview: document.querySelector("#session-overview"),
  taskPrompt: document.querySelector("#task-prompt"),
  runtimeUserContext: document.querySelector("#runtime-user-context"),
  baseInstructions: document.querySelector("#base-instructions"),
  developerMessages: document.querySelector("#developer-messages"),
  runtimeSettings: document.querySelector("#runtime-settings"),
  sessionCoverage: document.querySelector("#session-coverage"),
  timeline: document.querySelector("#timeline"),
  previousStep: document.querySelector("#previous-step"),
  nextStep: document.querySelector("#next-step"),
  stepCounter: document.querySelector("#step-counter"),
  stepTitle: document.querySelector("#step-title"),
  activityList: document.querySelector("#activity-list"),
  actionStatus: document.querySelector("#action-status"),
  actionRequest: document.querySelector("#action-request"),
  actionResponse: document.querySelector("#action-response"),
  observationTitle: document.querySelector("#observation-title"),
  observationImages: document.querySelector("#observation-images"),
  observationDownloads: document.querySelector("#observation-downloads"),
  observationState: document.querySelector("#observation-state"),
  observationProprioception: document.querySelector("#observation-proprioception"),
  observationConventions: document.querySelector("#observation-conventions"),
  tailPanel: document.querySelector("#tail-panel"),
  tailActivity: document.querySelector("#tail-activity"),
  activityTemplate: document.querySelector("#activity-template"),
};

function apiUrl(path, params = {}) {
  const url = new URL(path, window.location.href);
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) url.searchParams.set(key, String(value));
  }
  return url.toString();
}

function artifactUrl(artifact, download = false) {
  return apiUrl("api/artifact", {
    run: state.selectedRun,
    path: artifact,
    download: download ? 1 : 0,
  });
}

async function getJson(path, params = {}) {
  const response = await fetch(apiUrl(path, params), { cache: "no-store" });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}

function setLoading(active) {
  elements.loading.classList.toggle("hidden", !active);
  if (active) elements.content.classList.add("hidden");
}

function showError(error) {
  elements.error.textContent = error instanceof Error ? error.message : String(error);
  elements.error.classList.remove("hidden");
  elements.loading.classList.add("hidden");
  elements.content.classList.add("hidden");
}

function clearError() {
  elements.error.classList.add("hidden");
  elements.error.textContent = "";
}

function json(value) {
  return JSON.stringify(value ?? {}, null, 2);
}

function formatSeconds(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  return `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
}

function formatInteger(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString() : "—";
}

function summaryItem(label, value) {
  const item = document.createElement("div");
  item.className = "summary-item";
  const name = document.createElement("span");
  name.textContent = label;
  const content = document.createElement("strong");
  content.textContent = value ?? "—";
  content.title = content.textContent;
  item.append(name, content);
  return item;
}

function renderRunHeader(detail) {
  const summary = detail.summary || {};
  const session = detail.session || {};
  const completedStages = Number(summary.completed_required_stage_count);
  const requiredStages = Number(summary.required_stage_count);
  const orderedStageItems =
    summary.completed_required_stage_count != null &&
    summary.required_stage_count != null &&
    Number.isFinite(completedStages) &&
    Number.isFinite(requiredStages)
      ? [
          summaryItem(
            "Ordered stages",
            `${completedStages}/${requiredStages}` +
              (Number.isFinite(Number(summary.stage_score_percent))
                ? ` (${Number(summary.stage_score_percent).toFixed(1)}%)`
                : ""),
          ),
        ]
      : [];
  const success = summary.success === true;
  const failed = summary.success === false;
  elements.status.className = `status-pill ${success ? "success" : failed ? "failure" : ""}`;
  elements.status.textContent = success
    ? "✓ official success"
    : failed
      ? "× official failure"
      : summary.status || "in progress";
  elements.taskTitle.textContent =
    summary.curriculum_name || summary.task_instruction || summary.name || summary.id;
  elements.subtitle.textContent = `${summary.id} · ${summary.suite || "unknown suite"} / task ${summary.task_id ?? "?"}`;
  elements.summaryGrid.replaceChildren(
    summaryItem("Profile", summary.profile),
    summaryItem("ICL", summary.icl),
    summaryItem("Context", summary.experience_context_id),
    summaryItem("Action interface", summary.action_interface),
    summaryItem("Robot steps", summary.accepted_agent_steps ?? summary.action_count),
    ...orderedStageItems,
    summaryItem("Session", session.session_id || "unavailable"),
    summaryItem("CLI", session.cli_version),
    summaryItem("Tokens", formatInteger(session.token_usage?.total_tokens)),
    summaryItem("Duration", formatSeconds((session.completion?.duration_ms || 0) / 1000)),
    summaryItem(
      "Viewer coverage",
      session.coverage?.viewer_complete ? "complete" : "warning",
    ),
    summaryItem(
      "Alignment",
      `${detail.alignment?.matched_robot_commands ?? 0}/${detail.alignment?.action_records ?? 0}`,
    ),
  );
}

function renderVideo(detail, episodeIndex = null) {
  const selected =
    (detail.videos || []).find((item) => item.episode_index === episodeIndex) ||
    detail.video;
  if (selected?.artifact) {
    const source = artifactUrl(selected.artifact);
    if (elements.video.src !== source) elements.video.src = source;
    elements.video.classList.remove("hidden");
    elements.noVideo.classList.add("hidden");
  } else {
    elements.video.removeAttribute("src");
    elements.video.load();
    elements.video.classList.add("hidden");
    elements.noVideo.classList.remove("hidden");
  }
}

function renderSession(detail) {
  const session = detail.session || {};
  const facts = document.createElement("div");
  facts.className = "session-facts";
  const entries = [
    ["session", session.session_id],
    ["cwd", session.cwd],
    ["provider", session.model_provider],
    ["origin", session.originator],
    ["source", session.source],
    ["thread source", session.thread_source],
    ["resumable", String(session.episode_resumable)],
  ];
  for (const [name, value] of entries) {
    if (!value || value === "undefined") continue;
    const item = document.createElement("span");
    item.className = "session-fact";
    item.textContent = `${name}: ${value}`;
    facts.append(item);
  }
  elements.sessionOverview.replaceChildren(facts);
  elements.taskPrompt.textContent =
    (session.task_user_messages || []).join("\n\n---\n\n") || "Unavailable";
  elements.runtimeUserContext.textContent =
    (session.runtime_user_messages || []).join("\n\n---\n\n") || "None recorded";
  elements.baseInstructions.textContent = session.base_instructions || "Unavailable";
  elements.developerMessages.textContent =
    (session.developer_messages || []).join("\n\n---\n\n") || "Unavailable";
  elements.runtimeSettings.textContent = json(session.runtime_settings);
  elements.sessionCoverage.textContent = json(session.coverage);
}

function renderTimeline() {
  const steps = state.detail?.steps || [];
  elements.timeline.replaceChildren();
  let previousEpisode = null;
  for (const step of steps) {
    if (step.episode_index !== null && step.episode_index !== previousEpisode) {
      const marker = document.createElement("span");
      marker.className = "timeline-episode";
      marker.textContent = `Episode ${step.episode_index + 1}`;
      elements.timeline.append(marker);
      previousEpisode = step.episode_index;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = `timeline-step ${step.index === state.selectedStep ? "active" : ""}`;
    const number = document.createElement("span");
    number.className = "step-number";
    number.textContent =
      step.episode_index === null
        ? `A${step.index}`
        : `E${step.episode_index + 1}·A${step.episode_action_index}`;
    const command = document.createElement("span");
    command.className = "step-command";
    command.textContent = step.command;
    button.append(number, command);
    button.addEventListener("click", () => selectStep(step.index));
    elements.timeline.append(button);
  }
  const active = elements.timeline.querySelector(".timeline-step.active");
  if (active) active.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
}

function elapsedLabel(value) {
  const seconds = Number(value);
  return Number.isFinite(seconds) ? `+${formatSeconds(seconds)}` : "";
}

function renderActivities(container, activity) {
  container.replaceChildren();
  if (!activity?.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "这段时间没有公开的 Codex activity。";
    container.append(empty);
    return;
  }
  for (const item of activity) {
    const fragment = elements.activityTemplate.content.cloneNode(true);
    const card = fragment.querySelector(".activity-card");
    card.classList.add(String(item.kind || "activity").replaceAll("_", "-"));
    fragment.querySelector(".activity-kind").textContent = item.label || item.kind;
    fragment.querySelector(".activity-time").textContent = elapsedLabel(item.elapsed_seconds);
    const title = fragment.querySelector(".activity-title");
    title.textContent = item.title || "";
    title.classList.toggle("hidden", !item.title);
    const parts = fragment.querySelector(".activity-parts");
    for (const text of item.parts || []) {
      const paragraph = document.createElement("p");
      paragraph.textContent = text;
      parts.append(paragraph);
    }
    if (item.artifact) {
      const image = document.createElement("img");
      image.loading = "lazy";
      image.alt = item.title || "Viewed image";
      image.src = artifactUrl(item.artifact);
      fragment.querySelector(".activity-image").append(image);
    }
    const details = fragment.querySelector(".activity-details");
    if (item.details && Object.keys(item.details).length) {
      details.querySelector("pre").textContent = json(item.details);
    } else {
      details.classList.add("hidden-details");
    }
    container.append(fragment);
  }
}

function renderObservation(observation) {
  elements.observationImages.replaceChildren();
  elements.observationDownloads.replaceChildren();
  if (!observation) {
    elements.observationTitle.textContent = "No returned observation";
    elements.observationState.textContent = "{}";
    elements.observationProprioception.textContent = "{}";
    elements.observationConventions.textContent = "{}";
    return;
  }
  elements.observationTitle.textContent = `${observation.observation_id} · ${observation.profile || ""}`;
  for (const image of observation.images || []) {
    if (!image.artifact) continue;
    const card = document.createElement("figure");
    card.className = "image-card";
    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = image.label;
    img.src = artifactUrl(image.artifact);
    const label = document.createElement("figcaption");
    label.className = "image-label";
    label.textContent = image.label;
    card.append(img, label);
    elements.observationImages.append(card);
  }
  for (const download of observation.downloads || []) {
    if (!download.artifact) continue;
    const link = document.createElement("a");
    link.href = artifactUrl(download.artifact, true);
    link.textContent = `下载 ${download.label}`;
    elements.observationDownloads.append(link);
  }
  if (observation.observation_file) {
    const link = document.createElement("a");
    link.href = artifactUrl(observation.observation_file, true);
    link.textContent = "下载 observation.json";
    elements.observationDownloads.append(link);
  }
  elements.observationState.textContent = json(observation.state);
  elements.observationProprioception.textContent = json(observation.proprioception);
  elements.observationConventions.textContent = json(observation.coordinate_conventions);
}

function renderSelectedStep() {
  const steps = state.detail?.steps || [];
  const step = steps[state.selectedStep];
  if (!step) return;
  elements.stepCounter.textContent = `${state.selectedStep + 1} / ${steps.length}`;
  elements.previousStep.disabled = state.selectedStep <= 0;
  elements.nextStep.disabled = state.selectedStep >= steps.length - 1;
  const actionLabel =
    step.episode_index === null
      ? `A${step.index}`
      : `E${step.episode_index + 1}·A${step.episode_action_index}`;
  elements.stepTitle.textContent = `${actionLabel} · activity before robot ${step.command}`;
  renderVideo(state.detail, step.episode_index);
  renderActivities(elements.activityList, step.agent_activity || []);
  elements.actionStatus.replaceChildren();
  const status = document.createElement("span");
  status.className = `mini-pill ${step.ok ? "success" : "failure"}`;
  status.textContent = step.ok ? "✓ host accepted" : "× rejected / failed";
  elements.actionStatus.append(status);
  elements.actionRequest.textContent = json(step.request);
  elements.actionResponse.textContent = json(step.response);
  renderObservation(step.output_observation);
}

function renderTail(detail) {
  const tail = detail.tail_activity || [];
  elements.tailPanel.classList.toggle("hidden", tail.length === 0);
  if (tail.length) renderActivities(elements.tailActivity, tail);
}

function selectStep(index) {
  const steps = state.detail?.steps || [];
  if (!steps.length) return;
  state.selectedStep = Math.max(0, Math.min(index, steps.length - 1));
  renderTimeline();
  renderSelectedStep();
}

function renderDetail(detail) {
  state.detail = detail;
  state.selectedStep = 0;
  renderRunHeader(detail);
  renderVideo(detail);
  renderSession(detail);
  renderTimeline();
  renderSelectedStep();
  renderTail(detail);
  elements.loading.classList.add("hidden");
  elements.content.classList.remove("hidden");
}

async function loadRun(runId) {
  if (!runId) return;
  clearError();
  setLoading(true);
  state.selectedRun = runId;
  elements.runSelect.value = runId;
  const url = new URL(window.location.href);
  url.searchParams.set("run", runId);
  window.history.replaceState({}, "", url);
  try {
    renderDetail(await getJson("api/run", { run: runId }));
  } catch (error) {
    showError(error);
  }
}

async function loadRuns() {
  clearError();
  setLoading(true);
  try {
    const response = await getJson("api/runs");
    state.runs = response.runs || [];
    elements.runSelect.replaceChildren();
    for (const run of state.runs) {
      const option = document.createElement("option");
      option.value = run.id;
      const result = run.success === true ? "✓" : run.success === false ? "×" : "…";
      option.textContent = `${result} ${run.name} · ${run.profile || "?"} · ${run.icl || "none"} · ${run.action_interface || "legacy"}`;
      elements.runSelect.append(option);
    }
    if (!state.runs.length) throw new Error("Runs root 中没有可显示的 LIBERO Agent run。 ");
    const requested = new URL(window.location.href).searchParams.get("run");
    const selected = state.runs.some((run) => run.id === requested)
      ? requested
      : state.selectedRun && state.runs.some((run) => run.id === state.selectedRun)
        ? state.selectedRun
        : state.runs[0].id;
    await loadRun(selected);
  } catch (error) {
    showError(error);
  }
}

elements.runSelect.addEventListener("change", () => loadRun(elements.runSelect.value));
elements.refresh.addEventListener("click", loadRuns);
elements.previousStep.addEventListener("click", () => selectStep(state.selectedStep - 1));
elements.nextStep.addEventListener("click", () => selectStep(state.selectedStep + 1));
window.addEventListener("keydown", (event) => {
  if (event.key === "ArrowLeft") selectStep(state.selectedStep - 1);
  if (event.key === "ArrowRight") selectStep(state.selectedStep + 1);
});

loadRuns();
