const state = {
  selectedPipeline: null,
  steps: [],
  contextStack: [],
  toolCatalog: { order: [], byServer: {} },
  parameterData: null,
  editingPath: null,
  isBuilt: false,
  parametersReady: false,
  mode: "builder",
  logStream: { runId: null, lastId: -1, timer: null, status: "idle" },
  shutdownScheduled: false,
  // Updated Chat State, adding sessions and currentSessionId
  chat: { 
    history: [], 
    running: false,
    sessions: [], // Stores { id, title, messages: [] }
    currentSessionId: null
  },
};

const LOG_POLL_INTERVAL = 1500;

const els = {
  // View Containers
  mainRoot: document.querySelector(".content-wrapper"),
  pipelineForm: document.getElementById("pipeline-form"),
  parameterPanel: document.getElementById("parameter-panel"),
  chatView: document.getElementById("chat-view"),
  runView: document.getElementById("run-view"),
  
  // Logs
  log: document.getElementById("log"),
  runTerminal: document.getElementById("run-terminal"),
  runSpinner: document.getElementById("run-spinner"),

  // Controls
  name: document.getElementById("pipeline-name"),
  flowCanvas: document.getElementById("flow-canvas"),
  contextControls: document.getElementById("context-controls"),
  pipelinePreview: document.getElementById("pipeline-preview"),
  stepEditor: document.getElementById("step-editor"),
  stepEditorValue: document.getElementById("step-editor-value"),
  clearSteps: document.getElementById("clear-steps"),
  savePipeline: document.getElementById("save-pipeline"),
  buildPipeline: document.getElementById("build-pipeline"),
  deletePipeline: document.getElementById("delete-pipeline"),
  pipelineDropdownBtn: document.getElementById("pipelineDropdownBtn"),
  pipelineMenu: document.getElementById("pipeline-menu"),
  refreshPipelines: document.getElementById("refresh-pipelines"),
  newPipelineBtn: document.getElementById("new-pipeline-btn"),
  shutdownApp: document.getElementById("shutdown-app"),
  heroSelectedPipeline: document.getElementById("hero-selected-pipeline"),
  heroStatus: document.getElementById("hero-status"),
  
  // Parameter Controls
  parameterForm: document.getElementById("parameter-form"),
  parameterSave: document.getElementById("parameter-save"),
  parameterBack: document.getElementById("parameter-back"),
  parameterRun: document.getElementById("parameter-run"),
  parameterChat: document.getElementById("parameter-chat"),
  
  // Run Back
  runBack: document.getElementById("run-back"),
  
  // Chat Controls (Updated)
  chatPipelineName: document.getElementById("chat-pipeline-name"),
  chatBack: document.getElementById("chat-back"),
  chatHistory: document.getElementById("chat-history"),
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
  chatStatus: document.getElementById("chat-status"),
  chatSend: document.getElementById("chat-send"),
  chatNewBtn: document.getElementById("chat-new-btn"), // New
  chatSessionList: document.getElementById("chat-session-list"), // New
  
  // Node Picker
  nodePickerModal: document.getElementById("nodePickerModal"),
  nodePickerTabs: document.querySelectorAll("[data-node-mode]"),
  nodePickerServer: document.getElementById("node-picker-server"),
  nodePickerTool: document.getElementById("node-picker-tool"),
  nodePickerBranchCases: document.getElementById("node-picker-branch-cases"),
  nodePickerLoopTimes: document.getElementById("node-picker-loop-times"),
  nodePickerCustom: document.getElementById("node-picker-custom"),
  nodePickerPanels: {
    tool: document.getElementById("node-picker-tool-panel"),
    branch: document.getElementById("node-picker-branch-panel"),
    loop: document.getElementById("node-picker-loop-panel"),
    custom: document.getElementById("node-picker-custom-panel"),
  },
  nodePickerError: document.getElementById("node-picker-error"),
  nodePickerConfirm: document.getElementById("nodePickerConfirm"),
};

const Modes = {
  BUILDER: "builder",
  PARAMETERS: "parameters",
  RUN: "run",
  CHAT: "chat",
};

const nodePickerState = {
  mode: "tool",
  server: null,
  tool: null,
  branchCases: "case1, case2",
  loopTimes: 2,
  customValue: "",
};

let nodePickerModalInstance = null;
let pendingInsert = null;

// --- Logging ---
function log(message) {
  const stamp = new Date().toLocaleTimeString();
  const msg = `> [${stamp}] ${message}`;
  if (els.log) { els.log.textContent += msg + "\n"; els.log.scrollTop = els.log.scrollHeight; }
  if (state.mode === Modes.RUN && els.runTerminal) logToTerminal(msg); 
  else console.log(msg);
}
function logToTerminal(msg) {
    if (!els.runTerminal) return;
    els.runTerminal.textContent += msg + "\n";
    const container = els.runTerminal.parentElement;
    if (container) container.scrollTop = container.scrollHeight;
}

function createNewPipeline() {
  if (state.steps.length > 0) {
    if (!confirm("Create new pipeline? Unsaved changes will be lost.")) return;
  }
  state.selectedPipeline = null; state.parameterData = null; state.steps = []; state.isBuilt = false; state.parametersReady = false;
  els.name.value = ""; if (els.pipelineDropdownBtn) els.pipelineDropdownBtn.textContent = "Select Pipeline";
  setHeroPipelineLabel(""); setHeroStatusLabel("idle");
  resetContextStack(); renderSteps(); updatePipelinePreview(); setMode(Modes.BUILDER); updateActionButtons();
  log("Created new blank pipeline.");
}

// --- Chat Session Management ---

function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

function createNewChatSession() {
    // If the current session is not empty, save it before creating a new one
    if (state.chat.history.length > 0) {
        saveCurrentSession(true); // Force save for existing, non-empty session
    }
    
    state.chat.currentSessionId = generateId();
    state.chat.history = [];
    renderChatHistory();
    renderChatSidebar();
    setChatStatus("Ready", "ready");
    if(els.chatInput) els.chatInput.focus();
    log("Started new chat session.");
}

function loadChatSession(sessionId) {
    if (state.chat.running) return; // Don't switch while generating
    
    // Save current session before loading a new one, in case it was modified
    saveCurrentSession(false); 
    
    const session = state.chat.sessions.find(s => s.id === sessionId);
    if (!session) return;

    state.chat.currentSessionId = session.id;
    state.chat.history = [...session.messages]; // Copy messages
    renderChatHistory();
    renderChatSidebar();
    setChatStatus("Ready", "ready");
    log(`Loaded chat session: ${session.title}`);
}

function saveCurrentSession(force = false) {
    if (!state.chat.currentSessionId) return;

    // Only save non-empty sessions, unless forced
    if (!force && state.chat.history.length === 0) {
        // If an empty session exists in the list, remove it
        state.chat.sessions = state.chat.sessions.filter(s => s.id !== state.chat.currentSessionId);
        renderChatSidebar();
        return;
    }
    
    let session = state.chat.sessions.find(s => s.id === state.chat.currentSessionId);
    
    // Generate Title from first user message if it's a new session or title is default
    let title = "New Chat";
    const firstUserMsg = state.chat.history.find(m => m.role === 'user');
    if (firstUserMsg) {
        title = firstUserMsg.text.slice(0, 20) + (firstUserMsg.text.length > 20 ? "..." : "");
    }

    if (!session) {
        session = { id: state.chat.currentSessionId, title: title, messages: [] };
        state.chat.sessions.unshift(session); // Add to top
    } else {
        // Update existing
        // Move to top on update
        state.chat.sessions = state.chat.sessions.filter(s => s.id !== state.chat.currentSessionId);
        state.chat.sessions.unshift(session);
        
        if (session.title === "New Chat" || (session.messages.length === 0 && firstUserMsg)) {
             session.title = title;
        }
    }
    session.messages = [...state.chat.history]; // Update messages
    renderChatSidebar();
}

function renderChatSidebar() {
    if (!els.chatSessionList) return;
    els.chatSessionList.innerHTML = "";
    
    state.chat.sessions.forEach(session => {
        const btn = document.createElement("button");
        btn.className = `chat-session-item ${session.id === state.chat.currentSessionId ? 'active' : ''}`;
        btn.textContent = session.title || "Untitled Chat";
        btn.onclick = () => loadChatSession(session.id);
        els.chatSessionList.appendChild(btn);
    });
}

// --- Chat Logic ---
function resetChatSession() {
    // This is for pipeline reset, clears all chat data
    state.chat.history = [];
    state.chat.running = false;
    state.chat.sessions = [];
    state.chat.currentSessionId = null;
    renderChatHistory(); 
    renderChatSidebar();
    setChatStatus("Ready", "ready");
}

function appendChatMessage(role, text, meta = {}) {
  const entry = { role, text, meta, timestamp: new Date().toISOString() };
  state.chat.history.push(entry);
  renderChatHistory();
  
  // Save to session list immediately
  saveCurrentSession(); 
}

function renderChatHistory() {
  if (!els.chatHistory) return;
  els.chatHistory.innerHTML = "";
  if (state.chat.history.length === 0) { els.chatHistory.innerHTML = '<div class="text-center mt-5 pt-5 text-muted small"><p>Ready to start a conversation.</p></div>'; return; }
  state.chat.history.forEach((entry) => {
    const bubble = document.createElement("div"); bubble.className = `chat-bubble ${entry.role}`;
    const content = document.createElement("div"); content.textContent = entry.text; bubble.appendChild(content);
    if (entry.meta && entry.meta.hint) {
        const metaLine = document.createElement("small"); metaLine.className = "text-muted d-block mt-1";
        metaLine.style.fontSize = "0.7em"; metaLine.textContent = entry.meta.hint; bubble.appendChild(metaLine);
    }
    els.chatHistory.appendChild(bubble);
  });
  els.chatHistory.scrollTop = els.chatHistory.scrollHeight;
}
function setChatStatus(message, variant = "info") {
  if (!els.chatStatus) return;
  const badge = els.chatStatus;
  const variants = { info: "bg-light text-dark", ready: "bg-light text-dark", running: "bg-primary text-white", success: "bg-success text-white", warn: "bg-warning text-dark", error: "bg-danger text-white" };
  badge.className = `badge rounded-pill border ${variants[variant] || variants.info}`; badge.textContent = message || "";
}
function setChatRunning(isRunning) {
  state.chat.running = isRunning;
  if (els.chatInput) els.chatInput.disabled = isRunning;
  if (els.chatSend) els.chatSend.disabled = isRunning;
  if (els.chatBack) els.chatBack.disabled = isRunning;
  if (isRunning) setChatStatus("Thinking...", "running"); else updateActionButtons();
}
function canUseChat() { return Boolean(state.isBuilt && state.selectedPipeline && state.parameterData); }
function openChatView() {
  if (!canUseChat()) { log("Please build and save parameters first."); return; }
  if (els.chatPipelineName) els.chatPipelineName.textContent = state.selectedPipeline || "—";
  
  // Initialize session if needed
  if (!state.chat.currentSessionId) {
      createNewChatSession();
  }
  
  renderChatHistory();
  renderChatSidebar(); // Render sidebar
  setMode(Modes.CHAT);
  setChatRunning(state.chat.running);
  
  if (!state.chat.running && state.chat.history.length === 0) setChatStatus("Ready", "ready");
  if (!state.chat.running && els.chatInput) els.chatInput.focus();
}
async function handleChatSubmit(event) {
  event.preventDefault();
  if (!canUseChat()) return; if (state.chat.running) return;
  const question = (els.chatInput ? els.chatInput.value : "").trim(); if (!question) return;
  if (els.chatInput) els.chatInput.value = "";
  appendChatMessage("user", question); setChatRunning(true);
  try {
    if (!state.parametersReady) await persistParameterData({ silent: true });
    
    // Pass session history with request
    const endpoint = `/api/pipelines/${encodeURIComponent(state.selectedPipeline)}/chat`;
    const body = JSON.stringify({ question, history: state.chat.history });
    
    const resp = await fetchJSON(endpoint, { method: "POST", body: body });
    const status = resp.status || "unknown"; const answer = resp.answer || resp.result || "No answer received";
    const hints = []; if (resp.dataset_path) hints.push(`Dataset: ${resp.dataset_path}`); if (resp.memory_path) hints.push(`Memory: ${resp.memory_path}`);
    appendChatMessage("assistant", answer, { hint: hints.join(" | ") });
    if (status !== "succeeded") appendChatMessage("system", `Ended with status: ${status} ${resp.error || ''}`);
    setChatStatus("Done", status === "succeeded" ? "success" : "warn");
  } catch (err) { appendChatMessage("system", `Error: ${err.message || err}`); setChatStatus("Error", "error"); } finally { setChatRunning(false); }
}

// --- Status & Log Stream ---
function resetLogView() { if (els.log) els.log.textContent = ""; if (els.runTerminal) els.runTerminal.textContent = ""; }
function setHeroPipelineLabel(name) { if (els.heroSelectedPipeline) els.heroSelectedPipeline.textContent = name ? name : "No Pipeline Selected"; }
function setHeroStatusLabel(status) {
  if (!els.heroStatus) return;
  els.heroStatus.dataset.status = status; els.heroStatus.textContent = status.toUpperCase();
  if (els.runSpinner) { if (status === "running") els.runSpinner.classList.remove("d-none"); else els.runSpinner.classList.add("d-none"); }
}
function scheduleWindowClose() {
  if (state.shutdownScheduled) return; state.shutdownScheduled = true; log("Shutdown command sent. Closing window...");
  setTimeout(() => { try { window.close(); } catch (e) {} window.location.replace("about:blank"); }, 800);
}
function requestShutdown() { if (!window.confirm("Exit UltraRAG UI?")) return; stopRunLogStream(); fetch("/api/system/shutdown", { method: "POST" }).finally(scheduleWindowClose); }
function stopRunLogStream(finalStatus = "idle") {
  if (state.logStream.timer) clearTimeout(state.logStream.timer);
  state.logStream.timer = null; state.logStream.runId = null; state.logStream.lastId = -1; state.logStream.status = finalStatus;
  setHeroStatusLabel(finalStatus);
}
async function pollRunLogs() {
  if (!state.logStream.runId) return;
  const params = new URLSearchParams(); params.set("since", String(state.logStream.lastId)); params.set("run_id", state.logStream.runId);
  try {
    const data = await fetchJSON(`/api/logs/run?${params.toString()}`);
    if (data.reset) { resetLogView(); state.logStream.lastId = -1; }
    const entries = data.entries || [];
    entries.forEach((entry) => { state.logStream.lastId = Math.max(state.logStream.lastId, entry.id); if (entry.message) log(entry.message); });
    const status = data.status || {}; const stateValue = status.state || "running"; state.logStream.status = stateValue; setHeroStatusLabel(stateValue);
    if (stateValue === "running") { state.logStream.timer = window.setTimeout(pollRunLogs, LOG_POLL_INTERVAL); } else { stopRunLogStream(stateValue); }
  } catch (err) { stopRunLogStream("failed"); }
}
function startRunLogStream(runId) { if (!runId) return; stopRunLogStream("idle"); state.logStream.runId = runId; state.logStream.lastId = -1; state.logStream.status = "running"; setHeroStatusLabel("running"); pollRunLogs(); }

async function fetchJSON(url, options = {}) {
  const resp = await fetch(url, { headers: { "Content-Type": "application/json" }, ...options });
  if (!resp.ok) { const text = await resp.text(); throw new Error(text || resp.statusText); }
  return resp.json();
}

async function persistParameterData({ silent = false } = {}) {
  if (!state.selectedPipeline || !state.parameterData) throw new Error("No parameters to save");
  await fetchJSON(`/api/pipelines/${encodeURIComponent(state.selectedPipeline)}/parameters`, { method: "PUT", body: JSON.stringify(state.parameterData) });
  state.parametersReady = true; updateActionButtons(); if (!silent) log("Parameters saved.");
}

// --- Helpers ---
function cloneDeep(value) { return value === undefined ? undefined : JSON.parse(JSON.stringify(value)); }
function createLocation(segments = []) { return { segments: segments.map((seg) => ({ ...seg })) }; }
function locationsEqual(a, b) { return JSON.stringify((a && a.segments) || []) === JSON.stringify((b && b.segments) || []); }
function getContextKind(location) {
  const segments = (location && location.segments) || []; if (!segments.length) return "root";
  const last = segments[segments.length - 1];
  if (last.type === "loop") return "loop"; if (last.type === "branch") return last.section === "router" ? "branch-router" : "branch-case";
  return "root";
}
function resolveSteps(location) {
  let steps = state.steps; const segments = (location && location.segments) || [];
  for (const seg of segments) {
    const entry = steps[seg.index]; if (!entry) return steps;
    if (seg.type === "loop" && entry.loop) { entry.loop.steps = entry.loop.steps || []; steps = entry.loop.steps; }
    else if (seg.type === "branch" && entry.branch) {
      entry.branch.router = entry.branch.router || []; entry.branch.branches = entry.branch.branches || {};
      if (seg.section === "router") steps = entry.branch.router; else if (seg.section === "branch") steps = entry.branch.branches[seg.branchKey] || [];
    }
  }
  return steps;
}
function resolveParentSteps(stepPath) { return resolveSteps(createLocation(stepPath.parentSegments || [])); }
function createStepPath(parentLocation, index) { return { parentSegments: (parentLocation.segments || []).map((seg) => ({ ...seg })), index }; }
function getStepByPath(stepPath) { const steps = resolveParentSteps(stepPath); return steps[stepPath.index]; }
function setStepByPath(stepPath, value) { const steps = resolveParentSteps(stepPath); steps[stepPath.index] = value; markPipelineDirty(); }
function removeStepByPath(stepPath) { const steps = resolveParentSteps(stepPath); steps.splice(stepPath.index, 1); }
function ensureContextInitialized() { if (!state.contextStack.length) state.contextStack = [createLocation([])]; }
function getActiveLocation() { ensureContextInitialized(); return state.contextStack[state.contextStack.length - 1]; }
function setActiveLocation(location) {
  const segments = (location && location.segments) || []; const newStack = [createLocation([])];
  for (let i = 0; i < segments.length; i += 1) newStack.push(createLocation(segments.slice(0, i + 1)));
  state.contextStack = newStack; renderContextControls(); renderSteps(); updatePipelinePreview();
}
function resetContextStack() { state.contextStack = [createLocation([])]; renderContextControls(); }

// --- YAML ---
function yamlScalar(value) {
    if (value === null || value === undefined) return "null";
    if (typeof value === "boolean") return value ? "true" : "false";
    if (typeof value === "number") return Number.isFinite(value) ? String(value) : "null";
    if (typeof value === "string") return value; return JSON.stringify(value);
}
function yamlStringify(value, indent = 0) {
    const pad = "  ".repeat(indent);
    if (Array.isArray(value)) {
        if (!value.length) return `${pad}[]`;
        return value.map(item => { if (item && typeof item === "object") return `${pad}-\n${yamlStringify(item, indent + 1)}`; return `${pad}- ${yamlScalar(item)}`; }).join("\n");
    }
    if (value && typeof value === "object") {
        const entries = Object.entries(value); if (!entries.length) return `${pad}{}`;
        return entries.map(([k, v]) => { if (v && typeof v === "object") return `${pad}${k}:\n${yamlStringify(v, indent + 1)}`; return `${pad}${k}: ${yamlScalar(v)}`; }).join("\n");
    }
    return `${pad}${yamlScalar(value)}`;
}
function collectServersFromSteps(steps, set = new Set()) {
    for (const step of steps) {
        if (typeof step === "string") { const parts = step.split("."); if (parts.length > 1) set.add(parts[0]); }
        else if (step && typeof step === "object") {
            if (step.loop && Array.isArray(step.loop.steps)) collectServersFromSteps(step.loop.steps, set);
            else if (step.branch) { collectServersFromSteps(step.branch.router || [], set); Object.values(step.branch.branches || {}).forEach(bs => collectServersFromSteps(bs || [], set)); }
        }
    }
    return set;
}
function buildServersMapping(steps) { const mapping = {}; collectServersFromSteps(steps, new Set()).forEach((name) => { mapping[name] = `servers/${name}`; }); return mapping; }
function buildPipelinePayloadForPreview() { return { servers: buildServersMapping(state.steps), pipeline: cloneDeep(state.steps) }; }
function updatePipelinePreview() { if (els.pipelinePreview) els.pipelinePreview.textContent = yamlStringify(buildPipelinePayloadForPreview()); }

// --- View Switching ---
function setMode(mode) {
  state.mode = mode;
  if (els.pipelineForm) els.pipelineForm.classList.toggle("d-none", mode !== Modes.BUILDER);
  if (els.parameterPanel) els.parameterPanel.classList.toggle("d-none", mode !== Modes.PARAMETERS);
  if (els.chatView) els.chatView.classList.toggle("d-none", mode !== Modes.CHAT);
  if (els.runView) els.runView.classList.toggle("d-none", mode !== Modes.RUN);
}

// --- Node Picker ---
function getNodePickerModal() {
    const modalElement = els.nodePickerModal; if (!modalElement) return null;
    if (!nodePickerModalInstance) {
        // Fallback for missing Bootstrap in environment
        if (typeof window.bootstrap !== 'undefined' && window.bootstrap.Modal) {
            nodePickerModalInstance = new window.bootstrap.Modal(modalElement, { backdrop: "static" });
            modalElement.addEventListener("hidden.bs.modal", () => { pendingInsert = null; clearNodePickerError(); });
        } else {
            const body = document.body;
            let fallbackHandlers = {
                show() {
                    modalElement.classList.add("show"); modalElement.style.display = "block"; modalElement.removeAttribute("aria-hidden");
                    let backdrop = document.querySelector('.modal-backdrop'); if (!backdrop) { backdrop = document.createElement('div'); backdrop.className = 'modal-backdrop fade show'; body.appendChild(backdrop); }
                    body.classList.add("modal-open"); body.style.overflow = "hidden";
                },
                hide() {
                    modalElement.classList.remove("show"); modalElement.style.display = "none"; modalElement.setAttribute("aria-hidden", "true");
                    const backdrop = document.querySelector('.modal-backdrop'); if (backdrop) backdrop.remove();
                    body.classList.remove("modal-open"); body.style.overflow = ""; pendingInsert = null; clearNodePickerError();
                }
            };
            modalElement.querySelectorAll('[data-bs-dismiss="modal"]').forEach(btn => btn.onclick = () => fallbackHandlers.hide());
            nodePickerModalInstance = fallbackHandlers;
        }
    }
    return nodePickerModalInstance;
}
function clearNodePickerError() { if (els.nodePickerError) els.nodePickerError.classList.add("d-none"); }
function showNodePickerError(msg) { if (els.nodePickerError) { els.nodePickerError.textContent = msg; els.nodePickerError.classList.remove("d-none"); } }
function populateNodePickerTools() {
    if (!els.nodePickerTool) return;
    const select = els.nodePickerTool; select.innerHTML = "";
    const server = nodePickerState.server; const tools = (server && state.toolCatalog.byServer[server]) || [];
    if (!tools.length) { const option = document.createElement("option"); option.textContent = server ? "No tools" : "Select Server"; select.appendChild(option); select.disabled = true; nodePickerState.tool = null; return; }
    select.disabled = false; if (!nodePickerState.tool) nodePickerState.tool = tools[0].tool;
    tools.forEach(t => { const option = document.createElement("option"); option.value = t.tool; option.textContent = t.tool; select.appendChild(option); });
    select.value = nodePickerState.tool || "";
}
function populateNodePickerServers() {
    if (!els.nodePickerServer) return;
    const select = els.nodePickerServer; select.innerHTML = "";
    const servers = state.toolCatalog.order || [];
    if (!servers.length) { const option = document.createElement("option"); option.textContent = "No Servers"; select.appendChild(option); select.disabled = true; return; }
    select.disabled = false; if (!nodePickerState.server) nodePickerState.server = servers[0];
    servers.forEach(s => { const option = document.createElement("option"); option.value = s; option.textContent = s; select.appendChild(option); });
    select.value = nodePickerState.server; populateNodePickerTools();
}
function updateNodePickerInputs() {
  if (els.nodePickerBranchCases) els.nodePickerBranchCases.value = nodePickerState.branchCases || "case1, case2";
  if (els.nodePickerLoopTimes) els.nodePickerLoopTimes.value = nodePickerState.loopTimes || 2;
  if (els.nodePickerCustom) els.nodePickerCustom.value = nodePickerState.customValue || "";
}
function setNodePickerMode(mode) {
  if (!mode) return; nodePickerState.mode = mode;
  if (els.nodePickerTabs) els.nodePickerTabs.forEach(t => t.classList.toggle("active", t.dataset.nodeMode === mode));
  Object.entries(els.nodePickerPanels).forEach(([key, panel]) => { if (panel) panel.classList.toggle("d-none", key !== mode); });
  clearNodePickerError(); if (mode === "tool") populateNodePickerServers(); updateNodePickerInputs();
}
function openNodePicker(location, insertIndex) {
  pendingInsert = { location, index: insertIndex }; if (!nodePickerState.mode) nodePickerState.mode = "tool";
  populateNodePickerServers(); updateNodePickerInputs(); setNodePickerMode(nodePickerState.mode);
  const modal = getNodePickerModal(); if (modal) modal.show();
}
function handleNodePickerConfirm() {
    if (!pendingInsert) { getNodePickerModal()?.hide(); return; }
    const { location, index } = pendingInsert;
    try {
        switch (nodePickerState.mode) {
            case "tool": if (!nodePickerState.server || !nodePickerState.tool) throw new Error("Select a tool"); insertStepAt(location, index, `${nodePickerState.server}.${nodePickerState.tool}`); break;
            case "loop": const times = Math.max(1, Number(nodePickerState.loopTimes) || 1); const p = insertStepAt(location, index, { loop: { times, steps: [] } }); enterStructureContext("loop", p); break;
            case "branch": const cases = (nodePickerState.branchCases || "").split(",").map(c => c.trim()).filter(B => B); const step = { branch: { router: [], branches: {} } }; (cases.length ? cases : ["c1", "c2"]).forEach(k => step.branch.branches[k] = []); const p2 = insertStepAt(location, index, step); enterStructureContext("branch", p2); break;
            case "custom": if (!nodePickerState.customValue) throw new Error("Custom value cannot be empty"); insertStepAt(location, index, parseStepInput(nodePickerState.customValue)); break;
        }
        getNodePickerModal()?.hide(); pendingInsert = null;
    } catch (e) { showNodePickerError(e.message); }
}

// --- Actions ---
function markPipelineDirty() { stopRunLogStream(); state.isBuilt = false; state.parametersReady = false; if (state.mode !== Modes.BUILDER) setMode(Modes.BUILDER); updateActionButtons(); }
function setSteps(steps) { state.steps = Array.isArray(steps) ? cloneDeep(steps) : []; state.parameterData = null; resetChatSession(); markPipelineDirty(); resetContextStack(); renderSteps(); updatePipelinePreview(); }
function updateActionButtons() {
  if (els.parameterRun) els.parameterRun.disabled = !(state.isBuilt && state.parametersReady && state.selectedPipeline);
  if (els.parameterSave) els.parameterSave.disabled = !(state.isBuilt && state.selectedPipeline);
  if (els.parameterChat) els.parameterChat.disabled = state.mode === Modes.CHAT || !canUseChat();
}
function insertStepAt(location, insertIndex, stepValue) {
  const stepsArray = resolveSteps(location); const index = Math.max(0, Math.min(insertIndex, stepsArray.length));
  stepsArray.splice(index, 0, cloneDeep(stepValue)); markPipelineDirty(); setActiveLocation(location); return createStepPath(location, index);
}
function removeStep(stepPath) { removeStepByPath(stepPath); markPipelineDirty(); resetContextStack(); renderSteps(); updatePipelinePreview(); }
function openStepEditor(stepPath) { state.editingPath = stepPath; const step = getStepByPath(stepPath); els.stepEditorValue.value = typeof step === "string" ? step : JSON.stringify(step, null, 2); els.stepEditor.hidden = false; }
function closeStepEditor() { state.editingPath = null; els.stepEditor.hidden = true; }
function parseStepInput(raw) { const t = (raw||"").trim(); if (!t) throw new Error("Empty"); if ((t.startsWith("{")&&t.endsWith("}")) || (t.startsWith("[")&&t.endsWith("]"))) return JSON.parse(t); return t; }
function createInsertControl(location, insertIndex, { prominent = false, compact = false } = {}) {
  const holder = document.createElement("div"); holder.className = "flow-insert-control"; if (prominent) holder.classList.add("prominent");
  const button = document.createElement("button"); button.type = "button"; button.className = "flow-insert-button"; button.title = "Insert Node Here"; button.innerHTML = '<span>+</span><span>Add Node</span>';
  button.addEventListener("click", () => { const pendingLocation = createLocation((location.segments || []).map((seg) => ({ ...seg }))); openNodePicker(pendingLocation, insertIndex); });
  holder.appendChild(button); return holder;
}

function renderToolNode(identifier, stepPath) {
  const card = document.createElement("div"); card.className = "flow-node";
  const header = document.createElement("div"); header.className = "flow-node-header d-flex justify-content-between align-items-center";
  const title = document.createElement("h6"); title.className = "flow-node-title"; title.textContent = identifier; header.appendChild(title);
  const body = document.createElement("div"); body.className = "flow-node-body"; body.textContent = identifier;
  const actions = document.createElement("div"); actions.className = "step-actions";
  const editBtn = document.createElement("button"); editBtn.className = "btn btn-outline-primary btn-sm me-1"; editBtn.textContent = "Edit"; editBtn.onclick = (e) => { e.stopPropagation(); openStepEditor(stepPath); };
  const removeBtn = document.createElement("button"); removeBtn.className = "btn btn-outline-danger btn-sm"; removeBtn.textContent = "Delete"; removeBtn.onclick = (e) => { e.stopPropagation(); removeStep(stepPath); };
  actions.append(editBtn, removeBtn); card.append(header, body, actions); return card;
}
function renderLoopNode(step, parentLocation, index) {
  const loopLocation = createLocation([...(parentLocation.segments || []), { type: "loop", index }]);
  const container = document.createElement("div"); container.className = "loop-container";
  const header = document.createElement("div"); header.className = "loop-header";
  const title = document.createElement("h6"); title.textContent = `LOOP (${step.loop.times}x)`;
  const enterBtn = document.createElement("button"); enterBtn.className = "btn btn-sm btn-link text-decoration-none p-0"; enterBtn.textContent = "Open Context →"; enterBtn.onclick = () => setActiveLocation(loopLocation);
  header.append(title, enterBtn);
  const actions = document.createElement("div"); actions.className = "mt-2 d-flex justify-content-end gap-2";
  const editBtn = document.createElement("button"); editBtn.className = "btn btn-sm btn-outline-secondary border-0"; editBtn.textContent = "Edit"; editBtn.onclick = () => openStepEditor(createStepPath(parentLocation, index));
  const delBtn = document.createElement("button"); delBtn.className = "btn btn-sm btn-outline-danger border-0"; delBtn.textContent = "Delete"; delBtn.onclick = () => removeStep(createStepPath(parentLocation, index));
  actions.append(editBtn, delBtn);
  const list = renderStepList(step.loop.steps || [], loopLocation, { placeholderText: "Empty Loop", compact: true });
  container.append(header, list, actions); if (locationsEqual(loopLocation, getActiveLocation())) container.classList.add("active"); return container;
}
function renderBranchNode(step, parentLocation, index) {
    step.branch.router = step.branch.router || []; step.branch.branches = step.branch.branches || {};
    const branchBase = createLocation([...(parentLocation.segments || []), { type: "branch", index, section: "router" }]);
    const container = document.createElement("div"); container.className = "branch-container";
    const header = document.createElement("div"); header.className = "branch-header"; header.innerHTML = `<h6>BRANCH</h6>`;
    const enterBtn = document.createElement("button"); enterBtn.className = "btn btn-sm btn-link text-decoration-none p-0"; enterBtn.textContent = "Open Router →";
    enterBtn.onclick = () => setActiveLocation(branchBase); 
    header.appendChild(enterBtn);
    const routerDiv = document.createElement("div"); routerDiv.className = "branch-router " + (locationsEqual(branchBase, getActiveLocation()) ? "active" : "");
    routerDiv.appendChild(renderStepList(step.branch.router, branchBase, { placeholderText: "Router Logic", compact: true }));
    const casesDiv = document.createElement("div"); casesDiv.className = "branch-cases mt-3";
    Object.keys(step.branch.branches).forEach(k => {
        const loc = createLocation([...(parentLocation.segments||[]), { type: "branch", index, section: "branch", branchKey: k }]);
        const cCard = document.createElement("div"); cCard.className = "branch-case " + (locationsEqual(loc, getActiveLocation()) ? "active" : "");
        const cHeader = document.createElement("div"); cHeader.className = "d-flex justify-content-between mb-2";
        const cTitle = document.createElement("span"); cTitle.className = "fw-bold text-xs text-uppercase"; cTitle.textContent = `Case: ${k}`;
        const cBtn = document.createElement("button"); cBtn.className = "btn btn-link btn-sm p-0 text-decoration-none"; cBtn.textContent = "Open"; cBtn.onclick = () => setActiveLocation(loc);
        cHeader.append(cTitle, cBtn); cCard.append(cHeader, renderStepList(step.branch.branches[k], loc, { placeholderText: "Empty Case", compact: true })); casesDiv.appendChild(cCard);
    });
    const actions = document.createElement("div"); actions.className = "mt-2 d-flex justify-content-end gap-2";
    const addBtn = document.createElement("button"); addBtn.className = "btn btn-sm btn-light border"; addBtn.textContent = "+ Case"; addBtn.onclick = () => addBranchCase(parentLocation, index);
    const delBtn = document.createElement("button"); delBtn.className = "btn btn-sm btn-text text-danger"; delBtn.textContent = "Delete Branch"; delBtn.onclick = () => removeStep(createStepPath(parentLocation, index));
    actions.append(addBtn, delBtn); container.append(header, routerDiv, casesDiv, actions); return container;
}
function renderStepNode(step, parentLocation, index) {
  const stepPath = createStepPath(parentLocation, index);
  if (typeof step === "string") return renderToolNode(step, stepPath);
  if (step && typeof step === "object" && step.loop) return renderLoopNode(step, parentLocation, index);
  if (step && typeof step === "object" && step.branch) return renderBranchNode(step, parentLocation, index);
  const card = renderToolNode("Custom Object", stepPath); card.querySelector(".flow-node-body").textContent = JSON.stringify(step); return card;
}
function renderStepList(steps, location, options = {}) {
  const wrapper = document.createElement("div"); wrapper.className = "step-list";
  if (!steps.length) {
    const placeholder = document.createElement("div"); placeholder.className = "flow-placeholder";
    const control = createInsertControl(location, 0, { prominent: true }); placeholder.appendChild(control); wrapper.appendChild(placeholder); return wrapper;
  }
  steps.forEach((step, index) => { wrapper.appendChild(createInsertControl(location, index, { compact: options.compact })); wrapper.appendChild(renderStepNode(step, location, index)); });
  wrapper.appendChild(createInsertControl(location, steps.length, { compact: options.compact })); return wrapper;
}
function renderSteps() { els.flowCanvas.innerHTML = ""; const rootLocation = createLocation([]); els.flowCanvas.appendChild(renderStepList(state.steps, rootLocation)); }
function renderContextControls() {
  if (!els.contextControls) return; els.contextControls.innerHTML = ""; ensureContextInitialized();
  const breadcrumb = document.createElement("div"); breadcrumb.className = "context-breadcrumb d-flex flex-wrap gap-2 align-items-center";
  state.contextStack.forEach((loc, idx) => {
      const btn = document.createElement("button"); btn.className = `btn btn-sm rounded-pill ${idx === state.contextStack.length-1 ? "btn-dark" : "btn-light border"}`;
      btn.textContent = ctxLabel(loc, idx); btn.onclick = () => setActiveLocation(createLocation(loc.segments || []));
      breadcrumb.appendChild(btn); if (idx < state.contextStack.length - 1) { const sep = document.createElement("span"); sep.className = "text-muted small"; sep.textContent = "/"; breadcrumb.appendChild(sep); }
  });
  els.contextControls.appendChild(breadcrumb);
  const active = getActiveLocation(); const kind = getContextKind(active);
  if (kind !== "root") {
      const exitBtn = document.createElement("button"); exitBtn.className = "btn btn-sm btn-link text-danger text-decoration-none mt-2"; exitBtn.textContent = "Exit Context ✕";
      exitBtn.onclick = () => { setActiveLocation(createLocation((active.segments||[]).slice(0, -1))); }; els.contextControls.appendChild(exitBtn);
  }
}
function ctxLabel(location, idx) {
  if (idx === 0) return "Root"; const last = (location.segments||[])[location.segments.length - 1];
  if (!last) return "Root"; if (last.type === "loop") return "Loop"; if (last.type === "branch") return last.section === "router" ? "Router" : `Case:${last.branchKey}`; return "Node";
}
function addBranchCase(parentLocation, branchIndex) {
    const steps = resolveSteps(parentLocation); const entry = steps[branchIndex]; if (!entry?.branch) return;
    entry.branch.branches = entry.branch.branches || {}; let c = Object.keys(entry.branch.branches).length + 1; let key = `case${c}`; while (entry.branch.branches[key]) { c++; key = `case${c}`; }
    entry.branch.branches[key] = []; markPipelineDirty(); const segs = [...(parentLocation.segments||[]), { type: "branch", index: branchIndex, section: "branch", branchKey: key }]; setActiveLocation(createLocation(segs));
}
function enterStructureContext(type, stepPath, announce = true) {
    if (!stepPath) return; const segs = [...(stepPath.parentSegments||[]), { type, index: stepPath.index, ...(type==="branch"?{section:"router"}:{}) }]; setActiveLocation(createLocation(segs));
}

async function refreshPipelines() { const pipelines = await fetchJSON("/api/pipelines"); renderPipelineMenu(pipelines); }
function renderPipelineMenu(items) {
    els.pipelineMenu.innerHTML = ""; if (!items.length) { const li = document.createElement("li"); li.innerHTML = '<span class="dropdown-item text-muted small">No pipelines</span>'; els.pipelineMenu.appendChild(li); return; }
    items.forEach(i => {
        const li = document.createElement("li"); const btn = document.createElement("button"); btn.type = "button"; btn.className = "dropdown-item small"; btn.textContent = i.name;
        btn.onclick = () => { loadPipeline(i.name); btn.blur(); }; li.appendChild(btn); els.pipelineMenu.appendChild(li);
    });
}
async function loadPipeline(name) {
    const cfg = await fetchJSON(`/api/pipelines/${encodeURIComponent(name)}`); state.selectedPipeline = name; els.name.value = name; setSteps(cfg.pipeline || []);
    if (els.pipelineDropdownBtn) els.pipelineDropdownBtn.textContent = name; setHeroPipelineLabel(name);
}
function handleSubmit(e) {
    e.preventDefault(); const name = els.name.value.trim(); if (!name) return log("Pipeline name is required");
    fetchJSON("/api/pipelines", { method: "POST", body: JSON.stringify({ name, pipeline: cloneDeep(state.steps) }) })
    .then(s => { state.selectedPipeline=s.name||name; refreshPipelines(); log("Pipeline saved."); loadPipeline(s.name||name); }).catch(e=>log(e.message));
}
function buildSelectedPipeline() {
    if(!state.selectedPipeline) return log("Please save the pipeline first.");
    fetchJSON(`/api/pipelines/${encodeURIComponent(state.selectedPipeline)}/build`, { method: "POST" })
    .then(() => { state.isBuilt=true; state.parametersReady=false; updateActionButtons(); log("Pipeline built."); showParameterPanel(true); }).catch(e=>log(e.message));
}
function runSelectedPipeline() {
    if(!state.selectedPipeline || !state.parametersReady) return log("Please configure parameters first.");
    stopRunLogStream(); resetLogView(); setMode(Modes.RUN); log("Run initiated...");
    fetchJSON(`/api/pipelines/${encodeURIComponent(state.selectedPipeline)}/run`, { method: "POST" })
    .then(r => { if (r?.run_id) startRunLogStream(r.run_id); else log("No run_id returned"); }).catch(e=>log(e.message));
}
function deleteSelectedPipeline() {
    if(!state.selectedPipeline || !confirm("Delete pipeline?")) return;
    fetchJSON(`/api/pipelines/${encodeURIComponent(state.selectedPipeline)}`, { method: "DELETE" })
    .then(() => { state.selectedPipeline=null; els.name.value=""; setSteps([]); refreshPipelines(); }).catch(e=>log(e.message));
}
function flattenParameters(obj, prefix = "") {
    const entries = []; if (!obj || typeof obj !== "object") return entries;
    Object.keys(obj).sort().forEach(key => {
        const path = prefix ? `${prefix}.${key}` : key; const val = obj[key];
        if (val!==null && typeof val==="object" && !Array.isArray(val)) entries.push(...flattenParameters(val, path));
        else entries.push({ path, value: val, type: Array.isArray(val) ? "array" : (val===null?"null":typeof val) });
    }); return entries;
}
function setNestedValue(obj, path, val) {
    const p = path.split("."); let c = obj; for (let i=0; i<p.length-1; i++) { if (!c[p[i]]) c[p[i]]={}; c=c[p[i]]; } c[p[p.length-1]] = val;
}

// --- Updated Parameter Renderer ---
function renderParameterForm() {
    const container = els.parameterForm; container.innerHTML = "";
    if (!state.parameterData || typeof state.parameterData !== "object") { container.innerHTML = '<div class="col-12"><p class="text-muted text-center">No parameters available for configuration.</p></div>'; return; }
    const entries = flattenParameters(state.parameterData);
    if (!entries.length) { container.innerHTML = '<div class="col-12"><p class="text-muted text-center">The current Pipeline has no editable parameters.</p></div>'; return; }
    entries.forEach(e => {
        const grp = document.createElement("div"); grp.className = "form-group-styled";
        const isComplex = e.type === "array" || e.type === "object";
        if (isComplex) grp.classList.add("full-width");
        const label = document.createElement("label"); label.textContent = e.path;
        let ctrl;
        if (isComplex) { ctrl = document.createElement("textarea"); ctrl.rows = 4; ctrl.value = JSON.stringify(e.value, null, 2); }
        else { ctrl = document.createElement("input"); ctrl.type = "text"; ctrl.value = String(e.value ?? ""); }
        ctrl.className = "form-control code-font";
        ctrl.onchange = (ev) => {
            let val = ev.target.value;
            if (e.type === "number") val = Number(val); if (e.type === "boolean") val = val.toLowerCase() === "true";
            try { if (isComplex) val = JSON.parse(val); } catch (err) {}
            e.value = val; setNestedValue(state.parameterData, e.path, val); state.parametersReady = false; updateActionButtons();
        };
        grp.append(label, ctrl); container.appendChild(grp);
    });
}
async function showParameterPanel(force = false) {
    if (!state.isBuilt) return log("Please build the pipeline first.");
    if (force || !state.parameterData) { try { state.parameterData = cloneDeep(await fetchJSON(`/api/pipelines/${encodeURIComponent(state.selectedPipeline)}/parameters`)); } catch(e){ return log(e.message); } }
    renderParameterForm(); setMode(Modes.PARAMETERS);
}
function saveParameterForm() { persistParameterData(); }
async function refreshTools() {
    const tools = await fetchJSON("/api/tools"); const grouped = {};
    tools.forEach(t => { const s = t.server || "Unnamed"; if (!grouped[s]) grouped[s] = []; grouped[s].push(t); });
    state.toolCatalog = { order: Object.keys(grouped).sort(), byServer: grouped }; nodePickerState.server = null;
}

function bindEvents() {
    els.pipelineForm.addEventListener("submit", handleSubmit);
    els.clearSteps.addEventListener("click", () => { if(confirm("Clear steps?")) setSteps([]); });
    els.buildPipeline.addEventListener("click", buildSelectedPipeline);
    els.deletePipeline.addEventListener("click", deleteSelectedPipeline);
    if (els.newPipelineBtn) els.newPipelineBtn.addEventListener("click", createNewPipeline); 
    if (els.shutdownApp) els.shutdownApp.onclick = requestShutdown;
    
    // Bind Back Buttons
    if (els.parameterSave) els.parameterSave.onclick = saveParameterForm;
    if (els.parameterBack) els.parameterBack.onclick = () => setMode(Modes.BUILDER);
    if (els.parameterRun) els.parameterRun.onclick = runSelectedPipeline;
    if (els.parameterChat) els.parameterChat.onclick = openChatView;
    if (els.runBack) els.runBack.onclick = () => setMode(Modes.PARAMETERS);
    
    // Chat View
    if (els.chatForm) els.chatForm.onsubmit = handleChatSubmit;
    if (els.chatBack) els.chatBack.onclick = () => { saveCurrentSession(true); setChatRunning(false); setMode(Modes.PARAMETERS); };
    if (els.chatNewBtn) els.chatNewBtn.onclick = createNewChatSession;
    
    document.getElementById("step-editor-save").onclick = () => {
        if (!state.editingPath) return;
        try { setStepByPath(state.editingPath, parseStepInput(els.stepEditorValue.value)); closeStepEditor(); renderSteps(); updatePipelinePreview(); } catch(e){ log(e.message); }
    };
    document.getElementById("step-editor-cancel").onclick = closeStepEditor;
    
    els.refreshPipelines.onclick = refreshPipelines;
    els.name.oninput = updatePipelinePreview;
    
    els.nodePickerTabs.forEach(t => t.onclick = () => setNodePickerMode(t.dataset.nodeMode));
    if (els.nodePickerServer) els.nodePickerServer.onchange = () => { nodePickerState.server = els.nodePickerServer.value; populateNodePickerTools(); };
    if (els.nodePickerTool) els.nodePickerTool.onchange = () => nodePickerState.tool = els.nodePickerTool.value;
    if (els.nodePickerBranchCases) els.nodePickerBranchCases.oninput = (e) => nodePickerState.branchCases = e.target.value;
    if (els.nodePickerLoopTimes) els.nodePickerLoopTimes.oninput = (e) => nodePickerState.loopTimes = e.target.value;
    if (els.nodePickerCustom) els.nodePickerCustom.oninput = (e) => nodePickerState.customValue = e.target.value;
    if (els.nodePickerConfirm) els.nodePickerConfirm.onclick = handleNodePickerConfirm;
}

async function bootstrap() {
  setMode(Modes.BUILDER); resetContextStack(); renderSteps(); updatePipelinePreview(); bindEvents(); updateActionButtons();
  setHeroPipelineLabel(state.selectedPipeline || ""); // Ensure label is set on load
  try { await Promise.all([refreshPipelines(), refreshTools()]); log("UI Ready."); } catch (err) { log(`Initialization error: ${err.message}`); }
}

bootstrap();