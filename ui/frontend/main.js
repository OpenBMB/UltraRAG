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
  
  // [修改] 聊天状态管理
  chat: { 
    history: [], 
    running: false,
    sessions: [], 
    currentSessionId: null,

    controller: null,
    
    // 引擎连接状态
    engineSessionId: null, // 当前选中的 Pipeline 对应的 SessionID
    activeEngines: {},     // 映射表: { "pipelineName": "sessionId" }
    
    demoLoading: false
  },
};

const els = {
  // View Containers
  mainRoot: document.querySelector(".content-wrapper"),
  pipelineForm: document.getElementById("pipeline-form"),
  parameterPanel: document.getElementById("parameter-panel"),
  chatView: document.getElementById("chat-view"),
  // runView 已删除
  
  // Logs
  log: document.getElementById("log"),
  // runTerminal, runSpinner 已删除

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

  directChatBtn: document.getElementById("direct-chat-btn"),
  
  // Parameter Controls
  parameterForm: document.getElementById("parameter-form"),
  parameterSave: document.getElementById("parameter-save"),
  parameterBack: document.getElementById("parameter-back"),
  // parameterRun 已删除
  parameterChat: document.getElementById("parameter-chat"),
  
  // Chat Controls
  chatPipelineName: document.getElementById("chat-pipeline-name"),
  chatBack: document.getElementById("chat-back"),
  chatHistory: document.getElementById("chat-history"),
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
  chatStatus: document.getElementById("chat-status"),
  chatSend: document.getElementById("chat-send"),
  chatNewBtn: document.getElementById("chat-new-btn"),
  chatSessionList: document.getElementById("chat-session-list"),
  demoToggleBtn: document.getElementById("demo-toggle-btn"), // 引擎开关

  // [新增] Chat 顶部控件
  chatPipelineLabel: document.getElementById("chat-pipeline-label"),
  chatPipelineMenu: document.getElementById("chat-pipeline-menu"),
  
  // Node Picker (保留原样)
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

// --- Utilities ---
function uuidv4() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    var r = Math.random() * 16 | 0, v = c == 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

function log(message) {
  const stamp = new Date().toLocaleTimeString();
  const msg = `> [${stamp}] ${message}`;
  if (els.log) { els.log.textContent += msg + "\n"; els.log.scrollTop = els.log.scrollHeight; }
  console.log(msg);
}

// Markdown 渲染（带简单降级）
let markdownConfigured = false;
const MARKDOWN_LANGS = ["markdown", "md", "mdx"];

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderInlineMarkdown(str) {
  let text = escapeHtml(str);
  text = text.replace(/`([^`]+)`/g, "<code>$1</code>");
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  text = text.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  text = text.replace(/_([^_]+)_/g, "<em>$1</em>");
  text = text.replace(/~~([^~]+)~~/g, "<del>$1</del>");
  return text;
}

function basicMarkdown(text, { allowCodeBlock } = { allowCodeBlock: true }) {
  const lines = text.split(/\r?\n/);
  let html = "";
  let inCode = false;
  let codeBuffer = [];
  let paragraph = [];
  let listType = null;
  const closeList = () => {
    if (listType) { html += `</${listType}>`; listType = null; }
  };
  const flushParagraph = () => {
    if (paragraph.length) {
      const para = paragraph.join(" ").trim();
      if (para) html += `<p>${renderInlineMarkdown(para)}</p>`;
      paragraph = [];
    }
  };

  for (const rawLine of lines) {
    const line = rawLine.replace(/\s+$/, "");
    if (line.trim().startsWith("```")) {
      if (inCode) {
        const codeHtml = codeBuffer.map(escapeHtml).join("\n");
        if (allowCodeBlock) {
          html += `<pre><code>${codeHtml}</code></pre>`;
        } else {
          html += `<p>${codeHtml.replace(/\n/g, "<br>")}</p>`;
        }
        codeBuffer = [];
        inCode = false;
      } else {
        flushParagraph(); closeList();
        inCode = true;
      }
      continue;
    }
    if (inCode) { codeBuffer.push(line); continue; }

    if (/^#{1,6}\s/.test(line)) {
      flushParagraph(); closeList();
      const level = line.match(/^#{1,6}/)[0].length;
      const content = line.slice(level).trim();
      html += `<h${level}>${renderInlineMarkdown(content)}</h${level}>`;
      continue;
    }
    if (/^([-*_])\1{2,}\s*$/.test(line.trim())) {
      flushParagraph(); closeList();
      html += "<hr>";
      continue;
    }
    if (/^>\s?/.test(line)) {
      flushParagraph(); closeList();
      const content = line.replace(/^>\s?/, "");
      html += `<blockquote>${renderInlineMarkdown(content)}</blockquote>`;
      continue;
    }
    if (/^(\*|-|\+)\s+/.test(line)) {
      flushParagraph();
      if (listType !== "ul") { closeList(); listType = "ul"; html += "<ul>"; }
      const content = line.replace(/^(\*|-|\+)\s+/, "");
      html += `<li>${renderInlineMarkdown(content)}</li>`;
      continue;
    }
    if (/^\d+\.\s+/.test(line)) {
      flushParagraph();
      if (listType !== "ol") { closeList(); listType = "ol"; html += "<ol>"; }
      const content = line.replace(/^\d+\.\s+/, "");
      html += `<li>${renderInlineMarkdown(content)}</li>`;
      continue;
    }
    if (/^\s*$/.test(line)) {
      flushParagraph(); closeList();
      continue;
    }
    paragraph.push(line);
  }
  flushParagraph(); closeList();
  if (inCode) {
    const codeHtml = codeBuffer.map(escapeHtml).join("\n");
    if (allowCodeBlock) html += `<pre><code>${codeHtml}</code></pre>`;
    else html += `<p>${codeHtml.replace(/\n/g, "<br>")}</p>`;
  }
  return html || escapeHtml(text).replace(/\n/g, "<br>");
}

function unwrapLanguageBlocks(text, languages = []) {
  if (!languages.length) return text;
  const pattern = new RegExp("```\\s*(" + languages.join("|") + ")\\s*(?:\\r?\\n)([\\s\\S]*?)```", "gi");
  return text.replace(pattern, (_, __, body) => body.trim());
}

function stripLeadingLanguageFence(text, languages = []) {
  if (!languages.length || !text) return text;
  const startPattern = new RegExp("^\\s*```\\s*(?:" + languages.join("|") + ")\\s*(?:\\r?\\n)?", "i");
  let stripped = text;
  if (startPattern.test(stripped)) {
    stripped = stripped.replace(startPattern, "");
    stripped = stripped.replace(/\r?\n?```\s*$/i, "");
  }
  return stripped;
}

function isPendingLanguageFence(text, languages = []) {
  if (!languages.length || !text) return false;
  const trimmed = text.trimStart().toLowerCase();
  if (!trimmed.startsWith("```")) return false;
  const newlineIndex = trimmed.indexOf("\n");
  if (newlineIndex !== -1) return false;
  const langFragment = trimmed.slice(3).trim();
  return languages.some(lang => lang.startsWith(langFragment));
}

function renderMarkdown(text, { allowCodeBlock = true, unwrapLanguages = [] } = {}) {
  if (!text) return "";
  if (unwrapLanguages.length) {
    text = unwrapLanguageBlocks(text, unwrapLanguages);
  }
  if (window.marked) {
    if (!markdownConfigured) {
      window.marked.setOptions({ breaks: true, gfm: true });
      markdownConfigured = true;
    }
    let rendererOptions = undefined;
    if (!allowCodeBlock && window.marked) {
      const renderer = new window.marked.Renderer();
      renderer.code = (code) => `<p>${escapeHtml(code).replace(/\n/g, "<br>")}</p>`;
      rendererOptions = { renderer };
    }
    const rawHtml = window.marked.parse(text, rendererOptions);
    return window.DOMPurify ? DOMPurify.sanitize(rawHtml) : rawHtml;
  }
  return basicMarkdown(text, { allowCodeBlock });
}

// --- Lifecycle ---
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

// --- Demo Session / Engine Control ---

async function toggleDemoSession() {
  if (state.chat.demoLoading) return;
  const pipelineName = state.selectedPipeline;
  if (!pipelineName) return;

  state.chat.demoLoading = true;
  updateDemoControls();

  try {
    if (state.chat.engineSessionId) {
      // === STOP ===
      const sid = state.chat.engineSessionId;
      await fetchJSON(`/api/pipelines/demo/stop`, { 
          method: "POST", body: JSON.stringify({ session_id: sid }) 
      });
      state.chat.engineSessionId = null;
      // [新增] 移除记录
      delete state.chat.activeEngines[pipelineName];
      
      setChatStatus("Offline", "info");
      log("Demo engine stopped.");
    } else {
      // === START ===
      const newSid = uuidv4(); 
      setChatStatus("Starting...", "warn");
      
      await fetchJSON(`/api/pipelines/${encodeURIComponent(pipelineName)}/demo/start`, { 
          method: "POST", body: JSON.stringify({ session_id: newSid })
      });
      
      state.chat.engineSessionId = newSid;
      // [新增] 添加记录
      state.chat.activeEngines[pipelineName] = newSid;
      
      setChatStatus("Engine Ready", "ready");
      log("Demo engine started.");
    }
  } catch (err) {
    log(`Engine error: ${err.message}`);
    setChatStatus("Error", "error");
    if(!state.chat.engineSessionId) state.chat.engineSessionId = null;
  } finally {
    state.chat.demoLoading = false;
    updateDemoControls();
  }
}

function updateDemoControls() {
  if (!els.demoToggleBtn) return;
  const btn = els.demoToggleBtn;
  const isActive = !!state.chat.engineSessionId;
  
  if (state.chat.demoLoading) {
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...`;
    btn.className = "btn btn-sm btn-outline-secondary rounded-pill px-3 fw-bold";
    return;
  }

  btn.disabled = false;
  if (isActive) {
    // 引擎已启动
    btn.className = "btn btn-sm btn-outline-danger rounded-pill px-3 fw-bold";
    btn.innerHTML = "■ Stop Engine";
    
    // [修改] 移除对 chatInput 和 chatSend 的 running 状态干预
    // 这里只负责“引擎在线就解锁输入框”，至于 running 时锁不锁，交给 setChatRunning 管
    if (els.chatInput) els.chatInput.disabled = false; 
    if (els.chatSend) els.chatSend.disabled = false;
    
  } else {
    // 引擎未启动
    btn.className = "btn btn-sm btn-outline-success rounded-pill px-3 fw-bold";
    btn.innerHTML = "▶ Start Engine";
    
    // 引擎离线时，彻底锁死输入
    if (els.chatInput) els.chatInput.disabled = true;
    if (els.chatSend) els.chatSend.disabled = true;
  }
}

// --- Chat Logic (Updated with Streaming) ---

// [新增] 渲染 Pipeline 列表到下拉菜单
async function renderChatPipelineMenu() {
    if (!els.chatPipelineMenu) return;
    
    // 获取最新列表
    const pipelines = await fetchJSON("/api/pipelines");
    
    els.chatPipelineMenu.innerHTML = "";
    
    // [关键修改] 过滤掉还没有 Build (没有参数文件) 的 Pipeline
    const readyPipelines = pipelines.filter(p => p.is_ready);

    if (readyPipelines.length === 0) {
        els.chatPipelineMenu.innerHTML = '<li class="dropdown-item text-muted small">No ready pipelines</li>';
        return;
    }
    
    readyPipelines.forEach(p => {
        const li = document.createElement("li");
        const btn = document.createElement("button");
        const isActive = p.name === state.selectedPipeline;
        btn.className = `dropdown-item ${isActive ? 'active' : ''}`;
        
        btn.textContent = `${p.name}`;
        
        btn.onclick = () => switchChatPipeline(p.name);
        
        li.appendChild(btn);
        els.chatPipelineMenu.appendChild(li);
    });
    
    // 更新顶部 Label
    if (els.chatPipelineLabel) {
        els.chatPipelineLabel.textContent = state.selectedPipeline || "Select Pipeline";
    }
}

// [新增] 切换 Pipeline (核心逻辑)
async function switchChatPipeline(name) {
    if (name === state.selectedPipeline) return;
    if (state.chat.running) {
        alert("Please wait for the current response to finish.");
        return;
    }
    
    log(`Switching context to ${name}...`);
    
    // 1. 保存旧会话
    saveCurrentSession(true);
    
    // 2. 加载新 Pipeline 结构 (这会触发 setSteps -> resetChatSession)
    await loadPipeline(name); 
    
    // 3. 关键：加载该 Pipeline 的参数到内存
    // 这样 Chat 时 execute_pipeline 才能读到正确的配置
    try {
        state.parameterData = await fetchJSON(`/api/pipelines/${encodeURIComponent(name)}/parameters`);
        state.parametersReady = true;
    } catch (e) {
        console.warn("Parameters not found for this pipeline.");
        state.parametersReady = false;
    }

    // 4. 为新 Pipeline 创建一个空白聊天窗口 (或者恢复上次的？这里简化为新建)
    createNewChatSession();
    
    // 5. 刷新 UI
    renderChatPipelineMenu(); 
    // updateDemoControls 已在 resetChatSession 中触发，但再调一次也无妨
    updateDemoControls();
}

function resetChatSession() {
    state.chat.history = [];
    state.chat.running = false;
    state.chat.currentSessionId = null;
    
    // 切换 Pipeline 时，清空旧的聊天列表显示 (不清除 activeEngines)
    state.chat.sessions = []; 
    
    // [关键修改] 检查 activeEngines，恢复 Session ID
    const currentName = state.selectedPipeline;
    if (currentName && state.chat.activeEngines[currentName]) {
        state.chat.engineSessionId = state.chat.activeEngines[currentName];
        log(`Resumed active engine for ${currentName}`);
    } else {
        state.chat.engineSessionId = null;
    }
    
    renderChatHistory(); 
    renderChatSidebar();
    
    // 更新状态显示
    if (state.chat.engineSessionId) {
        setChatStatus("Engine Ready", "ready");
    } else {
        setChatStatus("Engine Offline", "info");
    }
    
    updateDemoControls(); 
}

function generateChatId() { return Date.now().toString(36) + Math.random().toString(36).substr(2); }

function createNewChatSession() {
    if (state.chat.history.length > 0) saveCurrentSession(true);
    state.chat.currentSessionId = generateChatId();
    state.chat.history = [];
    renderChatHistory(); renderChatSidebar();
    setChatStatus("Ready", "ready");
    if(els.chatInput && state.chat.engineSessionId) els.chatInput.focus();
}

function loadChatSession(sessionId) {
    if (state.chat.running) return;
    saveCurrentSession(false); 
    const session = state.chat.sessions.find(s => s.id === sessionId);
    if (!session) return;
    state.chat.currentSessionId = session.id;
    state.chat.history = [...session.messages];
    renderChatHistory(); renderChatSidebar();
    setChatStatus("Ready", "ready");
}

function saveCurrentSession(force = false) {
    if (!state.chat.currentSessionId) return;
    if (!force && state.chat.history.length === 0) {
        state.chat.sessions = state.chat.sessions.filter(s => s.id !== state.chat.currentSessionId);
        renderChatSidebar(); return;
    }
    let session = state.chat.sessions.find(s => s.id === state.chat.currentSessionId);
    let title = "New Chat";
    const firstUserMsg = state.chat.history.find(m => m.role === 'user');
    if (firstUserMsg) title = firstUserMsg.text.slice(0, 20) + (firstUserMsg.text.length > 20 ? "..." : "");

    if (!session) {
        session = { id: state.chat.currentSessionId, title: title, messages: [] };
        state.chat.sessions.unshift(session);
    } else {
        state.chat.sessions = state.chat.sessions.filter(s => s.id !== state.chat.currentSessionId);
        state.chat.sessions.unshift(session);
        if (session.title === "New Chat" || (session.messages.length === 0 && firstUserMsg)) session.title = title;
    }
    session.messages = [...state.chat.history];
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

function appendChatMessage(role, text, meta = {}) {
  const entry = { role, text, meta, timestamp: new Date().toISOString() };
  state.chat.history.push(entry);
  renderChatHistory();
  saveCurrentSession(); 
}

function renderChatHistory() {
  if (!els.chatHistory) return;
  els.chatHistory.innerHTML = "";
  if (state.chat.history.length === 0) { 
      els.chatHistory.innerHTML = '<div class="text-center mt-5 pt-5 text-muted small"><p>Ready to start.</p></div>'; 
      return; 
  }
  state.chat.history.forEach((entry) => {
    const bubble = document.createElement("div"); bubble.className = `chat-bubble ${entry.role}`;
    const content = document.createElement("div"); 
    content.className = "msg-content";
    let textToRender = entry.text;
    let mdOptions = {};
    if (entry.role === "assistant") {
      textToRender = stripLeadingLanguageFence(textToRender, MARKDOWN_LANGS);
      mdOptions = { unwrapLanguages: MARKDOWN_LANGS };
    }
    content.innerHTML = renderMarkdown(textToRender, mdOptions); 
    bubble.appendChild(content);
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
  
  // 1. 先更新引擎按钮状态 (但这会重置 input/send 的状态，所以要在下面覆盖)
  updateDemoControls();
  
  const btn = els.chatSend;
  const icon = document.getElementById("chat-send-icon");
  
  if (isRunning) {
      setChatStatus("Thinking...", "running");
      
      // [修改] 生成中：输入框锁定，但按钮必须可用（为了点击停止）
      if (els.chatInput) els.chatInput.disabled = true;
      if (els.chatSend) els.chatSend.disabled = false; // <--- 关键！必须解开！

      // 样式变为红色停止
      if (btn) btn.classList.add("stop");
      if (icon) {
          icon.textContent = "";
          icon.className = "icon-stop";
      }
  } else {
      updateActionButtons();
      
      // 闲置状态：输入框解锁（前提是引擎在线，updateDemoControls 已处理）
      // 样式变回箭头
      if (btn) btn.classList.remove("stop");
      if (icon) {
          icon.className = "";
          icon.textContent = "↑";
      }
  }
}

function canUseChat() { return Boolean(state.isBuilt && state.selectedPipeline && state.parameterData); }

function openChatView() {
  if (!canUseChat()) { log("Please build and save parameters first."); return; }
  if (els.chatPipelineName) els.chatPipelineName.textContent = state.selectedPipeline || "—";

  renderChatPipelineMenu();
  
  if (!state.chat.currentSessionId) createNewChatSession();
  
  renderChatHistory();
  renderChatSidebar();
  setMode(Modes.CHAT);
  
  // 检查引擎状态并更新UI
  updateDemoControls();
  if(!state.chat.engineSessionId) setChatStatus("Engine Offline", "info");
  else setChatStatus("Ready", "ready");
}

async function stopGeneration() {
    if (!state.chat.running) return;

    // 1. 前端断开连接 (停止接收数据流)
    // 这会让 fetch 抛出 AbortError，跳到 catch 块
    if (state.chat.controller) {
        state.chat.controller.abort();
        state.chat.controller = null;
    }

    // 2. 通知后端 Python 停止 Loop (释放 Session 锁)
    try {
        if (state.chat.engineSessionId) {
            // 发送停止信号，但不等待返回，直接结束 UI 状态
            fetchJSON(`/api/pipelines/chat/stop`, {
                method: "POST",
                body: JSON.stringify({ session_id: state.chat.engineSessionId })
            }).catch(e => console.warn("Backend stop signal failed:", e));
        }
    } catch (e) { console.error(e); }

    log("Generation stopped by user.");
    
    // UI 立即恢复
    setChatRunning(false);
    appendChatMessage("system", "Generation interrupted.");
}

function updateProcessUI(entryIndex, eventData) {
    // 1. 找到对应的 Chat Bubble (最后一个 assistant 气泡)
    const container = document.getElementById("chat-history");
    const bubbles = container.querySelectorAll(".chat-bubble.assistant");
    const lastBubble = bubbles[bubbles.length - 1];
    if (!lastBubble) return;

    // 2. 检查或创建 Process Container
    let procDiv = lastBubble.querySelector(".process-container");
    if (!procDiv) {
        procDiv = document.createElement("div");
        procDiv.className = "process-container"; 
        // 默认展开结构
        procDiv.innerHTML = `
            <div class="process-header" onclick="this.parentNode.classList.toggle('collapsed')">
                <span>Show Thinking</span>
                <span style="font-size:0.8em">▼</span>
            </div>
            <div class="process-body"></div>
        `;
        // 插在气泡最前面
        lastBubble.insertBefore(procDiv, lastBubble.firstChild);
    }
    
    const body = procDiv.querySelector(".process-body");

    // 3. 处理不同事件
    if (eventData.type === "step_start") {
        const stepDiv = document.createElement("div");
        stepDiv.className = "process-step";
        stepDiv.dataset.stepName = eventData.name;
        stepDiv.innerHTML = `
            <div class="step-title">
                <span class="step-spinner"></span>
                <span>${eventData.name}</span>
            </div>
            <div class="step-content-stream"></div> `;
        body.appendChild(stepDiv);
        // 自动滚动到底部
        body.scrollTop = body.scrollHeight;

    } else if (eventData.type === "token") {
        // 如果 Token 不是 final 的，就显示在思考过程里 (作为详细日志)
        // 找到当前正在运行的步骤
        const steps = body.querySelectorAll(".process-step");
        const currentStep = steps[steps.length - 1];
        if (currentStep) {
            const streamDiv = currentStep.querySelector(".step-content-stream");
            if (streamDiv) {
                // 简单的追加文本
                const span = document.createElement("span");
                span.textContent = eventData.content;
                streamDiv.appendChild(span);
            }
        }

    } else if (eventData.type === "step_end") {
        const steps = body.querySelectorAll(".process-step");
        const currentStep = steps[steps.length - 1];
        
        if (currentStep) {
            // 1. Spinner -> Checkmark
            const spinner = currentStep.querySelector(".step-spinner");
            if (spinner) {
                spinner.remove(); // 直接移除元素
            }
            
            // 2. 显示摘要 (output)
            // 如果之前有流式内容(step-content-stream)，可以选择保留或者被摘要覆盖
            // 这里我们选择追加摘要作为总结
            if (eventData.output) {
                const details = document.createElement("div");
                details.className = "step-details";
                details.textContent = eventData.output;
                currentStep.appendChild(details);
                
                // (可选) 隐藏流式过程，只看结果? 
                // currentStep.querySelector(".step-content-stream").style.display = 'none';
            }
        }
    }
}

async function handleChatSubmit(event) {
  // 1. 防止表单默认提交刷新页面
  if (event) event.preventDefault();
  
  // 2. [停止拦截] 如果当前正在生成，再次点击按钮（此时按钮是红色停止状态）视为“停止”
  if (state.chat.running) {
      await stopGeneration();
      return;
  }

  // 3. 基础校验
  if (!canUseChat()) return;
  
  if (!state.chat.engineSessionId) {
      alert("Please click 'Start Engine' first to initialize the backend.");
      return;
  }

  const question = (els.chatInput ? els.chatInput.value : "").trim();
  if (!question) return;
  
  // 清空输入框并显示用户提问
  if (els.chatInput) els.chatInput.value = "";
  appendChatMessage("user", question);
  
  // 设置 UI 为“运行中”状态
  setChatRunning(true);
  
  // 4. [初始化] 创建 AbortController 用于中断请求
  state.chat.controller = new AbortController();
  
  try {
    // 确保参数已保存
    if (!state.parametersReady) await persistParameterData({ silent: true });
    
    const endpoint = `/api/pipelines/${encodeURIComponent(state.selectedPipeline)}/chat`;
    
    // 预留动态参数接口 (例如文件上传后的 collection_name)
    const dynamicParams = {}; 
    
    const body = JSON.stringify({ 
        question, 
        history: state.chat.history,
        is_demo: true,
        session_id: state.chat.engineSessionId,
        dynamic_params: dynamicParams
    });
    
    // 5. [请求] 发送 Fetch 请求，绑定 signal
    const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body,
        signal: state.chat.controller.signal // <--- 关键：绑定中断信号
    });

    if (!response.ok) throw new Error(response.statusText);

    // [UI准备] 预先添加 Assistant 气泡 (占位)
    const entryIndex = state.chat.history.length;
    // 推入空对象占位
    state.chat.history.push({ role: "assistant", text: "", meta: {} });
    
    // 手动操作 DOM 添加气泡结构（包含消息内容容器 msg-content）
    const chatContainer = document.getElementById("chat-history");
    const bubble = document.createElement("div");
    bubble.className = "chat-bubble assistant";
    const contentDiv = document.createElement("div");
    contentDiv.className = "msg-content";
    bubble.appendChild(contentDiv);
    chatContainer.appendChild(bubble);

    let currentText = "";
    
    // 准备流式读取
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    // 6. [流式读取] 循环处理 SSE 数据
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n"); // SSE 标准分隔符
      buffer = lines.pop(); // 保留最后一行不完整的数据

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const jsonStr = line.slice(6);
            const data = JSON.parse(jsonStr);
            
            // --- A. 思考过程事件 (Start / End) ---
            if (data.type === "step_start" || data.type === "step_end") {
                updateProcessUI(entryIndex, data);
            }
            // --- B. Token 事件 ---
            else if (data.type === "token") {
                // 如果不是 Final Step，也可以选择在 Process 里显示日志（可选）
                if (!data.is_final) {
                    updateProcessUI(entryIndex, data);
                }

                // 只有 Final Step 的 Token 才上主屏幕
                if (data.is_final) {
                    currentText += data.content;
                    if (isPendingLanguageFence(currentText, MARKDOWN_LANGS)) {
                        contentDiv.innerHTML = "";
                        continue;
                    }
                    const normalized = stripLeadingLanguageFence(currentText, MARKDOWN_LANGS);
                    contentDiv.innerHTML = renderMarkdown(normalized, { unwrapLanguages: MARKDOWN_LANGS });
                    // 滚动到底部
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            } 
            // --- C. 最终结果汇总 ---
            else if (data.type === "final") {
                const final = data.data;
                
                // 更新 state 数据 (确保刷新页面后内容还在)
                let finalText = currentText || final.answer || "";
                finalText = stripLeadingLanguageFence(finalText, MARKDOWN_LANGS);
                state.chat.history[entryIndex].text = finalText;
                contentDiv.innerHTML = renderMarkdown(finalText, { unwrapLanguages: MARKDOWN_LANGS });
                
                const hints = [];
                if (final.dataset_path) hints.push(`Dataset: ${final.dataset_path}`);
                
                state.chat.history[entryIndex].meta = { hint: hints.join(" | ") };
                
                // 思考过程折叠 (生成完了自动折叠，保持界面清爽)
                const procDiv = bubble.querySelector(".process-container");
                if (procDiv) procDiv.classList.add("collapsed");
                
                setChatStatus("Ready", "ready");
                
            } 
            // --- D. 后端报错 ---
            else if (data.type === "error") {
                appendChatMessage("system", `Backend Error: ${data.message}`);
                setChatStatus("Error", "error");
            }

          } catch (e) { 
            console.error("JSON Parse error", e); 
          }
        }
      }
    }

  } catch (err) { 
      // 7. [错误处理] 忽略用户主动中断的错误
      if (err.name === 'AbortError') {
          console.log("Fetch aborted by user.");
          return; // 直接退出，后续 UI 由 stopGeneration 处理
      }
      
      console.error(err);
      appendChatMessage("system", `Network Error: ${err.message}`); 
      setChatStatus("Error", "error"); 
      
  } finally { 
      // 8. [清理]
      // 如果 controller 还在（说明不是通过 stopGeneration 触发的中断），则正常重置状态
      if (state.chat.controller) {
          state.chat.controller = null;
          setChatRunning(false);
      }
      saveCurrentSession();
  }
}

// --- Common Logic (Mode Switching, Node Picker, etc.) ---
function resetLogView() { if (els.log) els.log.textContent = ""; }
function setHeroPipelineLabel(name) { if (els.heroSelectedPipeline) els.heroSelectedPipeline.textContent = name ? name : "No Pipeline Selected"; }
function setHeroStatusLabel(status) {
  if (!els.heroStatus) return;
  els.heroStatus.dataset.status = status; els.heroStatus.textContent = status.toUpperCase();
}
function requestShutdown() { if (!window.confirm("Exit UltraRAG UI?")) return; fetch("/api/system/shutdown", { method: "POST" }); setTimeout(() => window.close(), 800); }

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

// --- Action Helpers (Removed Run Logic) ---
function cloneDeep(value) { return value === undefined ? undefined : JSON.parse(JSON.stringify(value)); }
// ... (Location/Step Helpers stay same) ...
function createLocation(segments = []) { return { segments: segments.map((seg) => ({ ...seg })) }; }
function locationsEqual(a, b) { return JSON.stringify((a && a.segments) || []) === JSON.stringify((b && b.segments) || []); }
function getContextKind(location) {
  const segments = (location && location.segments) || []; if (!segments.length) return "root";
  const last = segments[segments.length - 1];
  if (last.type === "loop") return "loop"; if (last.type === "branch") return last.section === "router" ? "branch-router" : "branch-case";
  return "root";
}
function resolveSteps(location) {
  let steps = state.steps; 
  const segments = (location && location.segments) || [];
  
  for (const seg of segments) {
    if (!Array.isArray(steps)) return [];
    
    const entry = steps[seg.index]; 
    if (!entry) return steps; // 或 return []
    
    if (seg.type === "loop" && entry.loop) { 
        entry.loop.steps = entry.loop.steps || []; 
        steps = entry.loop.steps; 
    }
    else if (seg.type === "branch" && entry.branch) {
      entry.branch.router = entry.branch.router || []; 
      entry.branch.branches = entry.branch.branches || {};
      
      if (seg.section === "router") {
          steps = entry.branch.router; 
      } else if (seg.section === "branch") {
          if (!entry.branch.branches[seg.branchKey]) {
              entry.branch.branches[seg.branchKey] = [];
          }
          steps = entry.branch.branches[seg.branchKey];
      }
    }
  }
  return Array.isArray(steps) ? steps : [];
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

// ... (YAML Helpers stay same) ...
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

function setMode(mode) {
  state.mode = mode;
  if (els.pipelineForm) els.pipelineForm.classList.toggle("d-none", mode !== Modes.BUILDER);
  if (els.parameterPanel) els.parameterPanel.classList.toggle("d-none", mode !== Modes.PARAMETERS);
  if (els.chatView) els.chatView.classList.toggle("d-none", mode !== Modes.CHAT);
}

// ... (Node Picker Helpers - keep same) ...
function getNodePickerModal() {
    const modalElement = els.nodePickerModal; if (!modalElement) return null;
    if (!nodePickerModalInstance) {
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

function markPipelineDirty() { 
    state.isBuilt = false; 
    state.parametersReady = false; 
    
    // [新增] 如果当前 Pipeline 被修改，强制废弃其 Engine Session
    const currentName = state.selectedPipeline;
    if (currentName && state.chat.activeEngines[currentName]) {
        const sid = state.chat.activeEngines[currentName];
        // 尝试后台停止
        fetchJSON(`/api/pipelines/demo/stop`, { 
            method: "POST", body: JSON.stringify({ session_id: sid }) 
        }).catch(() => {});

        delete state.chat.activeEngines[currentName];
        if (state.chat.engineSessionId === sid) {
            state.chat.engineSessionId = null;
        }
        log(`Pipeline '${currentName}' modified. Engine invalidated.`);
    }

    if (state.mode !== Modes.BUILDER) setMode(Modes.BUILDER); 
    updateActionButtons(); 
}
function setSteps(steps) { 
    state.steps = Array.isArray(steps) ? cloneDeep(steps) : []; 
    
    state.parameterData = null; 
    state.isBuilt = false; 
    state.parametersReady = false;

    resetChatSession(); 
    
    resetContextStack(); 
    renderSteps(); 
    updatePipelinePreview(); 
    updateActionButtons();
}
function updateActionButtons() {
  // 控制 Parameter 面板里的按钮 (保持不变)
  if (els.parameterSave) els.parameterSave.disabled = !(state.isBuilt && state.selectedPipeline);
  if (els.parameterChat) els.parameterChat.disabled = state.mode === Modes.CHAT || !canUseChat();

  // [新增] 控制 Builder 界面悬浮条里的 Chat 按钮
  if (els.directChatBtn) {
      // 只有当 Pipeline 已构建且参数已就绪时，才显示 Chat 按钮
      if (state.isBuilt && state.parametersReady && state.selectedPipeline) {
          els.directChatBtn.classList.remove("d-none");
      } else {
          els.directChatBtn.classList.add("d-none");
      }
  }
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

// ... (Render Helpers - Tool/Loop/Branch Nodes - keep same) ...
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
  const loopSteps = Array.isArray(step.loop.steps) ? step.loop.steps : [];

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
  
  const list = renderStepList(loopSteps, loopLocation, { placeholderText: "Empty Loop", compact: true });
  
  container.append(header, list, actions); if (locationsEqual(loopLocation, getActiveLocation())) container.classList.add("active"); return container;
}
function renderBranchNode(step, parentLocation, index) {
    step.branch.router = Array.isArray(step.branch.router) ? step.branch.router : [];
    step.branch.branches = (step.branch.branches && typeof step.branch.branches === 'object') ? step.branch.branches : {};

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
  const safeSteps = Array.isArray(steps) ? steps : [];
  
  const wrapper = document.createElement("div"); 
  wrapper.className = "step-list";
  
  if (!safeSteps.length) {
    const placeholder = document.createElement("div"); 
    placeholder.className = "flow-placeholder";
    const control = createInsertControl(location, 0, { prominent: !options.compact }); 
    placeholder.appendChild(control); 
    wrapper.appendChild(placeholder); 
    return wrapper;
  }
  
  safeSteps.forEach((step, index) => { 
      wrapper.appendChild(createInsertControl(location, index, { compact: options.compact })); 
      wrapper.appendChild(renderStepNode(step, location, index)); 
  });
  
  wrapper.appendChild(createInsertControl(location, safeSteps.length, { compact: options.compact })); 
  return wrapper;
}
function renderSteps() {
  els.flowCanvas.innerHTML = "";
  
  const activeLocation = getActiveLocation();
  
  const currentSteps = resolveSteps(activeLocation);
  
  els.flowCanvas.appendChild(renderStepList(currentSteps, activeLocation));
}
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
    try {
        console.log(`[UI] Loading pipeline: ${name}`);
        const cfg = await fetchJSON(`/api/pipelines/${encodeURIComponent(name)}`);
        
        state.selectedPipeline = name;
        els.name.value = name;
        

        let safeSteps = [];
        
        if (Array.isArray(cfg)) {
            safeSteps = cfg;
        } else if (cfg && typeof cfg === 'object') {
            if (Array.isArray(cfg.pipeline)) {
                safeSteps = cfg.pipeline;
            } else if (Array.isArray(cfg.steps)) {
                safeSteps = cfg.steps;
            }
        }
        
        safeSteps = cloneDeep(safeSteps);
        
        if (safeSteps.length === 0) {
            console.warn(`[Warn] Loaded pipeline '${name}' appears to be empty. Raw config:`, cfg);
        }

        setSteps(safeSteps);

        if (els.pipelineDropdownBtn) els.pipelineDropdownBtn.textContent = name;
        setHeroPipelineLabel(name);

        // [新增] 自动检查是否 Ready (跳过 Build 的关键)
        checkPipelineReadiness(name);
        
    } catch (err) {
        log(`Failed to load pipeline: ${err.message}`);
        console.error(err);
    }
}

// [新增辅助函数] 检查 Pipeline 是否已配置参数
async function checkPipelineReadiness(name) {
    try {
        // 尝试获取参数
        // 注意：后端 get_parameters 如果文件不存在会报错，所以要 catch
        // 或者我们可以直接利用 list_pipelines 返回的 is_ready 字段，但这里为了确保是最新的，发个请求也无妨
        // 这里为了性能，也可以改用 list 接口查。但直接 fetch 参数最稳妥。
        
        const params = await fetchJSON(`/api/pipelines/${encodeURIComponent(name)}/parameters`);
        
        // 如果成功获取到参数
        state.parameterData = params;
        state.isBuilt = true;         // 视为已构建
        state.parametersReady = true; // 视为参数已就绪
        
        log(`Pipeline '${name}' parameters loaded. Ready to Chat.`);
        updateActionButtons(); // 这会点亮 "Enter Chat Mode" 按钮
        
    } catch (e) {
        // 参数文件不存在，说明需要 Build
        state.isBuilt = false;
        state.parametersReady = false;
        state.parameterData = null;
        updateActionButtons(); // 禁用 Chat 按钮
    }
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
    
    if (els.parameterSave) els.parameterSave.onclick = saveParameterForm;
    if (els.parameterBack) els.parameterBack.onclick = () => setMode(Modes.BUILDER);
    if (els.parameterChat) els.parameterChat.onclick = openChatView;

    if (els.directChatBtn) els.directChatBtn.onclick = openChatView;

    if (els.chatBack) {
        els.chatBack.onclick = async () => {
            // 1. 保存当前对话
            try {
                saveCurrentSession(true);
            } catch (e) {
                console.error(e);
            }
            
            setChatRunning(false);
            
            // [关键修复] 不要直接 setMode，而是调用 showParameterPanel()
            // 这个函数会负责：检查数据 -> 生成 HTML 表单 -> 切换视图
            await showParameterPanel(); 
        };
    }
    
    if (els.chatForm) els.chatForm.onsubmit = handleChatSubmit;
    if (els.chatSend) els.chatSend.onclick = handleChatSubmit;

    if (els.chatNewBtn) els.chatNewBtn.onclick = createNewChatSession;
    if (els.demoToggleBtn) els.demoToggleBtn.onclick = toggleDemoSession;
    
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
  setHeroPipelineLabel(state.selectedPipeline || "");
  try { await Promise.all([refreshPipelines(), refreshTools()]); log("UI Ready."); } catch (err) { log(`Initialization error: ${err.message}`); }
}

bootstrap();
