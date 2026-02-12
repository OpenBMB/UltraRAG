const ACTIVE_ENGINE_STORAGE_KEY = "ultrarag_active_engines";

function loadActiveEnginesFromStorage() {
    try {
        const raw = localStorage.getItem(ACTIVE_ENGINE_STORAGE_KEY);
        if (!raw) return {};
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object") return parsed;
    } catch (e) {
        console.warn("Failed to load active engine cache", e);
    }
    return {};
}

function persistActiveEngines() {
    try {
        localStorage.setItem(
            ACTIVE_ENGINE_STORAGE_KEY,
            JSON.stringify(state.chat?.activeEngines || {})
        );
    } catch (e) {
        console.warn("Failed to persist active engines", e);
    }
}

const UI_LANGUAGE_STORAGE_KEY = "ultrarag_ui_language";
const UI_LANGUAGE_MAP = { en: "English", zh: "中文" };

const LOCALES = window.I18N_LOCALES || {};
const DEFAULT_LOCALE = "en";

function t(key) {
    const lang = state.uiLanguage || DEFAULT_LOCALE;
    const table = LOCALES[lang] || LOCALES[DEFAULT_LOCALE] || {};
    const fallback = LOCALES[DEFAULT_LOCALE] || {};
    return table[key] || fallback[key] || key;
}

function formatTemplate(template, params = {}) {
    return String(template).replace(/\{(\w+)\}/g, (_, key) =>
        params[key] !== undefined ? String(params[key]) : `{${key}}`
    );
}

function updateI18nTexts() {
    document.querySelectorAll('[data-i18n-html]').forEach((el) => {
        const key = el.getAttribute("data-i18n-html");
        if (!key) return;
        el.innerHTML = t(key);
    });

    document.querySelectorAll('[data-i18n]').forEach((el) => {
        const key = el.getAttribute("data-i18n");
        if (!key) return;
        if (el.hasAttribute("data-i18n-html")) return;
        if (el.dataset.i18nOverride === "true") return;
        if (key === "knowledge_base") return; // handled below to avoid overwriting selection
        if (key === "builder_select_pipeline") return; // handled below to preserve selection
        if (key === "builder_no_pipeline_selected") return; // handled below to preserve selection
        el.textContent = t(key);
    });

    // Title attributes
    document.querySelectorAll('[data-i18n-title]').forEach((el) => {
        const key = el.getAttribute("data-i18n-title");
        if (key) el.title = t(key);
    });

    // Placeholder attributes
    document.querySelectorAll('[data-i18n-placeholder]').forEach((el) => {
        const key = el.getAttribute("data-i18n-placeholder");
        if (key) el.placeholder = t(key);
    });

    // Aria-label attributes
    document.querySelectorAll('[data-i18n-aria-label]').forEach((el) => {
        const key = el.getAttribute("data-i18n-aria-label");
        if (key) el.setAttribute("aria-label", t(key));
    });

    // Prompt data attributes
    document.querySelectorAll('[data-i18n-prompt]').forEach((el) => {
        const key = el.getAttribute("data-i18n-prompt");
        if (key) el.setAttribute("data-prompt", t(key));
    });

    // Knowledge base label: only update when no selection
    document.querySelectorAll('[data-i18n="knowledge_base"]').forEach((el) => {
        const hasSelection = el.dataset.selected === "true";
        if (!hasSelection) el.textContent = t("knowledge_base");
    });

    // Refresh DB status text for KB page on language switch
    if (currentDbStatus) {
        updateDbStatusUI(currentDbStatus, currentDbConfig);
    }

    // Refresh KB cards to update vector labels on language switch
    if (Array.isArray(indexCollectionsCache) && indexCollectionsCache.length > 0) {
        renderCollectionList(null, indexCollectionsCache);
    }

    // Select pipeline label: keep selected pipeline name if exists
    const chatPipelineLabel = document.getElementById("chat-pipeline-label");
    if (chatPipelineLabel) {
        const fallbackName = chatPipelineLabel.dataset.currentName;
        const name = state.selectedPipeline || fallbackName;
        if (name) {
            chatPipelineLabel.textContent = name;
            chatPipelineLabel.dataset.selected = "true";
            chatPipelineLabel.dataset.currentName = name;
        } else {
            chatPipelineLabel.textContent = t("select_pipeline");
            chatPipelineLabel.dataset.selected = "false";
        }
    }

    const heroPipelineLabel = document.getElementById("hero-selected-pipeline");
    if (heroPipelineLabel) {
        const fallbackName = heroPipelineLabel.dataset.currentName;
        const name = state.selectedPipeline || fallbackName;
        if (name) {
            heroPipelineLabel.textContent = name;
            heroPipelineLabel.dataset.selected = "true";
            heroPipelineLabel.dataset.currentName = name;
        } else {
            heroPipelineLabel.textContent = t("builder_no_pipeline_selected");
            heroPipelineLabel.dataset.selected = "false";
        }
    }

    const pipelineDropdown = document.getElementById("pipelineDropdownBtn");
    if (pipelineDropdown) {
        const fallbackName = pipelineDropdown.dataset.currentName;
        const name = state.selectedPipeline || fallbackName;
        if (name) {
            pipelineDropdown.textContent = name;
            pipelineDropdown.dataset.selected = "true";
            pipelineDropdown.dataset.currentName = name;
        } else {
            pipelineDropdown.textContent = t("builder_select_pipeline");
            pipelineDropdown.dataset.selected = "false";
        }
    }

    if (typeof updateAIConnectionStatus === "function") {
        updateAIConnectionStatus();
    }
    if (typeof updateAIContextBanner === "function") {
        updateAIContextBanner("i18n");
    }
    if (typeof renderAISessionList === "function") {
        renderAISessionList();
    }
    if (typeof renderPromptTabs === "function") {
        renderPromptTabs();
    }
    if (typeof renderPromptList === "function" && typeof workspaceState !== "undefined") {
        renderPromptList(workspaceState?.prompts?.files || []);
    }

    // Placeholder updates
    const chatInput = document.getElementById("chat-input");
    if (chatInput) {
        chatInput.placeholder = t("placeholder_chat_input");
    }

    // Empty state greeting
    const greeting = document.querySelector(".greeting-gradient");
    if (greeting) {
        greeting.textContent = t("greeting_explore");
    }

    // Update chat history i18n texts (Show Thinking, Cited References, Other Retrieved)
    document.querySelectorAll(".process-header > span[data-i18n-key='chat_show_thinking']").forEach((el) => {
        el.textContent = t("chat_show_thinking");
    });
    document.querySelectorAll(".ref-header[data-i18n-key='chat_cited_references']").forEach((el) => {
        const count = el.dataset.count || 0;
        el.textContent = `${t("chat_cited_references")} (${count})`;
    });
    document.querySelectorAll(".unused-header span[data-i18n-key='chat_other_retrieved']").forEach((el) => {
        const count = el.dataset.count || 0;
        el.textContent = `${t("chat_other_retrieved")} (${count})`;
    });
}

function resolveInitialLanguage() {
    try {
        const stored = localStorage.getItem(UI_LANGUAGE_STORAGE_KEY);
        if (stored && UI_LANGUAGE_MAP[stored]) {
            return stored;
        }
    } catch (e) {
        console.warn("Failed to load UI language", e);
    }

    // Default to English
    return "en";
}

const state = {
    selectedPipeline: null,
    steps: [],
    contextStack: [],
    toolCatalog: { order: [], byServer: {} },
    parameterData: null,
    pipelineConfig: null,  // Store complete pipeline configuration (including servers)
    editingPath: null,
    isBuilt: false,
    parametersReady: false,
    lastSavedYaml: "",
    unsavedChanges: false,
    mode: "builder",

    // Application mode: true = admin (full interface), false = chat-only
    adminMode: true,

    // Simplified parameter display mode: true = show only essential params, false = show all
    simplifiedParams: true,

    uiLanguage: resolveInitialLanguage(),

    // [Modified] Chat state management
    chat: {
        history: [],
        running: false,
        sessions: [],
        currentSessionId: null,

        controller: null,

        // Engine connection state
        engineSessionId: null, // SessionID corresponding to currently selected Pipeline
        activeEngines: loadActiveEnginesFromStorage(),     // Mapping table: { "pipelineName": "sessionId" }
        engineStartSeq: 0,     // Used to avoid state disorder caused by concurrent startup
        engineStartingFor: null, // Pipeline name currently being started
        engineStartPromise: null, // Promise for engine startup (used for serialization)

        demoLoading: false
    },
};

const WORKSPACE_PANE_WIDTH_KEY = 'ultrarag_workspace_pane_width';
const DEFAULT_WORKSPACE_PANE_WIDTH = 320;
const MIN_WORKSPACE_PANE_WIDTH = 240;
const MIN_WORKSPACE_CONTENT_WIDTH = 320;

// Default Vanilla LLM pipeline template
const VANILLA_PIPELINE_TEMPLATE = {
    servers: {
        benchmark: "servers/benchmark",
        generation: "servers/generation",
        prompt: "servers/prompt",
    },
    pipeline: [
        "benchmark.get_data",
        "generation.generation_init",
        "prompt.qa_boxed",
        "generation.generate",
    ],
};

const VANILLA_PIPELINE_STEPS = [...VANILLA_PIPELINE_TEMPLATE.pipeline];

const els = {
    // View Containers
    mainRoot: document.querySelector(".content-wrapper"),
    pipelineForm: document.getElementById("pipeline-form"),
    parameterPanel: document.getElementById("parameter-panel"),
    chatView: document.getElementById("chat-view"),
    // runView removed

    // Logs
    log: document.getElementById("log"),
    // runTerminal, runSpinner removed

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
    heroSelectedPipeline: document.getElementById("hero-selected-pipeline"),
    heroStatus: document.getElementById("hero-status"),
    builderLogo: document.getElementById("builder-logo-link"),
    workspaceChatBtn: document.getElementById("workspace-chat-btn"),
    workspaceAiBtn: document.getElementById("navbar-ai-btn"),

    // Parameter Controls
    parameterForm: document.getElementById("parameter-form"),
    parameterSave: document.getElementById("parameter-save"),
    parameterBack: document.getElementById("parameter-back"),
    // parameterRun removed
    parameterChat: document.getElementById("parameter-chat"),

    // Chat Controls
    chatPipelineName: document.getElementById("chat-pipeline-name"),
    chatHistory: document.getElementById("chat-history"),
    chatForm: document.getElementById("chat-form"),
    chatInput: document.getElementById("chat-input"),
    chatStatus: document.getElementById("chat-status"),
    chatSend: document.getElementById("chat-send"),
    chatNewBtn: document.getElementById("chat-new-btn"),
    chatSessionList: document.getElementById("chat-session-list"),
    clearAllChats: document.getElementById("clear-all-chats"),
    demoToggleBtn: document.getElementById("demo-toggle-btn"), // Engine toggle button
    chatCollectionSelect: document.getElementById("chat-collection-select"),
    settingsMenu: document.getElementById("settings-menu"),
    settingsMenuTrigger: document.getElementById("settings-menu-trigger"),
    settingsDropdown: document.getElementById("settings-dropdown"),
    settingsBuilder: document.getElementById("settings-builder"),
    settingsLanguageLabel: document.getElementById("settings-language-label"),

    // [New] View containers
    chatMainView: document.getElementById("chat-main-view"),
    kbMainView: document.getElementById("kb-main-view"),
    // [New] Buttons
    kbBtn: document.getElementById("kb-btn"),

    chatSidebar: document.querySelector(".chat-sidebar"),
    chatSidebarToggleBtn: document.getElementById("sidebar-toggle-btn"),
    chatLogoBtn: document.getElementById("chat-logo-btn"),

    // [New] Chat top controls
    chatPipelineLabel: document.getElementById("chat-pipeline-label"),
    chatPipelineMenu: document.getElementById("chat-pipeline-menu"),

    // Node Picker (keep as is)
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

    // --- [Start] Knowledge Base Elements ---
    listRaw: document.getElementById("list-raw"),
    listCorpus: document.getElementById("list-corpus"),
    listChunks: document.getElementById("list-chunks"),
    // File upload input (hidden, but we need to reference it to bind events or trigger clicks)
    fileUpload: document.getElementById("file-upload"),
    // Status bar
    taskStatusBar: document.getElementById("task-status-bar"),
    taskMsg: document.getElementById("task-msg"),
    taskSpinner: document.getElementById("task-spinner"),
    taskProgressWrapper: document.getElementById("task-progress-wrapper"),
    taskProgressCircle: document.getElementById("task-progress-circle"),
    taskProgressText: document.getElementById("task-progress-text"),
    // Database configuration elements
    dbConnectionStatus: document.getElementById("db-connection-status"),
    dbConnectionText: document.getElementById("db-connection-text"),
    dbConnectionChip: document.getElementById("db-connection-chip"),
    dbUriDisplay: document.getElementById("db-uri-display"),
    dbConfigModal: document.getElementById("db-config-modal"), // New configuration modal
    cfgUri: document.getElementById("cfg-uri"),                 // Configuration modal - URI input
    cfgToken: document.getElementById("cfg-token"),              // Configuration modal - Token input
    listIndexes: document.getElementById("list-indexes"),         // Collection list container
    modalTargetDb: document.getElementById("modal-target-db"),   // Hint text in Milvus modal

    // Milvus modal related (keep and confirm ID)
    milvusDialog: document.getElementById("milvus-dialog"),
    idxCollection: document.getElementById("idx-collection"),
    idxCollectionLabel: document.getElementById("idx-collection-label"),
    idxCollectionSelect: document.getElementById("idx-collection-select"),
    idxMode: document.getElementById("idx-mode"),
    // [New] Refresh button
    refreshCollectionsBtn: document.getElementById("refresh-collections-btn"),
    // --- [End] Knowledge Base Elements ---

    // --- [Start] YAML Editor Elements ---
    yamlEditor: document.getElementById("yaml-editor"),
    yamlLineNumbers: document.getElementById("yaml-line-numbers"),
    yamlSyncStatus: document.getElementById("yaml-sync-status"),
    yamlErrorBar: document.getElementById("yaml-error-bar"),
    yamlErrorMessage: document.getElementById("yaml-error-message"),
    yamlFormatBtn: document.getElementById("yaml-format-btn"),
    builderResizer: document.getElementById("builder-resizer"),
    // --- [End] YAML Editor Elements ---

    // --- [Start] Console Elements ---
    canvasConsole: document.getElementById("canvas-console"),
    consoleToggle: document.getElementById("console-toggle"),
    // --- [End] Console Elements ---
};

// Cache Build button's initial content for state switching recovery
let buildBtnDefaultHtml = els.buildPipeline ? els.buildPipeline.innerHTML : "";

const Modes = {
    BUILDER: "builder",
    PARAMETERS: "parameters",
    CHAT: "chat",
};

// ==========================================
// --- Unified Modal System ---
// ==========================================

/**
 * Show a modal dialog (replaces alert)
 * @param {string} message - The message to display
 * @param {object} options - Configuration options
 * @param {string} options.title - Modal title (default: "Notification")
 * @param {string} options.type - Icon type: 'info' | 'success' | 'warning' | 'error' (default: 'info')
 * @param {string} options.confirmText - Text for confirm button (default: "OK")
 * @returns {Promise<void>}
 */
function showModal(message, options = {}) {
    const {
        title = t("modal_notification"),
        type = "info",
        confirmText = t("common_ok")
    } = options;

    return new Promise((resolve) => {
        const modal = document.getElementById("unified-modal");
        const iconEl = document.getElementById("unified-modal-icon");
        const titleEl = document.getElementById("unified-modal-title");
        const messageEl = document.getElementById("unified-modal-message");
        const actionsEl = document.getElementById("unified-modal-actions");

        if (!modal) {
            console.warn("Unified modal not found, falling back to alert");
            window.alert(message);
            resolve();
            return;
        }

        // Set icon based on type
        iconEl.className = `unified-modal-icon ${type}`;
        const icons = {
            info: '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>',
            success: '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
            warning: '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>',
            error: '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>'
        };
        iconEl.innerHTML = icons[type] || icons.info;

        // Set content
        titleEl.textContent = title;
        messageEl.textContent = message;

        // Set actions (single OK button)
        actionsEl.innerHTML = `<button class="btn unified-modal-btn unified-modal-btn-primary" id="unified-modal-ok">${escapeHtml(confirmText)}</button>`;

        const okBtn = document.getElementById("unified-modal-ok");
        const closeHandler = () => {
            modal.close();
            resolve();
        };
        okBtn.onclick = closeHandler;

        // Close on backdrop click
        modal.onclick = (e) => {
            if (e.target === modal) closeHandler();
        };

        // Close on Escape key
        modal.onkeydown = (e) => {
            if (e.key === "Escape") closeHandler();
        };

        modal.showModal();
    });
}

/**
 * Show a confirm dialog (replaces confirm)
 * @param {string} message - The message to display
 * @param {object} options - Configuration options
 * @param {string} options.title - Modal title (default: "Confirm")
 * @param {string} options.type - Icon type: 'info' | 'warning' | 'confirm' (default: 'confirm')
 * @param {string} options.confirmText - Text for confirm button (default: "Confirm")
 * @param {string} options.cancelText - Text for cancel button (default: "Cancel")
 * @param {boolean} options.danger - Whether this is a dangerous action (default: false)
 * @returns {Promise<boolean>}
 */
function showConfirm(message, options = {}) {
    const {
        title = t("modal_confirm_title"),
        type = "confirm",
        confirmText = t("common_confirm"),
        cancelText = t("common_cancel"),
        danger = false
    } = options;

    return new Promise((resolve) => {
        const modal = document.getElementById("unified-modal");
        const iconEl = document.getElementById("unified-modal-icon");
        const titleEl = document.getElementById("unified-modal-title");
        const messageEl = document.getElementById("unified-modal-message");
        const actionsEl = document.getElementById("unified-modal-actions");

        if (!modal) {
            console.warn("Unified modal not found, falling back to confirm");
            resolve(window.confirm(message));
            return;
        }

        // Set icon based on type
        iconEl.className = `unified-modal-icon ${type}`;
        const icons = {
            info: '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>',
            warning: '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>',
            confirm: '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
        };
        iconEl.innerHTML = icons[type] || icons.confirm;

        // Set content
        titleEl.textContent = title;
        messageEl.textContent = message;

        // Set actions (Cancel + Confirm buttons)
        const confirmBtnClass = danger ? "unified-modal-btn-danger" : "unified-modal-btn-primary";
        actionsEl.innerHTML = `
      <button class="btn unified-modal-btn unified-modal-btn-secondary" id="unified-modal-cancel">${escapeHtml(cancelText)}</button>
      <button class="btn unified-modal-btn ${confirmBtnClass}" id="unified-modal-confirm">${escapeHtml(confirmText)}</button>
    `;

        const cancelBtn = document.getElementById("unified-modal-cancel");
        const confirmBtn = document.getElementById("unified-modal-confirm");

        let resolved = false;
        const closeWith = (result) => {
            if (resolved) return;
            resolved = true;
            modal.close();
            resolve(result);
        };

        cancelBtn.onclick = () => closeWith(false);
        confirmBtn.onclick = () => closeWith(true);

        // Close on backdrop click = cancel
        modal.onclick = (e) => {
            if (e.target === modal) closeWith(false);
        };

        // Escape = cancel
        modal.onkeydown = (e) => {
            if (e.key === "Escape") closeWith(false);
        };

        modal.showModal();
    });
}

/**
 * Show a choice dialog with two actions.
 * @param {string} message - The message to display
 * @param {object} options - Configuration options
 * @param {string} options.title - Modal title (default: "Choose an option")
 * @param {string} options.type - Icon type: 'info' | 'warning' | 'confirm' (default: 'warning')
 * @param {string} options.primaryText - Primary button label
 * @param {string} options.secondaryText - Secondary button label
 * @param {string} options.cancelText - Cancel button label
 * @param {string} options.primaryValue - Value returned for primary button
 * @param {string} options.secondaryValue - Value returned for secondary button
 * @param {boolean} options.dangerPrimary - Whether primary is dangerous
 * @param {boolean} options.dangerSecondary - Whether secondary is dangerous
 * @returns {Promise<string|null>}
 */
function showChoice(message, options = {}) {
    const {
        title = t("modal_choice_title"),
        type = "warning",
        primaryText = t("common_primary"),
        secondaryText = t("common_secondary"),
        cancelText = t("common_cancel"),
        primaryValue = "primary",
        secondaryValue = "secondary",
        dangerPrimary = false,
        dangerSecondary = false
    } = options;

    return new Promise((resolve) => {
        const modal = document.getElementById("unified-modal");
        const iconEl = document.getElementById("unified-modal-icon");
        const titleEl = document.getElementById("unified-modal-title");
        const messageEl = document.getElementById("unified-modal-message");
        const actionsEl = document.getElementById("unified-modal-actions");

        if (!modal) {
            console.warn("Unified modal not found, falling back to confirm");
            resolve(window.confirm(message) ? primaryValue : null);
            return;
        }

        // Set icon based on type (reuse confirm icon if unknown)
        iconEl.className = `unified-modal-icon ${type}`;
        const icons = {
            info: '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>',
            warning: '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>',
            confirm: '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
        };
        iconEl.innerHTML = icons[type] || icons.confirm;

        titleEl.textContent = title;
        messageEl.textContent = message;

        const primaryClass = dangerPrimary ? "unified-modal-btn-danger" : "unified-modal-btn-primary";
        const secondaryClass = dangerSecondary ? "unified-modal-btn-danger" : "unified-modal-btn-secondary";
        actionsEl.innerHTML = `
      <button class="btn unified-modal-btn unified-modal-btn-secondary" id="unified-modal-cancel">${escapeHtml(cancelText)}</button>
      <button class="btn unified-modal-btn ${secondaryClass}" id="unified-modal-secondary">${escapeHtml(secondaryText)}</button>
      <button class="btn unified-modal-btn ${primaryClass}" id="unified-modal-primary">${escapeHtml(primaryText)}</button>
    `;

        const cancelBtn = document.getElementById("unified-modal-cancel");
        const secondaryBtn = document.getElementById("unified-modal-secondary");
        const primaryBtn = document.getElementById("unified-modal-primary");

        let resolved = false;
        const closeWith = (result) => {
            if (resolved) return;
            resolved = true;
            modal.close();
            resolve(result);
        };

        cancelBtn.onclick = () => closeWith(null);
        secondaryBtn.onclick = () => closeWith(secondaryValue);
        primaryBtn.onclick = () => closeWith(primaryValue);

        // Close on backdrop click = cancel
        modal.onclick = (e) => {
            if (e.target === modal) closeWith(null);
        };

        // Escape = cancel
        modal.onkeydown = (e) => {
            if (e.key === "Escape") closeWith(null);
        };

        modal.showModal();
    });
}

// Make functions globally available
window.showModal = showModal;
window.showConfirm = showConfirm;

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

// ==========================================
// --- Knowledge Base Logic ---
// ==========================================


let currentTargetFile = null;
let existingFilesSnapshot = new Set();
let indexCollectionsCache = [];
let lastIndexMode = "new";
let currentDbConfig = null;
let currentDbStatus = null;

// Open import workspace modal
window.openImportModal = async function () {
    const modal = document.getElementById('import-modal');
    if (modal) modal.showModal();

    // 1. Clear snapshot first
    existingFilesSnapshot.clear();

    // 2. Get all current files to establish baseline
    try {
        const data = await fetchJSON('/api/kb/files');

        // Add all existing file paths to snapshot
        // Files already in the list are not considered "new"
        const recordFiles = (list) => list.forEach(f => existingFilesSnapshot.add(f.path));

        recordFiles(data.raw);
        recordFiles(data.corpus);
        recordFiles(data.chunks);

        // 3. Render immediately (no highlighting at this point since all are in snapshot)
        // Manually call render to avoid refreshKBFiles not finishing fetch
        refreshKBModalViews(data);

    } catch (e) {
        console.error("Init modal failed:", e);
    }
};

// Close import workspace modal
window.closeImportModal = function () {
    const modal = document.getElementById('import-modal');
    if (modal) modal.close();

    // Refresh main interface after closing to ensure newly created Collections appear on bookshelf immediately
    refreshKBFiles();
};

// Clear staging area
window.clearStagingArea = async function () {
    const confirmed = await showConfirm(t("kb_clear_staging_prompt"), {
        title: t("kb_clear_staging_title"),
        type: "warning",
        confirmText: t("kb_clear_all"),
        danger: true
    });
    if (!confirmed) return;

    try {
        const res = await fetch('/api/kb/staging/clear', { method: 'POST' });
        const data = await res.json();

        if (res.ok) {
            const total = data.total_deleted || 0;
            const counts = data.deleted_counts || {};
            let message = formatTemplate(t("kb_deleted_summary"), {
                raw: counts.raw || 0,
                corpus: counts.corpus || 0,
                chunks: counts.chunks || 0,
                total
            });

            if (data.errors && data.errors.length > 0) {
                message += `\n\n${formatTemplate(t("kb_note_errors"), {
                    errors: data.errors.slice(0, 3).join('\n')
                })}`;
                if (data.errors.length > 3) {
                    message += `\n${formatTemplate(t("kb_more_errors"), {
                        count: data.errors.length - 3
                    })}`;
                }
            }

            await showModal(message, { title: t("kb_staging_cleared_title"), type: "success" });
            await refreshKBFiles();
        } else {
            await showModal(t("kb_clear_failed") + (data.error || res.statusText), { title: t("status_error"), type: "error" });
        }
    } catch (e) {
        console.error(e);
        await showModal(t("kb_clear_error") + e.message, { title: t("status_error"), type: "error" });
    }
};



// Helper function specifically for refreshing modal views
function refreshKBModalViews(data) {
    renderKBList(document.getElementById('list-raw'), data.raw, 'build_text_corpus', t("kb_action_parse"));
    renderKBList(document.getElementById('list-corpus'), data.corpus, 'corpus_chunk', t("kb_action_chunk"));
    renderKBList(document.getElementById('list-chunks'), data.chunks, 'milvus_index', t("kb_action_index"));
}

// Refresh file list (main function)
async function refreshKBFiles() {
    try {
        const data = await fetchJSON('/api/kb/files');

        // 1. Refresh modal views
        refreshKBModalViews(data);

        // 2. Refresh main page bookshelf
        renderCollectionList(null, data.index);

        // 2.5 Refresh index collection cache for dropdown usage
        indexCollectionsCache = data.index || [];
        if (els.idxMode && els.idxMode.value !== "new") {
            renderIndexCollectionSelect(indexCollectionsCache, els.idxCollectionSelect ? els.idxCollectionSelect.value : "");
        }

        // 3. Update status
        updateDbStatusUI(data.db_status, data.db_config);

    } catch (e) {
        console.error("Failed to load KB files:", e);
    }
}

// Render pipeline list (apply snapshot highlighting)
function renderKBList(container, files, nextPipeline, actionLabel) {
    if (!container) return;
    container.innerHTML = '';

    if (!files || files.length === 0) {
        container.innerHTML = `<div class="text-muted small text-center mt-5 opacity-50">${t("kb_empty")}</div>`;
        return;
    }

    // Sort by time descending, newest files first
    files.sort((a, b) => (b.mtime || 0) - (a.mtime || 0));

    files.forEach(f => {
        const div = document.createElement('div');

        const isNew = !existingFilesSnapshot.has(f.path);

        div.className = `file-item ${isNew ? 'new-upload' : ''}`;

        // 1. Determine basic information
        const isFolder = f.type === 'folder';
        const displayText = f.display_name || f.name;
        const tooltipText = f.display_name && f.display_name !== f.name
            ? `${f.display_name}\n(${f.name})`
            : f.name;
        const sizeStr = (f.size / 1024).toFixed(1) + " KB";

        // 2. Icon (SVG)
        const svgFolder = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-blue-500"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>`;
        const svgFile = `<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-gray-400"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>`;
        const iconSvg = isFolder ? svgFolder : svgFile;

        // 3. Metadata row content
        let metaText = sizeStr;
        if (isFolder && f.file_count) {
            metaText = `${f.file_count} ${t("kb_files")} · ${sizeStr}`;
        }

        // 4. Action button
        let actionBtn = `<button class="btn btn-sm btn-light border ms-auto flex-shrink-0" style="font-size:0.75rem;" onclick="event.stopPropagation(); window.handleKBAction('${f.path}', '${nextPipeline}')">${actionLabel}</button>`;

        // 5. Delete button
        let deleteBtn = '';
        if (f.category !== 'collection') {
            deleteBtn = `<button class="btn btn-sm text-danger ms-2 btn-icon-only flex-shrink-0" onclick="event.stopPropagation(); deleteKBFile('${f.category}', '${f.name}')" title="${t("kb_delete")}">×</button>`;
        }

        // 6. Card click event
        let onClickAttr = "";
        if (isFolder) {
            // Pass display_name for modal title
            onClickAttr = `onclick="window.inspectFolder('${f.category}', '${f.name}', '${displayText.replace(/'/g, "\\'")}')"`;
        } else {
            // File click has no action for now, or can add preview
            onClickAttr = `onclick=""`;
        }

        // 7. Build HTML (Finder style: two-line layout)
        div.innerHTML = `
            <div class="file-item-inner" ${onClickAttr}>
                <div class="file-icon-wrapper">${iconSvg}</div>
                <div class="file-info-wrapper">
                    <div class="file-title" title="${tooltipText}">${displayText}</div>
                    <div class="file-meta">
                        ${metaText}
                    </div>
                </div>
                <div class="file-actions">
                    ${actionBtn}
                    ${deleteBtn}
                </div>
            </div>
        `;
        container.appendChild(div);
    });
}

// New inspection function (mounted to window)
// Added displayName parameter
window.inspectFolder = async function (category, folderName, displayName) {
    const modal = document.getElementById('folder-detail-modal');
    const listContainer = document.getElementById('folder-detail-list');
    const title = document.getElementById('folder-detail-title');

    // Set title (prefer display_name)
    if (title) title.textContent = displayName || folderName;

    // Show loading
    if (listContainer) listContainer.innerHTML = '<div class="text-center text-muted p-3">Loading...</div>';

    // Open modal
    if (modal) modal.showModal();

    try {
        const res = await fetch(`/api/kb/files/inspect?category=${category}&name=${encodeURIComponent(folderName)}`);

        const data = await res.json();

        if (data.files && data.files.length > 0) {
            // Filter out files starting with _ like _meta.json
            const visibleFiles = data.files.filter(f => !f.name.startsWith('_'));

            if (visibleFiles.length > 0) {
                listContainer.innerHTML = visibleFiles.map(f => `
                    <div class="folder-file-row">
                        <span class="file-row-icon">📄</span>
                        <span class="file-row-name text-truncate">${f.name}</span>
                        <span class="text-muted ms-auto" style="font-size:0.75rem;">${(f.size / 1024).toFixed(1)} KB</span>
                    </div>
                `).join('');
            } else {
                listContainer.innerHTML = `<div class="text-center text-muted small mt-3">${t("kb_empty_no_visible")}</div>`;
            }
        } else {
            listContainer.innerHTML = `<div class="text-center text-muted small mt-3">${t("kb_empty_folder")}</div>`;
        }
    } catch (e) {
        if (listContainer) listContainer.innerHTML = `<div class="text-danger small p-2">${t("status_error")}: ${e.message}</div>`;
        console.error(e);
    }
};

// Gradient palette and utility functions: make card colors stable and soft
const KB_COVER_PALETTE = [
    "#e0f2fe", // sky-100
    "#dcfce7", // green-100
    "#f3e8ff", // purple-100
    "#fee2e2", // red-100
    "#ffedd5", // orange-100
    "#f1f5f9", // slate-100
    "#fae8ff", // fuchsia-100
    "#e0e7ff", // indigo-100
];

const KB_TEXT_PALETTE = [
    "#0369a1", // sky-700
    "#15803d", // green-700
    "#7e22ce", // purple-700
    "#b91c1c", // red-700
    "#c2410c", // orange-700
    "#334155", // slate-700
    "#a21caf", // fuchsia-700
    "#4338ca", // indigo-700
];

function hashString(input = "") {
    let hash = 0;
    for (let i = 0; i < input.length; i++) {
        hash = (hash * 31 + input.charCodeAt(i)) >>> 0;
    }
    return hash;
}

function pickKbColors(key = "") {
    const idx = hashString(key) % KB_COVER_PALETTE.length;
    return {
        bg: KB_COVER_PALETTE[idx],
        text: KB_TEXT_PALETTE[idx]
    };
}

function getKbInitial(name = "") {
    const initial = name.trim().charAt(0);
    return initial ? initial.toUpperCase() : "?";
}

// Render Collection list -> bookshelf card mode
function renderCollectionList(container, collections) {
    const grid = document.getElementById('bookshelf-grid');
    if (!grid) return;

    grid.innerHTML = '';

    if (!collections || collections.length === 0) {
        grid.innerHTML = `
            <div class="col-12 text-center py-5 text-muted" style="grid-column: 1 / -1;">
                <div style="font-size:3rem; margin-bottom:1rem; opacity:0.3;">📚</div>
                <h5>${t("kb_library_empty_title")}</h5>
                <p>${t("kb_library_empty_hint")}</p>
            </div>
        `;
        return;
    }

    // Sort by display name
    const getLabel = c => (c.display_name || c.name || "").toLowerCase();
    collections = collections.slice().sort((a, b) => getLabel(a).localeCompare(getLabel(b)));

    collections.forEach(c => {
        const displayName = c.display_name || c.name || "Untitled";
        const card = document.createElement('div');
        card.className = 'collection-card kb-card';

        const countStr = c.count !== undefined ? `${c.count} ${t("kb_vectors")}` : t("kb_ready");
        const colors = pickKbColors(displayName || c.name || "collection");
        const coverInitial = getKbInitial(displayName || c.name || "C");

        // Render card
        card.innerHTML = `
            <div class="kb-card-main">
                <div class="kb-icon-box" style="background-color: ${colors.bg}; color: ${colors.text}">
                    ${coverInitial}
                </div>
                <div class="kb-info-box">
                     <div class="kb-card-title" title="${displayName}">${displayName}</div>
                     <div class="kb-meta-count">${countStr}</div>
                </div>
                
                <button class="btn-delete-book" onclick="event.stopPropagation(); deleteKBFile('collection', '${c.name}', '${displayName.replace(/'/g, "\\'")}')" title="${t("kb_delete_collection")}">
                     <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                </button>
            </div>
        `;

        grid.appendChild(card);
    });
}

// UI status update
function updateDbStatusUI(status, config) {
    currentDbConfig = config;
    currentDbStatus = status;

    const chip = els.dbConnectionChip || document.getElementById('db-connection-chip');
    const statusTextEl = els.dbConnectionText || document.getElementById('db-connection-text');

    if (!els.dbConnectionStatus || !els.dbUriDisplay || !chip || !statusTextEl) return;

    // Status dot & text
    const statusClass = status === 'connected' ? 'connected' : (status === 'connecting' ? 'connecting' : 'disconnected');
    els.dbConnectionStatus.className = `kb-conn-dot ${statusClass}`;
    statusTextEl.textContent = status === 'connected'
        ? t('kb_connected')
        : (status === 'connecting' ? t('kb_connecting') : t('kb_disconnected'));
    chip.setAttribute('data-status', statusClass);

    // URI display: use shortened version in main text, full address in tooltip
    const fullUri = (config && config.milvus && config.milvus.uri) ? config.milvus.uri : t("kb_not_configured");
    const shortUri = fullUri.length > 38 ? `${fullUri.slice(0, 16)}…${fullUri.slice(-12)}` : fullUri;
    els.dbUriDisplay.textContent = shortUri;
    chip.title = `${t("kb_endpoint")}: ${fullUri}`;
}

// Configuration modal logic (mounted to window)
window.openDbConfigModal = async function () {
    const res = await fetchJSON('/api/kb/config');
    const cfg = res;

    // Read from milvus field
    const milvus = cfg.milvus || {};

    if (els.cfgUri) els.cfgUri.value = milvus.uri || '';
    if (els.cfgToken) els.cfgToken.value = milvus.token || '';

    // Store full config structure for merging on save
    window._currentFullKbConfig = cfg;

    if (els.dbConfigModal) els.dbConfigModal.showModal();
};

window.saveDbConfig = async function () {
    if (!els.cfgUri) return;
    const uri = els.cfgUri.value.trim();
    const token = els.cfgToken.value.trim();

    if (!uri) { showModal(t("kb_uri_required"), { title: t("kb_validation_error"), type: "warning" }); return; }

    const fullConfig = window._currentFullKbConfig || {};
    if (!fullConfig.milvus) fullConfig.milvus = {};

    // Only update URI and Token, preserve other advanced fields
    fullConfig.milvus.uri = uri;
    fullConfig.milvus.token = token;

    // Send complete JSON structure back
    await fetch('/api/kb/config', {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(fullConfig)
    });

    if (els.dbConfigModal) els.dbConfigModal.close();
    refreshKBFiles(); // Refresh immediately to test connection status
};

// ==========================================
// --- Chunk Configuration Logic ---
// ==========================================

const CHUNK_CONFIG_STORAGE_KEY = "ultrarag_chunk_config";
const INDEX_CONFIG_STORAGE_KEY = "ultrarag_index_config";
const AI_SETTINGS_STORAGE_KEY = "ultrarag_ai_settings";
const DEFAULT_INDEX_CONFIG = {
    api_key: "",
    base_url: "https://api.openai.com/v1",
    model_name: "text-embedding-3-small"
};
const DEFAULT_AI_SETTINGS = {
    provider: "openai",
    baseUrl: "https://api.openai.com/v1",
    apiKey: "",
    model: "gpt-5-mini"
};

// Load Chunk configuration from localStorage
function loadChunkConfigFromStorage() {
    try {
        const raw = localStorage.getItem(CHUNK_CONFIG_STORAGE_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object") return parsed;
    } catch (e) {
        console.warn("Failed to load chunk config from storage", e);
    }
    return null;
}

// Save Chunk configuration to localStorage
function persistChunkConfig() {
    try {
        localStorage.setItem(CHUNK_CONFIG_STORAGE_KEY, JSON.stringify(chunkConfigState));
    } catch (e) {
        console.warn("Failed to persist chunk config", e);
    }
}

// Load Embedding configuration from localStorage
function loadIndexConfigFromStorage() {
    try {
        const raw = localStorage.getItem(INDEX_CONFIG_STORAGE_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object") return parsed;
    } catch (e) {
        console.warn("Failed to load index config from storage", e);
    }
    return null;
}

// Save Embedding configuration to localStorage
function persistIndexConfig() {
    try {
        localStorage.setItem(INDEX_CONFIG_STORAGE_KEY, JSON.stringify(indexConfigState));
    } catch (e) {
        console.warn("Failed to persist index config", e);
    }
}

// 1. Define default configuration state (try to restore from localStorage)
let chunkConfigState = loadChunkConfigFromStorage() || {
    chunk_backend: "token",
    tokenizer_or_token_counter: "gpt2",
    chunk_size: 500,
    use_title: true
};

// 2. Open configuration modal (echo current state)
window.openChunkConfigModal = function () {
    const modal = document.getElementById('chunk-config-modal');

    // Echo data
    document.getElementById('cfg-chunk-backend').value = chunkConfigState.chunk_backend;
    document.getElementById('cfg-chunk-tokenizer').value = chunkConfigState.tokenizer_or_token_counter;
    document.getElementById('cfg-chunk-size').value = chunkConfigState.chunk_size;
    document.getElementById('cfg-chunk-title').value = chunkConfigState.use_title ? "true" : "false";

    if (modal) modal.showModal();
};

// 3. Save configuration
window.saveChunkConfig = function () {
    const backend = document.getElementById('cfg-chunk-backend').value;
    const tokenizer = document.getElementById('cfg-chunk-tokenizer').value;
    const size = parseInt(document.getElementById('cfg-chunk-size').value, 10);
    const useTitleStr = document.getElementById('cfg-chunk-title').value;

    if (isNaN(size) || size <= 0) {
        showModal(t("kb_chunk_size_invalid"), { title: t("kb_validation_error"), type: "warning" });
        return;
    }

    // Update global state
    chunkConfigState = {
        chunk_backend: backend,
        tokenizer_or_token_counter: tokenizer,
        chunk_size: size,
        use_title: (useTitleStr === "true")
    };

    // Persist to localStorage
    persistChunkConfig();

    const modal = document.getElementById('chunk-config-modal');
    if (modal) modal.close();

    console.log("Chunk Config Updated & Saved:", chunkConfigState);
};

// ==========================================
// --- Index (Embedding) Configuration Logic ---
// ==========================================

const storedIndexConfig = loadIndexConfigFromStorage();
// 1. Define default configuration state (try to restore from localStorage)
let indexConfigState = { ...DEFAULT_INDEX_CONFIG, ...(storedIndexConfig || {}) };

// 2. Open configuration modal (echo current state)
window.openIndexConfigModal = function () {
    const modal = document.getElementById('index-config-modal');

    // Echo data
    document.getElementById('cfg-emb-api-key').value = indexConfigState.api_key;
    document.getElementById('cfg-emb-base-url').value = indexConfigState.base_url;
    document.getElementById('cfg-emb-model-name').value = indexConfigState.model_name;

    if (modal) modal.showModal();
};

// 3. Save configuration
window.saveIndexConfig = function () {
    const apiKey = document.getElementById('cfg-emb-api-key').value.trim();
    const baseUrl = document.getElementById('cfg-emb-base-url').value.trim();
    const modelName = document.getElementById('cfg-emb-model-name').value.trim();

    if (!baseUrl) {
        showModal(t("kb_base_url_required"), { title: t("kb_validation_error"), type: "warning" });
        return;
    }

    if (!modelName) {
        showModal(t("kb_model_name_required"), { title: t("kb_validation_error"), type: "warning" });
        return;
    }

    // Update global state
    indexConfigState = {
        api_key: apiKey,
        base_url: baseUrl,
        model_name: modelName
    };

    // Persist to localStorage
    persistIndexConfig();

    const modal = document.getElementById('index-config-modal');
    if (modal) modal.close();

    console.log("Index Config Updated & Saved:", indexConfigState);
};

// ==========================================

// Handle action button clicks (mounted to window)
window.handleKBAction = async function (filePath, pipelineName) {
    currentTargetFile = filePath;

    if (pipelineName === 'milvus_index') {
        // Update hint and default value in Milvus modal
        const uriTxt = els.dbUriDisplay ? els.dbUriDisplay.textContent : t("kb_current_db");
        if (els.modalTargetDb) {
            els.modalTargetDb.textContent = uriTxt;
            els.modalTargetDb.dataset.i18nOverride = "true";
        }

        // Auto-fill Collection name (use filename as default collection name)
        const fileName = filePath.split('/').pop().replace('.jsonl', '').replace('.', '_');
        if (els.idxCollection) els.idxCollection.value = fileName;

        if (els.idxMode) {
            els.idxMode.value = lastIndexMode || "new";
        }
        await syncIndexModeUI(els.idxMode ? els.idxMode.value : "new", { forceFetch: true });

        if (els.milvusDialog) els.milvusDialog.showModal();
        return;
    }

    let extraParams = {};
    if (pipelineName === 'corpus_chunk') {
        extraParams = { ...chunkConfigState }; // Spread object to pass
    }

    // Other tasks start directly
    runKBTask(pipelineName, filePath, extraParams);
};

// Confirm index creation (mounted to window)
function normalizeKbName(value) {
    if (value === null || value === undefined) return "";
    let text = String(value).trim();
    try {
        text = text.normalize("NFKC");
    } catch (e) {
        // Fallback for environments without normalize
    }
    return text.toLowerCase();
}

function getCollectionDisplayName(collection) {
    if (!collection) return "";
    return collection.display_name || collection.name || "";
}

function findMatchingCollection(collections, inputName) {
    const normalizedInput = normalizeKbName(inputName);
    if (!normalizedInput || !Array.isArray(collections)) return null;
    return collections.find(c => {
        const displayName = normalizeKbName(c.display_name || c.name);
        const rawName = normalizeKbName(c.name);
        return normalizedInput === displayName || normalizedInput === rawName;
    }) || null;
}

async function loadIndexCollections(options = {}) {
    const { forceFetch = false } = options;
    if (!forceFetch && Array.isArray(indexCollectionsCache) && indexCollectionsCache.length > 0) {
        return indexCollectionsCache;
    }
    try {
        const data = await fetchJSON('/api/kb/files');
        indexCollectionsCache = data.index || [];
        return indexCollectionsCache;
    } catch (e) {
        console.warn("Failed to load collections for index modal", e);
        return null;
    }
}

function renderIndexCollectionSelect(collections, selectedName = "") {
    if (!els.idxCollectionSelect) return;
    const select = els.idxCollectionSelect;
    select.innerHTML = "";

    if (!Array.isArray(collections)) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "Failed to load collections";
        opt.disabled = true;
        opt.selected = true;
        select.appendChild(opt);
        select.disabled = true;
        return;
    }

    if (collections.length === 0) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "No collections found";
        opt.disabled = true;
        opt.selected = true;
        select.appendChild(opt);
        select.disabled = true;
        return;
    }

    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "Select a collection";
    placeholder.disabled = true;
    select.appendChild(placeholder);

    collections.forEach(c => {
        const opt = document.createElement("option");
        const displayName = getCollectionDisplayName(c);
        opt.value = c.name;
        opt.textContent = displayName || c.name;
        if (c.name === selectedName) opt.selected = true;
        select.appendChild(opt);
    });

    if (!selectedName) placeholder.selected = true;
    select.disabled = false;
}

async function syncIndexModeUI(mode, options = {}) {
    const { forceFetch = false, collections = null, selectedName = "" } = options;
    const activeMode = mode || "new";
    lastIndexMode = activeMode;

    const isNew = activeMode === "new";
    if (els.idxCollectionLabel) {
        els.idxCollectionLabel.textContent = isNew ? t("kb_collection_name") : t("kb_existing_collection");
    }
    if (els.idxCollection) {
        els.idxCollection.classList.toggle("d-none", !isNew);
    }
    if (els.idxCollectionSelect) {
        els.idxCollectionSelect.classList.toggle("d-none", isNew);
    }

    if (!isNew) {
        const list = Array.isArray(collections)
            ? collections
            : await loadIndexCollections({ forceFetch });
        renderIndexCollectionSelect(list, selectedName || (els.idxCollectionSelect ? els.idxCollectionSelect.value : ""));
    }
}

function startIndexTask(collectionName, mode) {
    if (els.milvusDialog) els.milvusDialog.close();

    // Start task, pass embedding configuration
    runKBTask('milvus_index', currentTargetFile, {
        collection_name: collectionName,
        index_mode: mode,
        // OpenAI Embedding parameters
        emb_api_key: indexConfigState.api_key,
        emb_base_url: indexConfigState.base_url,
        emb_model_name: indexConfigState.model_name
    });
}

async function confirmIndexMode(mode, collection) {
    const label = getCollectionDisplayName(collection) || collection?.name || "";
    if (mode === "append") {
        return await showConfirm(
            formatTemplate(t("kb_confirm_append"), { label }),
            {
                title: t("kb_confirm_append_title"),
                type: "info",
                confirmText: t("kb_continue"),
                cancelText: t("kb_cancel")
            }
        );
    }
    if (mode === "overwrite") {
        return await showConfirm(
            formatTemplate(t("kb_confirm_overwrite"), { label }),
            {
                title: t("kb_confirm_overwrite_title"),
                type: "warning",
                confirmText: t("kb_mode_overwrite"),
                cancelText: t("kb_cancel"),
                danger: true
            }
        );
    }
    if (mode === "new") {
        return await showConfirm(
            formatTemplate(t("kb_confirm_new"), { label }),
            {
                title: t("kb_confirm_new_title"),
                type: "info",
                confirmText: t("kb_create"),
                cancelText: t("kb_cancel")
            }
        );
    }
    return true;
}

window.confirmIndexTask = async function () {
    if (!els.idxMode) return;
    const mode = els.idxMode.value || "new";

    const collections = await loadIndexCollections({ forceFetch: true });
    if (collections === null) {
        await showModal(t("kb_load_collections_failed"), {
            title: t("kb_load_failed_title"),
            type: "error"
        });
        return;
    }

    if (mode === "new") {
        if (!els.idxCollection) return;
        const inputName = els.idxCollection.value.trim();
        if (!inputName) {
            showModal(t("kb_collection_name_required"), { title: t("kb_validation_error"), type: "warning" });
            return;
        }

        const matched = findMatchingCollection(collections, inputName);
        if (matched) {
            const displayName = getCollectionDisplayName(matched);
            const displayNameSuffix = displayName && displayName !== inputName
                ? (state.uiLanguage === "zh"
                    ? `（别名“${displayName}”）`
                    : ` as "${displayName}"`)
                : "";
            const choice = await showChoice(
                formatTemplate(t("kb_collection_exists"), {
                    inputName,
                    displayName: displayNameSuffix
                }),
                {
                    title: t("kb_name_exists_title"),
                    type: "warning",
                    primaryText: t("kb_mode_append"),
                    secondaryText: t("kb_mode_overwrite"),
                    cancelText: t("kb_cancel"),
                    primaryValue: "append",
                    secondaryValue: "overwrite",
                    dangerSecondary: true
                }
            );

            if (!choice) {
                if (els.idxCollection) els.idxCollection.focus();
                return;
            }

            if (els.idxMode) els.idxMode.value = choice;
            await syncIndexModeUI(choice, { collections, selectedName: matched.name });

            const confirmed = await confirmIndexMode(choice, matched);
            if (!confirmed) return;

            startIndexTask(matched.name, choice);
            return;
        }

        const confirmedNew = await confirmIndexMode("new", { name: inputName, display_name: inputName });
        if (!confirmedNew) return;

        startIndexTask(inputName, "new");
        return;
    }

    if (!els.idxCollectionSelect) return;
    if (collections.length === 0) {
        await showModal(t("kb_no_collections_message"), {
            title: t("kb_no_collections_title"),
            type: "warning"
        });
        return;
    }

    const selectedName = els.idxCollectionSelect.value;
    if (!selectedName) {
        await showModal(t("kb_select_collection_message"), {
            title: t("kb_select_collection_title"),
            type: "warning"
        });
        return;
    }

    const selectedCollection = collections.find(c => c.name === selectedName) || { name: selectedName };
    const confirmed = await confirmIndexMode(mode, selectedCollection);
    if (!confirmed) return;

    startIndexTask(selectedCollection.name, mode);
};

// handleFileUpload restored (no longer needs sessionStems)
window.handleFileUpload = async function (input) {
    if (!input.files.length) return;

    // No need to manually record filenames, Snapshot logic handles it automatically

    const formData = new FormData();
    for (let i = 0; i < input.files.length; i++) {
        formData.append('file', input.files[i]);
    }

    updateKBStatus(true, t('kb_uploading'));
    try {
        const res = await fetch('/api/kb/upload', { method: 'POST', body: formData });
        if (res.ok) {
            await refreshKBFiles();
            updateKBStatus(false);
        } else {
            showModal(t("kb_upload_failed"), { title: t("status_error"), type: "error" });
            updateKBStatus(false);
        }
    } catch (e) {
        console.error(e);
        updateKBStatus(false);
        showModal(t("kb_upload_error") + e.message, { title: t("status_error"), type: "error" });
    } finally {
        input.value = '';
    }
};

// Delete file (mounted to window)
window.deleteKBFile = async function (category, filename, displayName = "") {
    const action = category === 'collection' ? t('kb_delete_collection_action') : t('kb_delete_file_action');
    const label = category === 'collection' ? (displayName || filename) : filename;
    const confirmed = await showConfirm(formatTemplate(t("kb_delete_confirm_message"), { action, name: label }), {
        title: t("kb_delete_confirm_title"),
        type: "warning",
        confirmText: t("kb_delete"),
        danger: true
    });
    if (!confirmed) return;

    try {
        const res = await fetch(`/api/kb/files/${category}/${filename}`, { method: 'DELETE' });
        if (res.ok) {
            refreshKBFiles();
        } else {
            const err = await res.json();
            showModal(t("kb_delete_failed") + (err.error || res.statusText), { title: t("status_error"), type: "error" });
        }
    } catch (e) {
        console.error(e);
    }
};



// Core: submit task and poll
async function runKBTask(pipelineName, filePath, extraParams = {}) {
    updateKBStatus(true, formatTemplate(t("kb_running_task"), { task: pipelineName }));

    try {
        // A. Submit task
        const res = await fetch('/api/kb/run', {
            method: 'POST',
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                pipeline_name: pipelineName,
                target_file: filePath,
                ...extraParams
            })
        });

        const data = await res.json();

        if (res.status === 202) {
            // B. Start polling
            pollTaskStatus(data.task_id);
        } else {
            throw new Error(data.error || t("kb_task_start_failed"));
        }
    } catch (e) {
        showModal(e.message, { title: t("kb_task_error_title"), type: "error" });
        updateKBStatus(false);
    }
}

// Progress ring helper
function setProgress(percent) {
    if (!els.taskProgressCircle || !els.taskProgressText) return;
    const radius = 9;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (percent / 100) * circumference;
    els.taskProgressCircle.style.strokeDashoffset = offset;
    els.taskProgressText.textContent = Math.round(percent) + '%';
}

// Polling logic with simulated progress
function pollTaskStatus(taskId) {
    let progress = 0;
    let pollCount = 0;
    
    // Show progress ring, hide spinner
    if (els.taskSpinner) els.taskSpinner.classList.add('hidden');
    if (els.taskProgressWrapper) els.taskProgressWrapper.classList.remove('hidden');
    setProgress(0);
    
    const interval = setInterval(async () => {
        try {
            pollCount++;
            
            // Simulated progress: fast at start, slow down as it approaches 95%
            // Uses logarithmic curve for natural feel
            if (progress < 95) {
                const increment = Math.max(0.5, (95 - progress) * 0.08);
                progress = Math.min(95, progress + increment);
                setProgress(progress);
            }
            
            const res = await fetch(`/api/kb/status/${taskId}`);
            const task = await res.json();

            if (task.status === 'success') {
                clearInterval(interval);
                setProgress(100);
                setTimeout(() => {
                    updateKBStatus(false);
                    // 1. Refresh KB management interface list
                    refreshKBFiles();

                    // 2. Also refresh Chat interface dropdown menu
                    // So that after indexing is complete, it can be selected immediately in the chat box
                    renderChatCollectionOptions();
                    console.log('Task Result:', task.result);
                }, 400);
            } else if (task.status === 'failed') {
                clearInterval(interval);
                updateKBStatus(false);
                showModal(
                    formatTemplate(t("kb_task_failed_message"), { error: task.error }),
                    { title: t("kb_task_failed_title"), type: "error" }
                );
            } else {
                // still running...
                console.log('Task running, progress:', Math.round(progress) + '%');
            }
        } catch (e) {
            clearInterval(interval);
            updateKBStatus(false);
        }
    }, 800); // Poll every 800ms for smoother progress
}

// UI utility: show/hide status bar
function updateKBStatus(show, msg = '') {
    if (!els.taskStatusBar || !els.taskMsg) return;
    if (show) {
        els.taskStatusBar.classList.remove('hidden');
        els.taskMsg.textContent = msg;
        // Reset to spinner mode when starting
        if (els.taskSpinner) els.taskSpinner.classList.remove('hidden');
        if (els.taskProgressWrapper) els.taskProgressWrapper.classList.add('hidden');
    } else {
        els.taskStatusBar.classList.add('hidden');
        // Reset progress state
        if (els.taskProgressWrapper) els.taskProgressWrapper.classList.add('hidden');
        if (els.taskSpinner) els.taskSpinner.classList.remove('hidden');
        setProgress(0);
    }
}

// ==========================================
// --- End Knowledge Base Logic ---
// ==========================================

// --- Utilities ---
function uuidv4() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
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

function expandConsole() {
    if (els.canvasConsole) {
        els.canvasConsole.classList.remove("collapsed");
    }
}

function setBuildButtonState(state = "idle", label = "") {
    if (!els.buildPipeline) return;
    if (!buildBtnDefaultHtml) buildBtnDefaultHtml = els.buildPipeline.innerHTML;

    const reset = () => {
        els.buildPipeline.disabled = false;
        els.buildPipeline.innerHTML = buildBtnDefaultHtml;
    };

    if (state === "running") {
        els.buildPipeline.disabled = true;
        els.buildPipeline.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>${label || t("builder_build_running")}`;
        return;
    }

    if (state === "success") {
        els.buildPipeline.disabled = false;
        els.buildPipeline.innerHTML = `<span class="text-success me-1">✓</span>${label || t("builder_build_success")}`;
        setTimeout(reset, 1200);
        return;
    }

    if (state === "error") {
        els.buildPipeline.disabled = false;
        els.buildPipeline.innerHTML = `<span class="text-danger me-1">⚠</span>${label || t("builder_build_failed")}`;
        setTimeout(reset, 1800);
        return;
    }

    reset();
}

function markUnsavedChanges() {
    state.unsavedChanges = true;
    setYamlSyncStatus('modified');
}

function clearUnsavedChanges() {
    state.unsavedChanges = false;
    setYamlSyncStatus('synced');
}

function snapshotSavedYaml(content = "") {
    state.lastSavedYaml = (content || "").trim();
    clearUnsavedChanges();
    showYamlError(null);
    setYamlSyncStatus('synced');
}

function updateUnsavedFromEditor() {
    if (!els.yamlEditor) return;
    const current = (els.yamlEditor.value || "").trim();
    const dirty = current !== state.lastSavedYaml;
    state.unsavedChanges = dirty;
    if (dirty) {
        state.isBuilt = false;
        state.parametersReady = false;
    }
}

async function confirmUnsavedChanges(actionLabel = t("builder_action_continue")) {
    if (!state.unsavedChanges) return true;

    const message = formatTemplate(t("builder_unsaved_changes_message"), { action: actionLabel });
    const confirmed = await showConfirm(
        message,
        {
            title: t("builder_unsaved_changes_title"),
            type: "warning",
            confirmText: t("common_continue"),
            cancelText: t("common_cancel")
        }
    );

    return confirmed;
}

// Markdown rendering (with simple fallback)
const MARKDOWN_LANGS = ["markdown", "md", "mdx"];

// Citation highlighting function moved to formatCitationHtml(html, messageIdx) definition later

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

function protectMath(text) {
    const mathBlocks = [];
    // Match: code blocks (skip), $$...$$, \[...\], \(...\), $...$
    const regex = /(```[\s\S]*?```|`[^`\n]+`)|(\$\$[\s\S]*?\$\$)|(\\\[[\s\S]*?\\\])|(\\\([\s\S]*?\\\))|(\$[^\$\n]+\$)/g;

    const protectedText = text.replace(regex, (match, code) => {
        if (code) return code; // Keep code blocks as-is
        mathBlocks.push(match);
        return `MATHPLACEHOLDER${mathBlocks.length - 1}END`;
    });
    return { text: protectedText, mathBlocks };
}

function restoreMath(html, mathBlocks) {
    if (!mathBlocks.length) return html;

    // Simply restore the original LaTeX with delimiters
    return html.replace(/MATHPLACEHOLDER(\d+)END/g, (match, id) => {
        return mathBlocks[parseInt(id, 10)] || match;
    });
}

function renderLatex(element) {
    if (!element || !window.renderMathInElement) {
        console.warn("KaTeX renderMathInElement not available");
        return;
    }
    try {
        window.renderMathInElement(element, {
            delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false },
                { left: '\\(', right: '\\)', display: false },
                { left: '\\[', right: '\\]', display: true }
            ],
            throwOnError: false,
            ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
        });
    } catch (e) {
        console.error("KaTeX rendering error:", e);
    }
}

// ChatGPT-style code block rendering
function renderCodeBlock(code, lang) {
    const escapedCode = escapeHtml(code);
    const langClass = lang ? `language-${lang.toLowerCase()}` : '';

    // Generate unique ID for copy functionality
    const blockId = 'code-' + Math.random().toString(36).substr(2, 9);

    return `<div class="code-block-wrapper">
    <button class="code-block-copy" data-code-id="${blockId}" onclick="copyCodeBlock(this)" title="Copy code">
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
      <span class="copy-text">Copy</span>
    </button>
    <pre><code id="${blockId}" class="${langClass}">${escapedCode}</code></pre>
  </div>`;
}

// Code block copy functionality
function copyCodeBlock(btn) {
    const codeId = btn.dataset.codeId;
    const codeEl = document.getElementById(codeId);
    if (!codeEl) return;

    const code = codeEl.textContent;
    navigator.clipboard.writeText(code).then(() => {
        btn.classList.add('copied');
        const textSpan = btn.querySelector('.copy-text');
        if (textSpan) textSpan.textContent = 'Copied!';

        setTimeout(() => {
            btn.classList.remove('copied');
            if (textSpan) textSpan.textContent = 'Copy';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
    });
}

// Apply code highlighting
function applyCodeHighlight(container) {
    if (!window.hljs) return;
    const codeBlocks = container.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        // Only apply highlighting to code blocks with language class or unhighlighted code blocks
        if (!block.classList.contains('hljs')) {
            window.hljs.highlightElement(block);
        }
    });
}

// Table beautification and copy button
function applyTableEnhancements(container) {
    if (!container) return;
    const tables = container.querySelectorAll('table');
    tables.forEach((table) => {
        if (table.closest('.table-block-wrapper')) return;

        const wrapper = document.createElement('div');
        wrapper.className = 'table-block-wrapper';

        const scroll = document.createElement('div');
        scroll.className = 'table-scroll';

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'table-copy-btn';
        btn.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
      <span class="table-copy-text">Copy Table</span>
    `;
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            copyTableToClipboard(table, btn);
        });

        const parent = table.parentNode;
        if (!parent) return;
        parent.insertBefore(wrapper, table);
        wrapper.appendChild(btn);
        wrapper.appendChild(scroll);
        scroll.appendChild(table);
    });
}

function normalizeTableCellText(text) {
    return text.replace(/\s+/g, ' ').trim();
}

function escapeMarkdownCell(text) {
    return text.replace(/\|/g, '\\|');
}

function tableToMarkdown(table) {
    const thead = table.querySelector('thead');
    const tbody = table.querySelector('tbody');
    const allRows = Array.from(table.rows);

    let headerCells = [];
    let bodyRows = [];

    if (thead) {
        const headRow = thead.querySelector('tr');
        if (headRow) {
            headerCells = Array.from(headRow.cells).map(cell =>
                escapeMarkdownCell(normalizeTableCellText(cell.innerText || ''))
            );
        }
        bodyRows = tbody ? Array.from(tbody.rows) : allRows.slice(1);
    } else {
        const firstRow = allRows[0];
        if (firstRow) {
            headerCells = Array.from(firstRow.cells).map(cell =>
                escapeMarkdownCell(normalizeTableCellText(cell.innerText || ''))
            );
            bodyRows = allRows.slice(1);
        }
    }

    if (!headerCells.length) return '';

    const headerLine = `| ${headerCells.join(' | ')} |`;
    const separatorLine = `| ${headerCells.map(() => '---').join(' | ')} |`;
    const bodyLines = bodyRows.map(row => {
        const cells = Array.from(row.cells).map(cell =>
            escapeMarkdownCell(normalizeTableCellText(cell.innerText || ''))
        );
        return `| ${cells.join(' | ')} |`;
    });

    return [headerLine, separatorLine, ...bodyLines].filter(Boolean).join('\n');
}

function copyTableToClipboard(table, btn) {
    if (!table) return;
    const text = tableToMarkdown(table);
    if (!text) return;

    const done = () => {
        btn.classList.add('copied');
        const textSpan = btn.querySelector('.table-copy-text');
        if (textSpan) textSpan.textContent = 'Copied!';
        setTimeout(() => {
            btn.classList.remove('copied');
            if (textSpan) textSpan.textContent = 'Copy Table';
        }, 2000);
    };

    if (navigator.clipboard?.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(() => fallbackCopy(text, done));
    } else {
        fallbackCopy(text, done);
    }
}

function fallbackCopy(text, onDone) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.top = '-1000px';
    textarea.style.left = '-1000px';
    document.body.appendChild(textarea);
    textarea.select();
    try {
        document.execCommand('copy');
        if (typeof onDone === 'function') onDone();
    } catch (e) {
        console.error('Copy failed:', e);
    }
    document.body.removeChild(textarea);
}

// Chat text copy button (placed after body text, before citations)
function ensureChatCopyRow(bubble, rawText) {
    if (!bubble) return;
    let row = bubble.querySelector('.chat-copy-row');
    const refContainer = bubble.querySelector('.reference-container');

    if (!row) {
        row = document.createElement('div');
        row.className = 'chat-copy-row';

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'chat-copy-btn';
        btn.title = 'Copy Text';
        btn.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
    `;
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            copyChatOriginalText(btn);
        });

        row.appendChild(btn);
        if (refContainer) {
            bubble.insertBefore(row, refContainer);
        } else {
            bubble.appendChild(row);
        }
    }

    const btn = row.querySelector('.chat-copy-btn');
    if (btn) btn.dataset.rawText = rawText || '';
}

function copyChatOriginalText(btn) {
    const text = btn?.dataset?.rawText || '';
    if (!text) return;
    const done = () => {
        btn.classList.add('copied');
        setTimeout(() => btn.classList.remove('copied'), 2000);
    };
    if (navigator.clipboard?.writeText) {
        navigator.clipboard.writeText(text).then(done).catch(() => fallbackCopy(text, done));
    } else {
        fallbackCopy(text, done);
    }
}

// Mount copyCodeBlock to window for onclick calls
window.copyCodeBlock = copyCodeBlock;

function renderMarkdown(text, { allowCodeBlock = true, unwrapLanguages = [] } = {}) {
    if (!text) return "";
    if (unwrapLanguages.length) {
        text = unwrapLanguageBlocks(text, unwrapLanguages);
    }
    if (window.marked) {
        // Create new renderer on each render to ensure code blocks use custom rendering
        const renderer = new window.marked.Renderer();
        renderer.code = function (codeObj, lang) {
            // marked v5+ passes object format { text, lang, escaped }
            let codeText, codeLang;
            if (typeof codeObj === 'object' && codeObj !== null) {
                codeText = codeObj.text || '';
                codeLang = codeObj.lang || '';
            } else {
                codeText = codeObj || '';
                codeLang = lang || '';
            }

            if (!allowCodeBlock) {
                return `<p>${escapeHtml(codeText).replace(/\n/g, "<br>")}</p>`;
            }
            return renderCodeBlock(codeText, codeLang);
        };

        // Protect Math from Markdown processing
        const { text: protectedText, mathBlocks } = protectMath(text);

        const rawHtml = window.marked.parse(protectedText, { breaks: true, gfm: true, renderer: renderer });
        let sanitized = window.DOMPurify ? DOMPurify.sanitize(rawHtml, {
            ADD_TAGS: ['button', 'svg', 'path', 'rect', 'line', 'polyline', 'circle', 'polygon'],
            ADD_ATTR: ['onclick', 'data-code-id', 'viewBox', 'fill', 'stroke', 'stroke-width', 'stroke-linecap', 'stroke-linejoin', 'd', 'x', 'y', 'width', 'height', 'rx', 'ry', 'cx', 'cy', 'r', 'x1', 'y1', 'x2', 'y2', 'points']
        }) : rawHtml;

        // Restore original LaTeX (with $$ delimiters intact)
        sanitized = restoreMath(sanitized, mathBlocks);

        return sanitized;
    }
    return basicMarkdown(text, { allowCodeBlock });
}

// --- Lifecycle ---

/**
 * Show input modal
 * @param {string} message - Prompt message
 * @param {object} options - Configuration options
 * @returns {Promise<string|null>} User input value, returns null if cancelled
 */
function showPrompt(message, options = {}) {
    const {
        title = t("modal_input_title"),
        placeholder = "",
        defaultValue = "",
        confirmText = t("common_ok"),
        cancelText = t("common_cancel")
    } = options;

    return new Promise((resolve) => {
        const modal = document.getElementById("unified-modal");
        const iconEl = document.getElementById("unified-modal-icon");
        const titleEl = document.getElementById("unified-modal-title");
        const messageEl = document.getElementById("unified-modal-message");
        const actionsEl = document.getElementById("unified-modal-actions");

        if (!modal) {
            const result = window.prompt(message, defaultValue);
            resolve(result);
            return;
        }

        iconEl.className = "unified-modal-icon info";
        iconEl.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>';

        titleEl.textContent = title;
        messageEl.innerHTML = `
      <p style="margin-bottom: 12px;">${message}</p>
      <input type="text" id="prompt-input" class="form-control" placeholder="${placeholder}" value="${defaultValue}" style="width: 100%; padding: 10px 12px; border: 1px solid #e5e7eb; border-radius: 8px; font-size: 0.95rem;">
    `;

        actionsEl.innerHTML = `
      <button class="btn unified-modal-btn unified-modal-btn-secondary" id="prompt-cancel">${cancelText}</button>
      <button class="btn unified-modal-btn unified-modal-btn-primary" id="prompt-confirm">${confirmText}</button>
    `;

        modal.showModal();

        const inputEl = document.getElementById("prompt-input");
        inputEl.focus();
        inputEl.select();

        const handleConfirm = () => {
            const value = inputEl.value.trim();
            modal.close();
            resolve(value || null);
        };

        const handleCancel = () => {
            modal.close();
            resolve(null);
        };

        document.getElementById("prompt-confirm").onclick = handleConfirm;
        document.getElementById("prompt-cancel").onclick = handleCancel;

        inputEl.onkeydown = (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                handleConfirm();
            } else if (e.key === "Escape") {
                handleCancel();
            }
        };
    });
}

async function createNewPipeline() {
    // If there are unsaved changes, confirm first
    if (state.unsavedChanges) {
        const confirmed = await confirmUnsavedChanges(t("builder_action_create_pipeline"));
        if (!confirmed) return;
    }

    // Show input dialog for user to enter Pipeline name
    const pipelineName = await showPrompt(t("builder_new_pipeline_prompt"), {
        title: t("builder_new_pipeline"),
        placeholder: "my_pipeline",
        confirmText: t("common_create")
    });

    if (!pipelineName) return; // User cancelled

    // Validate name format
    if (!/^[a-zA-Z_][a-zA-Z0-9_-]*$/.test(pipelineName)) {
        await showModal(t("builder_invalid_pipeline_name_full"), {
            title: t("builder_invalid_name_title"),
            type: "error"
        });
        return;
    }

    // Create new pipeline
    state.selectedPipeline = pipelineName;
    state.parameterData = null;
    state.pipelineConfig = cloneDeep(VANILLA_PIPELINE_TEMPLATE);
    state.isBuilt = false;
    state.parametersReady = false;

    els.name.value = pipelineName;
    if (els.pipelineDropdownBtn) els.pipelineDropdownBtn.textContent = pipelineName;
    setHeroPipelineLabel(pipelineName);
    setHeroStatusLabel("idle");
    updateAIContextBanner('pipeline-new');

    setSteps(VANILLA_PIPELINE_STEPS, { markUnsaved: true });
    setYamlSyncStatus('modified');
    showYamlError(null);
    setMode(Modes.BUILDER);
    updateActionButtons();

    log(`Created new pipeline: ${pipelineName}`);
}

// --- Demo Session / Engine Control ---

// Suspend all engines except target Pipeline to avoid port/resource conflicts
async function suspendOtherEngines(targetPipeline = null) {
    const entries = Object.entries(state.chat.activeEngines || {});
    if (!entries.length) return;

    const stopPromises = [];
    for (const [name, sid] of entries) {
        if (targetPipeline && name === targetPipeline) continue;

        stopPromises.push(
            fetchJSON(`/api/pipelines/demo/stop`, {
                method: "POST",
                body: JSON.stringify({ session_id: sid })
            }).catch(err => console.warn(`Suspend engine ${name} failed`, err))
        );

        if (state.chat.engineSessionId === sid) {
            state.chat.engineSessionId = null;
        }
        delete state.chat.activeEngines[name];
    }

    if (stopPromises.length) {
        setChatStatus("Suspending...", "warn");
        await Promise.allSettled(stopPromises);
    }
    persistActiveEngines();
}

// 1. Start engine (idempotent operation: ignore if already started)
async function startEngine(pipelineName) {
    if (!pipelineName) return;
    if (state.chat.engineStartPromise) {
        await state.chat.engineStartPromise;
    }

    const startPromise = (async () => {
        state.chat.engineStartingFor = pipelineName;

        // If current session belongs to another Pipeline, stop it first to ensure state consistency
        const currentPipeline = Object.entries(state.chat.activeEngines || {})
            .find(([, sid]) => sid === state.chat.engineSessionId)?.[0];
        if (state.chat.engineSessionId && currentPipeline && currentPipeline !== pipelineName) {
            await stopEngine({ allowDuringStart: true });
        }

        // Suspend engines of other Pipelines to prevent port conflicts
        await suspendOtherEngines(pipelineName);

        const existingSid = state.chat.activeEngines[pipelineName];

        // If target Pipeline already has active engine, reuse it and sync UI
        if (state.chat.engineSessionId && existingSid === state.chat.engineSessionId) {
            setChatStatus("Ready", "ready");
            updateDemoControls();
            return;
        }
        if (!state.chat.engineSessionId && existingSid) {
            state.chat.engineSessionId = existingSid;
            setChatStatus("Ready", "ready");
            updateDemoControls();
            return;
        }

        state.chat.demoLoading = true;
        updateDemoControls(); // Update UI to show "Loading..."
        setChatStatus("Initializing...", "warn");

        try {
            const newSid = uuidv4();

            // Call backend to start
            await fetchJSON(`/api/pipelines/${encodeURIComponent(pipelineName)}/demo/start`, {
                method: "POST", body: JSON.stringify({ session_id: newSid })
            });

            state.chat.engineSessionId = newSid;
            state.chat.activeEngines[pipelineName] = newSid;
            persistActiveEngines();

            setChatStatus("Ready", "ready");
            log(`Engine started for ${pipelineName}`);

        } catch (err) {
            console.error(err);
            setChatStatus("Engine Error", "error");
            state.chat.engineSessionId = null;
            const msg = err?.message ? String(err.message) : t("common_unknown_error");
            showModal(formatTemplate(t("chat_engine_init_failed_message"), { error: msg }), {
                title: t("chat_engine_error_title"),
                type: "error"
            });
        } finally {
            state.chat.demoLoading = false;
            updateDemoControls();
        }
    })();

    state.chat.engineStartPromise = startPromise;
    try {
        await startPromise;
    } finally {
        if (state.chat.engineStartPromise === startPromise) {
            state.chat.engineStartPromise = null;
        }
        if (state.chat.engineStartingFor === pipelineName) {
            state.chat.engineStartingFor = null;
        }
    }
}

// 2. Stop engine
async function stopEngine(options = {}) {
    const { allowDuringStart = false } = options;
    if (!allowDuringStart && state.chat.engineStartPromise) {
        await state.chat.engineStartPromise;
    }
    if (!state.chat.engineSessionId) return;

    const sid = state.chat.engineSessionId;
    const currentName = Object.keys(state.chat.activeEngines).find(key => state.chat.activeEngines[key] === sid);

    try {
        await fetchJSON(`/api/pipelines/demo/stop`, {
            method: "POST", body: JSON.stringify({ session_id: sid })
        });
    } catch (e) {
        console.warn("Stop engine failed (maybe already stopped)", e);
    }

    state.chat.engineSessionId = null;
    if (currentName) delete state.chat.activeEngines[currentName];
    persistActiveEngines();

    setChatStatus("Offline", "info");
    updateDemoControls();
}

// 4. Verify/restore engine (auto-reconnect)
async function ensureEngineReady(pipelineName, options = {}) {
    const { forceRestart = false } = options;
    if (!pipelineName) return false;

    // If there's a cached session id, sync to state first
    const cachedSid = state.chat.activeEngines?.[pipelineName];
    if (!state.chat.engineSessionId && cachedSid) {
        state.chat.engineSessionId = cachedSid;
    }

    // Try to verify existing session
    if (!forceRestart && state.chat.engineSessionId) {
        const ok = await verifyEngineSession(state.chat.engineSessionId);
        if (ok) {
            setChatStatus("Ready", "ready");
            updateDemoControls();
            return true;
        }

        // Current session has expired, clear cache
        const currentName = Object.keys(state.chat.activeEngines || {}).find(
            key => state.chat.activeEngines[key] === state.chat.engineSessionId
        );
        if (currentName) delete state.chat.activeEngines[currentName];
        state.chat.engineSessionId = null;
        persistActiveEngines();
        setChatStatus("Reconnecting...", "warn");
    }

    await startEngine(pipelineName);
    return Boolean(state.chat.engineSessionId);
}

async function verifyEngineSession(sessionId) {
    if (!sessionId) return false;
    try {
        const res = await fetch(`/api/pipelines/chat/history?session_id=${encodeURIComponent(sessionId)}`);
        if (res.ok) return true;
    } catch (e) {
        console.warn("Engine session verification failed:", e);
    }
    return false;
}

// 3. Original toggle function kept for compatibility, or can be removed
// (Kept here to prevent onclick errors in HTML, but we don't actually click it anymore)
async function toggleDemoSession() {
    if (state.chat.engineSessionId) await stopEngine();
    else await startEngine(state.selectedPipeline);
}

function updateDemoControls() {
    // 1. Always enable input box (as long as not loading)
    // We want users to be able to input at any time, if engine is not ready, prompt or auto-wait when clicking send
    if (els.chatInput) els.chatInput.disabled = state.chat.demoLoading;
    if (els.chatSend) els.chatSend.disabled = state.chat.demoLoading;

    // 2. Hide or change button text
    if (els.demoToggleBtn) {
        if (state.chat.demoLoading) {
            els.demoToggleBtn.innerHTML = "Connecting...";
            els.demoToggleBtn.disabled = true;
            els.demoToggleBtn.classList.remove("d-none"); // Show during loading
        } else {
            // Hide button after loading completes, since it's automatic and doesn't need user click
            els.demoToggleBtn.classList.add("d-none");
        }
    }
}

// --- Chat Logic (Updated with Streaming) ---

// Render Collection dropdown options for chat interface
async function renderChatCollectionOptions() {
    if (!els.chatCollectionSelect) return;

    // Save currently selected value to avoid reset after refresh
    const currentVal = els.chatCollectionSelect.value;

    try {
        // Reuse backend list API
        const data = await fetchJSON('/api/kb/files');
        let collections = data.index || []; // data.index contains collection list

        // Sort by display name to keep chat dropdown ordered
        const getLabel = c => (c.display_name || c.name || "").toLowerCase();
        collections = collections.slice().sort((a, b) => getLabel(a).localeCompare(getLabel(b)));

        // Render new custom dropdown menu
        renderKbDropdownOptions(collections);

        // Try to restore selected state
        if (currentVal) {
            const exists = collections.find(c => c.name === currentVal);
            if (exists) {
                els.chatCollectionSelect.value = currentVal;
                // Update dropdown menu selected state
                const menu = document.getElementById('kb-dropdown-menu');
                if (menu) {
                    menu.querySelectorAll('.kb-dropdown-item').forEach(item => {
                        item.classList.remove('selected');
                        if (item.dataset.value === currentVal) {
                            item.classList.add('selected');
                        }
                    });
                }
            }
        }

        // Manually trigger visual update after rendering options
        if (window.updateKbLabel && els.chatCollectionSelect) {
            window.updateKbLabel(els.chatCollectionSelect);
        }

    } catch (e) {
        console.error("Failed to load collections for chat:", e);
    }
}

// Switch to knowledge base view
function openKBView() {
    if (!els.chatMainView || !els.kbMainView) return;

    // If generating, show confirmation dialog
    if (state.chat.running) {
        showInterruptConfirmDialog(() => {
            interruptAndOpenKB();
        });
        return;
    }

    doOpenKBView();
}

// Actually execute opening KB view
function doOpenKBView() {
    // Refresh data
    refreshKBFiles();

    // Hide chat, show knowledge base
    els.chatMainView.classList.add("d-none");
    els.kbMainView.classList.remove("d-none");

    // Update button state
    if (els.kbBtn) els.kbBtn.classList.add("active");

    // Remove highlight from all Session list items
    const items = document.querySelectorAll(".chat-session-item");
    items.forEach(el => el.classList.remove("active"));
}

// Interrupt generation and open KB
function interruptAndOpenKB() {
    if (state.chat.controller) {
        state.chat.controller.abort();
        state.chat.controller = null;
    }
    setChatRunning(false);
    saveCurrentSession(true);
    doOpenKBView();
}

// Switch back to chat view (reset)
function backToChatView() {
    if (!els.chatMainView || !els.kbMainView) return;

    els.kbMainView.classList.add("d-none");
    els.chatMainView.classList.remove("d-none");

    if (els.kbBtn) els.kbBtn.classList.remove("active");

    // Re-render sidebar to restore current session highlight state
    renderChatCollectionOptions();
    renderChatSidebar();
}

// Render Pipeline list to dropdown menu
async function renderChatPipelineMenu() {
    if (!els.chatPipelineMenu) return;

    // Get latest list
    const pipelines = await fetchJSON("/api/pipelines");

    els.chatPipelineMenu.innerHTML = "";

    // Filter out Pipelines that haven't been Built (no parameter file), and sort by name
    const readyPipelines = pipelines
        .filter(p => p.is_ready)
        .sort((a, b) => (a.name || "").localeCompare(b.name || "", "en", { sensitivity: "base" }));

    // If no Pipeline is currently selected, auto-select and load the first available one
    if (!state.selectedPipeline && readyPipelines.length > 0) {
        const defaultName = readyPipelines[0].name;
        try {
            await loadPipeline(defaultName, { ignoreUnsaved: true });
            state.selectedPipeline = defaultName;
        } catch (e) {
            console.warn("Auto-select pipeline failed:", e);
        }
    }

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

    // Update top label
    if (els.chatPipelineLabel) {
        const fallbackName = els.chatPipelineLabel.dataset.currentName;
        const name = state.selectedPipeline || fallbackName;
        if (name) {
            els.chatPipelineLabel.textContent = name;
            els.chatPipelineLabel.dataset.selected = "true";
            els.chatPipelineLabel.dataset.currentName = name;
        } else {
            els.chatPipelineLabel.textContent = t("select_pipeline");
            els.chatPipelineLabel.dataset.selected = "false";
        }
    }
}

// Switch Pipeline (core logic)
async function switchChatPipeline(name) {
    if (name === state.selectedPipeline) return;
    if (state.chat.running) {
        showModal(t("chat_wait_for_response_message"), { title: t("chat_wait_for_response_title"), type: "info" });
        return;
    }

    if (state.chat.engineStartPromise) {
        await state.chat.engineStartPromise;
    }

    // 1. Save old session
    saveCurrentSession(true);

    // 2. Stop old engine (to save resources, stop the previous one when switching)
    // If you want to keep it running in background, remove this line, but recommended to stop to prevent port conflicts or resource usage
    await stopEngine();

    // 3. Load new Pipeline structure
    await loadPipeline(name);

    // 4. Load parameters
    try {
        state.parameterData = await fetchJSON(`/api/pipelines/${encodeURIComponent(name)}/parameters`);
        state.parametersReady = true;
    } catch (e) {
        console.warn(t("builder_params_not_found"));
        state.parametersReady = false;
    }

    // 5. Create new session UI
    createNewChatSession();
    renderChatPipelineMenu();

    // 6. Automatically start new engine!
    if (state.parametersReady) {
        await startEngine(name);
    } else {
        setChatStatus("Params Missing", "error");
    }
}

function resetChatSession() {
    state.chat.history = [];
    state.chat.running = false;
    state.chat.currentSessionId = null;

    // Critical bug fix: Remove the line below! Switching Pipeline should not clear all session list!
    // state.chat.sessions = [];  <-- Remove it

    // Should only clear currently displayed session, not all session data
    // But for UI logic simplicity, do nothing here, renderChatSidebar will handle display automatically

    // Check activeEngines: restore Session ID
    const currentName = state.selectedPipeline;
    if (currentName && state.chat.activeEngines[currentName]) {
        state.chat.engineSessionId = state.chat.activeEngines[currentName];
        log(`Resumed active engine for ${currentName}`);
    } else {
        state.chat.engineSessionId = null;
    }

    renderChatHistory();
    renderChatSidebar();

    // Update status display
    if (state.chat.engineSessionId) {
        setChatStatus("Engine Ready", "ready");
    } else {
        setChatStatus("Engine Offline", "info");
    }

    updateDemoControls();
}

function generateChatId() { return Date.now().toString(36) + Math.random().toString(36).substr(2); }

function createNewChatSession() {
    // If generating, show confirmation dialog
    if (state.chat.running) {
        showInterruptConfirmDialog(() => {
            interruptAndCreateNewChat();
        });
        return;
    }

    if (state.chat.history.length > 0) {
        saveCurrentSession(true);
    }

    state.chat.currentSessionId = generateChatId();
    state.chat.history = [];
    renderChatHistory(); renderChatSidebar();
    updateChatIdleStatus();
    if (els.chatInput && state.chat.engineSessionId) els.chatInput.focus();
    backToChatView();
}

// Interrupt generation and create new session
function interruptAndCreateNewChat() {
    if (state.chat.controller) {
        state.chat.controller.abort();
        state.chat.controller = null;
    }
    setChatRunning(false);
    saveCurrentSession(true);

    state.chat.currentSessionId = generateChatId();
    state.chat.history = [];
    renderChatHistory(); renderChatSidebar();
    updateChatIdleStatus();
    if (els.chatInput && state.chat.engineSessionId) els.chatInput.focus();
    backToChatView();
}

function loadChatSession(sessionId) {
    // If generating, show confirmation dialog
    if (state.chat.running) {
        showInterruptConfirmDialog(() => {
            interruptAndLoadSession(sessionId);
        });
        return;
    }

    // Save current ongoing session first
    saveCurrentSession(false);

    const session = state.chat.sessions.find(s => s.id === sessionId);
    if (!session) return;

    state.chat.currentSessionId = session.id;
    state.chat.history = cloneDeep(session.messages || []); // Deep copy restore

    // Optional: If you want to make it more advanced:
    // If this historical session belongs to another Pipeline (session.pipeline),
    // You can automatically call switchChatPipeline(session.pipeline) here to switch context.
    // But for simplicity, just load text content for now.

    renderChatHistory();
    renderChatSidebar();
    updateChatIdleStatus();

    // Mobile adaptation: auto-collapse sidebar after loading
    const sidebar = document.querySelector('.chat-sidebar');
    if (window.innerWidth < 768 && sidebar) {
        sidebar.classList.remove('show');
    }

    backToChatView();
}

// Interrupt generation and load session
function interruptAndLoadSession(sessionId) {
    if (state.chat.controller) {
        state.chat.controller.abort();
        state.chat.controller = null;
    }
    setChatRunning(false);
    saveCurrentSession(true);

    const session = state.chat.sessions.find(s => s.id === sessionId);
    if (!session) return;

    state.chat.currentSessionId = session.id;
    state.chat.history = cloneDeep(session.messages || []);

    renderChatHistory();
    renderChatSidebar();
    setChatStatus("Ready", "ready");

    const sidebar = document.querySelector('.chat-sidebar');
    if (window.innerWidth < 768 && sidebar) {
        sidebar.classList.remove('show');
    }
    backToChatView();
}

// Check if session content has changed (to avoid "update" and sorting when clicking session)
function hasHistoryChanged(session, history) {
    if (!session) return true;
    const prev = Array.isArray(session.messages) ? session.messages : [];
    const curr = Array.isArray(history) ? history : [];
    if (prev.length !== curr.length) return true;
    if (prev.length === 0) return false;
    const lastPrev = prev[prev.length - 1];
    const lastCurr = curr[curr.length - 1];
    // Only do shallow comparison of last message, performance-friendly and sufficient to identify newly generated content
    return JSON.stringify(lastPrev) !== JSON.stringify(lastCurr);
}

// Show interrupt confirmation dialog
async function showInterruptConfirmDialog(onConfirm) {
    const confirmed = await showConfirm(
        t("chat_interrupt_message"),
        {
            title: t("chat_interrupt_title"),
            type: "warning",
            confirmText: t("chat_interrupt_confirm"),
            danger: true
        }
    );
    if (confirmed && onConfirm) {
        onConfirm();
    }
}

function saveCurrentSession(force = false) {
    if (!state.chat.currentSessionId) return;

    // If no valid messages (including empty text "new" sessions), don't save to history
    const hasContent = state.chat.history.some(m => {
        if (!m) return false;
        if (typeof m.text === "string" && m.text.trim() !== "") return true;
        if (m.meta && Object.keys(m.meta).length > 0) return true;
        return false;
    });
    if (!hasContent) {
        state.chat.sessions = state.chat.sessions.filter(s => s.id !== state.chat.currentSessionId);
        renderChatSidebar();
        localStorage.setItem("ultrarag_sessions", JSON.stringify(state.chat.sessions));
        return;
    }

    let session = state.chat.sessions.find(s => s.id === state.chat.currentSessionId);

    // Only update timestamp / sorting when there's new content (or forced), avoid "click only" causing top placement
    const contentChanged = force || hasHistoryChanged(session, state.chat.history);
    if (!contentChanged) {
        if (state.chat.currentSessionId) {
            localStorage.setItem("ultrarag_last_active_id", state.chat.currentSessionId);
        }
        return;
    }

    // Generate title: take first 20 characters of first user message
    let title = "New Chat";
    const firstUserMsg = state.chat.history.find(m => m.role === 'user');
    if (firstUserMsg) {
        title = firstUserMsg.text.slice(0, 20) + (firstUserMsg.text.length > 20 ? "..." : "");
    }

    if (!session) {
        // Create new session object
        session = {
            id: state.chat.currentSessionId,
            title: title,
            messages: cloneDeep(state.chat.history), // Deep copy to prevent reference issues
            pipeline: state.selectedPipeline,        // Record which Pipeline was used for chat
            timestamp: Date.now()                    // Record time
        };
        state.chat.sessions.unshift(session); // Add to front
    } else {
        // Update existing session
        // Remove old position first, then insert at front (top)
        state.chat.sessions = state.chat.sessions.filter(s => s.id !== state.chat.currentSessionId);
        session.messages = cloneDeep(state.chat.history);
        session.timestamp = Date.now();

        // If it's default title, try to update to new title
        if (session.title === "New Chat" || session.title === "Untitled Chat") {
            session.title = title;
        }
        state.chat.sessions.unshift(session);
    }

    // Render sidebar and persist to LocalStorage
    renderChatSidebar();
    localStorage.setItem("ultrarag_sessions", JSON.stringify(state.chat.sessions));

    // Record current active session ID for recovery after refresh
    if (state.chat.currentSessionId) {
        localStorage.setItem("ultrarag_last_active_id", state.chat.currentSessionId);
    }
}

function renderChatSidebar() {
    if (!els.chatSessionList) return;
    els.chatSessionList.innerHTML = "";

    const displaySessions = state.chat.sessions.filter(s =>
        !s.pipeline || s.pipeline === state.selectedPipeline
    );

    if (state.chat.sessions.length === 0) {
        els.chatSessionList.innerHTML = '<div class="text-muted small" style="padding-left: 24px;">No history</div>';
        return;
    }

    state.chat.sessions.forEach(session => {
        // Container
        const itemDiv = document.createElement("div");
        itemDiv.className = `chat-session-item ${session.id === state.chat.currentSessionId ? 'active' : ''}`;

        // Content area (click to load)
        const contentDiv = document.createElement("div");
        contentDiv.className = "chat-session-content";
        contentDiv.innerHTML = `<span class="chat-session-title">${escapeHtml(session.title || "Untitled Chat")}</span>`;
        contentDiv.onclick = (e) => {
            e.stopPropagation();
            loadChatSession(session.id);
        };

        // Delete button (shown on hover)
        const delBtn = document.createElement("button");
        delBtn.className = "chat-session-delete-btn";
        delBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>`;
        delBtn.title = "Delete Chat";
        delBtn.onclick = (e) => {
            e.stopPropagation();
            deleteChatSession(session.id);
        };

        // Right-click menu
        itemDiv.oncontextmenu = (e) => showChatSessionContextMenu(e, session);

        itemDiv.appendChild(contentDiv);
        itemDiv.appendChild(delBtn);
        els.chatSessionList.appendChild(itemDiv);
    });
}

// Show chat session right-click menu
function showChatSessionContextMenu(event, session) {
    event.preventDefault();
    event.stopPropagation();
    const menu = document.getElementById('chat-session-context-menu');
    if (!menu || !session?.id) return;

    const menuWidth = 160;
    const menuHeight = 96;
    const left = Math.min(event.clientX, window.innerWidth - menuWidth - 12);
    const top = Math.min(event.clientY, window.innerHeight - menuHeight - 12);

    menu.dataset.sessionId = session.id;
    menu.style.left = `${left}px`;
    menu.style.top = `${top}px`;
    menu.classList.remove('d-none');
}

// Hide chat session right-click menu
function hideChatSessionContextMenu() {
    const menu = document.getElementById('chat-session-context-menu');
    if (!menu) return;
    menu.classList.add('d-none');
    menu.dataset.sessionId = '';
}

// Rename chat session
async function renameChatSession(sessionId) {
    if (!sessionId) return;
    const session = state.chat.sessions.find(s => s.id === sessionId);
    if (!session) return;

    const newTitle = await showPrompt(t("chat_rename_prompt"), {
        title: t("chat_rename_title"),
        placeholder: t("chat_rename_placeholder"),
        defaultValue: session.title || t("chat_untitled"),
        confirmText: t("common_rename")
    });

    if (!newTitle || newTitle.trim() === '' || newTitle.trim() === session.title) return;

    session.title = newTitle.trim();
    localStorage.setItem("ultrarag_sessions", JSON.stringify(state.chat.sessions));
    renderChatSidebar();
}

// Delete session helper function
async function deleteChatSession(sessionId) {
    const confirmed = await showConfirm(t("chat_delete_confirm"), {
        title: t("chat_delete_title"),
        type: "warning",
        confirmText: t("common_delete"),
        danger: true
    });
    if (!confirmed) return;

    const wasCurrent = state.chat.currentSessionId === sessionId;

    state.chat.sessions = state.chat.sessions.filter(s => s.id !== sessionId);
    localStorage.setItem("ultrarag_sessions", JSON.stringify(state.chat.sessions));

    // If current session was deleted, clear last_active_id
    if (wasCurrent) {
        localStorage.removeItem("ultrarag_last_active_id"); // Clear

        if (state.chat.controller) {
            state.chat.controller.abort();
            state.chat.controller = null;
        }
        setChatRunning(false);
        state.chat.currentSessionId = null;
        state.chat.history = [];
        renderChatSidebar();
        createNewChatSession();
    } else {
        renderChatSidebar();
    }
}

// Delete all sessions
async function deleteAllChatSessions() {
    if (state.chat.sessions.length === 0) return;
    const confirmed = await showConfirm(t("chat_delete_all_confirm"), {
        title: t("chat_delete_all_title"),
        type: "warning",
        confirmText: t("chat_delete_all_action"),
        danger: true
    });
    if (!confirmed) return;

    state.chat.sessions = [];
    localStorage.setItem("ultrarag_sessions", JSON.stringify(state.chat.sessions));
    localStorage.removeItem("ultrarag_last_active_id");
    createNewChatSession();
}

function appendChatMessage(role, text, meta = {}) {
    const entry = { role, text, meta, timestamp: new Date().toISOString() };
    state.chat.history.push(entry);
    renderChatHistory();
    saveCurrentSession();
}

function formatCitationHtml(html, messageIdx = null) {
    if (!html) return "";
    // Add onclick="scrollToReference(1, messageIdx)"
    // Note: scrollToReference function must be mounted to window or defined in global scope
    // messageIdx is used to locate specific message bubble, avoiding confusion with multiple message citations
    return html.replace(
        /\[(\d+)\]/g,
        (match, p1) => `<span class="citation-link" onclick="scrollToReference(${p1}, ${messageIdx})">[${p1}]</span>`
    );
}

// Render suggestion chips (disabled)
function renderSuggestionChips() {
    // Suggestion buttons removed, no longer rendered
}

// Remove suggestion chips
function removeSuggestionChips() {
    const existing = document.getElementById('suggestion-chips-container');
    if (existing) existing.remove();
}

// Main function: modified renderChatHistory
function renderChatHistory() {
    if (!els.chatHistory) return;
    els.chatHistory.innerHTML = "";

    // Get chat-container element to control empty state layout
    const chatContainer = els.chatHistory.parentElement;

    // Core modification: mimic Tongyi Qianwen empty welcome page - input box centered, suggestion buttons below
    if (state.chat.history.length === 0) {
        // Add empty state class to center input box
        if (chatContainer) chatContainer.classList.add('empty-state');

        els.chatHistory.innerHTML = `
            <div class="empty-state-wrapper fade-in-up">
                <div class="greeting-section">
                    <div class="greeting-text">
                        <span class="greeting-gradient">${t("greeting_explore")}</span>
                    </div>
                </div>
            </div>
        `;

        // Render suggestion buttons below input box
        renderSuggestionChips();
        return;
    }

    // Remove empty state class, restore normal layout
    if (chatContainer) chatContainer.classList.remove('empty-state');
    // Remove suggestion buttons
    removeSuggestionChips();
    state.chat.history.forEach((entry, index) => {
        const bubble = document.createElement("div");
        // Add fade-in animation class for better appearance
        bubble.className = `chat-bubble ${entry.role} fade-in-up`;
        // Add message index identifier for citation positioning
        bubble.setAttribute("data-message-idx", index);
        // Historical records displayed directly, no animation delay needed
        bubble.style.animationDelay = "0ms";

        const content = document.createElement("div");
        content.className = "msg-content";

        if (entry.role === "assistant") {
            let htmlContent = renderMarkdown(entry.text || "", { unwrapLanguages: MARKDOWN_LANGS });
            content.innerHTML = formatCitationHtml(htmlContent, index);
            renderLatex(content);
            applyCodeHighlight(content);
            applyTableEnhancements(content);
        } else {
            // User message: preserve line break effects
            // Convert line breaks to HTML, while escaping HTML special characters to prevent XSS
            const escapedText = entry.text
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
            content.innerHTML = escapedText.replace(/\n/g, '<br>');
        }
        bubble.appendChild(content);

        // Render Show Thinking step information
        if (entry.role === "assistant" && entry.meta && entry.meta.steps && entry.meta.steps.length > 0) {
            renderStepsFromHistory(bubble, entry.meta.steps, entry.meta.interrupted);
        }

        // Copy original text button (placed before citations)
        if (entry.role === "assistant") {
            ensureChatCopyRow(bubble, entry.text || "");
        }

        // Render citation cards at bottom
        if (entry.meta && entry.meta.sources) {
            // Calculate which citations are used
            const usedIds = new Set();
            const regex = /\[(\d+)\]/g;
            let match;
            while ((match = regex.exec(entry.text || "")) !== null) {
                usedIds.add(parseInt(match[1], 10));
            }

            renderSources(bubble, entry.meta.sources, usedIds);
        }

        // Debug information (Hint) disabled, no longer showing Dataset/Memory paths
        // if (entry.meta && entry.meta.hint) {
        //     const hintDiv = document.createElement("div");
        //     hintDiv.className = "text-xs text-muted mt-2 pt-2 border-top";
        //     hintDiv.textContent = entry.meta.hint;
        //     bubble.appendChild(hintDiv);
        // }

        els.chatHistory.appendChild(bubble);
    });
    els.chatHistory.scrollTop = els.chatHistory.scrollHeight;
}

// Helper function to quickly fill prompt text (can be placed anywhere in main.js)
window.setQuickPrompt = function (text) {
    if (els.chatInput) {
        els.chatInput.value = text;
        // Auto-adjust textarea height
        els.chatInput.style.height = 'auto';
        els.chatInput.style.height = (els.chatInput.scrollHeight) + 'px';
        els.chatInput.focus();
        // Optional: if you want to send directly on click, uncomment the line below
        // els.chatSend.click();
    }
};

function setChatStatus(message, variant = "info") {
    if (!els.chatStatus) return;
    const badge = els.chatStatus;
    const variants = { info: "bg-light text-dark", ready: "bg-light text-dark", running: "bg-primary text-white", success: "bg-success text-white", warn: "bg-warning text-dark", error: "bg-danger text-white" };
    const statusMap = {
        "Ready": "status_ready",
        "Offline": "status_offline",
        "Engine Offline": "status_engine_offline",
        "Thinking...": "status_thinking",
        "Initializing...": "status_initializing",
        "Reconnecting...": "status_reconnecting",
        "Params Missing": "status_params_missing",
        "Engine Ready": "status_engine_ready",
        "Engine Error": "status_engine_error",
        "Error": "status_error",
        "Interrupted": "status_interrupted",
        "Suspending...": "status_suspending",
    };
    const key = statusMap[message] || null;
    badge.className = `badge rounded-pill border ${variants[variant] || variants.info}`;
    badge.textContent = key ? t(key) : (message || "");
}

function updateChatIdleStatus() {
    if (state.chat.engineSessionId) {
        setChatStatus("Ready", "ready");
    } else {
        // Avoid overriding to offline if a pipeline is selected but engine state is ambiguous
        if (!state.selectedPipeline) {
            setChatStatus("Engine Offline", "info");
        }
    }
}

function setChatRunning(isRunning) {
    state.chat.running = isRunning;

    // 1. Update engine button state first (but this will reset input/send state, so override below)
    updateDemoControls();

    const btn = els.chatSend;
    const iconWrapper = document.getElementById("chat-send-icon");

    if (isRunning) {
        setChatStatus("Thinking...", "running");

        // Generating: input box locked, but button must be enabled (to click stop)
        if (els.chatInput) els.chatInput.disabled = true;
        if (els.chatSend) els.chatSend.disabled = false; // <--- Critical! Must enable!

        // Change style to stop button
        if (btn) btn.classList.add("stop");
        if (iconWrapper) {
            // Hide send arrow, show stop icon
            iconWrapper.innerHTML = '<span class="icon-stop"></span>';
        }
    } else {
        updateActionButtons();

        // Idle state: input box unlocked (assuming engine is online, updateDemoControls has handled it)
        // Change style back to arrow
        if (btn) btn.classList.remove("stop");
        if (iconWrapper) {
            // Restore send arrow icon
            iconWrapper.innerHTML = `
            <svg class="icon-send" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="12" y1="19" x2="12" y2="5"></line>
              <polyline points="5 12 12 5 19 12"></polyline>
            </svg>
          `;
        }
    }
}

function canUseChat() { return Boolean(state.isBuilt && state.selectedPipeline && state.parameterData); }

/**
 * Collect tool names from pipeline steps
 */
function collectToolNamesFromSteps(steps, set = new Set()) {
    for (const step of steps || []) {
        if (typeof step === "string") {
            const name = step.trim();
            if (name) set.add(name);
        } else if (isToolConfigStep(step)) {
            const toolName = Object.keys(step)[0];
            if (toolName) set.add(toolName);
        } else if (step && typeof step === "object") {
            if (step.loop && Array.isArray(step.loop.steps)) {
                collectToolNamesFromSteps(step.loop.steps, set);
            } else if (step.branch) {
                collectToolNamesFromSteps(step.branch.router || [], set);
                Object.values(step.branch.branches || {}).forEach(bs => collectToolNamesFromSteps(bs || [], set));
            }
        }
    }
    return set;
}

/**
 * Check if current pipeline requires knowledge base selection
 * Only enforce when retriever search tools are present (exclude websearch tools)
 */
function pipelineRequiresKnowledgeBase() {
    const steps = (state.pipelineConfig && Array.isArray(state.pipelineConfig.pipeline))
        ? state.pipelineConfig.pipeline
        : (Array.isArray(state.steps) ? state.steps : []);
    const toolNames = collectToolNamesFromSteps(steps, new Set());
    if (!toolNames.size) return false;

    const retrieverTools = [...toolNames]
        .map(name => name.toLowerCase())
        .filter(name => name.startsWith("retriever."));
    if (!retrieverTools.length) return false;

    const kbFreeRetrieverTools = new Set([
        "retriever.retriever_init",
        "retriever.retriever_embed",
        "retriever.retriever_index",
        "retriever.bm25_index",
        "retriever.retriever_websearch",
        "retriever.retriever_batch_websearch",
    ]);

    return retrieverTools.some(name => !kbFreeRetrieverTools.has(name));
}

/**
 * Check if this is the first turn of chat (user hasn't sent a message yet, or only one just sent)
 * Note: When calling this function, user message may not have been added to history yet
 */
function isFirstTurnChat() {
    // If history is empty, or history has no assistant reply, it's the first conversation
    const hasAssistantMessage = state.chat.history.some(msg => msg.role === 'assistant');
    return !hasAssistantMessage;
}

/**
 * Validate if knowledge base selection is required
 * Returns true if validation passes, false if user needs to select knowledge base
 */
function validateKnowledgeBaseSelection() {
    // 1. Check if this is first turn chat
    if (!isFirstTurnChat()) {
        // Not first turn, no validation needed
        return true;
    }

    // 2. Check if pipeline requires knowledge base
    if (!pipelineRequiresKnowledgeBase()) {
        // Pipeline doesn't need knowledge base selection
        return true;
    }

    // 3. Check if user has selected knowledge base
    const selectedCollection = els.chatCollectionSelect ? els.chatCollectionSelect.value : "";
    if (selectedCollection) {
        // Knowledge base selected, validation passes
        return true;
    }

    // Knowledge base not selected, show prompt
    showKnowledgeBaseAlert();
    return false;
}

/**
 * Show knowledge base selection prompt modal
 */
async function showKnowledgeBaseAlert() {
    await showModal(t("chat_kb_required_message"), {
        title: t("chat_kb_required_title"),
        type: "warning"
    });
    // Focus on knowledge base selection dropdown
    if (els.chatCollectionSelect) {
        els.chatCollectionSelect.focus();
    }
}

async function safeOpenChatView() {
    if (state.mode !== Modes.CHAT && state.unsavedChanges) {
        const proceed = await confirmUnsavedChanges(t("builder_action_enter_chat"));
        if (!proceed) return;
    }
    await openChatView();
}

async function openChatView() {
    if (!canUseChat()) {
        log("Please build and save parameters first.");
        showModal(t("builder_pipeline_not_ready_message"), { title: t("builder_pipeline_not_ready_title"), type: "warning" });
        return;
    }
    if (els.chatPipelineName) els.chatPipelineName.textContent = state.selectedPipeline || "—";

    await renderChatPipelineMenu();

    // Load latest Collection list when entering chat
    await renderChatCollectionOptions();

    if (!state.chat.currentSessionId) createNewChatSession();

    renderChatHistory();
    renderChatSidebar();
    setMode(Modes.CHAT);
    updateUrlForView(Modes.CHAT);

    backToChatView();

    // Ensure engine is available when entering interface (auto-reconnect/restart if necessary)
    if (state.selectedPipeline) {
        await ensureEngineReady(state.selectedPipeline);
    }
    updateDemoControls();

    // Initialize background tasks
    initBackgroundTasks();
}

async function stopGeneration() {
    if (!state.chat.running) return;

    // 1. Frontend disconnect (stop receiving data stream)
    // This will cause fetch to throw AbortError, jump to catch block
    // AbortError handling will save already generated content
    if (state.chat.controller) {
        state.chat.controller.abort();
        state.chat.controller = null;
    }

    // 2. Notify backend Python to stop Loop (release Session lock)
    try {
        if (state.chat.engineSessionId) {
            // Send stop signal, but don't wait for response, directly end UI state
            fetchJSON(`/api/pipelines/chat/stop`, {
                method: "POST",
                body: JSON.stringify({ session_id: state.chat.engineSessionId })
            }).catch(e => console.warn("Backend stop signal failed:", e));
        }
    } catch (e) { console.error(e); }

    log("Generation stopped by user.");

    // UI state will be updated in AbortError handling
}

// Helper function: clean dirty formatting from PDF extracted text
function cleanPDFText(text) {
    if (!text) return "";

    let clean = text;

    // 1. Normalize line breaks
    clean = clean.replace(/\r\n/g, '\n');

    // 2. Fix word hyphenation
    // Example: "communi-\ncation" -> "communication"
    // Logic: letter + hyphen + line break + letter -> merge directly
    clean = clean.replace(/([a-zA-Z])-\n([a-zA-Z])/g, '$1$2');

    // 3. Intelligently merge hard line breaks (Unwrap Lines)
    // PDF parsed text usually has \n on every line, we need to merge them into paragraphs
    // Strategy:
    // a. First protect real paragraphs (\n\n), replace with special placeholder
    clean = clean.replace(/\n\s*\n/g, '___PARAGRAPH_BREAK___');

    // b. Replace remaining single \n (usually PDF hard line breaks) with spaces
    clean = clean.replace(/\n/g, ' ');

    // c. Merge extra spaces (multiple spaces become one)
    clean = clean.replace(/  +/g, ' ');

    // d. Restore special placeholder to Markdown standard paragraph break (\n\n)
    clean = clean.replace(/___PARAGRAPH_BREAK___/g, '\n\n');

    return clean;
}

// Open right side detail panel
function showSourceDetail(title, content) {
    const panel = document.getElementById("source-detail-panel");
    const contentDiv = document.getElementById("source-detail-content");
    const titleDiv = panel.querySelector(".detail-title");

    if (panel && contentDiv) {
        // Fill content
        titleDiv.textContent = title || "Reference";

        // 2. Clean text (handle PDF garbled text)
        const rawText = content || "No content available.";
        let cleanedText = cleanPDFText(rawText);

        // remove bibkey:textidXX
        const bibkeyMatch = cleanedText.match(/^bibkey:\s*\S+\s+([\s\S]*)/i)
        if (bibkeyMatch) {
            cleanedText = bibkeyMatch[1].trim();
        }

        // 3. Check if contains Title: and Content: format, if yes render separately
        const titleMatch = cleanedText.match(/^Title:\s*(.+?)(?:\n|Content:)/i);
        const contentMatch = cleanedText.match(/Content:\s*([\s\S]*)/i);

        let renderedHtml = "";

        if (titleMatch && contentMatch) {
            // Has title and content format, render separately
            const docTitle = titleMatch[1].trim();
            const docContent = contentMatch[1].trim();

            // Display title in bold
            renderedHtml = `<div class="source-doc-title">${escapeHtmlForDetail(docTitle)}</div>`;

            // Render content normally
            if (typeof renderMarkdown === 'function') {
                renderedHtml += renderMarkdown(docContent);
            } else {
                renderedHtml += `<p>${escapeHtmlForDetail(docContent).replace(/\n/g, '<br>')}</p>`;
            }
        } else {
            // Normal format, render directly
            if (typeof renderMarkdown === 'function') {
                renderedHtml = renderMarkdown(cleanedText);
            } else {
                renderedHtml = `<p>${escapeHtmlForDetail(cleanedText).replace(/\n/g, '<br>')}</p>`;
            }
        }

        contentDiv.innerHTML = renderedHtml;
        renderLatex(contentDiv);
        applyCodeHighlight(contentDiv);
        applyTableEnhancements(contentDiv);

        // 4. Scroll back to top (prevent opening at bottom if last view was at bottom)
        contentDiv.scrollTop = 0;

        // Expand panel
        panel.classList.add("show");
    }
}

// HTML escape helper function (for detail panel)
function escapeHtmlForDetail(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Close right side detail panel (bound to HTML x button)
window.closeSourceDetail = function () {
    const panel = document.getElementById("source-detail-panel");
    if (panel) panel.classList.remove("show");
};

// Click citation [x] to highlight citation item and show details
// messageIdx parameter is used to locate specific message bubble, ensuring correct message citations are displayed
window.scrollToReference = function (refId, messageIdx = null) {
    let target = null;

    if (messageIdx !== null) {
        // Prioritize searching for citation within specified message bubble
        const bubble = document.querySelector(`[data-message-idx="${messageIdx}"]`);
        if (bubble) {
            target = bubble.querySelector(`[data-ref-id="${refId}"]`);
        }
    }

    // If not found, fall back to old logic (compatibility)
    if (!target) {
        const targetId = `ref-item-${refId}`;
        const allRefs = document.querySelectorAll(`[id='${targetId}']`);
        target = allRefs[allRefs.length - 1];
    }

    if (target) {
        // 1. Clear highlight from all citation items
        document.querySelectorAll(".ref-item").forEach(item => {
            item.classList.remove("active-highlight");
        });

        // 2. Highlight currently selected citation item
        target.classList.add("active-highlight");

        // 3. If in collapsed area, auto-expand
        const unusedSection = target.closest(".unused-refs-section");
        if (unusedSection && unusedSection.classList.contains("collapsed")) {
            unusedSection.classList.remove("collapsed");
        }

        // 4. Open right sidebar to show details
        if (target._sourceData) {
            const src = target._sourceData;
            showSourceDetail(`Reference [${src.id}]`, src.content);
        }
    }
};

// Render reference list (supports collapsing unused citations)
function renderSources(bubble, sources, usedIds = null) {
    if (!bubble || !sources || sources.length === 0) return;

    let refContainer = bubble.querySelector(".reference-container");

    // Clear and rebuild
    if (refContainer) {
        refContainer.remove();
    }

    refContainer = document.createElement("div");
    refContainer.className = "reference-container";

    // Separate used and unused
    const usedSources = [];
    const unusedSources = [];

    // Deduplicate: keep only first by ID
    const seenIds = new Set();
    sources.forEach(src => {
        const id = src.displayId || src.id;
        if (seenIds.has(id)) return; // Skip duplicates
        seenIds.add(id);

        if (usedIds && usedIds.has(id)) {
            usedSources.push(src);
        } else if (usedIds) {
            unusedSources.push(src);
        } else {
            // If no usedIds, treat all as used
            usedSources.push(src);
        }
    });

    // Helper function to create citation item
    const createRefItem = (src) => {
        const showId = src.displayId || src.id;
        const item = document.createElement("div");
        item.className = "ref-item";
        item.id = `ref-item-${showId}`;
        // Add data-ref-id attribute for precise citation positioning via messageIdx
        item.setAttribute("data-ref-id", showId);
        item._sourceData = src;
        item.onclick = (e) => {
            e.stopPropagation();
            // Clear other highlights
            document.querySelectorAll(".ref-item").forEach(el => el.classList.remove("active-highlight"));
            item.classList.add("active-highlight");
            showSourceDetail(`Reference [${showId}]`, src.content);
        };

        // Process citation item display text: title at beginning, followed by content
        let content = src.content || "";
        let displayText = "";

        // remove bibkey:textidXX
        const bibkeyMatch = content.match(/^bibkey:\s*\S+\s+([\s\S]*)/i)
        if (bibkeyMatch) {
            content = bibkeyMatch[1].trim();
        }

        // Check if content contains "Title:" and "Content:" format
        const titleMatch = content.match(/^Title:\s*(.+?)(?:\n|Content:)/i);
        const contentMatch = content.match(/Content:\s*([\s\S]*)/i);

        if (titleMatch && contentMatch) {
            // Has title and content format: title + content
            const docTitle = titleMatch[1].trim();
            const afterContent = contentMatch[1].trim();
            const firstLine = afterContent.split('\n')[0].trim();
            displayText = `${docTitle}: ${firstLine}`;
        } else if (contentMatch) {
            // Only Content: format
            const afterContent = contentMatch[1].trim();
            displayText = afterContent.split('\n')[0].trim();
        } else {
            // Normal format, use original title
            displayText = content.split('\n')[0].trim() || src.title || "";
        }

        item.innerHTML = `
            <span class="ref-id">[${showId}]</span>
            <span class="ref-title">${escapeHtml(displayText)}</span>
        `;
        return item;
    };

    // HTML escape helper function
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Render used citations
    if (usedSources.length > 0) {
        const usedHeader = document.createElement("div");
        usedHeader.className = "ref-header";
        usedHeader.setAttribute("data-i18n-key", "chat_cited_references");
        usedHeader.dataset.count = usedSources.length;
        usedHeader.textContent = `${t("chat_cited_references")} (${usedSources.length})`;
        refContainer.appendChild(usedHeader);

        const usedList = document.createElement("div");
        usedList.className = "ref-list";
        usedSources.forEach(src => {
            const item = createRefItem(src);
            item.classList.add("used");
            usedList.appendChild(item);
        });
        refContainer.appendChild(usedList);
    }

    // Render unused citations (collapsible)
    if (unusedSources.length > 0) {
        const unusedSection = document.createElement("div");
        unusedSection.className = "unused-refs-section collapsed";

        const unusedHeader = document.createElement("div");
        unusedHeader.className = "ref-header unused-header";
        unusedHeader.innerHTML = `
            <span data-i18n-key="chat_other_retrieved" data-count="${unusedSources.length}">${t("chat_other_retrieved")} (${unusedSources.length})</span>
            <span class="toggle-icon">▶</span>
        `;
        unusedHeader.onclick = () => {
            unusedSection.classList.toggle("collapsed");
        };
        unusedSection.appendChild(unusedHeader);

        const unusedList = document.createElement("div");
        unusedList.className = "ref-list unused-list";
        unusedSources.forEach(src => {
            const item = createRefItem(src);
            item.classList.add("unused");
            unusedList.appendChild(item);
        });
        unusedSection.appendChild(unusedList);

        refContainer.appendChild(unusedSection);
    }

    bubble.appendChild(refContainer);
}

// Format message text (highlight [1])
function formatMessageText(text) {
    if (!text) return "";
    // Escape HTML first to prevent injection
    let safeText = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");

    // Handle line breaks
    safeText = safeText.replace(/\n/g, "<br>");

    // Regex replace [number] with highlight tag
    // Match [1], [12], [1,2] etc.
    safeText = safeText.replace(/\[(\d+)\]/g, '<span class="citation-link">[$1]</span>');

    return safeText;
}

// Restore step information from history
function renderStepsFromHistory(bubble, steps, isInterrupted = false) {
    if (!steps || steps.length === 0) return;

    // Create Process Container (exactly matching real-time rendering format)
    const procDiv = document.createElement("div");
    procDiv.className = "process-container collapsed"; // Collapsed by default
    procDiv.innerHTML = `
        <div class="process-header" onclick="this.parentNode.classList.toggle('collapsed')">
            <span data-i18n-key="chat_show_thinking">${t("chat_show_thinking")}</span>
            <svg class="process-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="6 9 12 15 18 9"></polyline>
            </svg>
        </div>
        <div class="process-body"></div>
    `;

    const body = procDiv.querySelector(".process-body");

    // Parse steps, merge step_start and step_end
    const stepMap = new Map();
    const stepOrder = []; // Maintain order

    for (const step of steps) {
        if (step.type === 'step_start') {
            if (!stepMap.has(step.name)) {
                stepOrder.push(step.name);
            }
            stepMap.set(step.name, {
                name: step.name,
                tokens: step.tokens || '',
                output: '',
                completed: false
            });
        } else if (step.type === 'step_end') {
            const existing = stepMap.get(step.name);
            if (existing) {
                existing.completed = true;
                if (step.output) {
                    existing.output = step.output;
                }
            }
        }
    }

    // Render each step in order (exactly matching real-time rendering format)
    for (const name of stepOrder) {
        const stepData = stepMap.get(name);
        if (!stepData) continue;

        const stepDiv = document.createElement("div");
        stepDiv.className = "process-step";
        stepDiv.dataset.stepName = name;

        // Title section: completed shows empty (spinner removed), incomplete shows warning
        let titleContent = '';
        if (!stepData.completed && isInterrupted) {
            titleContent = '<span class="step-spinner" style="border-color: #f59e0b transparent transparent transparent;"></span>';
        }
        // Note: completed steps have spinner removed in real-time rendering, so don't show here either

        stepDiv.innerHTML = `
            <div class="step-title">
                ${titleContent}
                <span>${escapeHtml(name)}</span>
            </div>
        `;

        // Add streaming content (if any)
        if (stepData.tokens) {
            const streamDiv = document.createElement("div");
            streamDiv.className = "step-content-stream";
            // Use textContent to maintain consistency with real-time rendering
            streamDiv.textContent = stepData.tokens;
            stepDiv.appendChild(streamDiv);
        }

        // Add output summary (if any)
        if (stepData.output) {
            const detailsDiv = document.createElement("div");
            detailsDiv.className = "step-details";
            detailsDiv.textContent = stepData.output;
            stepDiv.appendChild(detailsDiv);
        }

        body.appendChild(stepDiv);
    }

    // Insert at front of bubble (consistent with real-time rendering)
    bubble.insertBefore(procDiv, bubble.firstChild);
}

// Track process-body scroll state
const processScrollState = {
    shouldAutoScroll: true
};

function updateProcessUI(entryIndex, eventData) {
    // 1. Find corresponding Chat Bubble (last assistant bubble)
    const container = document.getElementById("chat-history");
    const bubbles = container.querySelectorAll(".chat-bubble.assistant");
    const lastBubble = bubbles[bubbles.length - 1];
    if (!lastBubble) return;

    // 2. Check or create Process Container
    let procDiv = lastBubble.querySelector(".process-container");
    if (!procDiv) {
        procDiv = document.createElement("div");
        procDiv.className = "process-container";
        // Expanded structure by default
        procDiv.innerHTML = `
            <div class="process-header" onclick="this.parentNode.classList.toggle('collapsed')">
                <span data-i18n-key="chat_show_thinking">${t("chat_show_thinking")}</span>
                <svg class="process-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="6 9 12 15 18 9"></polyline>
                </svg>
            </div>
            <div class="process-body"></div>
        `;
        // Insert at front of bubble
        lastBubble.insertBefore(procDiv, lastBubble.firstChild);

        // Add scroll listener to process-body for smart auto-scroll
        const newBody = procDiv.querySelector(".process-body");
        if (newBody) {
            processScrollState.shouldAutoScroll = true; // Reset state
            newBody.addEventListener('scroll', function () {
                const threshold = 30;
                const distance = this.scrollHeight - this.scrollTop - this.clientHeight;
                processScrollState.shouldAutoScroll = distance <= threshold;
            });
        }
    }

    const body = procDiv.querySelector(".process-body");

    // Helper function: smart scroll to bottom
    const smartScrollToBottom = () => {
        if (processScrollState.shouldAutoScroll && body) {
            body.scrollTop = body.scrollHeight;
        }
    };

    // 3. Handle different events
    if (eventData.type === "step_start") {
        const stepDiv = document.createElement("div");
        stepDiv.className = "process-step";
        stepDiv.dataset.stepName = eventData.name;
        stepDiv.innerHTML = `
            <div class="step-title">
                <span class="step-spinner"></span>
                <span>${escapeHtml(eventData.name)}</span>
            </div>
            <div class="step-content-stream"></div> `;
        body.appendChild(stepDiv);
        // Auto scroll to bottom
        smartScrollToBottom();

    } else if (eventData.type === "token") {
        // If Token is not final, display in thinking process (as detailed log)
        // Find currently running step
        const steps = body.querySelectorAll(".process-step");
        const currentStep = steps[steps.length - 1];
        if (currentStep) {
            const streamDiv = currentStep.querySelector(".step-content-stream");
            if (streamDiv) {
                // Simple text append
                const span = document.createElement("span");
                span.textContent = eventData.content;
                streamDiv.appendChild(span);
                // Smart scroll follow
                smartScrollToBottom();
            }
        }

    } else if (eventData.type === "step_end") {
        const steps = body.querySelectorAll(".process-step");
        const currentStep = steps[steps.length - 1];

        if (currentStep) {
            // 1. Spinner -> Checkmark
            const spinner = currentStep.querySelector(".step-spinner");
            if (spinner) {
                spinner.remove(); // Remove element directly
            }

            // 2. Show summary (output)
            // If there was streaming content (step-content-stream) before, can choose to keep or be overwritten by summary
            // Here we choose to append summary as conclusion
            if (eventData.output) {
                const details = document.createElement("div");
                details.className = "step-details";
                details.textContent = eventData.output;
                currentStep.appendChild(details);
                // Smart scroll follow
                smartScrollToBottom();

                // (Optional) Hide streaming process, only show result? 
                // currentStep.querySelector(".step-content-stream").style.display = 'none';
            }
        }
    }
}


async function handleChatSubmit(event) {
    if (event) event.preventDefault();
    if (state.chat.running) { await stopGeneration(); return; }
    if (!canUseChat()) return;
    const engineReady = await ensureEngineReady(state.selectedPipeline);
    if (!engineReady) {
        showModal(t("chat_engine_required_message"), { title: t("chat_engine_required_title"), type: "warning" });
        return;
    }

    const question = els.chatInput.value.trim();
    if (!question) return;

    // Validate if knowledge base selection is required (only for first turn chat and when pipeline contains retriever)
    if (!validateKnowledgeBaseSelection()) {
        return;
    }

    // Check if background mode is enabled
    if (backgroundTaskState.backgroundModeEnabled) {
        els.chatInput.value = "";
        if (els.chatInput) els.chatInput.style.height = 'auto';
        await sendToBackground(question);
        // Disable background mode after sending
        backgroundTaskState.backgroundModeEnabled = false;
        const toggle = document.getElementById('bg-mode-toggle');
        if (toggle) toggle.classList.remove('active');
        return;
    }

    els.chatInput.value = "";
    // Reset textarea height
    if (els.chatInput) {
        els.chatInput.style.height = 'auto';
    }
    appendChatMessage("user", question);
    setChatRunning(true);
    state.chat.controller = new AbortController();

    try {
        if (!state.parametersReady) await persistParameterData({ silent: true });

        const selectedCollection = els.chatCollectionSelect ? els.chatCollectionSelect.value : "";

        const dynamicParams = {};
        if (selectedCollection) {
            dynamicParams["collection_name"] = selectedCollection;
        }

        const endpoint = `/api/pipelines/${encodeURIComponent(state.selectedPipeline)}/chat`;
        const body = JSON.stringify({
            question,
            history: state.chat.history,
            is_demo: true,
            session_id: state.chat.engineSessionId,
            dynamic_params: dynamicParams
        });

        const response = await fetch(endpoint, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: body, signal: state.chat.controller.signal
        });

        if (!response.ok) throw new Error(response.statusText);

        // Create placeholder first
        const entryIndex = state.chat.history.length;
        state.chat.history.push({ role: "assistant", text: "", meta: {} });

        // Fix: Lift these variables to higher scope so they can be saved on interrupt
        state.chat._streamingText = "";
        state.chat._streamingSources = [];
        state.chat._streamingSteps = []; // Save step information
        state.chat._streamingEntryIndex = entryIndex;

        const chatContainer = document.getElementById("chat-history");

        // Scroll optimization
        let shouldAutoScroll = true;
        const handleScroll = () => {
            const threshold = 30;
            const distance = chatContainer.scrollHeight - chatContainer.scrollTop - chatContainer.clientHeight;
            shouldAutoScroll = distance <= threshold;
        };
        chatContainer.addEventListener('scroll', handleScroll);

        const bubble = document.createElement("div");
        bubble.className = "chat-bubble assistant";
        // Add message index identifier for citation positioning
        bubble.setAttribute("data-message-idx", entryIndex);
        const contentDiv = document.createElement("div");
        contentDiv.className = "msg-content";
        bubble.appendChild(contentDiv);
        chatContainer.appendChild(bubble);

        let currentText = "";
        let allSources = [];
        let pendingRenderSources = [];

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n\n");
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    try {
                        const jsonStr = line.slice(6);
                        const data = JSON.parse(jsonStr);

                        if (data.type === "step_start" || data.type === "step_end") {
                            updateProcessUI(entryIndex, data);
                            // Save step information for recovery
                            if (!state.chat._streamingSteps) state.chat._streamingSteps = [];
                            const stepInfo = {
                                type: data.type,
                                name: data.name,
                                timestamp: Date.now()
                            };
                            // Save step_end output summary
                            if (data.type === "step_end" && data.output) {
                                stepInfo.output = data.output;
                            }
                            state.chat._streamingSteps.push(stepInfo);
                        }
                        else if (data.type === "sources") {
                            // Backend has assigned unique ID to each document, use directly
                            const docs = data.data.map((doc) => ({
                                ...doc,
                                displayId: doc.id
                            }));
                            allSources = allSources.concat(docs);
                            pendingRenderSources = pendingRenderSources.concat(docs);
                            // Fix: Synchronously update state so it can be saved on interrupt
                            state.chat._streamingSources = allSources;
                        }
                        else if (data.type === "token") {
                            if (!data.is_final) {
                                updateProcessUI(entryIndex, data);
                                // Save thinking content
                                if (!state.chat._streamingSteps) state.chat._streamingSteps = [];
                                // Find last step_start, append token content
                                const lastStep = state.chat._streamingSteps.filter(s => s.type === 'step_start').pop();
                                if (lastStep) {
                                    if (!lastStep.tokens) lastStep.tokens = "";
                                    lastStep.tokens += data.content;
                                }
                            }
                            if (data.is_final) {
                                currentText += data.content;
                                // Fix: Synchronously update state so it can be saved on interrupt
                                state.chat._streamingText = currentText;

                                if (typeof isPendingLanguageFence === 'function' && isPendingLanguageFence(currentText, MARKDOWN_LANGS)) continue;

                                let html = renderMarkdown(currentText, { unwrapLanguages: MARKDOWN_LANGS });
                                html = formatCitationHtml(html, entryIndex);
                                contentDiv.innerHTML = html;
                                renderLatex(contentDiv);
                                applyCodeHighlight(contentDiv);
                                applyTableEnhancements(contentDiv);
                                if (shouldAutoScroll) {
                                    chatContainer.scrollTop = chatContainer.scrollHeight;
                                }
                            }
                        }
                        else if (data.type === "final") {
                            const final = data.data;
                            let finalText = currentText || final.answer || "";
                            let html = renderMarkdown(finalText, { unwrapLanguages: MARKDOWN_LANGS });
                            html = formatCitationHtml(html, entryIndex);
                            contentDiv.innerHTML = html;
                            renderLatex(contentDiv);
                            applyCodeHighlight(contentDiv);
                            applyTableEnhancements(contentDiv);

                            // Calculate which citations are used
                            const usedIds = new Set();
                            const regex = /\[(\d+)\]/g;
                            let match;
                            while ((match = regex.exec(finalText)) !== null) {
                                usedIds.add(parseInt(match[1], 10));
                            }

                            // Copy original text button (before citations)
                            ensureChatCopyRow(bubble, finalText);

                            // Render reference cards (used ones on top, unused ones collapsed)
                            if (pendingRenderSources && pendingRenderSources.length > 0) {
                                renderSources(bubble, pendingRenderSources, usedIds);
                            }

                            if (shouldAutoScroll) {
                                chatContainer.scrollTop = chatContainer.scrollHeight;
                            }

                            // Update history
                            state.chat.history[entryIndex].text = finalText;
                            if (!state.chat.history[entryIndex].meta) state.chat.history[entryIndex].meta = {};
                            state.chat.history[entryIndex].meta.sources = allSources;
                            // Save step information
                            if (state.chat._streamingSteps && state.chat._streamingSteps.length > 0) {
                                state.chat.history[entryIndex].meta.steps = state.chat._streamingSteps;
                            }

                            // No longer show debug information (Dataset/Memory paths)
                            // const hints = [];
                            // if (final.dataset_path) hints.push(`Dataset: ${final.dataset_path}`);
                            // if (final.memory_path) hints.push(`Memory: ${final.memory_path}`);
                            // state.chat.history[entryIndex].meta.hint = hints.join(" | ");

                            const procDiv = bubble.querySelector(".process-container");
                            if (procDiv) procDiv.classList.add("collapsed");
                            setChatStatus("Ready", "ready");
                        }
                        else if (data.type === "error") {
                            const msg = data?.message ? String(data.message) : t("common_unknown_error");
                            showModal(formatTemplate(t("chat_backend_error_message"), { error: msg }), {
                                title: t("chat_error_title"),
                                type: "error"
                            });
                            setChatStatus("Error", "error");
                        }
                    } catch (e) { console.error(e); }
                }
            }
        }
    } catch (err) {
        if (err.name === 'AbortError') {
            // Fix: Save generated content on interrupt
            if (state.chat._streamingEntryIndex !== undefined) {
                const idx = state.chat._streamingEntryIndex;
                if (state.chat.history[idx]) {
                    // Save generated text
                    if (state.chat._streamingText) {
                        state.chat.history[idx].text = state.chat._streamingText + "\n\n*(Generation interrupted)*";
                    }
                    state.chat.history[idx].meta = state.chat.history[idx].meta || {};
                    state.chat.history[idx].meta.sources = state.chat._streamingSources || [];
                    state.chat.history[idx].meta.interrupted = true;
                    // Save step information
                    if (state.chat._streamingSteps && state.chat._streamingSteps.length > 0) {
                        state.chat.history[idx].meta.steps = state.chat._streamingSteps;
                    }
                }
            }
            // Clean up streaming state
            delete state.chat._streamingText;
            delete state.chat._streamingSources;
            delete state.chat._streamingSteps;
            delete state.chat._streamingEntryIndex;

            setChatRunning(false);
            setChatStatus("Interrupted", "info");
            saveCurrentSession();
            if (chatContainer && handleScroll) {
                chatContainer.removeEventListener('scroll', handleScroll);
            }
            return;
        }
        console.error(err);
        const msg = err?.message ? String(err.message) : t("common_unknown_error");
        showModal(formatTemplate(t("chat_network_error_message"), { error: msg }), {
            title: t("chat_error_title"),
            type: "error"
        });
        setChatStatus("Error", "error");
    } finally {
        if (state.chat.controller) {
            state.chat.controller = null;
            setChatRunning(false);
        }
        // Clean up streaming state
        delete state.chat._streamingText;
        delete state.chat._streamingSources;
        delete state.chat._streamingSteps;
        delete state.chat._streamingEntryIndex;

        saveCurrentSession();
        // Remove event listener (prevent memory leak)
        if (chatContainer && handleScroll) {
            chatContainer.removeEventListener('scroll', handleScroll);
        }
    }
}

// --- Common Logic (Mode Switching, Node Picker, etc.) ---
function resetLogView() { if (els.log) els.log.textContent = ""; }
function setHeroPipelineLabel(name) {
    if (!els.heroSelectedPipeline) return;
    if (name) {
        els.heroSelectedPipeline.textContent = name;
        els.heroSelectedPipeline.dataset.selected = "true";
        els.heroSelectedPipeline.dataset.currentName = name;
    } else {
        els.heroSelectedPipeline.textContent = t("builder_no_pipeline_selected");
        els.heroSelectedPipeline.dataset.selected = "false";
    }
}
function setHeroStatusLabel(status) {
    if (!els.heroStatus) return;
    els.heroStatus.dataset.status = status; els.heroStatus.textContent = status.toUpperCase();
}

async function fetchJSON(url, options = {}) {
    const resp = await fetch(url, { headers: { "Content-Type": "application/json" }, ...options });
    if (!resp.ok) { const text = await resp.text(); throw new Error(text || resp.statusText); }
    return resp.json();
}

async function persistParameterData({ silent = false } = {}) {
    if (!state.selectedPipeline || !state.parameterData) throw new Error("No parameters to save");
    await fetchJSON(`/api/pipelines/${encodeURIComponent(state.selectedPipeline)}/parameters`, { method: "PUT", body: JSON.stringify(state.parameterData) });
    state.parametersReady = true; updateActionButtons(); if (!silent) log(t("builder_params_saved"));
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
        if (!entry) return steps; // or return []

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

// ... (YAML Helpers - Corrected formatting) ...
function yamlScalar(value) {
    if (value === null || value === undefined) return "null";
    if (typeof value === "boolean") return value ? "true" : "false";
    if (typeof value === "number") return Number.isFinite(value) ? String(value) : "null";
    if (typeof value === "string") return value;
    return JSON.stringify(value);
}

/**
 * Convert JavaScript object to properly formatted YAML string
 * Fix: Objects in arrays should follow `-` immediately, not on new line
 */
function yamlStringify(value, indent = 0, isArrayItem = false) {
    const pad = "  ".repeat(indent);
    const itemPad = isArrayItem ? "" : pad; // First line of array item doesn't need indentation

    if (Array.isArray(value)) {
        if (!value.length) return `${itemPad}[]`;
        return value.map(item => {
            if (item && typeof item === "object") {
                // Object as array item: first key follows `-`
                const objStr = yamlStringify(item, indent + 1, true);
                return `${pad}- ${objStr}`;
            }
            return `${pad}- ${yamlScalar(item)}`;
        }).join("\n");
    }

    if (value && typeof value === "object") {
        const entries = Object.entries(value);
        if (!entries.length) return `${itemPad}{}`;

        return entries.map(([k, v], idx) => {
            const keyPad = (isArrayItem && idx === 0) ? "" : pad; // First key of array item doesn't need extra indentation

            if (v && typeof v === "object") {
                if (Array.isArray(v)) {
                    if (!v.length) {
                        return `${keyPad}${k}: []`;
                    }
                    return `${keyPad}${k}:\n${yamlStringify(v, indent)}`;
                }
                // Nested object
                const nestedStr = yamlStringify(v, indent + 1, false);
                return `${keyPad}${k}:\n${nestedStr}`;
            }
            return `${keyPad}${k}: ${yamlScalar(v)}`;
        }).join("\n");
    }

    return `${itemPad}${yamlScalar(value)}`;
}
function isToolConfigStep(step) {
    if (!step || typeof step !== "object" || Array.isArray(step)) return false;
    if (step.loop || step.branch) return false;
    const keys = Object.keys(step);
    return keys.length === 1 && typeof keys[0] === "string";
}

function collectServersFromSteps(steps, set = new Set()) {
    for (const step of steps || []) {
        if (typeof step === "string") {
            const parts = step.split(".");
            if (parts.length > 1) set.add(parts[0]);
        } else if (isToolConfigStep(step)) {
            const toolName = Object.keys(step)[0];
            const server = toolName.split(".")[0];
            if (server) set.add(server);
        } else if (step && typeof step === "object") {
            if (step.loop && Array.isArray(step.loop.steps)) collectServersFromSteps(step.loop.steps, set);
            else if (step.branch) {
                collectServersFromSteps(step.branch.router || [], set);
                Object.values(step.branch.branches || {}).forEach(bs => collectServersFromSteps(bs || [], set));
            }
        }
    }
    return set;
}

function buildServersMapping(steps) {
    const mapping = {};
    collectServersFromSteps(steps, new Set()).forEach((name) => { mapping[name] = `servers/${name}`; });
    return mapping;
}

function buildPipelinePayloadForPreview() {
    const derivedServers = buildServersMapping(state.steps);
    const baseServers = (state.pipelineConfig && typeof state.pipelineConfig === "object" && state.pipelineConfig.servers) ? state.pipelineConfig.servers : {};
    const mergedServers = { ...derivedServers, ...(baseServers || {}) };
    return { servers: cloneDeep(mergedServers), pipeline: cloneDeep(state.steps) };
}

// =========================================
// YAML Editor Sync System
// =========================================

// Flag: prevent circular updates
let yamlEditorSyncLock = false;
let yamlEditorDebounceTimer = null;

/**
 * Update YAML editor line number display
 */
function updateYamlLineNumbers() {
    if (!els.yamlEditor || !els.yamlLineNumbers) return;

    const content = els.yamlEditor.value || '';
    const lines = content.split('\n');
    const lineCount = Math.max(lines.length, 1);

    // Generate line number HTML, one div per line to ensure alignment
    let lineNumbersHtml = '';
    for (let i = 1; i <= lineCount; i++) {
        lineNumbersHtml += `<div class="line-number">${i}</div>`;
    }
    els.yamlLineNumbers.innerHTML = lineNumbersHtml;

    // Synchronize scroll position
    els.yamlLineNumbers.scrollTop = els.yamlEditor.scrollTop;
}

/**
 * Set sync status indicator
 * @param {'synced' | 'syncing' | 'error' | 'modified'} status
 */
function setYamlSyncStatus(status) {
    if (!els.yamlSyncStatus) return;

    els.yamlSyncStatus.classList.remove('synced', 'syncing', 'error', 'modified');
    els.yamlSyncStatus.classList.add(status);

    const icons = {
        synced: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>',
        syncing: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>',
        error: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>',
        modified: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>'
    };

    const titles = {
        synced: t('builder_yaml_synced_title'),
        syncing: t('builder_yaml_syncing_title'),
        error: t('builder_yaml_error_title'),
        modified: t('builder_yaml_modified_title')
    };

    els.yamlSyncStatus.innerHTML = icons[status] || icons.synced;
    els.yamlSyncStatus.title = titles[status] || '';
}

/**
 * Show/hide YAML error message
 * @param {string|null} message - Error message, null means clear error
 */
function showYamlError(message) {
    if (!els.yamlErrorBar || !els.yamlErrorMessage) return;

    if (message) {
        els.yamlErrorMessage.textContent = message;
        els.yamlErrorBar.classList.remove('d-none');
        setYamlSyncStatus('error');
    } else {
        els.yamlErrorBar.classList.add('d-none');
        setYamlSyncStatus(state.unsavedChanges ? 'modified' : 'synced');
    }
}

/**
 * Simple YAML parser (supports basic syntax)
 * @param {string} yamlStr - YAML string
 * @returns {object} Parsed object
 */
function parseSimpleYaml(yamlStr) {
    const lines = yamlStr.split('\n');
    const result = {};
    const stack = [{ indent: -1, obj: result, key: null }];
    let currentArray = null;
    let currentArrayIndent = -1;

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmed = line.trim();

        // Skip empty lines and comments
        if (!trimmed || trimmed.startsWith('#')) continue;

        // Calculate indentation
        const indent = line.search(/\S/);

        // Handle array items (starting with -)
        if (trimmed.startsWith('- ')) {
            const value = trimmed.substring(2).trim();

            // Find appropriate parent
            while (stack.length > 1 && stack[stack.length - 1].indent >= indent) {
                stack.pop();
            }

            const parent = stack[stack.length - 1];
            if (parent.key && !Array.isArray(parent.obj[parent.key])) {
                parent.obj[parent.key] = [];
            }

            if (parent.key && Array.isArray(parent.obj[parent.key])) {
                // Try to parse JSON
                if (value.startsWith('{') || value.startsWith('[')) {
                    try {
                        parent.obj[parent.key].push(JSON.parse(value));
                    } catch {
                        parent.obj[parent.key].push(value);
                    }
                } else {
                    parent.obj[parent.key].push(value || null);
                }
            }
            continue;
        }

        // Handle lines with only - (subsequent lines are nested objects)
        if (trimmed === '-') {
            while (stack.length > 1 && stack[stack.length - 1].indent >= indent) {
                stack.pop();
            }

            const parent = stack[stack.length - 1];
            if (parent.key && !Array.isArray(parent.obj[parent.key])) {
                parent.obj[parent.key] = [];
            }

            if (parent.key && Array.isArray(parent.obj[parent.key])) {
                const newObj = {};
                parent.obj[parent.key].push(newObj);
                stack.push({ indent: indent, obj: newObj, key: null });
            }
            continue;
        }

        // Handle key-value pairs
        const colonIndex = trimmed.indexOf(':');
        if (colonIndex > 0) {
            const key = trimmed.substring(0, colonIndex).trim();
            let value = trimmed.substring(colonIndex + 1).trim();

            // Pop stack back to appropriate level
            while (stack.length > 1 && stack[stack.length - 1].indent >= indent) {
                stack.pop();
            }

            const parent = stack[stack.length - 1].obj;

            if (value === '' || value === '[]' || value === '{}') {
                // Empty value or empty container, subsequent lines may have nesting
                if (value === '[]') {
                    parent[key] = [];
                } else if (value === '{}') {
                    parent[key] = {};
                } else {
                    parent[key] = {};
                }
                stack.push({ indent: indent, obj: parent, key: key });
            } else {
                // Has value
                if (value === 'true') value = true;
                else if (value === 'false') value = false;
                else if (value === 'null') value = null;
                else if (/^-?\d+$/.test(value)) value = parseInt(value, 10);
                else if (/^-?\d+\.\d+$/.test(value)) value = parseFloat(value);
                else if ((value.startsWith('"') && value.endsWith('"')) ||
                    (value.startsWith("'") && value.endsWith("'"))) {
                    value = value.slice(1, -1);
                }

                parent[key] = value;
            }
        }
    }

    return result;
}

async function parseYamlContent(yamlStr) {
    if (!yamlStr || !yamlStr.trim()) return {};
    try {
        const resp = await fetch("/api/pipelines/parse", {
            method: "POST",
            headers: { "Content-Type": "text/plain; charset=utf-8" },
            body: yamlStr
        });
        if (!resp.ok) {
            const detailText = await resp.text();
            let message = detailText;
            try {
                const parsedErr = JSON.parse(detailText);
                if (parsedErr && parsedErr.error) message = parsedErr.error;
            } catch (_) { }
            throw new Error(message || `Parse failed (${resp.status})`);
        }
        return await resp.json();
    } catch (err) {
        console.warn("Server YAML parse failed, fallback to simple parser:", err);
        // Try using frontend simple parser as fallback
        return parseSimpleYaml(yamlStr);
    }
}

function extractStepsFromParsedYaml(parsed) {
    let newSteps = [];
    if (Array.isArray(parsed)) {
        newSteps = parsed;
    } else if (parsed && typeof parsed === 'object') {
        if (Array.isArray(parsed.pipeline)) {
            newSteps = parsed.pipeline;
        } else if (Array.isArray(parsed.steps)) {
            newSteps = parsed.steps;
        }
    }

    if (!Array.isArray(newSteps)) {
        throw new Error(t('builder_yaml_invalid_format'));
    }

    return newSteps;
}

async function validateYamlEditorContent(options = {}) {
    const { showModalOnError = false } = options;
    if (!els.yamlEditor) return { valid: true, content: '', steps: [], parsed: {} };

    const yamlContent = els.yamlEditor.value;
    if (!yamlContent.trim()) {
        showYamlError(null);
        return { valid: true, content: '', steps: [], parsed: {} };
    }

    try {
        const parsed = await parseYamlContent(yamlContent);
        const steps = extractStepsFromParsedYaml(parsed);
        showYamlError(null);
        return { valid: true, content: yamlContent, steps, parsed };
    } catch (err) {
        const message = err?.message ? String(err.message) : t("builder_yaml_parse_failed");
        showYamlError(message);
        setYamlSyncStatus('error');
        if (showModalOnError) {
            showModal(formatTemplate(t("builder_yaml_parse_failed_message"), { error: message }), {
                title: t("builder_yaml_error_title"),
                type: "error"
            });
        }
        return { valid: false, error: message };
    }
}

/**
 * Sync from YAML editor to canvas (YAML → Canvas)
 * Preserve editor content (including comments), don't overwrite in reverse
 */
function syncYamlToCanvas() {
    return syncYamlToCanvasOnly();
}

/**
 * Only sync to canvas, don't trigger any editor updates
 * This preserves comments and other content in editor
 */
async function syncYamlToCanvasOnly(options = {}) {
    const { markUnsaved = true } = options;
    if (!els.yamlEditor) return;

    const yamlContent = els.yamlEditor.value;

    // Clear canvas when content is empty
    if (!yamlContent.trim()) {
        yamlEditorSyncLock = true;
        state.pipelineConfig = { servers: {}, pipeline: [], _raw_yaml: "" };
        setSteps([], { markUnsaved, skipPreview: true });
        showYamlError(null);
        setYamlSyncStatus('synced');
        yamlEditorSyncLock = false;
        log(t("builder_editor_synced_to_canvas"));
        return;
    }

    try {
        setYamlSyncStatus('syncing');

        const validation = await validateYamlEditorContent();
        if (!validation.valid) {
            yamlEditorSyncLock = false;
            return;
        }

        const newSteps = validation.steps;
        const parsedConfig = (validation.parsed && typeof validation.parsed === "object" && !Array.isArray(validation.parsed))
            ? { ...validation.parsed }
            : { pipeline: newSteps };
        parsedConfig._raw_yaml = yamlContent;

        yamlEditorSyncLock = true;
        state.pipelineConfig = parsedConfig;
        setSteps(newSteps, { markUnsaved, skipPreview: true });
        showYamlError(null);
        setYamlSyncStatus('synced');
        yamlEditorSyncLock = false;

        log(t("builder_editor_synced_to_canvas"));

    } catch (err) {
        showYamlError(err.message);
        setYamlSyncStatus('error');
        yamlEditorSyncLock = false;
    }
}

/**
 * Sync from canvas to YAML editor (Canvas → YAML)
 * This is an enhanced version of updatePipelinePreview
 */
function updatePipelinePreview() {
    // Update hidden preview element (maintain compatibility)
    if (els.pipelinePreview) {
        els.pipelinePreview.textContent = yamlStringify(buildPipelinePayloadForPreview());
    }

    // Update YAML editor
    if (els.yamlEditor && !yamlEditorSyncLock) {
        yamlEditorSyncLock = true;

        const payload = buildPipelinePayloadForPreview();
        const yamlContent = yamlStringify(payload);
        if (!state.pipelineConfig || typeof state.pipelineConfig !== "object") {
            state.pipelineConfig = {};
        }
        state.pipelineConfig._raw_yaml = yamlContent;
        els.yamlEditor.value = yamlContent;
        updateYamlLineNumbers();
        showYamlError(null);

        yamlEditorSyncLock = false;
    }
}

/**
 * Initialize YAML editor event listeners
 */
function initYamlEditor() {
    if (!els.yamlEditor) return;

    // Listen to input events, update line numbers and mark as modified
    els.yamlEditor.addEventListener('input', () => {
        updateYamlLineNumbers();
        // Mark editor as having unsynced changes
        setYamlSyncStatus('modified');
        updateUnsavedFromEditor();
        updateActionButtons();
    });

    // Listen to scroll events, synchronize line number scrolling
    els.yamlEditor.addEventListener('scroll', () => {
        if (els.yamlLineNumbers) {
            els.yamlLineNumbers.scrollTop = els.yamlEditor.scrollTop;
        }
    });

    // Listen to Tab key, insert spaces instead of switching focus
    els.yamlEditor.addEventListener('keydown', (e) => {
        const isSaveShortcut = (e.key === 's' || e.key === 'S') && (e.ctrlKey || e.metaKey);
        if (isSaveShortcut) {
            e.preventDefault();
            handleSubmit(e);
            return;
        }
        if (e.key === 'Tab') {
            e.preventDefault();
            const start = els.yamlEditor.selectionStart;
            const end = els.yamlEditor.selectionEnd;
            const spaces = '  '; // 2 space indentation

            els.yamlEditor.value =
                els.yamlEditor.value.substring(0, start) +
                spaces +
                els.yamlEditor.value.substring(end);

            els.yamlEditor.selectionStart = els.yamlEditor.selectionEnd = start + spaces.length;
            updateYamlLineNumbers();
        }
    });

    // Format button: regenerate YAML from Canvas state (restore to canvas state)
    if (els.yamlFormatBtn) {
        els.yamlFormatBtn.addEventListener('click', async () => {
            const confirmed = await showConfirm(
                t("builder_editor_reset_message"),
                {
                    title: t("builder_editor_reset_title"),
                    type: "warning",
                    confirmText: t("builder_editor_reset_action")
                }
            );
            if (confirmed) {
                updatePipelinePreview();
                log(t("builder_editor_synced_from_canvas"));
            }
        });
    }

    // Initialize line numbers
    updateYamlLineNumbers();
}

/**
 * 初始化分隔条拖拽功能
 */
function initBuilderResizer() {
    const resizer = els.builderResizer;
    if (!resizer) return;

    const splitView = document.querySelector('.canvas-split-layout');

    if (!splitView) return;

    let isResizing = false;
    let startX = 0;
    let startCanvasWidth = 0;

    resizer.addEventListener('mousedown', (e) => {
        isResizing = true;
        startX = e.clientX;
        startCanvasWidth = getWorkspacePaneWidth();

        resizer.classList.add('dragging');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';

        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;

        const dx = e.clientX - startX;
        const maxWidth = Math.max(
            MIN_WORKSPACE_PANE_WIDTH,
            splitView.offsetWidth - MIN_WORKSPACE_CONTENT_WIDTH
        );
        const newCanvasWidth = Math.min(Math.max(startCanvasWidth + dx, MIN_WORKSPACE_PANE_WIDTH), maxWidth);
        setWorkspacePaneWidth(newCanvasWidth);
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            resizer.classList.remove('dragging');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });
}

/**
 * 初始化可折叠的 Console
 */
function initConsole() {
    const consoleEl = els.canvasConsole;
    const toggleEl = els.consoleToggle;

    if (!consoleEl || !toggleEl) return;

    // Collapsed by default
    consoleEl.classList.add('collapsed');

    // Click to toggle collapse state
    toggleEl.addEventListener('click', () => {
        consoleEl.classList.toggle('collapsed');
    });
}

// Infer initial view from URL (supports /chat, /settings or ?view=chat)
function getInitialModeFromUrl() {
    const path = (window.location.pathname || "").toLowerCase();
    const hash = (window.location.hash || "").toLowerCase();
    const params = new URLSearchParams(window.location.search);
    const viewParam = (params.get('view') || "").toLowerCase();

    if (viewParam === 'chat' || path.startsWith('/chat') || hash === '#chat') {
        return Modes.CHAT;
    }
    if (
        viewParam === 'builder' ||
        viewParam === 'config' ||
        viewParam === 'settings' ||
        path.startsWith('/settings') ||
        path.startsWith('/config')
    ) {
        return Modes.BUILDER;
    }
    return null;
}

// Sync address bar based on current view, convenient for direct URL access
function updateUrlForView(mode, replace = false) {
    let targetPath = null;
    const currentPath = window.location.pathname;

    if (mode === Modes.CHAT) {
        targetPath = "/";
    } else if (mode === Modes.BUILDER) {
        targetPath = "/settings";
    }

    if (!targetPath || currentPath === targetPath) return;

    const url = new URL(window.location.href);
    const method = replace ? 'replaceState' : 'pushState';
    window.history[method](null, "", targetPath + url.search + url.hash);
}

function setMode(mode) {
    state.mode = mode;
    if (els.pipelineForm) els.pipelineForm.classList.toggle("d-none", mode !== Modes.BUILDER);
    if (els.parameterPanel) els.parameterPanel.classList.toggle("d-none", mode !== Modes.PARAMETERS);
    if (els.chatView) els.chatView.classList.toggle("d-none", mode !== Modes.CHAT);

    // AI assistant only shown in admin interface (non-Chat mode)
    const aiContainer = document.getElementById('ai-assistant-container');
    if (aiContainer) {
        aiContainer.classList.toggle('d-none', mode === Modes.CHAT);
    }

    // Sync background task control visibility (only shown in Chat mode)
    scheduleBackgroundTaskVisibilitySync();
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
    if (!tools.length) {
        const option = document.createElement("option");
        option.textContent = server ? t("builder_node_no_tools") : t("builder_node_select_server");
        select.appendChild(option);
        select.disabled = true;
        nodePickerState.tool = null;
        return;
    }
    select.disabled = false; if (!nodePickerState.tool) nodePickerState.tool = tools[0].tool;
    tools.forEach(t => { const option = document.createElement("option"); option.value = t.tool; option.textContent = t.tool; select.appendChild(option); });
    select.value = nodePickerState.tool || "";
}
function populateNodePickerServers() {
    if (!els.nodePickerServer) return;
    const select = els.nodePickerServer; select.innerHTML = "";
    const servers = state.toolCatalog.order || [];
    if (!servers.length) {
        const option = document.createElement("option");
        option.textContent = t("builder_node_no_servers");
        select.appendChild(option);
        select.disabled = true;
        return;
    }
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
            case "tool":
                if (!nodePickerState.server || !nodePickerState.tool) throw new Error(t("builder_node_select_tool_error"));
                insertStepAt(location, index, `${nodePickerState.server}.${nodePickerState.tool}`);
                break;
            case "loop": const times = Math.max(1, Number(nodePickerState.loopTimes) || 1); const p = insertStepAt(location, index, { loop: { times, steps: [] } }); enterStructureContext("loop", p); break;
            case "branch": const cases = (nodePickerState.branchCases || "").split(",").map(c => c.trim()).filter(B => B); const step = { branch: { router: [], branches: {} } }; (cases.length ? cases : ["c1", "c2"]).forEach(k => step.branch.branches[k] = []); const p2 = insertStepAt(location, index, step); enterStructureContext("branch", p2); break;
            case "custom":
                if (!nodePickerState.customValue) throw new Error(t("builder_node_custom_empty_error"));
                insertStepAt(location, index, parseStepInput(nodePickerState.customValue));
                break;
        }
        getNodePickerModal()?.hide(); pendingInsert = null;
    } catch (e) { showNodePickerError(e.message); }
}

function markPipelineDirty() {
    state.isBuilt = false;
    state.parametersReady = false;
    markUnsavedChanges();

    // If current Pipeline is modified, force invalidate its Engine Session
    const currentName = state.selectedPipeline;
    if (currentName && state.chat.activeEngines[currentName]) {
        const sid = state.chat.activeEngines[currentName];
        // Try to stop in background
        fetchJSON(`/api/pipelines/demo/stop`, {
            method: "POST", body: JSON.stringify({ session_id: sid })
        }).catch(() => { });

        delete state.chat.activeEngines[currentName];
        if (state.chat.engineSessionId === sid) {
            state.chat.engineSessionId = null;
        }
        persistActiveEngines();
        log(formatTemplate(t("builder_log_engine_invalidated"), { name: currentName }));
    }

    if (state.mode !== Modes.BUILDER) setMode(Modes.BUILDER);
    updateActionButtons();
}
function setSteps(steps, options = {}) {
    const { markUnsaved = false, skipPreview = false, snapshotContent } = options;
    state.steps = Array.isArray(steps) ? cloneDeep(steps) : [];

    // Sync pipeline config, preserve original extra fields (like _raw_yaml)
    const derivedServers = buildServersMapping(state.steps);
    if (!state.pipelineConfig || typeof state.pipelineConfig !== "object") {
        state.pipelineConfig = {
            servers: cloneDeep(derivedServers),
            pipeline: cloneDeep(state.steps),
        };
    } else {
        state.pipelineConfig.pipeline = cloneDeep(state.steps);
        state.pipelineConfig.servers = { ...derivedServers, ...(state.pipelineConfig.servers || {}) };
    }

    state.parameterData = null;
    state.isBuilt = false;
    state.parametersReady = false;

    if (markUnsaved) {
        markUnsavedChanges();
    }

    resetChatSession();

    resetContextStack();
    renderSteps();
    if (!skipPreview) updatePipelinePreview();
    updateActionButtons();

    if (!markUnsaved) {
        // After programmatic updates, align saved snapshot if appropriate
        const currentYaml = snapshotContent !== undefined
            ? snapshotContent
            : (els.yamlEditor ? els.yamlEditor.value : yamlStringify(buildPipelinePayloadForPreview()));
        snapshotSavedYaml(currentYaml);
    }
}
function updateActionButtons() {
    // Control buttons in Parameter panel (unchanged)
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
function parseStepInput(raw) {
    const trimmed = (raw || "").trim();
    if (!trimmed) throw new Error(t("builder_parse_empty_error"));
    if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))) return JSON.parse(trimmed);
    return trimmed;
}
function createInsertControl(location, insertIndex, { prominent = false, compact = false } = {}) {
    const holder = document.createElement("div");
    holder.className = "flow-insert-control";
    if (prominent) holder.classList.add("prominent");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "flow-insert-button";
    button.title = t("builder_node_insert_here");
    button.innerHTML = `<span>+</span><span>${t("builder_node_add")}</span>`;
    button.addEventListener("click", () => {
        const pendingLocation = createLocation((location.segments || []).map((seg) => ({ ...seg })));
        openNodePicker(pendingLocation, insertIndex);
    });
    holder.appendChild(button);
    return holder;
}

// ... (Render Helpers - Tool/Loop/Branch Nodes - keep same) ...
function truncateText(text, maxLen = 160) {
    if (!text) return "";
    return text.length > maxLen ? text.slice(0, maxLen - 3) + "..." : text;
}

function summarizeToolConfig(config) {
    if (!config || typeof config !== "object") return "";
    const parts = [];
    if (config.input) parts.push(`input: ${truncateText(JSON.stringify(config.input))}`);
    if (config.output) parts.push(`output: ${truncateText(JSON.stringify(config.output))}`);
    if (config.branch) parts.push("branch");
    if (config.loop) parts.push("loop");
    if (!parts.length) {
        const raw = truncateText(JSON.stringify(config));
        return raw === "{}" ? "" : raw;
    }
    return parts.join(" | ");
}

function renderToolNode(identifier, stepPath, meta = {}) {
    const raw = String(identifier || "").trim();
    const parts = raw ? raw.split(".") : [];
    const serverName = parts.length > 1 ? parts[0] : "custom";
    const toolName = parts.length > 1 ? parts.slice(1).join(".") : raw;

    const card = document.createElement("div"); card.className = "flow-node";
    const header = document.createElement("div"); header.className = "flow-node-header";
    const serverLabel = document.createElement("div"); serverLabel.className = "flow-node-server"; serverLabel.textContent = serverName;
    serverLabel.title = serverName;
    serverLabel.dataset.module = serverName.toLowerCase();
    header.appendChild(serverLabel);

    const body = document.createElement("div"); body.className = "flow-node-body"; body.textContent = toolName || raw;
    const actions = document.createElement("div"); actions.className = "step-actions";
    const removeBtn = document.createElement("button");
    removeBtn.className = "btn btn-outline-danger btn-sm";
    removeBtn.textContent = t("common_delete");
    removeBtn.onclick = (e) => { e.stopPropagation(); removeStep(stepPath); };
    actions.append(removeBtn); card.append(header, body, actions); return card;
}
function renderLoopNode(step, parentLocation, index) {
    const loopSteps = Array.isArray(step.loop.steps) ? step.loop.steps : [];

    const loopLocation = createLocation([...(parentLocation.segments || []), { type: "loop", index }]);
    const container = document.createElement("div"); container.className = "loop-container";
    const header = document.createElement("div"); header.className = "loop-header";
    const title = document.createElement("h6");
    title.textContent = formatTemplate(t("builder_loop_title"), { times: step.loop.times });
    const enterBtn = document.createElement("button");
    enterBtn.className = "btn btn-sm btn-link text-decoration-none p-0";
    enterBtn.textContent = t("builder_open_context");
    enterBtn.onclick = () => setActiveLocation(loopLocation);
    header.append(title, enterBtn);
    const actions = document.createElement("div"); actions.className = "mt-2 d-flex justify-content-end gap-2";
    const editBtn = document.createElement("button");
    editBtn.className = "btn btn-sm btn-outline-secondary border-0";
    editBtn.textContent = t("common_edit");
    editBtn.onclick = () => openStepEditor(createStepPath(parentLocation, index));
    const delBtn = document.createElement("button");
    delBtn.className = "btn btn-sm btn-outline-danger border-0";
    delBtn.textContent = t("common_delete");
    delBtn.onclick = () => removeStep(createStepPath(parentLocation, index));
    actions.append(editBtn, delBtn);

    const list = renderStepList(loopSteps, loopLocation, { placeholderText: t("builder_empty_loop"), compact: true });

    container.append(header, list, actions); if (locationsEqual(loopLocation, getActiveLocation())) container.classList.add("active"); return container;
}
function renderBranchNode(step, parentLocation, index) {
    step.branch.router = Array.isArray(step.branch.router) ? step.branch.router : [];
    step.branch.branches = (step.branch.branches && typeof step.branch.branches === 'object') ? step.branch.branches : {};

    const branchBase = createLocation([...(parentLocation.segments || []), { type: "branch", index, section: "router" }]);
    const container = document.createElement("div"); container.className = "branch-container";
    const header = document.createElement("div");
    header.className = "branch-header";
    header.innerHTML = `<h6>${t("builder_branch")}</h6>`;
    const enterBtn = document.createElement("button");
    enterBtn.className = "btn btn-sm btn-link text-decoration-none p-0";
    enterBtn.textContent = t("builder_open_router");
    enterBtn.onclick = () => setActiveLocation(branchBase);
    header.appendChild(enterBtn);
    const routerDiv = document.createElement("div"); routerDiv.className = "branch-router " + (locationsEqual(branchBase, getActiveLocation()) ? "active" : "");
    routerDiv.appendChild(renderStepList(step.branch.router, branchBase, { placeholderText: t("builder_router_logic"), compact: true }));
    const casesDiv = document.createElement("div"); casesDiv.className = "branch-cases mt-3";
    Object.keys(step.branch.branches).forEach(k => {
        const loc = createLocation([...(parentLocation.segments || []), { type: "branch", index, section: "branch", branchKey: k }]);
        const cCard = document.createElement("div"); cCard.className = "branch-case " + (locationsEqual(loc, getActiveLocation()) ? "active" : "");
        const cHeader = document.createElement("div"); cHeader.className = "d-flex justify-content-between mb-2";
        const cTitle = document.createElement("span");
        cTitle.className = "fw-bold text-xs";
        cTitle.textContent = formatTemplate(t("builder_case_label"), { name: k });
        const cBtn = document.createElement("button");
        cBtn.className = "btn btn-link btn-sm p-0 text-decoration-none";
        cBtn.textContent = t("common_open");
        cBtn.onclick = () => setActiveLocation(loc);
        cHeader.append(cTitle, cBtn);
        cCard.append(cHeader, renderStepList(step.branch.branches[k], loc, { placeholderText: t("builder_empty_case"), compact: true }));
        casesDiv.appendChild(cCard);
    });
    const actions = document.createElement("div"); actions.className = "mt-2 d-flex justify-content-end gap-2";
    const addBtn = document.createElement("button");
    addBtn.className = "btn btn-sm btn-light border";
    addBtn.textContent = t("builder_add_case");
    addBtn.onclick = () => addBranchCase(parentLocation, index);
    const delBtn = document.createElement("button");
    delBtn.className = "btn btn-sm btn-text text-danger";
    delBtn.textContent = t("builder_delete_branch");
    delBtn.onclick = () => removeStep(createStepPath(parentLocation, index));
    actions.append(addBtn, delBtn); container.append(header, routerDiv, casesDiv, actions); return container;
}
function renderStepNode(step, parentLocation, index) {
    const stepPath = createStepPath(parentLocation, index);
    if (typeof step === "string") return renderToolNode(step, stepPath);
    if (isToolConfigStep(step)) {
        const toolName = Object.keys(step)[0];
        return renderToolNode(toolName, stepPath);
    }
    if (step && typeof step === "object" && step.loop) return renderLoopNode(step, parentLocation, index);
    if (step && typeof step === "object" && step.branch) return renderBranchNode(step, parentLocation, index);
    const card = renderToolNode(t("builder_custom_object"), stepPath, { description: truncateText(JSON.stringify(step)) });
    return card;
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
        const btn = document.createElement("button"); btn.className = `btn btn-sm rounded-pill ${idx === state.contextStack.length - 1 ? "btn-dark" : "btn-light border"}`;
        btn.textContent = ctxLabel(loc, idx); btn.onclick = () => setActiveLocation(createLocation(loc.segments || []));
        breadcrumb.appendChild(btn); if (idx < state.contextStack.length - 1) { const sep = document.createElement("span"); sep.className = "text-muted small"; sep.textContent = "/"; breadcrumb.appendChild(sep); }
    });
    els.contextControls.appendChild(breadcrumb);
    const active = getActiveLocation(); const kind = getContextKind(active);
    if (kind !== "root") {
        const exitBtn = document.createElement("button");
        exitBtn.className = "btn btn-sm btn-link text-danger text-decoration-none mt-2";
        exitBtn.textContent = t("builder_exit_context");
        exitBtn.onclick = () => { setActiveLocation(createLocation((active.segments || []).slice(0, -1))); };
        els.contextControls.appendChild(exitBtn);
    }
}
function ctxLabel(location, idx) {
    if (idx === 0) return t("builder_context_root");
    const last = (location.segments || [])[location.segments.length - 1];
    if (!last) return t("builder_context_root");
    if (last.type === "loop") return t("builder_context_loop");
    if (last.type === "branch") {
        return last.section === "router"
            ? t("builder_context_router")
            : formatTemplate(t("builder_context_case"), { name: last.branchKey });
    }
    return t("builder_context_node");
}
function addBranchCase(parentLocation, branchIndex) {
    const steps = resolveSteps(parentLocation); const entry = steps[branchIndex]; if (!entry?.branch) return;
    entry.branch.branches = entry.branch.branches || {}; let c = Object.keys(entry.branch.branches).length + 1; let key = `case${c}`; while (entry.branch.branches[key]) { c++; key = `case${c}`; }
    entry.branch.branches[key] = []; markPipelineDirty(); const segs = [...(parentLocation.segments || []), { type: "branch", index: branchIndex, section: "branch", branchKey: key }]; setActiveLocation(createLocation(segs));
}
function enterStructureContext(type, stepPath, announce = true) {
    if (!stepPath) return; const segs = [...(stepPath.parentSegments || []), { type, index: stepPath.index, ...(type === "branch" ? { section: "router" } : {}) }]; setActiveLocation(createLocation(segs));
}

async function refreshPipelines() {
    const pipelines = await fetchJSON("/api/pipelines");
    renderPipelineMenu(pipelines);
    return pipelines;
}
function renderPipelineMenu(items) {
    els.pipelineMenu.innerHTML = "";
    if (!items.length) {
        const li = document.createElement("li");
        li.innerHTML = `<span class="dropdown-item text-muted small">${t("builder_no_pipelines")}</span>`;
        els.pipelineMenu.appendChild(li);
        return;
    }

    const sortedItems = items.slice().sort((a, b) =>
        (a.name || "").localeCompare(b.name || "", "en", { sensitivity: "base" })
    );

    sortedItems.forEach(i => {
        const li = document.createElement("li");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "dropdown-item small pipeline-menu-item d-flex align-items-center justify-content-between gap-2";

        const nameSpan = document.createElement("span");
        nameSpan.textContent = i.name;
        btn.appendChild(nameSpan);

        if (i.is_ready) {
            const readyDot = document.createElement("span");
            readyDot.className = "pipeline-ready-dot";
            readyDot.title = t("builder_pipeline_built_title");
            btn.appendChild(readyDot);
        }

        btn.onclick = () => { loadPipeline(i.name); btn.blur(); };
        li.appendChild(btn);
        els.pipelineMenu.appendChild(li);
    });
}
async function loadPipeline(name, options = {}) {
    const { ignoreUnsaved = false } = options;
    if (!ignoreUnsaved && state.unsavedChanges && state.selectedPipeline && state.selectedPipeline !== name) {
        const proceed = await confirmUnsavedChanges(t("builder_action_switch_pipeline"));
        if (!proceed) return;
    }

    try {
        console.log(`[UI] Loading pipeline: ${name}`);
        const cfg = await fetchJSON(`/api/pipelines/${encodeURIComponent(name)}`);

        state.selectedPipeline = name;
        els.name.value = name;

        // Store complete pipeline configuration (including servers)
        state.pipelineConfig = cfg;

        const rawYaml = typeof cfg._raw_yaml === "string" ? cfg._raw_yaml : null;

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

        // If backend returns raw YAML, directly inject into editor to ensure consistency with file
        if (els.yamlEditor && rawYaml !== null) {
            yamlEditorSyncLock = true;
            els.yamlEditor.value = rawYaml;
            updateYamlLineNumbers();
            yamlEditorSyncLock = false;
        }

        setSteps(safeSteps, { skipPreview: Boolean(rawYaml), snapshotContent: rawYaml || undefined });
        clearUnsavedChanges();
        showYamlError(null);
        setYamlSyncStatus('synced');

        if (els.pipelineDropdownBtn) els.pipelineDropdownBtn.textContent = name;
        setHeroPipelineLabel(name);

        // Automatically check if Ready (key to skipping Build)
        checkPipelineReadiness(name);
        updateAIContextBanner('pipeline-load');

    } catch (err) {
        log(formatTemplate(t("builder_load_pipeline_failed"), { error: err.message }));
        console.error(err);
    }
}

// Helper function: Check if Pipeline has configured parameters
async function checkPipelineReadiness(name) {
    try {
        // Try to get parameters
        // Note: backend get_parameters will error if file doesn't exist, so need to catch
        // Or we could directly use is_ready field from list_pipelines, but sending a request here ensures it's latest
        // For performance, could also use list interface. But directly fetching parameters is most reliable.

        const params = await fetchJSON(`/api/pipelines/${encodeURIComponent(name)}/parameters`);

        // If successfully got parameters
        state.parameterData = params;
        state.isBuilt = true;         // Consider as built
        state.parametersReady = true; // Consider parameters as ready

        log(formatTemplate(t("builder_pipeline_ready"), { name }));
        updateActionButtons(); // This will enable "Enter Chat Mode" button

    } catch (e) {
        // Parameter file doesn't exist, need to Build
        state.isBuilt = false;
        state.parametersReady = false;
        state.parameterData = null;
        updateActionButtons(); // Disable Chat button
    }
}



async function handleSubmit(e) {
    if (e) e.preventDefault();
    const name = els.name.value.trim();
    if (!name) return log(t("builder_pipeline_name_required"));

    // Get YAML content from editor and validate
    const validation = await validateYamlEditorContent({ showModalOnError: true });
    let yamlContent = validation.valid ? (validation.content || '') : null;
    if (yamlContent === null) {
        log(t("builder_save_aborted_yaml_error"));
        return;
    }

    // Sync latest parsed result to memory for subsequent build/preview
    if (validation.parsed) {
        const parsedCfg = (validation.parsed && typeof validation.parsed === "object" && !Array.isArray(validation.parsed))
            ? { ...validation.parsed }
            : { pipeline: validation.steps };
        parsedCfg._raw_yaml = yamlContent;
        state.pipelineConfig = parsedCfg;
    }

    // If editor has content, use YAML API to save directly
    if (yamlContent) {
        try {
            const res = await fetch(`/api/pipelines/${encodeURIComponent(name)}/yaml`, {
                method: "PUT",
                headers: { 'Content-Type': 'text/plain; charset=utf-8' },
                body: yamlContent,
            });
            if (!res.ok) throw new Error(formatTemplate(t("builder_save_failed_status"), { status: res.status }));
            await res.json();

            state.selectedPipeline = name;
            refreshPipelines();
            log(t("builder_pipeline_saved"));
            setYamlSyncStatus('synced');
            snapshotSavedYaml(yamlContent);

            // After successful save, automatically sync canvas (don't overwrite editor), but maintain "saved" state
            await syncYamlToCanvasOnly({ markUnsaved: false });
        } catch (err) {
            const msg = err?.message || t("common_unknown_error");
            log(formatTemplate(t("builder_error_message"), { error: msg }));
            showYamlError(msg);
            showModal(formatTemplate(t("builder_save_failed_message"), { error: msg }), { title: t("builder_save_error_title"), type: "error" });
        }
    } else {
        // Empty content, save using JSON method
        saveWithJson(name);
    }
}

function saveWithJson(name) {
    fetchJSON("/api/pipelines", {
        method: "POST",
        body: JSON.stringify({ name, pipeline: cloneDeep(state.steps) })
    })
        .then(s => {
            state.selectedPipeline = s.name || name;
            refreshPipelines();
            log(t("builder_pipeline_saved"));
            setYamlSyncStatus('synced');
            snapshotSavedYaml(yamlStringify(buildPipelinePayloadForPreview()));
            loadPipeline(s.name || name);
        })
        .catch(e => log(e.message));
}
async function buildSelectedPipeline() {
    if (state.unsavedChanges) {
        showModal(t("builder_save_before_build"), { title: t("builder_unsaved_changes_title"), type: "warning" });
        return;
    }
    if (!state.selectedPipeline) return log(t("builder_save_pipeline_first"));
    expandConsole();
    setBuildButtonState("running");
    log(formatTemplate(t("builder_building_pipeline"), { name: state.selectedPipeline }));
    try {
        await fetchJSON(`/api/pipelines/${encodeURIComponent(state.selectedPipeline)}/build`, { method: "POST" });

        state.isBuilt = true;
        state.parametersReady = false;
        updateActionButtons();
        setBuildButtonState("success");
        log(t("builder_pipeline_built_log"));

        // Load parameter data
        try {
            state.parameterData = cloneDeep(await fetchJSON(`/api/pipelines/${encodeURIComponent(state.selectedPipeline)}/parameters`));
        } catch (e) {
            log(formatTemplate(t("builder_load_parameters_failed"), { error: e.message }));
        }

        // Switch to Parameters panel
        if (typeof switchWorkspaceMode === 'function') {
            switchWorkspaceMode('parameters');
        }
    } catch (e) {
        setBuildButtonState("error");
        log(formatTemplate(t("builder_build_failed_message"), { error: e.message }));
        showModal(formatTemplate(t("builder_build_failed_message"), { error: e.message }), { title: t("builder_build_error_title"), type: "error" });
    }
}
async function deleteSelectedPipeline() {
    if (!state.selectedPipeline) return;
    const confirmed = await showConfirm(t("builder_delete_pipeline_confirm"), {
        title: t("builder_delete_pipeline_title"),
        type: "warning",
        confirmText: t("common_delete"),
        danger: true
    });
    if (!confirmed) return;
    fetchJSON(`/api/pipelines/${encodeURIComponent(state.selectedPipeline)}`, { method: "DELETE" })
        .then(async () => {
            state.selectedPipeline = null;
            els.name.value = "";
            setSteps([]);
            clearUnsavedChanges();
            showYamlError(null);
            setYamlSyncStatus('synced');
            const list = await refreshPipelines();
            if (Array.isArray(list) && list.length > 0) {
                loadPipeline(list[0].name, { ignoreUnsaved: true });
            }
        }).catch(e => log(e.message));
}

// Trigger rename when Pipeline name input loses focus
async function handlePipelineNameBlur() {
    const newName = els.name.value.trim();

    // If no pipeline selected or name unchanged, don't process
    if (!state.selectedPipeline || newName === state.selectedPipeline) {
        return;
    }

    // If name is empty, restore original name
    if (!newName) {
        els.name.value = state.selectedPipeline;
        return;
    }

    // Validate name format
    if (!/^[a-zA-Z_][a-zA-Z0-9_-]*$/.test(newName)) {
        log(t("builder_invalid_pipeline_name"));
        els.name.value = state.selectedPipeline;
        return;
    }

    try {
        await fetchJSON(`/api/pipelines/${encodeURIComponent(state.selectedPipeline)}/rename`, {
            method: "POST",
            body: JSON.stringify({ new_name: newName })
        });

        state.selectedPipeline = newName;
        await refreshPipelines();
        log(formatTemplate(t("builder_pipeline_renamed"), { name: newName }));
    } catch (e) {
        log(`Rename failed: ${e.message}`);
        els.name.value = state.selectedPipeline; // Restore original name
    }
}
function flattenParameters(obj, prefix = "") {
    const entries = []; if (!obj || typeof obj !== "object") return entries;
    Object.keys(obj).sort().forEach(key => {
        const path = prefix ? `${prefix}.${key}` : key; const val = obj[key];
        if (val !== null && typeof val === "object" && !Array.isArray(val)) entries.push(...flattenParameters(val, path));
        else entries.push({ path, value: val, type: Array.isArray(val) ? "array" : (val === null ? "null" : typeof val) });
    }); return entries;
}

// Filter parameters for simplified display
// For generation-type servers: hide backend, show only openai backend_configs, extra_params, sampling_params, system_prompt
// For retriever-type servers: hide backend, show only openai backend_configs, query_instruction, top_k
function filterServerParameters(entries) {
    // If simplified mode is disabled, return all entries with displayPath
    if (!state.simplifiedParams) {
        return entries.map(e => ({ ...e, displayPath: e.path }));
    }
    
    // Get server type mapping from pipeline config
    const servers = (state.pipelineConfig && state.pipelineConfig.servers) || {};
    
    // Build sets of server names that map to generation or retriever types
    const generationServers = new Set();
    const retrieverServers = new Set();
    
    Object.entries(servers).forEach(([name, path]) => {
        if (typeof path === 'string') {
            if (path === 'servers/generation' || path.endsWith('/generation')) {
                generationServers.add(name);
            } else if (path === 'servers/retriever' || path.endsWith('/retriever')) {
                retrieverServers.add(name);
            }
        }
    });
    
    // Helper function to get server name from path
    function getServerName(path) {
        const parts = path.split('.');
        return parts[0];
    }
    
    // Helper function to check if path matches generation-type allowed patterns
    function isGenerationAllowed(serverName, path) {
        const patterns = [
            new RegExp(`^${serverName}\\.backend_configs\\.openai\\.`),
            new RegExp(`^${serverName}\\.extra_params\\.`),
            new RegExp(`^${serverName}\\.sampling_params\\.`),
            new RegExp(`^${serverName}\\.system_prompt$`)
        ];
        return patterns.some(pattern => pattern.test(path));
    }
    
    // Helper function to check if path matches retriever-type allowed patterns
    function isRetrieverAllowed(serverName, path) {
        const patterns = [
            new RegExp(`^${serverName}\\.backend_configs\\.openai\\.`),
            new RegExp(`^${serverName}\\.query_instruction$`),
            new RegExp(`^${serverName}\\.top_k$`)
        ];
        return patterns.some(pattern => pattern.test(path));
    }
    
    return entries.filter(e => {
        const path = e.path;
        const serverName = getServerName(path);
        
        // For generation-type server parameters
        if (generationServers.has(serverName)) {
            // Skip the backend field itself
            if (path === `${serverName}.backend`) return false;
            // Skip non-openai backend_configs
            if (path.startsWith(`${serverName}.backend_configs.`) && !path.startsWith(`${serverName}.backend_configs.openai.`)) return false;
            // Check if matches allowed patterns
            return isGenerationAllowed(serverName, path);
        }
        
        // For retriever-type server parameters
        if (retrieverServers.has(serverName)) {
            // Skip the backend field itself
            if (path === `${serverName}.backend`) return false;
            // Skip non-openai backend_configs
            if (path.startsWith(`${serverName}.backend_configs.`) && !path.startsWith(`${serverName}.backend_configs.openai.`)) return false;
            // Skip index_backend related fields
            if (path === `${serverName}.index_backend`) return false;
            if (path.startsWith(`${serverName}.index_backend_configs.`)) return false;
            // Skip other non-essential fields
            const skipPattern = new RegExp(`^${serverName}\\.(batch_size|collection_name|corpus_path|gpu_ids|is_demo|is_multimodal|model_name_or_path)$`);
            if (skipPattern.test(path)) return false;
            // Check if matches allowed patterns
            return isRetrieverAllowed(serverName, path);
        }
        
        // Allow all other parameters
        return true;
    }).map(e => {
        // Simplify display path by removing 'openai' level from backend_configs
        let simplifiedPath = e.path;
        if (e.path.includes('.backend_configs.openai.')) {
            simplifiedPath = e.path.replace('.backend_configs.openai.', '.backend_configs.');
        }
        return { ...e, displayPath: simplifiedPath };
    });
}
function setNestedValue(obj, path, val) {
    const p = path.split("."); let c = obj; for (let i = 0; i < p.length - 1; i++) { if (!c[p[i]]) c[p[i]] = {}; c = c[p[i]]; } c[p[p.length - 1]] = val;
}

// Sync simplified parameter toggle state between the two toggle switches
function syncSimplifiedToggles() {
    const overlayToggle = document.getElementById('parameter-simplified-toggle');
    const inlineToggle = document.getElementById('params-simplified-toggle');
    if (overlayToggle) overlayToggle.checked = state.simplifiedParams;
    if (inlineToggle) inlineToggle.checked = state.simplifiedParams;
}

function renderParameterForm() {
    const container = els.parameterForm;
    const navContainer = document.getElementById('parameter-nav');
    container.innerHTML = "";
    if (navContainer) navContainer.innerHTML = "";

    if (!state.parameterData || typeof state.parameterData !== "object") {
        container.innerHTML = '<div class="parameter-empty"><p>No parameters available for configuration.</p></div>';
        return;
    }

    let entries = flattenParameters(state.parameterData).filter(e => !/^benchmark(\.|$)/i.test(e.path));
    // Apply server-specific parameter filtering for simplified display
    entries = filterServerParameters(entries);
    if (!entries.length) {
        container.innerHTML = '<div class="parameter-empty"><p>The current Pipeline has no editable parameters.</p></div>';
        return;
    }

    // Group parameters by server
    const grouped = {};
    entries.forEach(e => {
        const parts = (e.displayPath || e.path).split('.');
        const serverName = parts[0].toUpperCase();
        if (!grouped[serverName]) grouped[serverName] = [];
        grouped[serverName].push({
            ...e,
            displayPath: parts.slice(1).join('.') || parts[0], // Remove server prefix
            fullPath: e.path
        });
    });

    const serverNames = Object.keys(grouped).sort();

    // Render left navigation
    if (navContainer) {
        serverNames.forEach((serverName, idx) => {
            const navItem = document.createElement('button');
            navItem.className = 'parameter-nav-item';
            navItem.dataset.server = serverName;
            navItem.innerHTML = `
                <span class="nav-item-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
                </span>
                <span class="nav-item-name">${escapeHtml(serverName)}</span>
                <span class="nav-item-count">${grouped[serverName].length}</span>
            `;
            navItem.onclick = () => {
                // Scroll to corresponding section
                const section = document.getElementById(`param-section-${serverName}`);
                if (section) {
                    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    // Expand this section
                    section.classList.add('expanded');
                }
                // Highlight current navigation
                document.querySelectorAll('.parameter-nav-item').forEach(n => n.classList.remove('active'));
                navItem.classList.add('active');
            };
            navContainer.appendChild(navItem);
        });
    }

    // Render right parameter area
    serverNames.forEach((serverName, idx) => {
        const section = document.createElement('div');
        section.className = 'parameter-section expanded'; // Expanded by default
        section.id = `param-section-${serverName}`;

        // Section Header (collapsible)
        const header = document.createElement('div');
        header.className = 'parameter-section-header';
        header.innerHTML = `
            <div class="section-header-left">
                <svg class="section-chevron" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
                <span class="section-title">${escapeHtml(serverName)}</span>
                <span class="section-badge">${grouped[serverName].length} parameters</span>
            </div>
        `;
        header.onclick = () => {
            section.classList.toggle('expanded');
        };

        // Section Content
        const content = document.createElement('div');
        content.className = 'parameter-section-content';

        const grid = document.createElement('div');
        grid.className = 'parameter-grid';

        grouped[serverName].forEach(e => {
            const grp = document.createElement('div');
            grp.className = 'parameter-field';
            const isComplex = e.type === "array" || e.type === "object";
            if (isComplex) grp.classList.add('full-width');

            const label = document.createElement('label');
            label.className = 'parameter-label';
            label.textContent = e.displayPath;
            label.title = e.fullPath; // Full path as tooltip

            let ctrl;
            if (isComplex) {
                ctrl = document.createElement("textarea");
                ctrl.rows = 4;
                ctrl.value = JSON.stringify(e.value, null, 2);
            } else {
                ctrl = document.createElement("input");
                ctrl.type = "text";
                ctrl.value = String(e.value ?? "");
            }
            ctrl.className = "parameter-input";
            ctrl.onchange = (ev) => {
                let val = ev.target.value;
                if (e.type === "number") val = Number(val);
                if (e.type === "boolean") val = val.toLowerCase() === "true";
                try { if (isComplex) val = JSON.parse(val); } catch (err) { }
                e.value = val;
                setNestedValue(state.parameterData, e.fullPath, val);
                state.parametersReady = false;
                updateActionButtons();
            };

            grp.append(label, ctrl);
            grid.appendChild(grp);
        });

        content.appendChild(grid);
        section.append(header, content);
        container.appendChild(section);
    });
}
async function showParameterPanel(force = false) {
    if (!state.isBuilt) return log(t("builder_build_pipeline_first"));
    if (force || !state.parameterData) { try { state.parameterData = cloneDeep(await fetchJSON(`/api/pipelines/${encodeURIComponent(state.selectedPipeline)}/parameters`)); } catch (e) { return log(e.message); } }
    renderParameterForm(); setMode(Modes.PARAMETERS);
}
function saveParameterForm() { persistParameterData(); }
async function refreshTools() {
    const tools = await fetchJSON("/api/tools"); const grouped = {};
    tools.forEach(t => { const s = t.server || "Unnamed"; if (!grouped[s]) grouped[s] = []; grouped[s].push(t); });
    state.toolCatalog = { order: Object.keys(grouped).sort(), byServer: grouped }; nodePickerState.server = null;
}

function bindEvents() {
    // Save button click event (compatible with form submit and button click)
    if (els.pipelineForm) {
        els.pipelineForm.addEventListener("submit", handleSubmit);
    }
    if (els.savePipeline) {
        els.savePipeline.addEventListener("click", (e) => {
            e.preventDefault();
            handleSubmit(e);
        });
    }

    if (els.clearSteps) {
        els.clearSteps.addEventListener("click", async () => {
            const confirmed = await showConfirm(t("builder_clear_steps_confirm"), {
                title: t("builder_clear_steps_title"),
                type: "warning",
                confirmText: t("builder_clear_steps_action")
            });
            if (confirmed) setSteps([], { markUnsaved: true });
        });
    }
    els.buildPipeline.addEventListener("click", buildSelectedPipeline);
    els.deletePipeline.addEventListener("click", deleteSelectedPipeline);

    // Trigger rename when Pipeline name input loses focus
    els.name.addEventListener("blur", handlePipelineNameBlur);

    if (els.newPipelineBtn) els.newPipelineBtn.addEventListener("click", createNewPipeline);

    if (els.parameterSave) els.parameterSave.onclick = saveParameterForm;
    if (els.parameterBack) els.parameterBack.onclick = () => setMode(Modes.BUILDER);
    if (els.parameterChat) els.parameterChat.onclick = safeOpenChatView;

    // Parameter simplified toggle - overlay panel
    const parameterSimplifiedToggle = document.getElementById('parameter-simplified-toggle');
    if (parameterSimplifiedToggle) {
        parameterSimplifiedToggle.addEventListener('change', (e) => {
            state.simplifiedParams = e.target.checked;
            syncSimplifiedToggles();
            renderParameterForm();
        });
    }

    // Parameter simplified toggle - inline panel  
    const paramsSimplifiedToggle = document.getElementById('params-simplified-toggle');
    if (paramsSimplifiedToggle) {
        paramsSimplifiedToggle.addEventListener('change', (e) => {
            state.simplifiedParams = e.target.checked;
            syncSimplifiedToggles();
            renderParameterFormInline();
        });
    }

    if (els.workspaceChatBtn) {
        els.workspaceChatBtn.onclick = safeOpenChatView;
    }

    if (els.kbBtn) {
        els.kbBtn.onclick = openKBView;
    }

    if (els.builderLogo) {
        els.builderLogo.onclick = (e) => { e.preventDefault(); setMode(Modes.BUILDER); };
    }

    if (els.chatLogoBtn) {
        els.chatLogoBtn.onclick = (e) => { e.preventDefault(); createNewChatSession(); };
    }

    if (els.chatSidebarToggleBtn && els.chatSidebar) {
        els.chatSidebarToggleBtn.onclick = () => {
            const isCollapsed = els.chatSidebar.classList.toggle("collapsed");
            localStorage.setItem("ultrarag_sidebar_collapsed", isCollapsed);
        };
    }

    if (els.refreshCollectionsBtn) {
        els.refreshCollectionsBtn.onclick = async () => {
            log("Manually refreshing collections...");

            // Add visual feedback (optional)
            els.refreshCollectionsBtn.disabled = true;
            els.refreshCollectionsBtn.innerHTML = '⟳'; // Use a rotating icon instead

            try {
                await refreshKBFiles(); // Call refresh function that already includes sync logic
            } finally {
                // Restore button state
                els.refreshCollectionsBtn.disabled = false;
                els.refreshCollectionsBtn.innerHTML = '↻';
            }
        };
    }

    if (els.idxMode) {
        els.idxMode.addEventListener("change", async () => {
            const mode = els.idxMode.value || "new";
            await syncIndexModeUI(mode, { forceFetch: true });
        });
    }

    setupSettingsMenu();

    if (els.chatForm) els.chatForm.onsubmit = handleChatSubmit;
    if (els.chatSend) els.chatSend.onclick = handleChatSubmit;

    // Support Shift+Enter for newline, Enter for submit
    // Fix: Add IME composition state detection to prevent accidental send during Chinese input
    if (els.chatInput) {
        let isComposing = false; // Track if IME composition is in progress

        // Listen to IME composition start event (pinyin input starts)
        els.chatInput.addEventListener('compositionstart', function () {
            isComposing = true;
        });

        // Listen to IME composition end event (pinyin input ends, character selected or pinyin confirmed)
        els.chatInput.addEventListener('compositionend', function () {
            isComposing = false;
        });

        els.chatInput.addEventListener('keydown', function (e) {
            // If IME composition is in progress, don't trigger send
            if (e.key === 'Enter' && !e.shiftKey && !isComposing) {
                e.preventDefault();
                handleChatSubmit(e);
            }
            // Shift+Enter allows default behavior (newline)
        });

        // Auto-adjust textarea height
        els.chatInput.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    }

    if (els.chatNewBtn) els.chatNewBtn.onclick = createNewChatSession;
    if (els.clearAllChats) els.clearAllChats.onclick = deleteAllChatSessions;
    if (els.demoToggleBtn) els.demoToggleBtn.onclick = toggleDemoSession;

    // Chat session right-click menu initialization
    const chatSessionContextMenu = document.getElementById('chat-session-context-menu');
    if (chatSessionContextMenu) {
        chatSessionContextMenu.addEventListener('click', (e) => {
            const btn = e.target.closest('.chat-session-context-item');
            const action = btn?.dataset?.action;
            const sessionId = chatSessionContextMenu.dataset.sessionId;
            if (!action || !sessionId) return;
            hideChatSessionContextMenu();
            if (action === 'rename') {
                renameChatSession(sessionId);
            } else if (action === 'delete') {
                deleteChatSession(sessionId);
            }
        });
    }
    document.addEventListener('click', hideChatSessionContextMenu);
    document.addEventListener('scroll', hideChatSessionContextMenu, true);

    if (els.kbBtn) els.kbBtn.onclick = openKBView;

    document.getElementById("step-editor-save").onclick = () => {
        if (!state.editingPath) return;
        try { setStepByPath(state.editingPath, parseStepInput(els.stepEditorValue.value)); closeStepEditor(); renderSteps(); updatePipelinePreview(); } catch (e) { log(e.message); }
    };
    document.getElementById("step-editor-cancel").onclick = closeStepEditor;
    if (els.refreshPipelines) {
        els.refreshPipelines.onclick = async () => {
            const canProceed = state.unsavedChanges ? await confirmUnsavedChanges(t("builder_action_refresh_pipeline_list")) : true;
            if (!canProceed) return;

            await refreshPipelines();

            // If current pipeline has no unsaved changes, reload after refresh to prevent occasional canvas blank
            if (state.selectedPipeline && !state.unsavedChanges) {
                loadPipeline(state.selectedPipeline, { ignoreUnsaved: true });
            }
        };
    }
    els.name.oninput = updatePipelinePreview;

    els.nodePickerTabs.forEach(t => t.onclick = () => setNodePickerMode(t.dataset.nodeMode));
    if (els.nodePickerServer) els.nodePickerServer.onchange = () => { nodePickerState.server = els.nodePickerServer.value; populateNodePickerTools(); };
    if (els.nodePickerTool) els.nodePickerTool.onchange = () => nodePickerState.tool = els.nodePickerTool.value;
    if (els.nodePickerBranchCases) els.nodePickerBranchCases.oninput = (e) => nodePickerState.branchCases = e.target.value;
    if (els.nodePickerLoopTimes) els.nodePickerLoopTimes.oninput = (e) => nodePickerState.loopTimes = e.target.value;
    if (els.nodePickerCustom) els.nodePickerCustom.oninput = (e) => nodePickerState.customValue = e.target.value;
    if (els.nodePickerConfirm) els.nodePickerConfirm.onclick = handleNodePickerConfirm;

    if (els.milvusDialog) {
        els.milvusDialog.addEventListener('close', () => {
            if (els.modalTargetDb) {
                els.modalTargetDb.textContent = t("kb_loading_config");
                delete els.modalTargetDb.dataset.i18nOverride;
            }
        });
    }
}

// Knowledge base dropdown control function
window.toggleKbDropdown = function () {
    const wrapper = document.querySelector('.kb-dropdown-wrapper');
    if (wrapper) {
        wrapper.classList.toggle('open');
    }
};

function closeChatPipelineDropdown(target) {
    const trigger = document.getElementById('chatPipelineDropdown');
    const menu = document.getElementById('chat-pipeline-menu');
    if (!trigger || !menu) return;
    const isOpen = trigger.getAttribute('aria-expanded') === 'true' || menu.classList.contains('show');
    if (!isOpen) return;
    if (menu.contains(target) || trigger.contains(target)) return;
    if (window.bootstrap?.Dropdown) {
        const instance = window.bootstrap.Dropdown.getOrCreateInstance(trigger);
        instance.hide();
    } else {
        trigger.setAttribute('aria-expanded', 'false');
        menu.classList.remove('show');
    }
}

function closeBuilderPipelineDropdown(target) {
    const trigger = document.getElementById('pipelineDropdownBtn');
    const menu = document.getElementById('pipeline-menu');
    if (!trigger || !menu) return;
    const isOpen = trigger.getAttribute('aria-expanded') === 'true' || menu.classList.contains('show');
    if (!isOpen) return;
    if (menu.contains(target) || trigger.contains(target)) return;
    if (window.bootstrap?.Dropdown) {
        const instance = window.bootstrap.Dropdown.getOrCreateInstance(trigger);
        instance.hide();
    } else {
        trigger.setAttribute('aria-expanded', 'false');
        menu.classList.remove('show');
    }
}

function closeKbDropdown(target) {
    const wrapper = document.querySelector('.kb-dropdown-wrapper');
    const menu = document.getElementById('kb-dropdown-menu');
    const trigger = document.getElementById('kb-dropdown-trigger');
    if (!wrapper || !wrapper.classList.contains('open')) return;
    if ((menu && menu.contains(target)) || (trigger && trigger.contains(target))) return;
    wrapper.classList.remove('open');
}

function closeContextMenus(target) {
    const chatMenu = document.getElementById('chat-session-context-menu');
    if (chatMenu && !chatMenu.classList.contains('d-none') && !chatMenu.contains(target)) {
        hideChatSessionContextMenu();
    }
    const promptMenu = document.getElementById('prompt-context-menu');
    if (promptMenu && !promptMenu.classList.contains('d-none') && !promptMenu.contains(target)) {
        hidePromptContextMenu();
    }
}

// Pointer-down capture to close menus on blank area click
document.addEventListener('pointerdown', function (e) {
    const target = e.target;
    closeContextMenus(target);
    closeKbDropdown(target);
    closeChatPipelineDropdown(target);
    closeBuilderPipelineDropdown(target);
}, true);

// Clear knowledge base selection
window.clearKbSelection = function (e) {
    if (e) e.stopPropagation();
    const hiddenSelect = document.getElementById('chat-collection-select');
    if (hiddenSelect) {
        hiddenSelect.value = "";
    }
    // Simulate selecting empty item
    const mockItem = document.createElement('div');
    mockItem.dataset.value = "";
    mockItem.dataset.label = t("knowledge_base");
    selectKbOption(mockItem);
};

// Select knowledge base option
window.selectKbOption = function (itemEl) {
    const value = itemEl.dataset.value;
    const labelText = itemEl.dataset.label || itemEl.querySelector('.kb-item-text')?.textContent || "";
    const menu = document.getElementById('kb-dropdown-menu');
    const trigger = document.getElementById('kb-dropdown-trigger');
    const label = document.getElementById('kb-label-text');
    const hiddenSelect = document.getElementById('chat-collection-select');
    const clearBtn = document.getElementById('kb-clear-btn');

    // Update selected state of all options
    menu.querySelectorAll('.kb-dropdown-item').forEach(item => {
        item.classList.remove('selected');
    });
    // Only add selected class if itemEl actually exists in menu (avoid mockItem error)
    if (itemEl.parentNode === menu) {
        itemEl.classList.add('selected');
    } else if (value) {
        // If mockItem but has value, try to find corresponding item in menu and select
        const target = menu.querySelector(`.kb-dropdown-item[data-value="${value}"]`);
        if (target) target.classList.add('selected');
    }

    // Update hidden select value
    if (hiddenSelect) {
        hiddenSelect.value = value;
    }

    // Update trigger display
    if (value) {
        label.textContent = labelText;
        label.dataset.selected = "true";
        trigger.classList.add('active');
        if (clearBtn) clearBtn.style.display = 'inline-flex';
    } else {
        label.textContent = t("knowledge_base");
        label.dataset.selected = "false";
        trigger.classList.remove('active');
        if (clearBtn) clearBtn.style.display = 'none';
    }

    // Close dropdown menu
    document.querySelector('.kb-dropdown-wrapper').classList.remove('open');
};

// Render knowledge base dropdown options
function renderKbDropdownOptions(collections) {
    const menu = document.getElementById('kb-dropdown-menu');
    const hiddenSelect = document.getElementById('chat-collection-select');
    if (!menu) return;

    const currentVal = hiddenSelect ? hiddenSelect.value : '';

    // Clear and rebuild options
    menu.innerHTML = '';

    collections.forEach(c => {
        const isSelected = c.name === currentVal;
        const displayName = c.display_name || c.name;
        const item = document.createElement('div');
        item.className = `kb-dropdown-item ${isSelected ? 'selected' : ''}`;
        item.dataset.value = c.name;
        item.dataset.label = displayName;
        item.onclick = function () { selectKbOption(this); };
        item.innerHTML = `
            <span class="kb-item-check">✓</span>
            <span class="kb-item-text">${escapeHtmlForDropdown(displayName)}</span>
        `;
        menu.appendChild(item);
    });

    // Sync hidden select options
    if (hiddenSelect) {
        hiddenSelect.innerHTML = `<option value="">${t('no_knowledge_base')}</option>`;
        collections.forEach(c => {
            const displayName = c.display_name || c.name;
            const opt = document.createElement("option");
            opt.value = c.name;
            opt.textContent = `${displayName}`;
            if (c.name === currentVal) opt.selected = true;
            hiddenSelect.appendChild(opt);
        });
    }

    // Update trigger display state
    const trigger = document.getElementById('kb-dropdown-trigger');
    const label = document.getElementById('kb-label-text');
    const clearBtn = document.getElementById('kb-clear-btn');
    if (currentVal && trigger) {
        const selectedCollection = collections.find(c => c.name === currentVal);
        if (selectedCollection) {
            label.textContent = selectedCollection.display_name || selectedCollection.name;
            label.dataset.selected = "true";
            trigger.classList.add('active');
            if (clearBtn) clearBtn.style.display = 'inline-flex';
        } else {
            label.textContent = t("knowledge_base");
            label.dataset.selected = "false";
            trigger.classList.remove('active');
            if (clearBtn) clearBtn.style.display = 'none';
        }
    } else if (trigger) {
        label.textContent = t("knowledge_base");
        label.dataset.selected = "false";
        trigger.classList.remove('active');
        if (clearBtn) clearBtn.style.display = 'none';
    }
}

// HTML escape helper function (for dropdown menu)
function escapeHtmlForDropdown(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Compatibility: Update knowledge base selector display text (compatible with old code)
window.updateKbLabel = function (selectEl) {
    const label = document.getElementById('kb-label-text');
    const trigger = document.getElementById('kb-dropdown-trigger');
    if (!label || !trigger) return;

    const selectedVal = selectEl.value;

    if (!selectedVal) {
        label.textContent = t("knowledge_base");
        label.dataset.selected = "false";
        trigger.classList.remove('active');
    } else {
        const selectedText = selectEl.options[selectEl.selectedIndex].text;
        label.textContent = selectedText;
        label.dataset.selected = "true";
        trigger.classList.add('active');
    }
};

async function bootstrap() {
    const initialModeFromUrl = getInitialModeFromUrl();
    if (initialModeFromUrl) {
        setMode(initialModeFromUrl);
    }
    // 0. First get application mode configuration
    try {
        const modeConfig = await fetchJSON('/api/config/mode');
        state.adminMode = modeConfig.admin_mode === true;
    } catch (err) {
        console.warn("Failed to fetch app mode, defaulting to admin mode:", err);
        state.adminMode = true;
    }

    // Decide initial view based on mode and URL
    let initialMode;
    if (state.adminMode) {
        initialMode = initialModeFromUrl || Modes.CHAT;
    } else {
        initialMode = Modes.CHAT;
    }

    setMode(initialMode);
    applyUILanguage(state.uiLanguage);

    if (!state.adminMode && initialMode === Modes.CHAT) {
        // Chat-only mode: directly enter Chat view
        applyChatOnlyMode();
    }

    // Sync address bar so current view path is maintained after refresh (/ and /settings)
    updateUrlForView(initialMode, true);
    const startInChat = initialMode === Modes.CHAT;

    resetContextStack();
    renderSteps();
    updatePipelinePreview();

    // Initialize YAML editor, divider drag and Console
    initYamlEditor();
    initWorkspacePaneWidth();
    initBuilderResizer();
    initConsole();

    bindEvents();
    updateActionButtons();

    // 1. Load basic data first
    try {
        await Promise.all([refreshPipelines(), refreshTools()]);
        log("UI Ready.");
    } catch (err) {
        log(`Initialization error: ${err.message}`);
    }

    const wasCollapsed = localStorage.getItem("ultrarag_sidebar_collapsed") === "true";
    if (wasCollapsed && els.chatSidebar) {
        els.chatSidebar.classList.add("collapsed");
    }

    // 2. After data loading completes, restore historical sessions and selected state
    const savedSessions = localStorage.getItem("ultrarag_sessions");
    if (savedSessions) {
        try {
            state.chat.sessions = JSON.parse(savedSessions);
            state.chat.sessions.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));

            // Restore last active session
            const lastId = localStorage.getItem("ultrarag_last_active_id");
            if (lastId) {
                const session = state.chat.sessions.find(s => s.id === lastId);
                if (session) {
                    state.chat.currentSessionId = session.id;
                    state.chat.history = cloneDeep(session.messages || []);

                    // Only auto-load Pipeline in Admin mode (Chat-only mode handled in initChatOnlyView)
                    if (state.adminMode && session.pipeline) {
                        // refreshPipelines has completed at this point, UI is safe
                        loadPipeline(session.pipeline);
                    } else if (state.adminMode) {
                        // If no pipeline record, only set label
                        setHeroPipelineLabel(state.selectedPipeline || "");
                    }
                    log(`Restored last session: ${session.title}`);
                }
            }
        } catch (e) {
            console.warn("Failed to load history:", e);
            state.chat.sessions = [];
        }
    } else if (state.adminMode) {
        setHeroPipelineLabel(state.selectedPipeline || "");
    }

    // In Chat-only mode, initialize Chat interface
    if (startInChat) {
        await initChatOnlyView();
    }
}

// Hide admin-related buttons in Chat-only mode
function applyChatOnlyMode() {
    // Hide "Configure Pipeline" back button (not allowed to return to Builder)
    if (els.settingsBuilder) {
        els.settingsBuilder.style.display = "none";
    }
    if (els.settingsMenu) {
        els.settingsMenu.classList.remove("open");
    }
}

function applyUILanguage(lang) {
    const resolved = UI_LANGUAGE_MAP[lang] ? lang : "en";
    state.uiLanguage = resolved;

    try {
        localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, resolved);
    } catch (e) {
        console.warn("Failed to persist UI language", e);
    }

    document.documentElement.setAttribute("data-ui-lang", resolved);

    if (els.settingsLanguageLabel) {
        els.settingsLanguageLabel.textContent = UI_LANGUAGE_MAP[resolved];
    }

    updateI18nTexts();

    document.querySelectorAll("[data-ui-lang]").forEach((btn) => {
        const isActive = btn.getAttribute("data-ui-lang") === resolved;
        btn.classList.toggle("active", isActive);
    });
}

async function navigateBackToBuilder() {
    try {
        saveCurrentSession(true);
    } catch (e) {
        console.error(e);
    }

    setChatRunning(false);

    setMode(Modes.BUILDER);
    updateUrlForView(Modes.BUILDER);
    if (typeof switchWorkspaceMode === "function") {
        switchWorkspaceMode("pipeline");
    }
}

function setupSettingsMenu() {
    if (!els.settingsMenu) return;

    const closeMenu = () => {
        els.settingsMenu.classList.remove("open");
        const languageItem = document.getElementById("settings-language");
        if (languageItem) languageItem.classList.remove("submenu-open");
    };
    const toggleMenu = () => els.settingsMenu.classList.toggle("open");

    if (els.settingsMenuTrigger) {
        els.settingsMenuTrigger.addEventListener("click", (e) => {
            e.stopPropagation();
            toggleMenu();
        });
    }

    document.addEventListener("click", (e) => {
        if (!els.settingsMenu.contains(e.target)) {
            closeMenu();
        }
    });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            closeMenu();
        }
    });

    // Language submenu: keep open after hover/click until outside click
    const languageItem = document.getElementById("settings-language");
    if (languageItem) {
        const openSubmenu = () => languageItem.classList.add("submenu-open");
        const closeSubmenu = () => languageItem.classList.remove("submenu-open");

        languageItem.addEventListener("mouseenter", openSubmenu);
        languageItem.addEventListener("click", (e) => {
            e.stopPropagation();
            const isOpen = languageItem.classList.contains("submenu-open");
            if (isOpen) {
                closeSubmenu();
            } else {
                openSubmenu();
            }
        });

        // Keep submenu open when moving between item and submenu
        const languageSubmenu = document.getElementById("settings-language-submenu");
        if (languageSubmenu) {
            languageSubmenu.addEventListener("mouseenter", openSubmenu);
            languageSubmenu.addEventListener("click", (e) => e.stopPropagation());
        }

        // Close submenu on outside click
        document.addEventListener("click", (e) => {
            if (!languageItem.contains(e.target)) {
                closeSubmenu();
            }
        });
    }

    const languageButtons = document.querySelectorAll("[data-ui-lang]");
    languageButtons.forEach((btn) => {
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            const lang = btn.getAttribute("data-ui-lang");
            applyUILanguage(lang);
            closeMenu();
        });
    });

    if (els.settingsBuilder) {
        els.settingsBuilder.onclick = async (e) => {
            e?.stopPropagation();
            if (state.chat.running) {
                showInterruptConfirmDialog(async () => {
                    if (state.chat.controller) {
                        state.chat.controller.abort();
                        state.chat.controller = null;
                    }
                    await navigateBackToBuilder();
                });
                return;
            }

            await navigateBackToBuilder();
            closeMenu();
        };
    }

    // Ensure chat pipeline dropdown closes after selection
    const chatPipelineDropdown = document.getElementById("chatPipelineDropdown");
    const chatPipelineMenu = document.getElementById("chat-pipeline-menu");
    if (chatPipelineDropdown && chatPipelineMenu) {
        chatPipelineMenu.addEventListener("click", () => {
            if (window.bootstrap?.Dropdown) {
                const inst =
                    window.bootstrap.Dropdown.getInstance(chatPipelineDropdown) ||
                    new window.bootstrap.Dropdown(chatPipelineDropdown);
                inst.hide();
            } else {
                chatPipelineDropdown.classList.remove("show");
                chatPipelineMenu.classList.remove("show");
                chatPipelineDropdown.setAttribute("aria-expanded", "false");
            }
        });
    }
}

// ===== Background Task Management =====

// User ID management for isolating background tasks per user
const USER_ID_STORAGE_KEY = 'ultrarag_user_id';

function getUserId() {
    let userId = localStorage.getItem(USER_ID_STORAGE_KEY);
    if (!userId) {
        // Generate a unique user ID
        userId = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem(USER_ID_STORAGE_KEY, userId);
    }
    return userId;
}

// Background task state
const backgroundTaskState = {
    tasks: [],
    polling: null,
    panelOpen: false,
    notifiedTasks: new Set(), // Already notified task IDs
    backgroundModeEnabled: false, // Whether background mode is active
    cachedTasks: [], // Cached completed tasks from localStorage
    detailLoadingTaskId: null,
    loadToChatTaskId: null,
    loadToChatTarget: null,
};

// LocalStorage key for background tasks
const BG_TASKS_STORAGE_KEY = 'ultrarag_background_tasks';

// Schedule background task UI visibility sync to avoid TDZ issues
function scheduleBackgroundTaskVisibilitySync() {
    if (typeof setTimeout === 'function') {
        setTimeout(syncBackgroundTasksVisibility, 0);
    }
}

// Ensure background task controls only appear in Chat mode
function syncBackgroundTasksVisibility() {
    const fab = document.getElementById('bg-tasks-fab');
    const countEl = document.getElementById('bg-tasks-count');
    const panel = document.getElementById('background-tasks-panel');
    if (!fab || !countEl || !panel) return;

    const isChatMode = state.mode === Modes.CHAT;
    if (!isChatMode) {
        fab.classList.add('d-none');
        countEl.classList.add('d-none');
        panel.classList.add('d-none');
        backgroundTaskState.panelOpen = false;
        return;
    }

    updateBackgroundTasksCount();
}

// Load cached background tasks from localStorage
function loadCachedBackgroundTasks() {
    try {
        const cached = localStorage.getItem(BG_TASKS_STORAGE_KEY);
        if (cached) {
            backgroundTaskState.cachedTasks = JSON.parse(cached);
            // Mark already notified tasks to avoid duplicate notifications
            backgroundTaskState.cachedTasks.forEach(t => {
                if (t.status === 'completed' || t.status === 'failed') {
                    backgroundTaskState.notifiedTasks.add(t.task_id);
                }
            });
        }
    } catch (e) {
        console.warn('Failed to load cached background tasks:', e);
        backgroundTaskState.cachedTasks = [];
    }
}

// Save background tasks to localStorage
function saveCachedBackgroundTasks() {
    try {
        // Only cache completed and failed tasks (with full results)
        const toCache = backgroundTaskState.tasks.filter(t =>
            t.status === 'completed' || t.status === 'failed'
        );
        localStorage.setItem(BG_TASKS_STORAGE_KEY, JSON.stringify(toCache));
        backgroundTaskState.cachedTasks = toCache;
    } catch (e) {
        console.warn('Failed to save background tasks:', e);
    }
}

// Merge server tasks with cached tasks
function mergeBackgroundTasks(serverTasks) {
    const merged = [...serverTasks];
    const serverTaskIds = new Set(serverTasks.map(t => t.task_id));

    // Add cached tasks that are not on the server (e.g., server restarted)
    for (const cached of backgroundTaskState.cachedTasks) {
        if (!serverTaskIds.has(cached.task_id)) {
            // Mark as cached so we know it's from localStorage
            merged.push({ ...cached, fromCache: true });
        }
    }

    // Sort by created_at (newest first)
    merged.sort((a, b) => (b.created_at || 0) - (a.created_at || 0));

    return merged;
}

// Toggle background mode
window.toggleBackgroundMode = function () {
    backgroundTaskState.backgroundModeEnabled = !backgroundTaskState.backgroundModeEnabled;

    const toggle = document.getElementById('bg-mode-toggle');

    if (toggle) {
        toggle.classList.toggle('active', backgroundTaskState.backgroundModeEnabled);
    }
};

// Show notification toast
function showNotification(type, title, message, onClick = null) {
    const container = document.getElementById('notification-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `notification-toast ${type}`;

    const iconSvg = type === 'success'
        ? '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>'
        : type === 'error'
            ? '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>'
            : '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>';

    toast.innerHTML = `
        <div class="notification-icon">${iconSvg}</div>
        <div class="notification-content">
            <div class="notification-title">${title}</div>
            <div class="notification-message">${message}</div>
        </div>
        <button class="notification-close" onclick="event.stopPropagation(); this.parentElement.remove();">×</button>
    `;

    if (onClick) {
        toast.onclick = () => {
            onClick();
            toast.remove();
        };
    }

    container.appendChild(toast);

    // Auto remove
    setTimeout(() => {
        toast.style.animation = 'slideOutRight 0.3s ease-out forwards';
        setTimeout(() => toast.remove(), 300);
    }, 8000);

    // Try browser native notification
    requestBrowserNotification(title, message);
}

// Request browser notification permission and show notification
async function requestBrowserNotification(title, message) {
    if (!('Notification' in window)) return;

    if (Notification.permission === 'default') {
        await Notification.requestPermission();
    }

    if (Notification.permission === 'granted' && document.hidden) {
        const notification = new Notification(title, {
            body: message,
            icon: '/favicon.svg',
            tag: 'ultrarag-bg-task'
        });

        notification.onclick = () => {
            window.focus();
            notification.close();
        };
    }
}

// Toggle background tasks panel
window.toggleBackgroundPanel = function () {
    if (state.mode !== Modes.CHAT) {
        showNotification('info', t('bg_tasks_title'), t('bg_tasks_switch_chat'));
        return;
    }

    const panel = document.getElementById('background-tasks-panel');
    if (!panel) return;

    backgroundTaskState.panelOpen = !backgroundTaskState.panelOpen;
    panel.classList.toggle('d-none', !backgroundTaskState.panelOpen);

    if (backgroundTaskState.panelOpen) {
        refreshBackgroundTasks();
    }
};

// Refresh background tasks list
window.refreshBackgroundTasks = async function () {
    try {
        const userId = getUserId();
        const serverTasks = await fetchJSON(`/api/background-tasks?limit=20&user_id=${encodeURIComponent(userId)}`);

        // Merge with cached tasks (for tasks that may have been lost on server restart)
        backgroundTaskState.tasks = mergeBackgroundTasks(serverTasks);

        // Save completed tasks to cache
        saveCachedBackgroundTasks();

        renderBackgroundTasksList();
        updateBackgroundTasksCount();

        // Check for newly completed tasks and notify
        checkForCompletedTasks(backgroundTaskState.tasks);
    } catch (e) {
        console.error('Failed to refresh background tasks:', e);
        // If server is unavailable, show cached tasks
        if (backgroundTaskState.cachedTasks.length > 0) {
            backgroundTaskState.tasks = [...backgroundTaskState.cachedTasks];
            renderBackgroundTasksList();
            updateBackgroundTasksCount();
        }
    }
};

// Check for newly completed tasks
function checkForCompletedTasks(tasks) {
    for (const task of tasks) {
        if (task.status === 'completed' && !backgroundTaskState.notifiedTasks.has(task.task_id)) {
            backgroundTaskState.notifiedTasks.add(task.task_id);
            showNotification(
                'success',
                t('bg_task_completed_title'),
                task.question,
                () => showBackgroundTaskDetail(task.task_id)
            );
        } else if (task.status === 'failed' && !backgroundTaskState.notifiedTasks.has(task.task_id)) {
            backgroundTaskState.notifiedTasks.add(task.task_id);
            showNotification(
                'error',
                t('bg_task_failed_title'),
                task.error || task.question,
                () => showBackgroundTaskDetail(task.task_id)
            );
        }
    }
}

// Render background tasks list
function renderBackgroundTasksList() {
    const container = document.getElementById('bg-tasks-list');
    if (!container) return;

    if (backgroundTaskState.tasks.length === 0) {
        container.innerHTML = `<div class="text-muted text-center py-4 small">${t('bg_tasks_empty')}</div>`;
        return;
    }

    container.innerHTML = backgroundTaskState.tasks.map(task => {
        const time = task.created_at ? new Date(task.created_at * 1000).toLocaleTimeString() : '';
        const statusText = task.status === 'running'
            ? t('bg_task_running')
            : task.status === 'completed'
                ? t('bg_task_completed')
                : t('bg_task_failed');
        return `
            <div class="bg-task-item ${task.status}" onclick="showBackgroundTaskDetail('${task.task_id}')">
                <div class="bg-task-header">
                    <div class="bg-task-question">${escapeHtml(task.question)}</div>
                    <span class="bg-task-status ${task.status}">${statusText}</span>
                </div>
                <div class="bg-task-meta">
                    <span>${task.pipeline_name}</span>
                    <span>${time}</span>
                </div>
                ${task.status === 'completed' && task.result_preview ? `
                    <div class="bg-task-preview">${escapeHtml(task.result_preview)}</div>
                ` : ''}
            </div>
        `;
    }).join('');
}

// HTML escape
function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Update background tasks count
function updateBackgroundTasksCount() {
    const countEl = document.getElementById('bg-tasks-count');
    const fab = document.getElementById('bg-tasks-fab');
    if (!countEl || !fab) return;

    const isChatMode = state.mode === Modes.CHAT;
    if (!isChatMode) {
        countEl.classList.add('d-none');
        fab.classList.add('d-none');
        return;
    }

    const runningCount = backgroundTaskState.tasks.filter(t => t.status === 'running').length;

    if (runningCount > 0) {
        countEl.textContent = runningCount;
        countEl.classList.remove('d-none');
        fab.classList.remove('d-none');
    } else if (backgroundTaskState.tasks.length > 0) {
        countEl.classList.add('d-none');
        fab.classList.remove('d-none');
    } else {
        fab.classList.add('d-none');
    }
}

function renderTaskDetailLoading(modal, message = null) {
    if (!modal) return;
    const loadingMessage = message || t('bg_task_loading_details');
    modal.innerHTML = `
        <div style="padding: 32px; text-align: center;">
            <div class="spinner-border" style="color: #3b82f6; width: 2.5rem; height: 2.5rem;" role="status">
                <span class="visually-hidden">${t('bg_task_loading_details')}</span>
            </div>
            <div style="margin-top: 16px; color: var(--text-secondary);">${escapeHtml(loadingMessage)}</div>
        </div>
    `;
}

function setLoadButtonsLoading(taskId, isLoading, target) {
    const selector = target
        ? `[data-task-id="${taskId}"][data-load-target="${target}"]`
        : `[data-task-id="${taskId}"][data-load-target]`;
    document.querySelectorAll(selector).forEach(btn => {
        if (!(btn instanceof HTMLElement)) return;
        if (isLoading) {
            btn.dataset.originalText = btn.dataset.originalText || btn.innerHTML;
            btn.disabled = true;
            btn.classList.add('disabled');
            btn.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>${btn.dataset.loadTarget === 'new' ? t('bg_task_loading_to_new') : t('bg_task_loading_to_current')}`;
        } else {
            btn.disabled = false;
            btn.classList.remove('disabled');
            if (btn.dataset.originalText) {
                btn.innerHTML = btn.dataset.originalText;
            }
        }
    });
}

// Show background task detail
window.showBackgroundTaskDetail = async function (taskId) {
    if (backgroundTaskState.detailLoadingTaskId) return;

    // Create or get detail modal early and show loading state
    let modal = document.getElementById('bg-task-detail-modal');
    if (!modal) {
        modal = document.createElement('dialog');
        modal.id = 'bg-task-detail-modal';
        modal.className = 'bg-task-detail-modal';
        document.body.appendChild(modal);
    }

    renderTaskDetailLoading(modal, t('bg_task_loading_details'));
    modal.showModal();
    backgroundTaskState.detailLoadingTaskId = taskId;

    try {
        // Try to get from server first, fallback to cache
        let task;
        const userId = getUserId();
        try {
            task = await fetchJSON(`/api/background-tasks/${taskId}?user_id=${encodeURIComponent(userId)}`);
        } catch (e) {
            // Server may not have this task (e.g., after restart), try cache
            task = backgroundTaskState.cachedTasks.find(t => t.task_id === taskId);
            if (!task) {
                task = backgroundTaskState.tasks.find(t => t.task_id === taskId);
            }
            if (!task) {
                showNotification('error', t('status_error'), t('bg_task_task_not_found'));
                if (modal) modal.close();
                backgroundTaskState.detailLoadingTaskId = null;
                return;
            }
        }

        const statusText = task.status === 'running'
            ? t('bg_task_running')
            : task.status === 'completed'
                ? t('bg_task_completed')
                : t('bg_task_failed');

        const actionButtons = `
            ${task.status === 'completed' ? `
                <button class="btn btn-primary" onclick="copyTaskResult('${taskId}')">${t('bg_task_copy_result')}</button>
                <button class="btn btn-outline-secondary" data-task-id="${taskId}" data-load-target="current" onclick="loadTaskToChat('${taskId}','current')">${t('bg_task_load_current')}</button>
                <button class="btn btn-outline-secondary" data-task-id="${taskId}" data-load-target="new" onclick="loadTaskToChat('${taskId}','new')">${t('bg_task_load_new')}</button>
            ` : ''}
            <button class="btn btn-outline-danger ms-auto" onclick="deleteBackgroundTask('${taskId}')">${t('bg_task_delete')}</button>
        `;

        modal.innerHTML = `
            <div class="bg-task-detail-header">
                <div>
                    <span class="bg-task-status ${task.status}">${statusText}</span>
                    <div class="text-muted">${task.pipeline_name}</div>
                </div>
                <button class="bg-modal-close-btn" onclick="document.getElementById('bg-task-detail-modal').close()">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                </button>
            </div>
            <div class="bg-task-detail-body">
                <div class="bg-task-detail-question">
                    <strong>${t('bg_task_question')}</strong>
                    ${escapeHtml(task.full_question || task.question)}
                </div>
                ${task.status === 'completed' ? `
                    <div class="bg-task-detail-answer">
                        ${typeof renderMarkdown === 'function' ? renderMarkdown(task.result || '') : escapeHtml(task.result || '')}
                    </div>
                ` : task.status === 'failed' ? `
                    <div class="bg-task-detail-question" style="background: rgba(239, 68, 68, 0.06); border: 1px solid rgba(239, 68, 68, 0.15);">
                        <strong style="color: #ef4444;">${t('bg_task_error')}</strong>
                        ${escapeHtml(task.error || t('bg_task_unknown_error'))}
                    </div>
                ` : `
                    <div style="text-align: center; padding: 40px 20px;">
                        <div class="spinner-border" style="color: #3b82f6; width: 2rem; height: 2rem;" role="status">
                            <span class="visually-hidden">${t('bg_task_loading_details')}</span>
                        </div>
                        <div style="margin-top: 16px; color: var(--text-secondary); font-size: 0.9rem;">${t('bg_task_processing')}</div>
                    </div>
                `}
            </div>
            <div class="bg-task-detail-actions d-flex gap-2 flex-wrap align-items-center">
                ${actionButtons}
            </div>
        `;

        modal.showModal();

        // If task is still running, refresh periodically
        if (task.status === 'running') {
            const refreshInterval = setInterval(async () => {
                const userId = getUserId();
                const updated = await fetchJSON(`/api/background-tasks/${taskId}?user_id=${encodeURIComponent(userId)}`);
                if (updated.status !== 'running') {
                    clearInterval(refreshInterval);
                    showBackgroundTaskDetail(taskId);
                }
            }, 2000);

            modal.addEventListener('close', () => clearInterval(refreshInterval), { once: true });
        }
    } catch (e) {
        console.error('Failed to load task detail:', e);
        renderTaskDetailLoading(modal, t('bg_task_loading_details_failed'));
    } finally {
        backgroundTaskState.detailLoadingTaskId = null;
    }
};

// Copy task result
window.copyTaskResult = async function (taskId) {
    try {
        // Try server first, fallback to cache
        let task;
        const userId = getUserId();
        try {
            task = await fetchJSON(`/api/background-tasks/${taskId}?user_id=${encodeURIComponent(userId)}`);
        } catch (e) {
            task = backgroundTaskState.cachedTasks.find(t => t.task_id === taskId) ||
                backgroundTaskState.tasks.find(t => t.task_id === taskId);
        }

        if (task && task.result) {
            await navigator.clipboard.writeText(task.result);
            showNotification('success', 'Copied', 'Result copied to clipboard');
        }
    } catch (e) {
        console.error('Failed to copy:', e);
    }
};

// Load task result to chat
window.loadTaskToChat = async function (taskId, target = 'current') {
    if (backgroundTaskState.loadToChatTaskId) return;

    backgroundTaskState.loadToChatTaskId = taskId;
    backgroundTaskState.loadToChatTarget = target;
    setLoadButtonsLoading(taskId, true, target);

    try {
        // Try server first, fallback to cache
        let task;
        const userId = getUserId();
        try {
            task = await fetchJSON(`/api/background-tasks/${taskId}?user_id=${encodeURIComponent(userId)}`);
        } catch (e) {
            task = backgroundTaskState.cachedTasks.find(t => t.task_id === taskId) ||
                backgroundTaskState.tasks.find(t => t.task_id === taskId);
        }

        if (!task || task.status !== 'completed') {
            showNotification('error', 'Not Ready', 'This task has not finished yet.');
            return;
        }

        // Ensure we are in Chat view before injecting messages
        if (state.mode !== Modes.CHAT) {
            await safeOpenChatView();
            if (state.mode !== Modes.CHAT) {
                showNotification('error', 'Chat Unavailable', 'Please switch to Chat to load the result.');
                return;
            }
        }

        // If currently generating, ask the user to confirm interruption first
        if (state.chat.running) {
            const confirmed = await showConfirm(
                t("chat_interrupt_message"),
                {
                    title: t("chat_interrupt_title"),
                    type: "warning",
                    confirmText: t("chat_interrupt_confirm"),
                    danger: true
                }
            );
            if (!confirmed) {
                return; // User cancelled — do not interrupt, do not load
            }
            // User confirmed: abort the running generation
            if (state.chat.controller) {
                state.chat.controller.abort();
                state.chat.controller = null;
            }
            setChatRunning(false);
        }

        if (target === 'new') {
            if (state.chat.history.length > 0) {
                saveCurrentSession(true);
            }
            // Directly create new session without going through createNewChatSession()
            // to avoid its redundant running check and interrupt dialog
            state.chat.currentSessionId = generateChatId();
            state.chat.history = [];
        } else if (!state.chat.currentSessionId) {
            state.chat.currentSessionId = generateChatId();
            state.chat.history = [];
        }

        const createdAt = task.created_at ? new Date(task.created_at * 1000).toISOString() : new Date().toISOString();
        const completedAt = task.completed_at ? new Date(task.completed_at * 1000).toISOString() : new Date().toISOString();

        state.chat.history.push({
            role: 'user',
            text: task.full_question || task.question,
            timestamp: createdAt
        });
        state.chat.history.push({
            role: 'assistant',
            text: task.result || '',
            meta: { sources: task.sources || [] },
            timestamp: completedAt
        });

        // Save and render
        saveCurrentSession(true);
        renderChatHistory();
        renderChatSidebar();
        backToChatView();
        updateChatIdleStatus();

        // Close background panel if open
        if (backgroundTaskState.panelOpen) {
            toggleBackgroundPanel();
        }

        // Close modal after successful load
        const modal = document.getElementById('bg-task-detail-modal');
        if (modal) modal.close();

        showNotification(
            'success',
            t('bg_task_loaded'),
            target === 'new' ? t('bg_task_loaded_new') : t('bg_task_loaded_current')
        );
    } catch (e) {
        console.error('Failed to load task to chat:', e);
        showNotification('error', t('bg_task_load_failed'), e.message || t('bg_task_load_failed'));
    } finally {
        backgroundTaskState.loadToChatTaskId = null;
        backgroundTaskState.loadToChatTarget = null;
        setLoadButtonsLoading(taskId, false, target);
    }
};

// Delete background task
window.deleteBackgroundTask = async function (taskId) {
    const confirmed = await showConfirm(t("bg_task_delete_confirm"), {
        title: t("bg_task_delete_title"),
        type: "warning",
        confirmText: t("common_delete"),
        danger: true
    });
    if (!confirmed) return;

    try {
        // Try to delete from server (may fail if task is only in cache)
        const userId = getUserId();
        try {
            await fetchJSON(`/api/background-tasks/${taskId}?user_id=${encodeURIComponent(userId)}`, { method: 'DELETE' });
        } catch (e) {
            // Ignore server error, may be a cached-only task
        }

        // Also remove from local cache
        backgroundTaskState.cachedTasks = backgroundTaskState.cachedTasks.filter(t => t.task_id !== taskId);
        backgroundTaskState.tasks = backgroundTaskState.tasks.filter(t => t.task_id !== taskId);
        localStorage.setItem(BG_TASKS_STORAGE_KEY, JSON.stringify(backgroundTaskState.cachedTasks));

        // Close detail modal
        const modal = document.getElementById('bg-task-detail-modal');
        if (modal) modal.close();

        // Update UI
        renderBackgroundTasksList();
        updateBackgroundTasksCount();
    } catch (e) {
        console.error('Failed to delete task:', e);
    }
};

// Clear completed tasks
window.clearCompletedTasks = async function () {
    try {
        const confirmed = await showConfirm(t("bg_tasks_clear_completed_confirm"), {
            title: t("bg_tasks_clear_completed_title"),
            type: "warning",
            confirmText: t("bg_tasks_clear"),
            cancelText: t("common_cancel"),
            danger: true
        });
        if (!confirmed) return;

        // Try to clear from server
        let serverCount = 0;
        const userId = getUserId();
        try {
            const result = await fetchJSON('/api/background-tasks/clear-completed', {
                method: 'POST',
                body: JSON.stringify({ user_id: userId })
            });
            serverCount = result.count || 0;
        } catch (e) {
            // Ignore server error
        }

        // Also clear from local cache
        const cachedCount = backgroundTaskState.cachedTasks.filter(t => t.status === 'completed').length;
        backgroundTaskState.cachedTasks = backgroundTaskState.cachedTasks.filter(t => t.status !== 'completed');
        backgroundTaskState.tasks = backgroundTaskState.tasks.filter(t => t.status !== 'completed');
        localStorage.setItem(BG_TASKS_STORAGE_KEY, JSON.stringify(backgroundTaskState.cachedTasks));

        const totalCleared = Math.max(serverCount, cachedCount);
        showNotification('success', 'Cleared', `Cleared ${totalCleared} completed tasks`);

        // Update UI
        renderBackgroundTasksList();
        updateBackgroundTasksCount();
    } catch (e) {
        console.error('Failed to clear tasks:', e);
    }
};

// Send to background
async function sendToBackground(question) {
    if (!state.chat.engineSessionId) {
        showModal(t("chat_engine_required_message"), { title: t("chat_engine_required_title"), type: "warning" });
        return null;
    }

    // Validate if knowledge base selection is required (only for first turn chat and when pipeline contains retriever)
    if (!validateKnowledgeBaseSelection()) {
        return null;
    }

    const selectedCollection = els.chatCollectionSelect ? els.chatCollectionSelect.value : '';
    const dynamicParams = {};
    if (selectedCollection) {
        dynamicParams['collection_name'] = selectedCollection;
    }

    try {
        const userId = getUserId();
        const response = await fetchJSON(
            `/api/pipelines/${encodeURIComponent(state.selectedPipeline)}/chat/background`,
            {
                method: 'POST',
                body: JSON.stringify({
                    question,
                    session_id: state.chat.engineSessionId,
                    dynamic_params: dynamicParams,
                    user_id: userId
                })
            }
        );

        showNotification('info', 'Task Submitted', 'Question sent to background, you will be notified when complete');

        // Show background tasks FAB
        const fab = document.getElementById('bg-tasks-fab');
        if (fab) fab.classList.remove('d-none');

        // Start polling
        startBackgroundPolling();

        return response.task_id;
    } catch (e) {
        console.error('Failed to send to background:', e);
        showNotification('error', 'Send Failed', e.message || 'Failed to send to background');
        return null;
    }
}

// Start background task polling
function startBackgroundPolling() {
    if (backgroundTaskState.polling) return;

    backgroundTaskState.polling = setInterval(async () => {
        await refreshBackgroundTasks();

        // Stop polling if no running tasks
        const hasRunning = backgroundTaskState.tasks.some(t => t.status === 'running');
        if (!hasRunning) {
            stopBackgroundPolling();
        }
    }, 3000);
}

// Stop background task polling
function stopBackgroundPolling() {
    if (backgroundTaskState.polling) {
        clearInterval(backgroundTaskState.polling);
        backgroundTaskState.polling = null;
    }
}

// Initialize Chat interface in Chat-only mode
async function initChatOnlyView() {
    // 1. Render Pipeline selection menu
    await renderChatPipelineMenu();

    // 2. Render knowledge base options
    renderChatCollectionOptions();

    // 3. Render sidebar session list
    renderChatSidebar();

    // 4. Render chat history
    renderChatHistory();

    // 5. If there are saved sessions with corresponding Pipeline, try to auto-load
    const lastId = localStorage.getItem("ultrarag_last_active_id");
    if (lastId) {
        const session = state.chat.sessions.find(s => s.id === lastId);
        if (session && session.pipeline) {
            // Try to load last used Pipeline
            try {
                await switchChatPipeline(session.pipeline);
            } catch (e) {
                console.warn("Failed to restore last pipeline:", e);
            }
        }
    }

    // 6. Ensure engine is ready (auto-start if not started)
    if (state.selectedPipeline) {
        await ensureEngineReady(state.selectedPipeline);
    } else {
        updateChatIdleStatus();
    }

    // 7. Update Demo control button state
    updateDemoControls();

    // 8. Initialize background tasks state
    initBackgroundTasks();
}

// Initialize background tasks functionality
async function initBackgroundTasks() {
    // Load cached tasks from localStorage first
    loadCachedBackgroundTasks();

    // Request browser notification permission
    if ('Notification' in window && Notification.permission === 'default') {
        // Delay request to avoid disturbing user
        setTimeout(() => {
            Notification.requestPermission();
        }, 5000);
    }

    // Load background tasks list
    await refreshBackgroundTasks();

    // Start polling if there are running tasks
    if (backgroundTaskState.tasks.some(t => t.status === 'running')) {
        startBackgroundPolling();
    }
}

// Handle page refresh/close
window.addEventListener('beforeunload', function (e) {
    // If generating, show browser confirmation dialog
    if (state.chat.running) {
        // Save currently generated content
        saveCurrentSession(true);
        // Show browser native confirmation dialog
        e.preventDefault();
        e.returnValue = 'A response is being generated. Are you sure you want to leave?';
        return e.returnValue;
    }
    // Save session normally
    if (state.chat.history.length > 0 && state.chat.currentSessionId) {
        saveCurrentSession(true);
    }
});

// =========================================
// Workspace Mode Switching
// =========================================

const workspaceState = {
    currentMode: 'pipeline',  // 'pipeline' | 'parameters' | 'prompts'
    prompts: {
        files: [],
        currentFile: null,
        modified: false,
        originalContent: '',
        openTabs: [],
        tabState: {}
    }
};

function initWorkspace() {
    initWorkspacePaneWidth();
    // Bind feature toggle buttons
    document.querySelectorAll('.workspace-nav-btn[data-mode]').forEach(btn => {
        btn.addEventListener('click', async () => {
            const mode = btn.dataset.mode;
            await switchWorkspaceMode(mode);
        });
    });

    // Parameters panel events
    const paramsSaveBtn = document.getElementById('params-save-btn');
    if (paramsSaveBtn) {
        paramsSaveBtn.addEventListener('click', () => {
            persistParameterData();
            log(t("builder_params_saved"));
        });
    }

    const paramsRefreshBtn = document.getElementById('params-refresh-btn');
    if (paramsRefreshBtn) {
        paramsRefreshBtn.addEventListener('click', async () => {
            if (state.selectedPipeline && state.isBuilt) {
                try {
                    state.parameterData = cloneDeep(await fetchJSON(`/api/pipelines/${encodeURIComponent(state.selectedPipeline)}/parameters`));
                    renderParameterFormInline();
                    log(t("builder_params_reloaded"));
                } catch (e) {
                    log("Failed to reload parameters: " + e.message);
                }
            }
        });
    }

    // Prompts panel events
    initPromptEditor();
    initWorkspaceSideResizers();
}

function getWorkspacePaneWidth() {
    const stored = parseInt(localStorage.getItem(WORKSPACE_PANE_WIDTH_KEY), 10);
    if (Number.isFinite(stored) && stored > 0) return stored;
    return DEFAULT_WORKSPACE_PANE_WIDTH;
}

function setWorkspacePaneWidth(width, options = {}) {
    const { persist = true } = options;
    const next = Math.max(MIN_WORKSPACE_PANE_WIDTH, Math.round(width));
    document.documentElement.style.setProperty('--workspace-pane-width', `${next}px`);
    if (persist) {
        localStorage.setItem(WORKSPACE_PANE_WIDTH_KEY, String(next));
    }
}

function initWorkspacePaneWidth() {
    setWorkspacePaneWidth(getWorkspacePaneWidth(), { persist: false });
}

function initWorkspaceSideResizers() {
    const paramsResizer = document.getElementById('params-resizer');
    const promptsResizer = document.getElementById('prompts-resizer');
    initWorkspaceResizer(paramsResizer, document.querySelector('.params-layout'));
    initWorkspaceResizer(promptsResizer, document.querySelector('.prompts-layout'));
}

function initWorkspaceResizer(resizer, container) {
    if (!resizer || !container) return;

    let isResizing = false;
    let startX = 0;
    let startWidth = 0;

    resizer.addEventListener('mousedown', (e) => {
        isResizing = true;
        startX = e.clientX;
        startWidth = getWorkspacePaneWidth();
        resizer.classList.add('dragging');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        const dx = e.clientX - startX;
        const maxWidth = Math.max(
            MIN_WORKSPACE_PANE_WIDTH,
            container.offsetWidth - MIN_WORKSPACE_CONTENT_WIDTH
        );
        const next = Math.min(Math.max(startWidth + dx, MIN_WORKSPACE_PANE_WIDTH), maxWidth);
        setWorkspacePaneWidth(next);
    });

    document.addEventListener('mouseup', () => {
        if (!isResizing) return;
        isResizing = false;
        resizer.classList.remove('dragging');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
    });
}

async function switchWorkspaceMode(mode) {
    if (!mode) return;
    if (mode === workspaceState.currentMode) return;

    if (workspaceState.currentMode === 'pipeline' && mode !== 'pipeline' && state.unsavedChanges) {
        const ok = await showConfirm(t("builder_workspace_switch_unsaved_message"), {
            title: t("builder_unsaved_changes_title"),
            type: "warning",
            confirmText: t("common_continue"),
            cancelText: t("common_cancel")
        });
        if (!ok) return;

        // Discard unsaved changes: restore editor to last saved state
        if (els.yamlEditor && state.lastSavedYaml !== undefined) {
            els.yamlEditor.value = state.lastSavedYaml;
            updateYamlLineNumbers();
            setYamlSyncStatus('synced');
            await syncYamlToCanvasOnly({ markUnsaved: false });
        }
        clearUnsavedChanges();
    }

    workspaceState.currentMode = mode;

    updateAIContextBanner('mode-change');

    // Update navigation button state
    document.querySelectorAll('.workspace-nav-btn[data-mode]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    // Toggle panel display
    document.querySelectorAll('.workspace-panel').forEach(panel => {
        panel.classList.add('d-none');
    });

    const targetPanel = document.getElementById(`panel-${mode}`);
    if (targetPanel) {
        targetPanel.classList.remove('d-none');
    }

    // Execute specific operations based on mode
    if (mode === 'parameters') {
        loadParametersInline();
    } else if (mode === 'prompts') {
        loadPromptList();
    }

    const modeLabels = {
        pipeline: t("builder_mode_pipeline"),
        parameters: t("builder_mode_parameters"),
        prompts: t("builder_mode_prompts")
    };
    const modeLabel = modeLabels[mode] || mode;
    log(formatTemplate(t("builder_switched_mode"), { mode: modeLabel }));
}

// =========================================
// Inline Parameter Form (Integrated version)
// =========================================

function loadParametersInline() {
    const emptyEl = document.getElementById('params-empty');
    const formEl = document.getElementById('parameter-form');

    if (!state.selectedPipeline || !state.isBuilt) {
        if (emptyEl) emptyEl.classList.remove('d-none');
        if (formEl) formEl.innerHTML = '';
        return;
    }

    // If data already exists, render directly
    if (state.parameterData) {
        if (emptyEl) emptyEl.classList.add('d-none');
        renderParameterFormInline();
        return;
    }

    // Load parameter data
    fetchJSON(`/api/pipelines/${encodeURIComponent(state.selectedPipeline)}/parameters`)
        .then(data => {
            state.parameterData = cloneDeep(data);
            if (emptyEl) emptyEl.classList.add('d-none');
            renderParameterFormInline();
            updateAIContextBanner('params-loaded');
        })
        .catch(e => {
            if (emptyEl) emptyEl.classList.remove('d-none');
            log("Failed to load parameters: " + e.message);
        });
}

function renderParameterFormInline() {
    const container = document.getElementById('parameter-form');
    const navContainer = document.getElementById('parameter-nav');
    if (!container) return;

    container.innerHTML = "";
    if (navContainer) navContainer.innerHTML = "";

    if (!state.parameterData || typeof state.parameterData !== "object") {
        container.innerHTML = '<div class="params-empty"><p>No parameters available.</p></div>';
        return;
    }

    let entries = flattenParameters(state.parameterData).filter(e => !/^benchmark(\.|$)/i.test(e.path));
    // Apply server-specific parameter filtering for simplified display
    entries = filterServerParameters(entries);
    if (!entries.length) {
        container.innerHTML = '<div class="params-empty"><p>No editable parameters.</p></div>';
        return;
    }

    // Group parameters by server
    const grouped = {};
    entries.forEach(e => {
        const parts = (e.displayPath || e.path).split('.');
        const serverName = parts[0].toUpperCase();
        if (!grouped[serverName]) grouped[serverName] = [];
        grouped[serverName].push({
            ...e,
            displayPath: parts.slice(1).join('.') || parts[0],
            fullPath: e.path
        });
    });

    const serverNames = Object.keys(grouped).sort();

    // Render left navigation
    if (navContainer) {
        serverNames.forEach(serverName => {
            const navItem = document.createElement('button');
            navItem.className = 'parameter-nav-item';
            navItem.dataset.server = serverName;
            navItem.innerHTML = `
                <span class="nav-item-name">${escapeHtml(serverName)}</span>
                <span class="nav-item-count">${grouped[serverName].length}</span>
            `;
            navItem.onclick = () => {
                const section = document.getElementById(`param-section-${serverName}`);
                if (section) {
                    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    section.classList.add('expanded');
                }
                document.querySelectorAll('.parameter-nav-item').forEach(n => n.classList.remove('active'));
                navItem.classList.add('active');
            };
            navContainer.appendChild(navItem);
        });
    }

    // Render parameter area
    serverNames.forEach(serverName => {
        const section = document.createElement('div');
        section.className = 'parameter-section expanded';
        section.id = `param-section-${serverName}`;

        const header = document.createElement('div');
        header.className = 'parameter-section-header';
        header.innerHTML = `
            <div class="section-header-left">
                <svg class="section-chevron" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
                <span class="section-title">${escapeHtml(serverName)}</span>
                <span class="section-badge">${grouped[serverName].length} params</span>
            </div>
        `;
        header.onclick = () => section.classList.toggle('expanded');

        const content = document.createElement('div');
        content.className = 'parameter-section-content';

        const grid = document.createElement('div');
        grid.className = 'parameter-grid';

        grouped[serverName].forEach(e => {
            const grp = document.createElement('div');
            grp.className = 'parameter-field';
            const isComplex = e.type === "array" || e.type === "object";
            if (isComplex) grp.classList.add('full-width');

            const label = document.createElement('label');
            label.className = 'parameter-label';
            label.textContent = e.displayPath;
            label.title = e.fullPath;

            let ctrl;
            if (isComplex) {
                ctrl = document.createElement("textarea");
                ctrl.rows = 4;
                ctrl.value = JSON.stringify(e.value, null, 2);
            } else {
                ctrl = document.createElement("input");
                ctrl.type = "text";
                ctrl.value = String(e.value ?? "");
            }
            ctrl.className = "parameter-input";
            ctrl.onchange = (ev) => {
                let val = ev.target.value;
                if (e.type === "number") val = Number(val);
                if (e.type === "boolean") val = val.toLowerCase() === "true";
                try { if (isComplex) val = JSON.parse(val); } catch (err) { }
                e.value = val;
                setNestedValue(state.parameterData, e.fullPath, val);
                state.parametersReady = false;
                updateActionButtons();
            };

            grp.append(label, ctrl);
            grid.appendChild(grp);
        });

        content.appendChild(grid);
        section.append(header, content);
        container.appendChild(section);
    });
}

// =========================================
// Prompt Editor
// =========================================

function initPromptEditor() {
    const newBtn = document.getElementById('prompt-new-btn');
    const saveBtn = document.getElementById('prompt-save-btn');
    const deleteBtn = document.getElementById('prompt-delete-btn');
    const searchInput = document.getElementById('prompt-search');
    const editor = document.getElementById('prompt-editor');
    const contextMenu = document.getElementById('prompt-context-menu');

    if (newBtn) {
        newBtn.addEventListener('click', createNewPrompt);
    }

    if (saveBtn) {
        saveBtn.addEventListener('click', saveCurrentPrompt);
    }

    if (deleteBtn) {
        deleteBtn.addEventListener('click', deleteCurrentPrompt);
    }

    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            filterPromptList(e.target.value);
        });
    }

    if (editor) {
        editor.addEventListener('input', () => {
            syncPromptEditorState(editor.value);
            updatePromptLineNumbers();
        });

        editor.addEventListener('scroll', () => {
            const lineNumbers = document.getElementById('prompt-line-numbers');
            if (lineNumbers) {
                lineNumbers.scrollTop = editor.scrollTop;
            }
        });

        editor.addEventListener('keydown', (e) => {
            const isSaveShortcut = (e.key === 's' || e.key === 'S') && (e.ctrlKey || e.metaKey);
            if (isSaveShortcut) {
                e.preventDefault();
                if (saveBtn && !saveBtn.disabled) {
                    saveCurrentPrompt();
                }
                return;
            }
            if (e.key === 'Tab') {
                e.preventDefault();
                const start = editor.selectionStart;
                const end = editor.selectionEnd;
                editor.value = editor.value.substring(0, start) + '  ' + editor.value.substring(end);
                editor.selectionStart = editor.selectionEnd = start + 2;
                syncPromptEditorState(editor.value);
            }
        });
    }

    if (contextMenu) {
        contextMenu.addEventListener('click', (e) => {
            const action = e.target?.dataset?.action;
            const path = contextMenu.dataset.path;
            if (!action || !path) return;
            hidePromptContextMenu();
            if (action === 'rename') {
                renamePromptFile(path);
            } else if (action === 'delete') {
                deletePromptFile(path);
            }
        });
    }

    document.addEventListener('click', hidePromptContextMenu);
    document.addEventListener('scroll', hidePromptContextMenu, true);
}

function getPromptFileName(path = '') {
    return path.split('/').pop();
}

function ensurePromptTabState(path, initialContent = '') {
    const state = workspaceState.prompts.tabState;
    if (!state[path]) {
        state[path] = {
            content: initialContent,
            originalContent: initialContent,
            modified: false
        };
    }
    return state[path];
}

function syncPromptEditorState(content) {
    const current = workspaceState.prompts.currentFile;
    if (!current) return;
    const tabState = ensurePromptTabState(current, content);
    tabState.content = content;
    tabState.modified = content !== tabState.originalContent;
    workspaceState.prompts.modified = tabState.modified;
    workspaceState.prompts.originalContent = tabState.originalContent;
    updatePromptUI();
    renderPromptTabs();
}

function renderPromptTabs() {
    const tabsEl = document.getElementById('prompt-tabs');
    if (!tabsEl) return;

    const { openTabs, tabState, currentFile } = workspaceState.prompts;
    if (!openTabs.length) {
        tabsEl.innerHTML = '';
        tabsEl.classList.add('d-none');
        return;
    }

    tabsEl.classList.remove('d-none');
    tabsEl.innerHTML = '';
    openTabs.forEach(path => {
        const file = workspaceState.prompts.files.find(f => f.path === path);
        const name = file?.name || getPromptFileName(path);
        const tab = document.createElement('div');
        tab.className = 'prompt-tab';
        if (path === currentFile) tab.classList.add('active');
        if (tabState[path]?.modified) tab.classList.add('unsaved');
        tab.innerHTML = `
            <span class="prompt-tab-name">${name}</span>
            <button class="prompt-tab-close" type="button" title="${t("common_close")}">×</button>
        `;
        tab.onclick = () => setActivePromptTab(path);
        tab.querySelector('.prompt-tab-close')?.addEventListener('click', (e) => {
            e.stopPropagation();
            closePromptTab(path);
        });
        tabsEl.appendChild(tab);
    });
}

function prunePromptTabs(files) {
    const validPaths = new Set(files.map(f => f.path));
    const { openTabs, tabState } = workspaceState.prompts;
    workspaceState.prompts.openTabs = openTabs.filter(p => validPaths.has(p));
    Object.keys(tabState).forEach(path => {
        if (!validPaths.has(path)) {
            delete tabState[path];
        }
    });
    if (workspaceState.prompts.currentFile && !validPaths.has(workspaceState.prompts.currentFile)) {
        workspaceState.prompts.currentFile = null;
    }
    renderPromptTabs();
    updatePromptUI();
    if (!workspaceState.prompts.currentFile) {
        document.getElementById('prompt-editor-wrapper')?.classList.add('d-none');
        document.getElementById('prompt-empty')?.classList.remove('d-none');
    }
}

async function openPromptTab(file) {
    if (!file?.path) return;
    if (!workspaceState.prompts.openTabs.includes(file.path)) {
        workspaceState.prompts.openTabs.push(file.path);
    }
    await setActivePromptTab(file.path, file);
    renderPromptTabs();
}

async function setActivePromptTab(path, fileHint = null) {
    const editor = document.getElementById('prompt-editor');
    if (!editor || !path) return;

    let tabState = workspaceState.prompts.tabState[path];
    if (!tabState) {
        try {
            const content = await fetchJSON(`/api/prompts/${encodeURIComponent(path)}`);
            tabState = ensurePromptTabState(path, content.content || '');
            tabState.content = content.content || '';
            tabState.originalContent = content.content || '';
            tabState.modified = false;
        } catch (e) {
            log(formatTemplate(t("builder_prompt_load_failed"), { error: e.message }));
            return;
        }
    }

    workspaceState.prompts.currentFile = path;
    workspaceState.prompts.modified = tabState.modified;
    workspaceState.prompts.originalContent = tabState.originalContent;

    editor.value = tabState.content || '';
    updatePromptLineNumbers();
    updatePromptUI();
    renderPromptTabs();
    renderPromptList(workspaceState.prompts.files);
    updateAIContextBanner('prompt-select');

    document.getElementById('prompt-empty')?.classList.add('d-none');
    document.getElementById('prompt-editor-wrapper')?.classList.remove('d-none');
}

async function closePromptTab(path, options = {}) {
    if (!path) return;
    const tabState = workspaceState.prompts.tabState[path];
    if (tabState?.modified && !options.skipConfirm) {
        const confirmed = await showConfirm(formatTemplate(t("builder_prompt_discard_confirm"), { name: getPromptFileName(path) }), {
            title: t("builder_unsaved_changes_title"),
            type: "warning",
            confirmText: t("builder_discard")
        });
        if (!confirmed) return;
    }

    workspaceState.prompts.openTabs = workspaceState.prompts.openTabs.filter(p => p !== path);
    delete workspaceState.prompts.tabState[path];

    if (workspaceState.prompts.currentFile === path) {
        const next = workspaceState.prompts.openTabs[0];
        if (next) {
            await setActivePromptTab(next);
        } else {
            workspaceState.prompts.currentFile = null;
            workspaceState.prompts.modified = false;
            workspaceState.prompts.originalContent = '';
            document.getElementById('prompt-editor-wrapper')?.classList.add('d-none');
            document.getElementById('prompt-empty')?.classList.remove('d-none');
        }
    }

    renderPromptTabs();
    updatePromptUI();
    renderPromptList(workspaceState.prompts.files);
}

function showPromptContextMenu(event, file) {
    event.preventDefault();
    const menu = document.getElementById('prompt-context-menu');
    if (!menu || !file?.path) return;

    const menuWidth = 180;
    const menuHeight = 96;
    const left = Math.min(event.clientX, window.innerWidth - menuWidth - 12);
    const top = Math.min(event.clientY, window.innerHeight - menuHeight - 12);

    menu.dataset.path = file.path;
    menu.style.left = `${left}px`;
    menu.style.top = `${top}px`;
    menu.classList.remove('d-none');
}

function hidePromptContextMenu() {
    const menu = document.getElementById('prompt-context-menu');
    if (!menu) return;
    menu.classList.add('d-none');
    menu.dataset.path = '';
}

async function deletePromptFile(path) {
    if (!path) return;
    const confirmed = await showConfirm(formatTemplate(t("builder_prompt_delete_confirm"), { name: getPromptFileName(path) }), {
        title: t("builder_prompt_delete_title"),
        type: "warning",
        confirmText: t("common_delete"),
        danger: true
    });
    if (!confirmed) return;

    try {
        await fetchJSON(`/api/prompts/${encodeURIComponent(path)}`, {
            method: 'DELETE'
        });
        log(formatTemplate(t("builder_prompt_deleted"), { path }));

        await closePromptTab(path, { skipConfirm: true });
        await loadPromptList();
    } catch (e) {
        log(formatTemplate(t("builder_prompt_delete_failed"), { error: e.message }));
    }
}

async function renamePromptFile(path) {
    if (!path) return;
    const currentName = getPromptFileName(path);
    const newNameInput = await showPrompt(t("builder_prompt_rename_prompt"), {
        title: t("builder_prompt_rename_title"),
        placeholder: t("builder_prompt_rename_placeholder"),
        defaultValue: currentName,
        confirmText: t("common_rename")
    });
    if (!newNameInput) return;

    let newName = newNameInput.trim();
    if (!newName || newName === currentName) return;
    if (!newName.endsWith('.jinja2') && !newName.endsWith('.jinja')) {
        newName += '.jinja';
    }

    try {
        await fetchJSON(`/api/prompts/${encodeURIComponent(path)}/rename`, {
            method: 'POST',
            body: JSON.stringify({ new_name: newName })
        });
        log(formatTemplate(t("builder_prompt_renamed"), { name: newName }));

        await loadPromptList();
        const newFile = workspaceState.prompts.files.find(f => f.name === newName);
        if (newFile) {
            const { openTabs, tabState } = workspaceState.prompts;
            const idx = openTabs.indexOf(path);
            if (idx >= 0) {
                openTabs[idx] = newFile.path;
            }
            if (tabState[path]) {
                tabState[newFile.path] = { ...tabState[path] };
                delete tabState[path];
            }
            if (workspaceState.prompts.currentFile === path) {
                workspaceState.prompts.currentFile = newFile.path;
            }
            renderPromptTabs();
            renderPromptList(workspaceState.prompts.files);
            if (workspaceState.prompts.currentFile) {
                setActivePromptTab(workspaceState.prompts.currentFile);
            }
        }
    } catch (e) {
        log(formatTemplate(t("builder_prompt_rename_failed"), { error: e.message }));
    }
}

async function loadPromptList() {
    const listEl = document.getElementById('prompt-list');
    if (!listEl) return;

    try {
        const files = await fetchJSON('/api/prompts');
        workspaceState.prompts.files = files;
        renderPromptList(files);
        prunePromptTabs(files);
    } catch (e) {
        log(formatTemplate(t("builder_prompt_list_load_failed"), { error: e.message }));
        listEl.innerHTML = `<div class="prompts-empty"><p>${t("builder_prompt_list_load_failed_ui")}</p></div>`;
    }
}

function renderPromptList(files) {
    const listEl = document.getElementById('prompt-list');
    if (!listEl) return;

    listEl.innerHTML = '';

    if (!files.length) {
        listEl.innerHTML = `<div style="padding: 20px; text-align: center; color: #94a3b8; font-size: 0.85rem;">${t("builder_prompt_list_empty")}</div>`;
        return;
    }

    files.forEach(file => {
        const item = document.createElement('div');
        item.className = 'prompt-item';
        item.dataset.path = file.path;
        item.dataset.name = file.name;
        if (workspaceState.prompts.currentFile === file.path) {
            item.classList.add('active');
        }
        item.innerHTML = `
            <span class="prompt-item-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
            </span>
            <span class="prompt-item-name" title="${file.path}">${file.name}</span>
        `;
        item.onclick = () => selectPromptFile(file);
        item.oncontextmenu = (e) => showPromptContextMenu(e, file);
        listEl.appendChild(item);
    });
}

function filterPromptList(query) {
    const filtered = workspaceState.prompts.files.filter(f =>
        f.name.toLowerCase().includes(query.toLowerCase())
    );
    renderPromptList(filtered);
}

async function selectPromptFile(file) {
    await openPromptTab(file);
}

async function createNewPrompt() {
    const name = await showPrompt(t("builder_prompt_new_prompt"), {
        title: t("builder_new_prompt"),
        placeholder: t("builder_prompt_new_placeholder")
    });

    if (!name) return;

    // Ensure filename has correct extension
    let filename = name;
    if (!filename.endsWith('.jinja2') && !filename.endsWith('.jinja')) {
        filename += '.jinja';
    }

    try {
        await fetchJSON('/api/prompts', {
            method: 'POST',
            body: JSON.stringify({ name: filename, content: '# New Prompt Template\n\n' })
        });

        log(formatTemplate(t("builder_prompt_created"), { name: filename }));
        await loadPromptList();

        // Select newly created file
        const newFile = workspaceState.prompts.files.find(f => f.name === filename);
        if (newFile) {
            openPromptTab(newFile);
        }
    } catch (e) {
        log(formatTemplate(t("builder_prompt_create_failed"), { error: e.message }));
    }
}

async function saveCurrentPrompt() {
    if (!workspaceState.prompts.currentFile) return;

    const editor = document.getElementById('prompt-editor');
    if (!editor) return;

    try {
        await fetchJSON(`/api/prompts/${encodeURIComponent(workspaceState.prompts.currentFile)}`, {
            method: 'PUT',
            body: JSON.stringify({ content: editor.value })
        });

        const tabState = ensurePromptTabState(workspaceState.prompts.currentFile, editor.value);
        tabState.content = editor.value;
        tabState.originalContent = editor.value;
        tabState.modified = false;
        workspaceState.prompts.originalContent = editor.value;
        workspaceState.prompts.modified = false;
        updatePromptUI();
        renderPromptTabs();
        log(formatTemplate(t("builder_prompt_saved"), { path: workspaceState.prompts.currentFile }));
    } catch (e) {
        log(formatTemplate(t("builder_prompt_save_failed"), { error: e.message }));
    }
}

async function deleteCurrentPrompt() {
    if (!workspaceState.prompts.currentFile) return;
    await deletePromptFile(workspaceState.prompts.currentFile);
}

// Trigger rename when Prompt filename input loses focus
async function handlePromptFilenameBlur() {
    const filenameInput = document.getElementById('prompt-filename');
    if (!filenameInput || !workspaceState.prompts.currentFile) return;

    let newName = filenameInput.value.trim();
    const currentName = workspaceState.prompts.currentFile.split('/').pop();

    // If name unchanged, don't process
    if (!newName || newName === currentName) {
        filenameInput.value = currentName;
        return;
    }

    // Ensure correct extension
    if (!newName.endsWith('.jinja2') && !newName.endsWith('.jinja')) {
        newName += '.jinja';
        filenameInput.value = newName;
    }

    try {
        await fetchJSON(`/api/prompts/${encodeURIComponent(workspaceState.prompts.currentFile)}/rename`, {
            method: 'POST',
            body: JSON.stringify({ new_name: newName })
        });

        log(`Renamed to: ${newName}`);
        workspaceState.prompts.currentFile = newName;

        await loadPromptList();

        // Re-select file
        const newFile = workspaceState.prompts.files.find(f => f.name === newName);
        if (newFile) {
            workspaceState.prompts.currentFile = newFile.path;
            renderPromptList(workspaceState.prompts.files);
        }
    } catch (e) {
        log("Failed to rename prompt: " + e.message);
        filenameInput.value = currentName; // Restore original name
    }
}

function updatePromptUI() {
    const modifiedEl = document.getElementById('prompt-modified');
    const saveBtn = document.getElementById('prompt-save-btn');
    const deleteBtn = document.getElementById('prompt-delete-btn');

    if (modifiedEl) {
        modifiedEl.classList.toggle('d-none', !workspaceState.prompts.modified);
    }

    if (saveBtn) {
        saveBtn.disabled = !workspaceState.prompts.currentFile;
    }

    if (deleteBtn) {
        deleteBtn.disabled = !workspaceState.prompts.currentFile;
    }
}

function updatePromptLineNumbers() {
    const editor = document.getElementById('prompt-editor');
    const lineNumbers = document.getElementById('prompt-line-numbers');
    if (!editor || !lineNumbers) return;

    const lines = editor.value.split('\n').length;
    lineNumbers.innerHTML = Array.from({ length: lines }, (_, i) =>
        `<div class="line-number">${i + 1}</div>`
    ).join('');
}

bootstrap();

// Initialize workspace
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(initWorkspace, 100);
    setTimeout(initAIAssistant, 200);
});

// =========================================
// AI Assistant
// =========================================

const AI_HISTORY_KEY = 'ultrarag_ai_history';

const aiState = {
    isOpen: false,
    isConnected: false,
    isLoading: false,
    controller: null,
    isComposing: false,
    view: 'home',
    sessions: [],
    currentSessionId: null,
    lastUserMessage: null,
    lastContextSnapshot: null,
    settings: {
        ...DEFAULT_AI_SETTINGS
    },
    messages: [],
    conversationHistory: []
};

let cachedUiDefaults = null;
let uiDefaultsLoaded = false;

async function fetchUIDefaults() {
    if (uiDefaultsLoaded) return cachedUiDefaults;
    try {
        const resp = await fetch('/api/defaults/ai');
        if (!resp.ok) {
            uiDefaultsLoaded = true;
            return null;
        }
        const payload = await resp.json();
        cachedUiDefaults = (payload && typeof payload === 'object') ? payload : null;
    } catch (e) {
        console.warn('Failed to load UI defaults:', e);
        cachedUiDefaults = null;
    }
    uiDefaultsLoaded = true;
    return cachedUiDefaults;
}

function applyEnvDefaultsToIndex(indexDefaults) {
    if (!indexDefaults || typeof indexDefaults !== 'object') return;

    // Respect existing localStorage settings; only fill missing values.
    const hasSavedIndex = !!storedIndexConfig;
    if (!hasSavedIndex || !indexConfigState.api_key) {
        indexConfigState.api_key = indexDefaults.api_key || indexConfigState.api_key;
    }
    if (!hasSavedIndex || !indexConfigState.base_url) {
        indexConfigState.base_url = indexDefaults.base_url || indexConfigState.base_url;
    }
    if (!hasSavedIndex || !indexConfigState.model_name) {
        indexConfigState.model_name = indexDefaults.model_name || indexConfigState.model_name;
    }
}

const AI_WELCOME_HTML = `
    <div class="ai-welcome">
        <div class="ai-welcome-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z"/><circle cx="7.5" cy="14.5" r="1.5"/><circle cx="16.5" cy="14.5" r="1.5"/></svg>
        </div>
        <h4 data-i18n="builder_ai_welcome_title">UltraRAG AI Assistant</h4>
        <p data-i18n="builder_ai_welcome_desc">I can help you build pipelines, configure parameters, and edit prompts.</p>
        <p class="ai-welcome-hint" data-i18n="builder_ai_welcome_hint">Click the settings icon to configure your API connection.</p>

        <!-- Quick Action Cards -->
        <div class="ai-starter-chips">
            <button class="ai-starter-chip" data-prompt="Update the current RAG pipeline to include a citation module, ensuring the final output displays source references for fact-checking purposes." data-i18n-prompt="builder_ai_prompt_pipeline_adjustment">
                <span class="ai-starter-chip-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
                </span>
                <span class="ai-starter-chip-text" data-i18n="builder_ai_chip_pipeline_adjustment">Pipeline Adjustment</span>
            </button>
            <button class="ai-starter-chip" data-prompt="Optimize the system prompt for the [Insert Domain, e.g., Medical/Legal] domain. Please refine the instructions to ensure the generated responses strictly adhere to professional terminology and logical accuracy suitable for this field." data-i18n-prompt="builder_ai_prompt_prompt_adaptation">
                <span class="ai-starter-chip-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
                </span>
                <span class="ai-starter-chip-text" data-i18n="builder_ai_chip_prompt_adaptation">Prompt Adaptation</span>
            </button>
            <button class="ai-starter-chip" data-prompt="Reconfigure the generation backend. Switch the backend type to OpenAI, set the model name to [Insert Model Name, e.g., Llama-3-70B], and update the API endpoint to port [Insert Port, e.g., 8000]." data-i18n-prompt="builder_ai_prompt_parameter_settings">
                <span class="ai-starter-chip-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/></svg>
                </span>
                <span class="ai-starter-chip-text" data-i18n="builder_ai_chip_parameter_settings">Parameter Settings</span>
            </button>
            <button class="ai-starter-chip" data-prompt="I want to redesign my RAG workflow based on this article/paper: [Insert Link]. Please analyze its core methodologies and assist me in constructing a similar pipeline architecture." data-i18n-prompt="builder_ai_prompt_freeform_tuning">
                <span class="ai-starter-chip-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg>
                </span>
                <span class="ai-starter-chip-text" data-i18n="builder_ai_chip_freeform_tuning">Free-form Tuning</span>
            </button>
        </div>
    </div>
`;

async function initAIAssistant() {
    const trigger = document.getElementById('ai-assistant-trigger');
    const panel = document.getElementById('ai-assistant-panel');
    const sidebarBtn = document.getElementById('navbar-ai-btn');
    const closeBtn = document.getElementById('ai-close-btn');
    const settingsBtn = document.getElementById('ai-settings-btn');
    const settingsPanel = document.getElementById('ai-settings-panel');
    const settingsClose = document.getElementById('ai-settings-close');
    const clearBtn = document.getElementById('ai-clear-btn');
    const resizer = document.getElementById('ai-panel-resizer');
    const input = document.getElementById('ai-input');
    const sendBtn = document.getElementById('ai-send-btn');
    const connectionStatus = document.getElementById('ai-connection-status');

    if (!panel || (!trigger && !sidebarBtn)) return;

    const uiDefaults = await fetchUIDefaults();
    if (uiDefaults && typeof uiDefaults === 'object') {
        const aiDefaults = uiDefaults.ai || {};
        aiState.settings = {
            ...aiState.settings,
            provider: aiDefaults.provider || aiState.settings.provider,
            baseUrl: aiDefaults.baseUrl || aiState.settings.baseUrl,
            apiKey: aiDefaults.apiKey || aiState.settings.apiKey,
            model: aiDefaults.model || aiState.settings.model
        };
        applyEnvDefaultsToIndex(uiDefaults.embedding || {});
    }

    loadAISettings();
    updateAIConnectionStatus();
    loadAIConversation();
    ensureAISession();
    renderAIConversationFromState();
    setAIView('home');
    updateAIContextBanner('init');

    // If configured, auto-test connection
    if (aiState.settings.apiKey) {
        autoTestAIConnection();
    }

    // Click status bar to open settings interface
    connectionStatus?.addEventListener('click', () => {
        settingsPanel?.classList.add('open');
    });
    connectionStatus?.style.setProperty('cursor', 'pointer');

    const openHandler = () => toggleAIPanel();
    trigger?.addEventListener('click', openHandler);
    sidebarBtn?.addEventListener('click', openHandler);
    closeBtn?.addEventListener('click', () => closeAIPanel());
    document.getElementById('ai-back-btn')?.addEventListener('click', () => enterAIHome());

    settingsBtn?.addEventListener('click', () => settingsPanel?.classList.add('open'));
    settingsClose?.addEventListener('click', () => settingsPanel?.classList.remove('open'));

    clearBtn?.addEventListener('click', () => clearAIChat());
    document.getElementById('ai-session-new')?.addEventListener('click', () => {
        handleNewAISessionClick();
    });
    document.getElementById('ai-session-delete')?.addEventListener('click', () => {
        deleteAllAISessions();
    });

    initAIPanelResizer(resizer, panel);

    sendBtn?.addEventListener('click', () => {
        sendAIMessage();
    });

    if (input) {
        input.addEventListener('compositionstart', () => { aiState.isComposing = true; });
        input.addEventListener('compositionend', () => { aiState.isComposing = false; });
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey && !aiState.isComposing) {
                e.preventDefault();
                sendAIMessage();
            }
        });
        input.addEventListener('input', () => {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 120) + 'px';
            // Dynamically update send button state
            updateAISendButtonState();
        });
    }

    // Recommended question card click events
    initAIStarterChips();

    initAISettingsPanel();
}

// Update send button active state
function updateAISendButtonState() {
    const input = document.getElementById('ai-input');
    const sendBtn = document.getElementById('ai-send-btn');
    if (!sendBtn) return;

    const hasContent = input?.value?.trim().length > 0;
    if (hasContent) {
        sendBtn.classList.add('active');
    } else {
        sendBtn.classList.remove('active');
    }
}

// Initialize recommended question cards
function initAIStarterChips() {
    const chips = document.querySelectorAll('.ai-starter-chip');
    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            const prompt = chip.dataset.prompt;
            if (!prompt) return;

            const input = document.getElementById('ai-input');
            if (input) {
                input.value = prompt;
                input.style.height = 'auto';
                input.style.height = Math.min(input.scrollHeight, 120) + 'px';
                updateAISendButtonState();
                input.focus();
            }

            // Only fill input box, don't auto-send
        });
    });
}

function updateAIPanelOffset(panel) {
    if (!panel) return;
    const width = panel.getBoundingClientRect().width || parseInt(getComputedStyle(panel).width, 10) || 380;
    document.documentElement.style.setProperty('--ai-panel-width', `${width}px`);
    document.body.classList.add('ai-panel-open');
}

function toggleAIPanel() {
    if (aiState.isOpen) {
        closeAIPanel();
    } else {
        openAIPanel();
    }
}

function openAIPanel() {
    const trigger = document.getElementById('ai-assistant-trigger');
    const panel = document.getElementById('ai-assistant-panel');
    const sidebarBtn = document.getElementById('navbar-ai-btn');

    trigger?.classList.add('hidden');
    panel?.classList.add('open');
    sidebarBtn?.classList.add('active');
    aiState.isOpen = true;
    setAIView('home');
    updateAIPanelOffset(panel);

    setTimeout(() => {
        if (aiState.view === 'chat') {
            document.getElementById('ai-input')?.focus();
        }
    }, 100);
}

function closeAIPanel() {
    const trigger = document.getElementById('ai-assistant-trigger');
    const panel = document.getElementById('ai-assistant-panel');
    const sidebarBtn = document.getElementById('navbar-ai-btn');

    trigger?.classList.remove('hidden');
    panel?.classList.remove('open');
    document.body.classList.remove('ai-panel-open');
    sidebarBtn?.classList.remove('active');
    aiState.isOpen = false;
}

function initAIPanelResizer(resizer, panel) {
    if (!resizer || !panel) return;

    let isResizing = false;
    let startX = 0;
    let startWidth = 0;

    resizer.addEventListener('mousedown', (e) => {
        isResizing = true;
        startX = e.clientX;
        startWidth = panel.offsetWidth;
        resizer.classList.add('dragging');
        document.body.style.cursor = 'ew-resize';
        document.body.style.userSelect = 'none';
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;

        const diff = startX - e.clientX; // Reverse: dragging left is positive, should increase width
        const newWidth = Math.min(Math.max(startWidth + diff, 300), 600);
        panel.style.width = newWidth + 'px';
        updateAIPanelOffset(panel);
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            resizer.classList.remove('dragging');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        }
    });
}

function initAISettingsPanel() {
    const providerSelect = document.getElementById('ai-provider');
    const baseUrlInput = document.getElementById('ai-base-url');
    const apiKeyInput = document.getElementById('ai-api-key');
    const modelInput = document.getElementById('ai-model');
    const toggleKeyBtn = document.getElementById('ai-toggle-key');
    const saveBtn = document.getElementById('ai-save-settings');

    // Fill current settings
    if (providerSelect) providerSelect.value = aiState.settings.provider;
    if (baseUrlInput) baseUrlInput.value = aiState.settings.baseUrl;
    if (apiKeyInput) apiKeyInput.value = aiState.settings.apiKey;
    if (modelInput) modelInput.value = aiState.settings.model;

    // Update baseUrl when Provider changes
    providerSelect?.addEventListener('change', (e) => {
        const provider = e.target.value;
        const defaultUrls = {
            openai: 'https://api.openai.com/v1',
            azure: 'https://YOUR_RESOURCE.openai.azure.com',
            anthropic: 'https://api.anthropic.com/v1',
            custom: ''
        };
        if (baseUrlInput && defaultUrls[provider]) {
            baseUrlInput.value = defaultUrls[provider];
        }
    });

    // Toggle key visibility
    toggleKeyBtn?.addEventListener('click', () => {
        if (apiKeyInput) {
            apiKeyInput.type = apiKeyInput.type === 'password' ? 'text' : 'password';
        }
    });

    // Save and test connection
    saveBtn?.addEventListener('click', async () => {
        const saved = saveAISettings({ silent: true });
        if (!saved) return;
        await testAIConnection(saveBtn);
    });
}

function loadAISettings() {
    try {
        const saved = localStorage.getItem(AI_SETTINGS_STORAGE_KEY);
        if (saved) {
            const parsed = JSON.parse(saved);
            aiState.settings = { ...aiState.settings, ...parsed };
            updateAIConnectionStatus();
            return true;
        }
    } catch (e) {
        console.error('Failed to load AI settings:', e);
    }
    return false;
}

function saveAISettings(options = {}) {
    const { silent = false } = options;
    const providerSelect = document.getElementById('ai-provider');
    const baseUrlInput = document.getElementById('ai-base-url');
    const apiKeyInput = document.getElementById('ai-api-key');
    const modelInput = document.getElementById('ai-model');
    const statusEl = document.getElementById('ai-settings-status');

    aiState.settings = {
        provider: providerSelect?.value || 'openai',
        baseUrl: baseUrlInput?.value || '',
        apiKey: apiKeyInput?.value || '',
        model: modelInput?.value || 'gpt-5-mini'
    };

    try {
        localStorage.setItem(AI_SETTINGS_STORAGE_KEY, JSON.stringify(aiState.settings));

        if (statusEl && !silent) {
            statusEl.textContent = t('builder_ai_settings_saved_success');
            statusEl.className = 'ai-settings-status success';
            setTimeout(() => {
                statusEl.className = 'ai-settings-status';
            }, 3000);
        }

        updateAIConnectionStatus();
        return true;
    } catch (e) {
        if (statusEl) {
            statusEl.textContent = formatTemplate(t('builder_ai_settings_save_failed'), { error: e.message });
            statusEl.className = 'ai-settings-status error';
        }
        return false;
    }
}

async function testAIConnection(triggerBtn = null) {
    const statusEl = document.getElementById('ai-settings-status');
    const actionBtn = triggerBtn || document.getElementById('ai-test-connection');

    if (!aiState.settings.apiKey) {
        if (statusEl) {
            statusEl.textContent = t('builder_ai_settings_api_key_required');
            statusEl.className = 'ai-settings-status error';
        }
        return;
    }

    if (actionBtn) actionBtn.disabled = true;
    if (statusEl) {
        statusEl.textContent = t('builder_ai_settings_testing');
        statusEl.className = 'ai-settings-status';
        statusEl.style.display = 'block';
    }

    try {
        const response = await fetch('/api/ai/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(aiState.settings)
        });

        const result = await response.json();

        if (result.success) {
            aiState.isConnected = true;
            if (statusEl) {
                statusEl.textContent = formatTemplate(t('builder_ai_settings_connection_success'), {
                    model: result.model || aiState.settings.model
                });
                statusEl.className = 'ai-settings-status success';
            }
        } else {
            aiState.isConnected = false;
            if (statusEl) {
                statusEl.textContent = formatTemplate(t('builder_ai_settings_connection_failed'), {
                    error: result.error || t('common_unknown_error')
                });
                statusEl.className = 'ai-settings-status error';
            }
        }

        updateAIConnectionStatus();
    } catch (e) {
        aiState.isConnected = false;
        if (statusEl) {
            statusEl.textContent = formatTemplate(t('builder_ai_settings_connection_failed'), { error: e.message });
            statusEl.className = 'ai-settings-status error';
        }
    } finally {
        if (actionBtn) actionBtn.disabled = false;
    }
}

// Silent test connection - automatically detect saved configuration on page load
async function autoTestAIConnection() {
    if (!aiState.settings.apiKey) return;

    try {
        const response = await fetch('/api/ai/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(aiState.settings)
        });

        const result = await response.json();

        if (result.success) {
            aiState.isConnected = true;
        } else {
            aiState.isConnected = false;
        }

        updateAIConnectionStatus();
    } catch (e) {
        aiState.isConnected = false;
        updateAIConnectionStatus();
    }
}

function updateAIConnectionStatus() {
    const statusEl = document.getElementById('ai-connection-status');
    if (!statusEl) return;

    const dot = statusEl.querySelector('.ai-status-dot');
    const text = statusEl.querySelector('.ai-status-text');

    if (aiState.settings.apiKey) {
        if (aiState.isConnected) {
            dot?.classList.remove('disconnected', 'connecting');
            dot?.classList.add('connected');
            if (text) text.textContent = formatTemplate(t("builder_ai_status_connected"), { model: aiState.settings.model });
        } else {
            dot?.classList.remove('connected', 'connecting');
            dot?.classList.add('disconnected');
            if (text) text.textContent = t("builder_ai_status_configured_pending");
        }
    } else {
        dot?.classList.remove('connected', 'connecting');
        dot?.classList.add('disconnected');
        if (text) text.textContent = t("builder_ai_status_not_configured");
    }
}

function setAIView(view) {
    const home = document.getElementById('ai-home-view');
    const chat = document.getElementById('ai-chat-view');
    const backBtn = document.getElementById('ai-back-btn');
    aiState.view = view;
    if (home) {
        home.classList.toggle('active', view === 'home');
        home.classList.toggle('d-none', view !== 'home');
    }
    if (chat) {
        chat.classList.toggle('active', view === 'chat');
        chat.classList.toggle('d-none', view !== 'chat');
    }
    if (backBtn) {
        backBtn.classList.toggle('d-none', view === 'home');
    }
}

function enterAIHome() {
    setAIView('home');
    renderAISessionList();
}

function enterAIChat(sessionId) {
    if (sessionId) {
        switchAISession(sessionId);
    } else {
        setAIView('chat');
        renderAIConversationFromState();
    }
}

function currentSessionHasContent() {
    const current = aiState.sessions.find(s => s.id === aiState.currentSessionId);
    return Boolean(current && current.messages && current.messages.length);
}

function deriveAISessionTitle(messages = []) {
    const firstUser = messages.find(m => m.role === 'user');
    if (!firstUser || !firstUser.content) return t("builder_ai_session_new");
    const text = firstUser.content.trim().replace(/\s+/g, ' ');
    if (text.length <= 24) return text;
    return text.slice(0, 24) + '…';
}

function syncCurrentAISession() {
    if (!aiState.currentSessionId) return;
    const idx = aiState.sessions.findIndex(s => s.id === aiState.currentSessionId);
    if (idx === -1) return;
    const session = aiState.sessions[idx];
    session.messages = [...aiState.messages];
    session.conversationHistory = [...aiState.conversationHistory];
    session.updatedAt = Date.now();
    session.title = deriveAISessionTitle(session.messages);
    aiState.sessions[idx] = session;
}

function renderAISessionList() {
    const listEl = document.getElementById('ai-session-list');
    const deleteBtn = document.getElementById('ai-session-delete');
    if (!listEl) return;
    listEl.innerHTML = '';

    // Filter out empty sessions (sessions without messages are not displayed)
    const nonEmptySessions = aiState.sessions.filter(session =>
        session.messages && session.messages.length > 0
    );

    if (!nonEmptySessions.length) {
        const empty = document.createElement('div');
        empty.className = 'ai-session-empty';
        empty.textContent = t("builder_ai_sessions_empty");
        listEl.appendChild(empty);
        if (deleteBtn) deleteBtn.disabled = true;
        return;
    }
    nonEmptySessions.forEach(session => {
        const btn = document.createElement('div');
        btn.className = 'ai-session-item' + (session.id === aiState.currentSessionId ? ' active' : '');
        const title = escapeHtml(session.title || 'New Session');
        const time = session.updatedAt ? new Date(session.updatedAt).toLocaleString() : '';
        btn.innerHTML = `
            <div class="ai-session-content">
                <div class="ai-session-title-text">${title}</div>
                <div class="ai-session-meta">${time}</div>
            </div>
            <button class="ai-session-delete-btn" title="${t("builder_ai_delete_session")}">
                <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
            </button>
        `;
        // Click session content to switch session
        const contentEl = btn.querySelector('.ai-session-content');
        contentEl?.addEventListener('click', () => {
            switchAISession(session.id);
        });
        // Click delete button to delete session
        const deleteBtnEl = btn.querySelector('.ai-session-delete-btn');
        deleteBtnEl?.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteAISession(session.id);
        });
        listEl.appendChild(btn);
    });
    if (deleteBtn) deleteBtn.disabled = nonEmptySessions.length === 0;
}

function applyAISessionState(session, options = {}) {
    const keepView = options.keepView;
    aiState.messages = session.messages || [];
    aiState.conversationHistory = session.conversationHistory || [];
    aiState.currentSessionId = session.id;
    aiState.isLoading = false;
    aiState.controller = null;
    renderAIConversationFromState();
    setAIRunning(false);
    renderAISessionList();
    if (!keepView) setAIView('chat');
    persistAIConversation();
}

function createNewAISession(skipRender = false) {
    const id = `ai_${Date.now()}`;
    const session = {
        id,
        title: 'New Session',
        messages: [],
        conversationHistory: [],
        updatedAt: Date.now()
    };
    aiState.sessions.unshift(session);
    aiState.currentSessionId = id;
    if (!skipRender) {
        applyAISessionState(session);
    }
    renderAISessionList();
}

function handleNewAISessionClick() {
    const hasContent = currentSessionHasContent();
    if (!hasContent) {
        setAIView('chat');
        renderAIConversationFromState();
        return;
    }
    createNewAISession();
    setAIView('chat');
}

function switchAISession(id) {
    if (aiState.controller) {
        aiState.controller.abort();
        aiState.controller = null;
    }
    syncCurrentAISession();
    const session = aiState.sessions.find(s => s.id === id);
    if (session) {
        applyAISessionState(session);
    }
}

async function deleteAISession(id) {
    const targetId = id || aiState.currentSessionId;
    if (!targetId) return;

    const confirmed = await showConfirm(t('ai_session_delete_confirm'), {
        title: t('ai_session_delete_title'),
        confirmText: t('common_delete'),
        cancelText: t('common_cancel'),
        danger: true
    });

    if (!confirmed) return;

    aiState.sessions = aiState.sessions.filter(s => s.id !== targetId);
    if (!aiState.sessions.length) {
        createNewAISession(true);
    }
    const next = aiState.sessions[0];
    applyAISessionState(next, { keepView: true });
    setAIView('home');
}

async function deleteAllAISessions() {
    const nonEmptySessions = aiState.sessions.filter(s => s.messages && s.messages.length > 0);
    if (!nonEmptySessions.length) return;

    const confirmed = await showConfirm(formatTemplate(t('ai_session_delete_all_confirm'), { count: nonEmptySessions.length }), {
        title: t('ai_session_delete_all_title'),
        confirmText: t('ai_session_delete_all_action'),
        cancelText: t('common_cancel'),
        danger: true
    });

    if (!confirmed) return;

    aiState.sessions = [];
    createNewAISession(true);
    const next = aiState.sessions[0];
    applyAISessionState(next, { keepView: true });
    setAIView('home');
}

function ensureAISession() {
    if (!aiState.sessions.length) {
        createNewAISession(true);
    }
    if (!aiState.currentSessionId) {
        aiState.currentSessionId = aiState.sessions[0].id;
    }
    const current = aiState.sessions.find(s => s.id === aiState.currentSessionId);
    if (current) {
        applyAISessionState(current, { keepView: true });
    } else {
        createNewAISession(false);
    }
}

function persistAIConversation() {
    try {
        syncCurrentAISession();
        localStorage.setItem(AI_HISTORY_KEY, JSON.stringify({
            sessions: aiState.sessions,
            currentSessionId: aiState.currentSessionId
        }));
    } catch (e) {
        console.warn('Failed to persist AI conversation', e);
    }
}

function loadAIConversation() {
    let parsed = null;
    try {
        const saved = localStorage.getItem(AI_HISTORY_KEY);
        if (saved) {
            parsed = JSON.parse(saved);
            aiState.sessions = parsed.sessions || [];
            aiState.currentSessionId = parsed.currentSessionId || null;
        }
        // Compatible with old data: only stored messages
        if ((!aiState.sessions || !aiState.sessions.length) && parsed?.messages) {
            aiState.sessions = [{
                id: `ai_${Date.now()}`,
                title: deriveAISessionTitle(parsed.messages),
                messages: parsed.messages || [],
                conversationHistory: parsed.conversationHistory || [],
                updatedAt: Date.now()
            }];
            aiState.currentSessionId = aiState.sessions[0].id;
        }
    } catch (e) {
        console.warn('Failed to load AI conversation', e);
    }
}

function renderAIConversationFromState() {
    const messagesEl = document.getElementById('ai-messages');
    if (!messagesEl) return;

    messagesEl.innerHTML = '';
    if (!aiState.messages.length) {
        messagesEl.innerHTML = AI_WELCOME_HTML;
        updateI18nTexts();
        // Re-bind recommended question card events
        initAIStarterChips();
        return;
    }

    aiState.messages.forEach(msg => {
        renderAIMessage(msg.role, msg.content, msg.actions || []);
    });
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function clearAIChat() {
    if (aiState.controller) {
        aiState.controller.abort();
        aiState.controller = null;
    }
    aiState.messages = [];
    aiState.conversationHistory = [];
    aiState.controller = null;
    aiState.isLoading = false;

    const messagesEl = document.getElementById('ai-messages');
    if (messagesEl) {
        messagesEl.innerHTML = AI_WELCOME_HTML;
        updateI18nTexts();
        // Re-bind recommended question card events
        initAIStarterChips();
    }
    persistAIConversation();
    setAIRunning(false);
    renderAISessionList();
}

function stopAIResponse() {
    if (aiState.controller) {
        aiState.controller.abort();
    }
}

function setAIRunning(isRunning) {
    const sendBtn = document.getElementById('ai-send-btn');
    const iconWrapper = document.getElementById('ai-send-icon');
    const input = document.getElementById('ai-input');

    aiState.isLoading = isRunning;

    if (isRunning) {
        if (input) input.disabled = true;
        if (sendBtn) sendBtn.disabled = false;
        sendBtn?.classList.add('stop');
        sendBtn?.classList.add('active'); // Show active state while running
        if (iconWrapper) {
            iconWrapper.innerHTML = '<span class="icon-stop"></span>';
        }
    } else {
        if (input) input.disabled = false;
        sendBtn?.classList.remove('stop');
        if (iconWrapper) {
            iconWrapper.innerHTML = `
              <svg class="icon-send" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="12" y1="19" x2="12" y2="5"></line>
                <polyline points="5 12 12 5 19 12"></polyline>
              </svg>
            `;
        }
        // After restore, update button state based on input content
        updateAISendButtonState();
    }
}

function dedupeAIActionList(actions = []) {
    if (!Array.isArray(actions)) return [];
    const seen = new Set();
    const unique = [];

    actions.forEach(action => {
        let key;
        try {
            key = JSON.stringify({
                type: action?.type,
                filename: action?.filename,
                path: action?.path,
                preview: action?.preview,
                content: action?.content,
                value: action?.value
            });
        } catch (e) {
            key = `${action?.type || 'unknown'}::${action?.filename || action?.path || ''}::${action?.preview || action?.content || ''}`;
        }
        if (seen.has(key)) return;
        seen.add(key);
        unique.push(action);
    });

    return unique;
}

function buildAIActionHtml(actions) {
    const uniqueActions = dedupeAIActionList(actions);
    if (!uniqueActions.length) return '';

    let actionsHtml = '';
    uniqueActions.forEach((action, index) => {
        const typeLabel = {
            'modify_pipeline': t('builder_ai_action_modify_pipeline'),
            'modify_prompt': t('builder_ai_action_modify_prompt'),
            'modify_parameter': t('builder_ai_action_modify_parameter')
        }[action.type] || t('builder_ai_action_modify_generic');

        const typeIcon = {
            'modify_pipeline': '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>',
            'modify_prompt': '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>',
            'modify_parameter': '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle></svg>'
        }[action.type] || '';

        actionsHtml += `
            <div class="ai-action-block" data-action-index="${index}">
                <div class="ai-action-header">
                    <span class="ai-action-type">${typeIcon} ${typeLabel}</span>
                    <div class="ai-action-buttons">
                        <button class="ai-action-btn apply" data-action="apply" data-index="${index}">${t("common_apply")}</button>
                        <button class="ai-action-btn reject" data-action="reject" data-index="${index}">${t("common_reject")}</button>
                    </div>
                </div>
                <pre class="ai-action-preview">${escapeHtml(action.preview || action.content || '')}</pre>
            </div>
        `;
    });
    return actionsHtml;
}

function bindAIActionButtons(container, actions) {
    if (!container) return;
    const actionList = dedupeAIActionList(actions);
    container.querySelectorAll('.ai-action-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const actionType = e.target.dataset.action;
            const actionIndex = parseInt(e.target.dataset.index);
            const action = actionList[actionIndex];

            if (actionType === 'apply') {
                applyAIAction(action, e.target.closest('.ai-action-block'));
            } else {
                rejectAIAction(e.target.closest('.ai-action-block'));
            }
        });
    });
}

function renderAIMessage(role, content, actions = []) {
    const messagesEl = document.getElementById('ai-messages');
    if (!messagesEl) return null;

    const welcome = messagesEl.querySelector('.ai-welcome');
    if (welcome) welcome.remove();

    const messageEl = document.createElement('div');
    messageEl.className = `ai-message ${role}`;

    const avatarSvg = role === 'assistant'
        ? '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z"/></svg>'
        : '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>';

    const renderedContent = renderMarkdown(content || '');
    const normalizedActions = dedupeAIActionList(actions);
    const actionsHtml = normalizedActions.length ? buildAIActionHtml(normalizedActions) : '';

    messageEl.innerHTML = `
        <div class="ai-message-avatar">${avatarSvg}</div>
        <div class="ai-message-content">${renderedContent}${actionsHtml}</div>
    `;

    messagesEl.appendChild(messageEl);
    messagesEl.scrollTop = messagesEl.scrollHeight;

    // Apply code highlighting
    applyCodeHighlight(messageEl);
    applyTableEnhancements(messageEl);

    if (normalizedActions.length) {
        bindAIActionButtons(messageEl, normalizedActions);
    }

    return messageEl;
}

function addAIMessage(role, content, options = {}) {
    const { actions = [], addToHistory = false, skipState = false, persist = true } = options;
    const normalizedActions = dedupeAIActionList(actions);
    const el = renderAIMessage(role, content, normalizedActions);
    if (skipState) return el;

    aiState.messages.push({ role, content, actions: normalizedActions });
    if (addToHistory) {
        aiState.conversationHistory.push({ role, content });
    }
    if (persist) persistAIConversation();
    renderAISessionList();
    return el;
}

function addAIMessageWithActions(role, content, actions) {
    return addAIMessage(role, content, { actions: actions || [], addToHistory: role === 'assistant' });
}

async function sendAIMessage() {
    const input = document.getElementById('ai-input');
    const messagesEl = document.getElementById('ai-messages');

    if (aiState.isLoading) {
        stopAIResponse();
        return;
    }

    const message = input?.value?.trim();
    if (!message) return;

    aiState.lastUserMessage = message;

    if (!aiState.settings.apiKey) {
        addAIMessage('assistant', 'Please configure your API settings first. Click the settings icon in the top right.');
        return;
    }

    setAIView('chat');

    if (input) {
        input.value = '';
        input.style.height = 'auto';
    }

    addAIMessage('user', message, { addToHistory: true });
    showAIThinking();

    aiState.controller = new AbortController();
    aiState.isLoading = true;
    setAIRunning(true);

    const controller = aiState.controller;
    const context = buildAIContext();
    aiState.lastContextSnapshot = context;
    let accumulated = '';
    let finalActions = [];
    let placeholderEl = null;

    try {
        const response = await fetch('/api/ai/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                settings: aiState.settings,
                messages: aiState.conversationHistory,
                context: context,
                stream: true
            }),
            signal: controller.signal
        });

        hideAIThinking();

        if (!response.ok) {
            throw new Error(response.statusText || t('builder_ai_error_request_failed'));
        }

        const isStream = (response.headers.get('content-type') || '').includes('text/event-stream');

        if (!isStream) {
            const result = await response.json();
            if (result.error) throw new Error(result.error);
            accumulated = result.content || result.message || t('builder_ai_no_response');
            finalActions = result.actions || [];
            renderAIStreamingResult(accumulated, finalActions);
            aiState.isConnected = true;
            updateAIConnectionStatus();
            return;
        }

        placeholderEl = renderAIMessage('assistant', '', []);
        const contentEl = placeholderEl?.querySelector('.ai-message-content');
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const chunks = buffer.split("\n\n");
            buffer = chunks.pop();

            for (const chunk of chunks) {
                const trimmed = chunk.trim();
                if (!trimmed.startsWith('data:')) continue;
                const jsonStr = trimmed.slice(5).trim();
                if (!jsonStr) continue;

                let data;
                try { data = JSON.parse(jsonStr); } catch (_) { continue; }

                if (data.type === 'token') {
                    if (data.content) {
                        accumulated += data.content;
                        if (contentEl) {
                            contentEl.innerHTML = renderMarkdown(accumulated);
                            applyCodeHighlight(contentEl);
                            applyTableEnhancements(contentEl);
                            messagesEl.scrollTop = messagesEl.scrollHeight;
                        }
                    }
                } else if (data.type === 'final') {
                    accumulated = data.content || accumulated;
                    finalActions = data.actions || [];
                } else if (data.type === 'error') {
                    throw new Error(data.message || t('common_unknown_error'));
                }
            }
        }

        renderAIStreamingResult(accumulated, finalActions, placeholderEl);
        aiState.isConnected = true;
        updateAIConnectionStatus();
    } catch (e) {
        hideAIThinking();
        if (e.name === 'AbortError') {
            if (accumulated) {
                renderAIStreamingResult(`${accumulated}\n\n*(Stopped)*`, finalActions, placeholderEl);
            } else if (placeholderEl) {
                placeholderEl.remove();
            }
        } else {
            aiState.isConnected = false;
            updateAIConnectionStatus();
            showAIErrorWithRetry(e.message);
        }
    } finally {
        aiState.isLoading = false;
        setAIRunning(false);
        aiState.controller = null;
    }
}

function renderAIStreamingResult(content, actions = [], placeholderEl) {
    const targetEl = placeholderEl || renderAIMessage('assistant', '', []);
    const contentEl = targetEl?.querySelector('.ai-message-content');
    const finalText = content || t('builder_ai_no_response');
    const normalizedActions = dedupeAIActionList(actions);

    if (contentEl) {
        const actionsHtml = normalizedActions.length ? buildAIActionHtml(normalizedActions) : '';
        contentEl.innerHTML = `${renderMarkdown(finalText)}${actionsHtml}`;
        applyCodeHighlight(contentEl);
        applyTableEnhancements(contentEl);
        if (normalizedActions.length) {
            bindAIActionButtons(targetEl, normalizedActions);
        }
    }

    aiState.messages.push({ role: 'assistant', content: finalText, actions: normalizedActions });
    aiState.conversationHistory.push({ role: 'assistant', content: finalText });
    persistAIConversation();
    renderAISessionList();

    const messagesEl = document.getElementById('ai-messages');
    if (messagesEl) messagesEl.scrollTop = messagesEl.scrollHeight;
}

function describeAIContext(snapshot) {
    if (!snapshot) return t('builder_ai_context_none');
    const { currentMode, selectedPipeline, currentPromptFile } = snapshot;
    if (currentMode === 'prompts') {
        if (currentPromptFile) return formatTemplate(t('builder_ai_context_prompt_editing'), { file: currentPromptFile });
        return t('builder_ai_context_prompt_panel');
    }
    if (currentMode === 'parameters') {
        if (selectedPipeline) return formatTemplate(t('builder_ai_context_parameters_editing'), { pipeline: selectedPipeline });
        return t('builder_ai_context_parameters_no_pipeline');
    }
    if (currentMode === 'pipeline') {
        if (selectedPipeline) return formatTemplate(t('builder_ai_context_pipeline_editing'), { pipeline: selectedPipeline });
        return t('builder_ai_context_pipeline_canvas');
    }
    return t('builder_ai_context_none');
}

function getAIContextSnapshot() {
    const snapshot = {
        currentMode: workspaceState.currentMode,
        selectedPipeline: state.selectedPipeline,
        isBuilt: state.isBuilt
    };

    if (workspaceState.currentMode === 'pipeline') {
        const yamlEditor = document.getElementById('yaml-editor');
        if (yamlEditor) {
            snapshot.pipelineYaml = yamlEditor.value;
        }
    } else if (workspaceState.currentMode === 'parameters') {
        snapshot.parameters = state.parameterData;
    } else if (workspaceState.currentMode === 'prompts') {
        const promptEditor = document.getElementById('prompt-editor');
        if (promptEditor && workspaceState.prompts.currentFile) {
            snapshot.currentPromptFile = workspaceState.prompts.currentFile;
            snapshot.promptContent = promptEditor.value;
            snapshot.promptModified = Boolean(workspaceState.prompts.modified);
        }
    }

    snapshot.focusHint = describeAIContext(snapshot);
    aiState.lastContextSnapshot = snapshot;
    return snapshot;
}

function updateAIContextBanner(reason = '') {
    const hintEl = document.getElementById('ai-context-hint');
    const snapshot = getAIContextSnapshot();
    if (hintEl) {
        hintEl.textContent = snapshot && snapshot.focusHint
            ? `${snapshot.focusHint}`
            : t('builder_ai_context_none');
    }
}

function buildAIContext() {
    return getAIContextSnapshot();
}

function showAIThinking() {
    const messagesEl = document.getElementById('ai-messages');
    if (!messagesEl) return;

    const thinkingEl = document.createElement('div');
    thinkingEl.className = 'ai-message assistant';
    thinkingEl.id = 'ai-thinking';
    thinkingEl.innerHTML = `
        <div class="ai-message-avatar">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z"/></svg>
        </div>
        <div class="ai-message-content">
            <div class="ai-thinking">
                <span class="ai-thinking-dot"></span>
                <span class="ai-thinking-dot"></span>
                <span class="ai-thinking-dot"></span>
            </div>
        </div>
    `;

    messagesEl.appendChild(thinkingEl);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function hideAIThinking() {
    const thinkingEl = document.getElementById('ai-thinking');
    if (thinkingEl) thinkingEl.remove();
}

function flashHighlight(el) {
    if (!el) return;
    el.classList.add('ai-apply-highlight');
    setTimeout(() => el.classList.remove('ai-apply-highlight'), 1600);
}

function findParameterFieldByPath(path) {
    if (!path) return null;
    try {
        const selector = `.parameter-label[title="${CSS?.escape ? CSS.escape(path) : path}"]`;
        const label = document.querySelector(selector);
        return label ? label.closest('.parameter-field') : null;
    } catch (_) {
        return null;
    }
}

function mapAIErrorMessage(rawMessage = '') {
    const msg = String(rawMessage || '').toLowerCase();
    if (!rawMessage) return t('builder_ai_error_generic');
    if (msg.includes('list index out of range')) {
        return t('builder_ai_error_list_index');
    }
    if (msg.includes('timeout')) {
        return t('builder_ai_error_timeout');
    }
    if (msg.includes('network') || msg.includes('fetch')) {
        return t('builder_ai_error_network');
    }
    return formatTemplate(t('builder_ai_error_with_message'), { error: rawMessage });
}

function showAIErrorWithRetry(rawMessage) {
    const friendly = mapAIErrorMessage(rawMessage);
    const msgEl = addAIMessage('assistant', friendly);
    if (!msgEl) return;
    const contentEl = msgEl.querySelector('.ai-message-content');
    if (!contentEl) return;

    const card = document.createElement('div');
    card.className = 'ai-error-card';
    card.innerHTML = `
        <span>${friendly}</span>
        <button class="ai-retry-btn" type="button">${t("common_retry")}</button>
    `;
    const btn = card.querySelector('.ai-retry-btn');
    btn?.addEventListener('click', () => retryLastAIRequest());
    contentEl.appendChild(card);
}

async function retryLastAIRequest() {
    if (!aiState.lastUserMessage) return;
    const input = document.getElementById('ai-input');
    if (input) {
        input.value = aiState.lastUserMessage;
    }
    await sendAIMessage();
}

async function applyAIAction(action, blockEl) {
    try {
        let success = false;

        if (action.type === 'modify_pipeline' && workspaceState.currentMode === 'parameters') {
            showNotification(
                'info',
                t('builder_ai_pipeline_change_ignored'),
                t('builder_ai_pipeline_change_ignored_hint'),
                () => switchWorkspaceMode('pipeline')
            );
            return;
        }

        switch (action.type) {
            case 'modify_pipeline':
                success = await applyPipelineModification(action);
                break;
            case 'modify_prompt':
                success = await applyPromptModification(action);
                break;
            case 'modify_parameter':
                success = await applyParameterModification(action);
                break;
            default:
                log(formatTemplate(t('builder_ai_action_unknown'), { type: action.type }));
        }

        if (success) {
            if (blockEl) {
                blockEl.classList.add('applied');
                const btns = blockEl.querySelector('.ai-action-buttons');
                if (btns) {
                    btns.innerHTML = `<span style="color: #22c55e; font-size: 0.75rem;">${t("builder_ai_action_applied")}</span>`;
                }
                flashHighlight(blockEl);
            }
            log(t('builder_ai_action_applied_log'));
        }
    } catch (e) {
        log(formatTemplate(t('builder_ai_action_apply_failed'), { error: e.message }));
    }
}

function rejectAIAction(blockEl) {
    if (!blockEl) return;
    blockEl.classList.add('rejected');
    const btns = blockEl.querySelector('.ai-action-buttons');
    if (btns) {
        btns.innerHTML = `<span style="color: #94a3b8; font-size: 0.75rem;">${t("builder_ai_action_rejected")}</span>`;
    }
}

async function applyPipelineModification(action) {
    const yamlEditor = document.getElementById('yaml-editor');
    if (!yamlEditor) return false;

    if (action.content) {
        yamlEditor.value = action.content;
        updateYamlLineNumbers();
        await syncYamlToCanvasOnly();

        // Auto-save
        if (state.selectedPipeline) {
            await handleSubmit();
        }

        flashHighlight(yamlEditor);
        updateAIContextBanner('pipeline-apply');
        showNotification(
            'success',
            t('builder_ai_pipeline_updated_title'),
            state.selectedPipeline
                ? formatTemplate(t('builder_ai_pipeline_updated_message'), { name: state.selectedPipeline })
                : t('builder_ai_pipeline_updated_message_default'),
            () => switchWorkspaceMode('pipeline')
        );
    }

    return true;
}

async function applyPromptModification(action) {
    const promptEditor = document.getElementById('prompt-editor');
    if (!promptEditor) return false;

    // If filename is specified, switch to that file first
    if (action.filename && action.filename !== workspaceState.prompts.currentFile) {
        const file = workspaceState.prompts.files.find(f => f.name === action.filename || f.path === action.filename);
        if (file) {
            await selectPromptFile(file);
        }
    }

    if (action.content) {
        promptEditor.value = action.content;
        workspaceState.prompts.modified = true;
        updatePromptLineNumbers();
        updatePromptUI();

        // Auto-save
        await saveCurrentPrompt();
        flashHighlight(promptEditor);
        updateAIContextBanner('prompt-apply');
        showNotification(
            'success',
            t('builder_ai_prompt_updated_title'),
            action.filename
                ? formatTemplate(t('builder_ai_prompt_updated_message'), { name: action.filename })
                : t('builder_ai_prompt_updated_message_default'),
            () => switchWorkspaceMode('prompts')
        );
    }

    return true;
}

async function applyParameterModification(action) {
    if (!action.path || action.value === undefined) return false;

    // Ensure parameter data is present
    if (!state.parameterData) {
        if (state.selectedPipeline) {
            try {
                state.parameterData = cloneDeep(await fetchJSON(`/api/pipelines/${encodeURIComponent(state.selectedPipeline)}/parameters`));
            } catch (e) {
                state.parameterData = {};
                log(formatTemplate(t('builder_ai_params_load_failed'), { error: e.message }));
            }
        } else {
            state.parameterData = {};
        }
    }

    setNestedValue(state.parameterData, action.path, action.value);
    renderParameterFormInline();

    // Auto-save
    await persistParameterData();
    const fieldEl = findParameterFieldByPath(action.path) || document.getElementById('parameter-form');
    flashHighlight(fieldEl);
    updateAIContextBanner('params-apply');
    showNotification(
        'success',
        t('builder_ai_parameter_updated_title'),
        formatTemplate(t('builder_ai_parameter_updated_message'), { path: action.path }),
        () => switchWorkspaceMode('parameters')
    );

    return true;
}
