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
  chatCollectionSelect: document.getElementById("chat-collection-select"),

  // [新增] 视图容器
  chatMainView: document.getElementById("chat-main-view"),
  kbMainView: document.getElementById("kb-main-view"),
  // [新增] 按钮
  kbBtn: document.getElementById("kb-btn"),

  chatSidebar: document.querySelector(".chat-sidebar"),
  chatSidebarToggleBtn: document.getElementById("sidebar-toggle-btn"),

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

  // --- [Start] Knowledge Base Elements ---
  listRaw: document.getElementById("list-raw"),
  listCorpus: document.getElementById("list-corpus"),
  listChunks: document.getElementById("list-chunks"),
  // 文件上传 input (虽然是隐藏的，但我们需要引用它来绑定事件或触发点击)
  fileUpload: document.getElementById("file-upload"),
  // 状态条
  taskStatusBar: document.getElementById("task-status-bar"),
  taskMsg: document.getElementById("task-msg"),
  // 数据库配置元素
  dbConnectionStatus: document.getElementById("db-connection-status"),
  dbUriDisplay: document.getElementById("db-uri-display"),
  dbConfigModal: document.getElementById("db-config-modal"), // 新增的配置弹窗
  cfgUri: document.getElementById("cfg-uri"),                 // 配置弹窗 - URI输入
  cfgToken: document.getElementById("cfg-token"),              // 配置弹窗 - Token输入
  listIndexes: document.getElementById("list-indexes"),         // Collection 列表容器
  modalTargetDb: document.getElementById("modal-target-db"),   // Milvus 弹窗中的提示文本
  
  // Milvus 弹窗相关 (保留并确认 ID)
  milvusDialog: document.getElementById("milvus-dialog"),
  idxCollection: document.getElementById("idx-collection"),
  idxMode: document.getElementById("idx-mode"),
  // [新增] 刷新按钮
  refreshCollectionsBtn: document.getElementById("refresh-collections-btn"),
  // --- [End] Knowledge Base Elements ---
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

// ==========================================
// --- Knowledge Base Logic ---
// ==========================================


let currentTargetFile = null;
let existingFilesSnapshot = new Set();

// [修改] 打开导入工作台
window.openImportModal = async function() {
    const modal = document.getElementById('import-modal');
    if (modal) modal.showModal();
    
    // 1. 先清空快照
    existingFilesSnapshot.clear();
    
    // 2. 获取当前所有文件，建立“基准线”
    try {
        const data = await fetchJSON('/api/kb/files');
        
        // 把当前已有的所有文件路径加入快照
        // 这样，凡是现在就在列表里的，都不是“新”的
        const recordFiles = (list) => list.forEach(f => existingFilesSnapshot.add(f.path));
        
        recordFiles(data.raw);
        recordFiles(data.corpus);
        recordFiles(data.chunks);
        
        // 3. 立即渲染一次（此时不会有高亮，因为都在快照里）
        // 这里手动调用 render 避免 refreshKBFiles 还没拉取完
        refreshKBModalViews(data);
        
    } catch(e) {
        console.error("Init modal failed:", e);
    }
};

// [新增] 关闭导入工作台
window.closeImportModal = function() {
    const modal = document.getElementById('import-modal');
    if (modal) modal.close();
    
    // 关闭后刷新主界面，确保新生成的 Collection 立即出现在书架上
    refreshKBFiles();
};

// [新增] 清空暂存区
window.clearStagingArea = async function() {
    if (!confirm("Are you sure you want to clear ALL temporary files (Raw, Corpus, Chunks)?")) return;
    
    try {
        const res = await fetch('/api/kb/staging/clear', { method: 'POST' });
        const data = await res.json();
        
        if (res.ok) {
            const total = data.total_deleted || 0;
            const counts = data.deleted_counts || {};
            let message = `Successfully cleared staging area!\n\nDeleted:\n`;
            message += `- Raw: ${counts.raw || 0} items\n`;
            message += `- Corpus: ${counts.corpus || 0} items\n`;
            message += `- Chunks: ${counts.chunks || 0} items\n`;
            message += `\nTotal: ${total} items`;
            
            if (data.errors && data.errors.length > 0) {
                message += `\n\nNote: Some errors occurred:\n${data.errors.slice(0, 3).join('\n')}`;
                if (data.errors.length > 3) {
                    message += `\n... and ${data.errors.length - 3} more errors`;
                }
            }
            
            alert(message);
            await refreshKBFiles();
        } else {
            alert("Clear failed: " + (data.error || res.statusText));
        }
    } catch(e) {
        console.error(e);
        alert("Clear error: " + e.message);
    }
};



// [新增] 专门用于刷新弹窗视图的辅助函数
function refreshKBModalViews(data) {
    renderKBList(document.getElementById('list-raw'), data.raw, 'build_text_corpus', 'Parse');
    renderKBList(document.getElementById('list-corpus'), data.corpus, 'corpus_chunk', 'Chunk');
    renderKBList(document.getElementById('list-chunks'), data.chunks, 'milvus_index', 'Index');
}

// [修改] 刷新文件列表 (主函数)
async function refreshKBFiles() {
    try {
        const data = await fetchJSON('/api/kb/files');
        
        // 1. 刷新弹窗视图
        refreshKBModalViews(data);
        
        // 2. 刷新主页书架
        renderCollectionList(null, data.index); 
        
        // 3. 更新状态
        updateDbStatusUI(data.db_status, data.db_config);
        
    } catch (e) {
        console.error("Failed to load KB files:", e);
    }
}

// [修改] 渲染流水线列表 (应用快照高亮)
function renderKBList(container, files, nextPipeline, actionLabel) {
    if (!container) return;
    container.innerHTML = '';
    
    if (!files || files.length === 0) {
        container.innerHTML = '<div class="text-muted small text-center mt-5 opacity-50">Empty</div>';
        return;
    }

    // 按时间倒序，新文件排前面
    files.sort((a, b) => (b.mtime || 0) - (a.mtime || 0));

    files.forEach(f => {
        const div = document.createElement('div');
        
        // [核心修复] 高亮逻辑
        // 如果这个文件的路径 不在 打开弹窗时的快照里，那它就是新的！
        const isNew = !existingFilesSnapshot.has(f.path);
        
        div.className = `file-item ${isNew ? 'new-upload' : ''}`;
        
        // --- 以下 UI 生成代码保持不变 ---
        const isFolder = f.type === 'folder';
        const svgFolder = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>`;
        const svgFile = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>`;
        const iconSvg = isFolder ? svgFolder : svgFile;

        // View 按钮
        let viewBtn = '';
        if (isFolder) {
            viewBtn = `<button class="btn btn-sm btn-link text-muted p-0 me-2" onclick="window.inspectFolder('${f.category}', '${f.name}')" title="View"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg></button>`;
        }

        // Action 按钮
        let actionBtn = `<button class="btn btn-sm btn-light border ms-auto" style="font-size:0.75rem;" onclick="window.handleKBAction('${f.path}', '${nextPipeline}')">${actionLabel}</button>`;
        
        // Delete 按钮
        let deleteBtn = '';
        if (f.category !== 'collection') {
            deleteBtn = `<button class="btn btn-sm text-danger ms-2" onclick="deleteKBFile('${f.category}', '${f.name}')">×</button>`;
        }

        div.innerHTML = `
            <div class="d-flex align-items-center w-100">
                <div class="text-muted me-2">${iconSvg}</div>
                <div class="text-truncate small text-dark" style="max-width: 130px;" title="${f.name}">${f.name}</div>
                ${isFolder && f.file_count ? `<span class="badge bg-light text-secondary border ms-1" style="font-size:0.6rem">${f.file_count}</span>` : ''}
                ${viewBtn}
                ${actionBtn}
                ${deleteBtn}
            </div>
        `;
        container.appendChild(div);
    });
}

// 2. 新增查看函数 (挂载到 window)
window.inspectFolder = async function(category, folderName) {
    const modal = document.getElementById('folder-detail-modal');
    const listContainer = document.getElementById('folder-detail-list');
    const title = document.getElementById('folder-detail-title');
    
    // 设置标题
    if (title) title.textContent = folderName;
    
    // 显示 Loading
    if (listContainer) listContainer.innerHTML = '<div class="text-center text-muted p-3">Loading...</div>';
    
    // 打开弹窗
    if (modal) modal.showModal();

    try {
        const res = await fetch(`/api/kb/files/inspect?category=${category}&name=${encodeURIComponent(folderName)}`);
        
        const data = await res.json();

        if (data.files && data.files.length > 0) {
            listContainer.innerHTML = data.files.map(f => `
                <div class="folder-file-row">
                    <span>📄 ${f.name}</span>
                    <span class="text-muted">${(f.size/1024).toFixed(1)} KB</span>
                </div>
            `).join('');
        } else {
            listContainer.innerHTML = '<div class="text-center text-muted small mt-3">Empty Folder</div>';
        }
    } catch (e) {
        if (listContainer) listContainer.innerHTML = `<div class="text-danger small p-2">Error: ${e.message}</div>`;
        console.error(e);
    }
};

// [修改] 渲染 Collection 列表 -> 书架卡片模式
function renderCollectionList(container, collections) {
    const grid = document.getElementById('bookshelf-grid');
    if (!grid) return;
    
    grid.innerHTML = '';

    if (!collections || collections.length === 0) {
        grid.innerHTML = `
            <div class="col-12 text-center py-5 text-muted" style="grid-column: 1 / -1;">
                <div style="font-size:3rem; margin-bottom:1rem; opacity:0.3;">📚</div>
                <h5>Library is empty</h5>
                <p>Click "New Collection" to import documents.</p>
            </div>
        `;
        return;
    }

    collections.forEach(c => {
        const card = document.createElement('div');
        card.className = 'collection-card';
        
        const countStr = c.count !== undefined ? `${c.count} vectors` : 'Ready';

        // [修改] 定义一个精致的书本 SVG (Stroke 风格)
        const bookSvg = `
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path>
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path>
            </svg>
        `;
        
        // 渲染卡片
        card.innerHTML = `
            <div class="book-cover">
                <div class="book-icon">${bookSvg}</div>
                
                <button class="btn-delete-book" onclick="event.stopPropagation(); deleteKBFile('collection', '${c.name}')" title="Delete Collection">
                     <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                </button>
            </div>
            <div class="book-info">
                <div class="book-title" title="${c.name}">${c.name}</div>
                <div class="book-meta">
                    <span>${countStr}</span>
                    <span class="badge bg-light text-dark border">Vector DB</span>
                </div>
            </div>
        `;
        
        grid.appendChild(card);
    });
}

// 4. UI 状态更新 (新增)
function updateDbStatusUI(status, config) {
    currentDbConfig = config; 
    
    if (!els.dbConnectionStatus || !els.dbUriDisplay) return;

    // 状态 Badge
    if (status === 'connected') {
        els.dbConnectionStatus.className = 'badge rounded-pill bg-success';
        els.dbConnectionStatus.textContent = 'Connected';
    } else {
        els.dbConnectionStatus.className = 'badge rounded-pill bg-danger';
        els.dbConnectionStatus.textContent = 'Disconnected';
    }
    
    // URI 显示
    let uri = config.milvus.uri || "Not configured";
    if (uri.length > 50) uri = '...' + uri.slice(-45); // 截断长 URI
    els.dbUriDisplay.textContent = uri;
}

// 5. 配置弹窗逻辑 (新增 - 挂载到 window)
window.openDbConfigModal = async function() {
    const res = await fetchJSON('/api/kb/config');
    const cfg = res;
    
    // 从 milvus 字段下读取
    const milvus = cfg.milvus || {};
    
    if (els.cfgUri) els.cfgUri.value = milvus.uri || '';
    if (els.cfgToken) els.cfgToken.value = milvus.token || '';
    
    // 暂存完整配置结构，以便保存时合并
    window._currentFullKbConfig = cfg;
    
    if (els.dbConfigModal) els.dbConfigModal.showModal();
};

window.saveDbConfig = async function() {
    if (!els.cfgUri) return;
    const uri = els.cfgUri.value.trim();
    const token = els.cfgToken.value.trim();
    
    if(!uri) { alert("URI is required"); return; }
    
    const fullConfig = window._currentFullKbConfig || {};
    if (!fullConfig.milvus) fullConfig.milvus = {};

    // 只更新 URI 和 Token，保留其他高级字段
    fullConfig.milvus.uri = uri;
    fullConfig.milvus.token = token;
    
    // 发送完整的 JSON 结构回去
    await fetch('/api/kb/config', {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(fullConfig)
    });
    
    if (els.dbConfigModal) els.dbConfigModal.close();
    refreshKBFiles(); // 立即刷新，测试连接状态
};

// ==========================================
// --- Chunk Configuration Logic ---
// ==========================================

// 1. 定义默认配置状态
let chunkConfigState = {
    chunk_backend: "sentence",
    tokenizer_or_token_counter: "character",
    chunk_size: 512,
    use_title: true
};

// 2. 打开配置弹窗 (回显当前状态)
window.openChunkConfigModal = function() {
    const modal = document.getElementById('chunk-config-modal');
    
    // 回显数据
    document.getElementById('cfg-chunk-backend').value = chunkConfigState.chunk_backend;
    document.getElementById('cfg-chunk-tokenizer').value = chunkConfigState.tokenizer_or_token_counter;
    document.getElementById('cfg-chunk-size').value = chunkConfigState.chunk_size;
    document.getElementById('cfg-chunk-title').value = chunkConfigState.use_title ? "true" : "false";
    
    if (modal) modal.showModal();
};

// 3. 保存配置
window.saveChunkConfig = function() {
    const backend = document.getElementById('cfg-chunk-backend').value;
    const tokenizer = document.getElementById('cfg-chunk-tokenizer').value;
    const size = parseInt(document.getElementById('cfg-chunk-size').value, 10);
    const useTitleStr = document.getElementById('cfg-chunk-title').value;

    if (isNaN(size) || size <= 0) {
        alert("Chunk size must be a positive number");
        return;
    }

    // 更新全局状态
    chunkConfigState = {
        chunk_backend: backend,
        tokenizer_or_token_counter: tokenizer,
        chunk_size: size,
        use_title: (useTitleStr === "true")
    };

    const modal = document.getElementById('chunk-config-modal');
    if (modal) modal.close();
    
    // 可选：给个提示
    // alert("Chunk configuration saved!"); 
    console.log("Chunk Config Updated:", chunkConfigState);
};

// ==========================================

// 6. 处理操作按钮点击 (修改 - 挂载到 window)
window.handleKBAction = function(filePath, pipelineName) {
    currentTargetFile = filePath;
    
    if (pipelineName === 'milvus_index') {
        // 更新 Milvus 弹窗里的提示和默认值
        const uriTxt = els.dbUriDisplay ? els.dbUriDisplay.textContent : "Current DB";
        if (els.modalTargetDb) els.modalTargetDb.textContent = uriTxt;

        // 自动填充 Collection 名 (使用文件名作为默认集合名)
        const fileName = filePath.split('/').pop().replace('.jsonl', '').replace('.', '_');
        if (els.idxCollection) els.idxCollection.value = fileName;
        
        if (els.milvusDialog) els.milvusDialog.showModal();
        return;
    }

    let extraParams = {};
    if (pipelineName === 'corpus_chunk') {
        extraParams = { ...chunkConfigState }; // 展开对象传递
    }
    
    // 其他任务直接开始
    runKBTask(pipelineName, filePath, extraParams);
};

// 7. 确认建索引 (修改 - 挂载到 window)
window.confirmIndexTask = function() {
    if (!els.idxCollection || !els.idxMode) return;
    const collName = els.idxCollection.value.trim();
    
    if (!collName) { alert("Collection name required"); return; }
    const mode = els.idxMode.value;
    
    if (els.milvusDialog) els.milvusDialog.close();
    
    // 发起任务
    runKBTask('milvus_index', currentTargetFile, {
        collection_name: collName,
        index_mode: mode
    });
};

// [修改] handleFileUpload 恢复原样 (不再需要 sessionStems)
window.handleFileUpload = async function(input) {
    if (!input.files.length) return;
    
    // 不需要手动记录文件名了，Snapshot 逻辑会自动处理

    const formData = new FormData();
    for (let i = 0; i < input.files.length; i++) {
        formData.append('file', input.files[i]);
    }

    updateKBStatus(true, 'Uploading...');
    try {
        const res = await fetch('/api/kb/upload', { method: 'POST', body: formData });
        if (res.ok) {
            await refreshKBFiles(); 
            updateKBStatus(false);
        } else {
            alert('Upload failed');
            updateKBStatus(false);
        }
    } catch (e) {
        console.error(e);
        updateKBStatus(false);
        alert("Upload error: " + e.message);
    } finally {
        input.value = '';
    }
};

// 9. 删除文件 (新增 - 挂载到 window)
window.deleteKBFile = async function(category, filename) {
    const action = category === 'collection' ? 'drop this collection' : 'delete this file';
    if(!confirm(`Permanently ${action} (${filename})?`)) return;
    
    try {
        const res = await fetch(`/api/kb/files/${category}/${filename}`, { method: 'DELETE' });
        if(res.ok) {
            refreshKBFiles();
        } else {
            const err = await res.json();
            alert("Delete failed: " + (err.error || res.statusText));
        }
    } catch(e) {
        console.error(e);
    }
};



// 6. 核心：提交任务并轮询
async function runKBTask(pipelineName, filePath, extraParams = {}) {
    updateKBStatus(true, `Running ${pipelineName}...`);
    
    try {
        // A. 提交任务
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
            // B. 开始轮询
            pollTaskStatus(data.task_id);
        } else {
            throw new Error(data.error || 'Task start failed');
        }
    } catch (e) {
        alert(e.message);
        updateKBStatus(false);
    }
}

// 7. 轮询逻辑
function pollTaskStatus(taskId) {
    const interval = setInterval(async () => {
        try {
            const res = await fetch(`/api/kb/status/${taskId}`);
            const task = await res.json();
            
            if (task.status === 'success') {
                clearInterval(interval);
                updateKBStatus(false);
                // 1. 刷新 KB 管理界面的列表
                refreshKBFiles(); 
                
                // [新增] 2. 同时也刷新 Chat 界面的下拉菜单
                // 这样当你建好索引后，聊天框里马上就能选到了
                renderChatCollectionOptions();
                console.log('Task Result:', task.result);
            } else if (task.status === 'failed') {
                clearInterval(interval);
                updateKBStatus(false);
                alert(`Task Failed: ${task.error}`);
            } else {
                // still running...
                console.log('Task running...');
            }
        } catch (e) {
            clearInterval(interval);
            updateKBStatus(false);
        }
    }, 1500); // 每 1.5 秒查一次
}

// UI 工具：显示/隐藏状态条
function updateKBStatus(show, msg = '') {
    if (!els.taskStatusBar || !els.taskMsg) return;
    if (show) {
        els.taskStatusBar.classList.remove('hidden');
        els.taskMsg.textContent = msg;
    } else {
        els.taskStatusBar.classList.add('hidden');
    }
}

// ==========================================
// --- End Knowledge Base Logic ---
// ==========================================

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

// 引用高亮函数 - 将 [1] 替换为可点击的引用链接
function formatCitationHtml(html) {
    if (!html) return "";
    return html.replace(/\[(\d+)\]/g, (match, p1) => {
        const id = parseInt(p1, 10);
        return `<span class="citation-link" onclick="scrollToReference(${id})">[${id}]</span>`;
    });
}

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
        {left: '$$', right: '$$', display: true},
        {left: '$', right: '$', display: false},
        {left: '\\(', right: '\\)', display: false},
        {left: '\\[', right: '\\]', display: true}
      ],
      throwOnError: false,
      ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    });
  } catch (e) {
    console.error("KaTeX rendering error:", e);
  }
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
    
    // Protect Math from Markdown processing
    const { text: protectedText, mathBlocks } = protectMath(text);

    let rendererOptions = undefined;
    if (!allowCodeBlock && window.marked) {
      const renderer = new window.marked.Renderer();
      renderer.code = (code) => `<p>${escapeHtml(code).replace(/\n/g, "<br>")}</p>`;
      rendererOptions = { renderer };
    }
    
    const rawHtml = window.marked.parse(protectedText, rendererOptions);
    let sanitized = window.DOMPurify ? DOMPurify.sanitize(rawHtml) : rawHtml;
    
    // Restore original LaTeX (with $$ delimiters intact)
    sanitized = restoreMath(sanitized, mathBlocks);

    return sanitized;
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

// 1. 启动引擎 (幂等操作：如果已启动则忽略)
async function startEngine(pipelineName) {
    if (!pipelineName) return;
    
    // 如果当前已经有正在运行且是同一个 Pipeline，直接返回
    if (state.chat.engineSessionId && state.chat.activeEngines[pipelineName] === state.chat.engineSessionId) {
        return;
    }

    // 如果有别的引擎在跑，先停掉
    if (state.chat.engineSessionId) {
        await stopEngine();
    }

    state.chat.demoLoading = true;
    updateDemoControls(); // 更新 UI 显示 "Loading..."
    setChatStatus("Initializing...", "warn");

    try {
        const newSid = uuidv4();
        
        // 调用后端启动
        await fetchJSON(`/api/pipelines/${encodeURIComponent(pipelineName)}/demo/start`, { 
             method: "POST", body: JSON.stringify({ session_id: newSid })
        });
        
        state.chat.engineSessionId = newSid;
        state.chat.activeEngines[pipelineName] = newSid;
        
        setChatStatus("Ready", "ready");
        log(`Engine started for ${pipelineName}`);
        
    } catch (err) {
        console.error(err);
        setChatStatus("Engine Error", "error");
        state.chat.engineSessionId = null;
    } finally {
        state.chat.demoLoading = false;
        updateDemoControls();
    }
}

// 2. 停止引擎
async function stopEngine() {
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
    
    setChatStatus("Offline", "info");
    updateDemoControls();
}

// 3. 原来的 toggle 函数保留作为兼容，或者直接废弃
// (这里留着是为了防止 HTML 里有 onclick 报错，但实际上我们不再点它了)
async function toggleDemoSession() {
    if (state.chat.engineSessionId) await stopEngine();
    else await startEngine(state.selectedPipeline);
}

function updateDemoControls() {
    // 1. 总是启用输入框 (只要不是 loading)
    // 因为我们希望用户随时能输入，如果引擎没好，点击发送时再提示或者自动等待
    if (els.chatInput) els.chatInput.disabled = state.chat.demoLoading;
    if (els.chatSend) els.chatSend.disabled = state.chat.demoLoading;

    // 2. 隐藏或改变按钮文本
    if (els.demoToggleBtn) {
        if (state.chat.demoLoading) {
            els.demoToggleBtn.innerHTML = "Connecting...";
            els.demoToggleBtn.disabled = true;
            els.demoToggleBtn.classList.remove("d-none"); // 加载时显示一下
        } else {
            // 加载完了就隐藏按钮，因为是自动的，不需要用户点
            els.demoToggleBtn.classList.add("d-none"); 
        }
    }
}

// --- Chat Logic (Updated with Streaming) ---

// [新增] 渲染聊天界面的 Collection 下拉选项
async function renderChatCollectionOptions() {
    if (!els.chatCollectionSelect) return;
    
    // 保存当前选中的值，以免刷新后重置
    const currentVal = els.chatCollectionSelect.value;
    
    try {
        // 复用后端的列表接口
        const data = await fetchJSON('/api/kb/files');
        const collections = data.index || []; // data.index 存放的是 collection 列表
        
        els.chatCollectionSelect.innerHTML = '<option value="">No Knowledge Base</option>';
        
        collections.forEach(c => {
            const opt = document.createElement("option");
            opt.value = c.name;
            // 显示名字和数据量
            opt.textContent = `${c.name} (${c.count || 0})`;
            els.chatCollectionSelect.appendChild(opt);
        });
        
        // 尝试恢复选中状态
        if (currentVal) {
            // 检查新列表里是否还有这个值
            const exists = collections.find(c => c.name === currentVal);
            if (exists) els.chatCollectionSelect.value = currentVal;
        }

        // 渲染完 options 后，手动触发一次视觉更新
        // 确保 "Knowledge Base" 这里的文字显示正确（比如维持之前选中的状态）
        if (window.updateKbLabel && els.chatCollectionSelect) {
            window.updateKbLabel(els.chatCollectionSelect);
        }
        
    } catch (e) {
        console.error("Failed to load collections for chat:", e);
        // 出错时不覆盖默认选项，或者显示错误
    }
}

// [新增] 切换到知识库视图
function openKBView() {
    if (!els.chatMainView || !els.kbMainView) return;

    // 刷新数据 [新增]
    refreshKBFiles();
    
    // 隐藏聊天，显示知识库
    els.chatMainView.classList.add("d-none");
    els.kbMainView.classList.remove("d-none");
    
    // 更新按钮状态
    if (els.kbBtn) els.kbBtn.classList.add("active");
    
    // 取消所有 Session 列表的高亮 (视觉上告诉用户现在没在聊任何会话)
    const items = document.querySelectorAll(".chat-session-item");
    items.forEach(el => el.classList.remove("active"));
}

// [新增] 切换回聊天视图 (复位)
function backToChatView() {
    if (!els.chatMainView || !els.kbMainView) return;

    els.kbMainView.classList.add("d-none");
    els.chatMainView.classList.remove("d-none");
    
    if (els.kbBtn) els.kbBtn.classList.remove("active");
    
    // 重新渲染侧边栏以恢复当前会话的高亮状态
    renderChatCollectionOptions();
    renderChatSidebar(); 
}

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
    
    // 1. 保存旧会话
    saveCurrentSession(true);
    
    // 2. 停止旧引擎 (为了节省资源，切换时先停掉上一个)
    // 如果你想保留后台运行，可以去掉这行，但推荐停掉以防端口冲突或资源占用
    await stopEngine(); 

    // 3. 加载新 Pipeline 结构
    await loadPipeline(name); 
    
    // 4. 加载参数
    try {
        state.parameterData = await fetchJSON(`/api/pipelines/${encodeURIComponent(name)}/parameters`);
        state.parametersReady = true;
    } catch (e) {
        console.warn("Parameters not found.");
        state.parametersReady = false;
    }

    // 5. 新建会话 UI
    createNewChatSession();
    renderChatPipelineMenu(); 
    
    // [核心新增] 6. 自动启动新引擎！
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
    
    // [严重错误修复] 删除下面这行！切换 Pipeline 不应该清空所有会话列表！
    // state.chat.sessions = [];  <-- 删除它
    
    // [新增] 应该只清除当前显示的会话，而不是所有会话数据
    // 但为了 UI 逻辑简单，这里什么都不做，renderChatSidebar 会自动处理显示
    
    // [检查 activeEngines] 恢复 Session ID
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
    backToChatView();
}

function loadChatSession(sessionId) {
    if (state.chat.running) return; // 正在生成时不许切
    
    // 先保存当前正在进行的会话
    saveCurrentSession(false); 
    
    const session = state.chat.sessions.find(s => s.id === sessionId);
    if (!session) return;

    state.chat.currentSessionId = session.id;
    state.chat.history = cloneDeep(session.messages || []); // 深拷贝恢复
    
    // [可选] 如果你想做的更高级：
    // 如果这个历史会话是属于另一个 Pipeline 的 (session.pipeline)，
    // 你可以在这里自动调用 switchChatPipeline(session.pipeline) 来切换上下文。
    // 但为了简单起见，目前只加载文本内容即可。
    
    renderChatHistory();
    renderChatSidebar();
    setChatStatus("Ready", "ready");
    
    // 移动端适配：加载后自动收起侧边栏
    const sidebar = document.querySelector('.chat-sidebar');
    if (window.innerWidth < 768 && sidebar) {
        sidebar.classList.remove('show');
    }

    backToChatView();
}

function saveCurrentSession(force = false) {
    if (!state.chat.currentSessionId) return;
    
    // 如果当前没有任何消息，且不是强制保存，则不保存（避免产生大量空会话）
    if (!force && state.chat.history.length === 0) {
        // 从列表中移除当前空会话
        state.chat.sessions = state.chat.sessions.filter(s => s.id !== state.chat.currentSessionId);
        // 更新 UI 和 本地存储
        renderChatSidebar();
        localStorage.setItem("ultrarag_sessions", JSON.stringify(state.chat.sessions));
        return;
    }
    
    let session = state.chat.sessions.find(s => s.id === state.chat.currentSessionId);
    
    // 生成标题：取第一条用户消息的前20个字
    let title = "New Chat";
    const firstUserMsg = state.chat.history.find(m => m.role === 'user');
    if (firstUserMsg) {
        title = firstUserMsg.text.slice(0, 20) + (firstUserMsg.text.length > 20 ? "..." : "");
    }

    if (!session) {
        // 新建会话对象
        session = { 
            id: state.chat.currentSessionId, 
            title: title, 
            messages: cloneDeep(state.chat.history), // 深拷贝防止引用问题
            pipeline: state.selectedPipeline,        // [新增] 记录是用哪个 Pipeline 聊的
            timestamp: Date.now()                    // [新增] 记录时间
        };
        state.chat.sessions.unshift(session); // 加到最前面
    } else {
        // 更新现有会话
        // 先移除旧位置，再插到最前面（置顶）
        state.chat.sessions = state.chat.sessions.filter(s => s.id !== state.chat.currentSessionId);
        session.messages = cloneDeep(state.chat.history);
        session.timestamp = Date.now();
        
        // 如果是默认标题，尝试更新为新标题
        if (session.title === "New Chat" || session.title === "Untitled Chat") {
             session.title = title;
        }
        state.chat.sessions.unshift(session);
    }

    // [关键] 渲染侧边栏 并 持久化到 LocalStorage
    renderChatSidebar();
    localStorage.setItem("ultrarag_sessions", JSON.stringify(state.chat.sessions));

    // [补全] 记录当前活跃的会话 ID，以便刷新后恢复
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
        els.chatSessionList.innerHTML = '<div class="text-muted small px-2">No history</div>';
        return;
    }
    
    state.chat.sessions.forEach(session => {
        // 容器
        const itemDiv = document.createElement("div");
        itemDiv.className = `chat-session-item d-flex justify-content-between align-items-center ${session.id === state.chat.currentSessionId ? 'active' : ''}`;
        
        // 标题按钮 (点击加载)
        const titleBtn = document.createElement("span");
        titleBtn.className = "text-truncate flex-grow-1";
        titleBtn.style.cursor = "pointer";
        titleBtn.textContent = session.title || "Untitled Chat";
        titleBtn.onclick = (e) => {
            e.stopPropagation(); // 防止冒泡
            loadChatSession(session.id);
        };
        
        // 删除按钮 (小垃圾桶)
        const delBtn = document.createElement("button");
        delBtn.className = "btn btn-sm btn-icon text-muted ms-2";
        delBtn.innerHTML = "×"; // 或者用图标
        delBtn.title = "Delete Chat";
        delBtn.style.width = "20px";
        delBtn.style.height = "20px";
        delBtn.style.lineHeight = "1";
        delBtn.onclick = (e) => {
            e.stopPropagation();
            deleteChatSession(session.id);
        };

        itemDiv.appendChild(titleBtn);
        itemDiv.appendChild(delBtn);
        els.chatSessionList.appendChild(itemDiv);
    });
}

// [新增] 删除会话辅助函数
function deleteChatSession(sessionId) {
    if (!confirm("Delete this chat?")) return;
    
    state.chat.sessions = state.chat.sessions.filter(s => s.id !== sessionId);
    localStorage.setItem("ultrarag_sessions", JSON.stringify(state.chat.sessions));
    
    // [补全] 如果删除了当前会话，清理 last_active_id
    if (state.chat.currentSessionId === sessionId) {
        localStorage.removeItem("ultrarag_last_active_id"); // 清理
        createNewChatSession();
    } else {
        renderChatSidebar();
    }
}

function appendChatMessage(role, text, meta = {}) {
  const entry = { role, text, meta, timestamp: new Date().toISOString() };
  state.chat.history.push(entry);
  renderChatHistory();
  saveCurrentSession(); 
}

function formatCitationHtml(html) {
    if (!html) return "";
    // [关键修改] 增加 onclick="scrollToReference(1)"
    // 注意：scrollToReference 函数必须挂在 window 上或定义在全局作用域
    return html.replace(
        /\[(\d+)\]/g, 
        '<span class="citation-link" onclick="scrollToReference($1)">[$1]</span>'
    );
}

// 2. [主函数] 修改后的 renderChatHistory
function renderChatHistory() {
    if (!els.chatHistory) return;
    els.chatHistory.innerHTML = "";
    // [核心修改] 仿 Gemini 空白欢迎页
    if (state.chat.history.length === 0) { 
        els.chatHistory.innerHTML = `
            <div class="empty-state-wrapper fade-in-up">
                <div class="greeting-section">
                    <div class="greeting-text">
                        <span class="greeting-gradient">Welcome back.</span>
                    </div>
                    <h1 class="greeting-title">Ready when you are.</h1>
                </div>
                
                <div class="suggestion-chips">
                    <button class="chip-btn" onclick="setQuickPrompt('Summarize this document')">
                        <span class="chip-icon">📝</span>
                        <span>Research</span>
                    </button>
                    <button class="chip-btn" onclick="setQuickPrompt('Write a Python script for RAG')">
                        <span class="chip-icon">💻</span>
                        <span>Write Code</span>
                    </button>
                    <button class="chip-btn" onclick="setQuickPrompt('Explain quantum computing')">
                        <span class="chip-icon">💡</span>
                        <span>Learn Concept</span>
                    </button>
                    <button class="chip-btn" onclick="setQuickPrompt('Brainstorm marketing ideas')">
                        <span class="chip-icon">🤯</span>
                        <span>Brainstorm</span>
                    </button>
                </div>
            </div>
        `;
        return; 
    }
    state.chat.history.forEach((entry) => {
        const bubble = document.createElement("div"); 
        // 加上 fade-in 动画类，稍微好看点
        bubble.className = `chat-bubble ${entry.role} fade-in-up`;
        // 历史记录直接展示，不需要动画延迟
        bubble.style.animationDelay = "0ms";

        const content = document.createElement("div"); 
        content.className = "msg-content";

        if (entry.role === "assistant") {
            let htmlContent = renderMarkdown(entry.text || "", { unwrapLanguages: MARKDOWN_LANGS });
            content.innerHTML = formatCitationHtml(htmlContent);
            renderLatex(content);
        } else {
            // 用户消息：保留换行效果
            // 将换行符转换为HTML，同时转义HTML特殊字符防止XSS
            const escapedText = entry.text
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
            content.innerHTML = escapedText.replace(/\n/g, '<br>');
        }
        bubble.appendChild(content);

        // 渲染底部的引用卡片
        if (entry.meta && entry.meta.sources) {
            // 注意：renderSources 内部通常会处理 DOM 生成
            // 确保 entry.meta.sources 里的对象已经包含了正确的 displayId (如 6, 7, 8)
            // 这样卡片的数字就能和正文里的 [6], [7], [8] 对应上
            renderSources(bubble, entry.meta.sources);
        }
        
        // 渲染调试信息 (Hint)
        if (entry.meta && entry.meta.hint) {
            const hintDiv = document.createElement("div");
            hintDiv.className = "text-xs text-muted mt-2 pt-2 border-top";
            hintDiv.textContent = entry.meta.hint;
            bubble.appendChild(hintDiv);
        }

        els.chatHistory.appendChild(bubble);
    });
    els.chatHistory.scrollTop = els.chatHistory.scrollHeight;
}

// [新增] 快速填入提示词的辅助函数 (加在 main.js 任意位置)
window.setQuickPrompt = function(text) {
    if (els.chatInput) {
        els.chatInput.value = text;
        // 自动调整textarea高度
        els.chatInput.style.height = 'auto';
        els.chatInput.style.height = (els.chatInput.scrollHeight) + 'px';
        els.chatInput.focus();
        // 可选：如果想点击直接发送，取消下面这行的注释
        // els.chatSend.click();
    }
};

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

  // [新增] 进入聊天时，加载最新的 Collection 列表
  renderChatCollectionOptions();
  
  if (!state.chat.currentSessionId) createNewChatSession();
  
  renderChatHistory();
  renderChatSidebar();
  setMode(Modes.CHAT);

  backToChatView();
  
  // [核心新增] 进入界面时，如果没有引擎在跑，就自动跑起来
    if (!state.chat.engineSessionId && state.selectedPipeline) {
        startEngine(state.selectedPipeline);
    } else {
        // 如果已经在跑，更新一下 UI 状态
        updateDemoControls();
    }

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

// [新增] 辅助函数：清洗 PDF 提取文本中的脏格式
function cleanPDFText(text) {
    if (!text) return "";

    let clean = text;

    // 1. 统一换行符
    clean = clean.replace(/\r\n/g, '\n');

    // 2. 修复单词断行 (Hyphenation)
    // 例如: "communi-\ncation" -> "communication"
    // 逻辑: 字母 + 连字符 + 换行 + 字母 -> 直接合并
    clean = clean.replace(/([a-zA-Z])-\n([a-zA-Z])/g, '$1$2');

    // 3. 智能合并硬换行 (Unwrap Lines)
    // PDF 解析出来的文本通常每一行都有 \n，我们需要把它们合并成一个段落
    // 策略：
    // a. 先把真正的段落 (\n\n) 保护起来，替换成一个特殊占位符
    clean = clean.replace(/\n\s*\n/g, '___PARAGRAPH_BREAK___');
    
    // b. 把剩下的单个 \n (通常是 PDF 的硬换行) 替换成空格
    clean = clean.replace(/\n/g, ' ');
    
    // c. 把多余的空格合并 (多个空格变一个)
    clean = clean.replace(/  +/g, ' ');

    // d. 把特殊占位符还原为 Markdown 的标准段落换行 (\n\n)
    clean = clean.replace(/___PARAGRAPH_BREAK___/g, '\n\n');

    return clean;
}

// [新增] 打开右侧详情栏
function showSourceDetail(title, content) {
    const panel = document.getElementById("source-detail-panel");
    const contentDiv = document.getElementById("source-detail-content");
    const titleDiv = panel.querySelector(".detail-title");

    if (panel && contentDiv) {
        // 填充内容
        titleDiv.textContent = title || "Reference";

        // 2. 清洗文本 (处理 PDF 乱码)
        const rawText = content || "No content available.";
        const cleanedText = cleanPDFText(rawText);

        if (typeof renderMarkdown === 'function') {
            contentDiv.innerHTML = renderMarkdown(cleanedText);
            renderLatex(contentDiv);
        } else {
            // 降级处理：直接显示清洗后的纯文本，也比之前好读很多
            contentDiv.innerText = cleanedText;
        }

        // 4. 滚动回顶部 (防止上次看到底部，这次打开还在底部)
        contentDiv.scrollTop = 0;
        
        // 展开面板
        panel.classList.add("show");
    }
}

// [新增] 关闭右侧详情栏 (绑定到了 HTML 的 x 按钮)
window.closeSourceDetail = function() {
    const panel = document.getElementById("source-detail-panel");
    if (panel) panel.classList.remove("show");
};

// [新增] 点击角标跳转函数
window.scrollToReference = function(refId) {
    const targetId = `ref-item-${refId}`;
    // 查找当前可见的引用列表项 (倒序查找最近的)
    const allRefs = document.querySelectorAll(`[id='${targetId}']`);
    const target = allRefs[allRefs.length - 1];

    if (target) {
        // 1. 视觉反馈：闪烁一下底部的列表项，告诉用户对应关系
        target.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        target.classList.remove("active-highlight");
        void target.offsetWidth; 
        target.classList.add("active-highlight");
        
        // 2. [新增] 打开右侧侧边栏显示详情
        if (target._sourceData) {
            const src = target._sourceData;
            const title = `Reference [${src.id}]`;
            showSourceDetail(title, src.content);
        }
    }
};

// [修改] 渲染参考资料列表 (支持追加模式)
function renderSources(bubble, sources, isAppend = false) {
    if (!bubble || !sources || sources.length === 0) return;

    let refContainer = bubble.querySelector(".reference-container");
    let list;

    if (!refContainer) {
        // 第一次创建容器
        refContainer = document.createElement("div");
        refContainer.className = "reference-container";
        refContainer.innerHTML = `<div class="ref-header">References</div>`;
        list = document.createElement("div");
        list.className = "ref-list";
        refContainer.appendChild(list);
        bubble.appendChild(refContainer);
    } else {
        list = refContainer.querySelector(".ref-list");
        // 如果不是追加模式（比如页面刷新重绘），先清空
        if (!isAppend) list.innerHTML = "";
    }

    sources.forEach(src => {
        // 使用计算好的全局 ID (displayId)
        const showId = src.displayId || src.id;
        
        // 防止重复添加
        if (list.querySelector(`#ref-item-${showId}`)) return;

        const item = document.createElement("div");
        item.className = "ref-item";
        item.id = `ref-item-${showId}`;
        item._sourceData = src; 
        item.onclick = () => {
            const title = `Reference [${showId}]`;
            showSourceDetail(title, src.content);
        };
        
        item.innerHTML = `
            <span class="ref-id">[${showId}]</span>
            <span class="ref-title">${src.title}</span>
        `;
        list.appendChild(item);
    });
}

// [新增] 格式化正文文本 (高亮 [1])
function formatMessageText(text) {
    if (!text) return "";
    // 先转义 HTML 防止注入
    let safeText = text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    
    // 处理换行
    safeText = safeText.replace(/\n/g, "<br>");
    
    // 正则替换 [数字] 为高亮标签
    // 匹配 [1], [12], [1,2] 等格式
    safeText = safeText.replace(/\[(\d+)\]/g, '<span class="citation-link">[$1]</span>');
    
    return safeText;
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
  if (event) event.preventDefault();
  if (state.chat.running) { await stopGeneration(); return; }
  if (!canUseChat()) return;
  if (!state.chat.engineSessionId) { alert("Start Engine first."); return; }

  const question = els.chatInput.value.trim();
  if (!question) return;
  els.chatInput.value = "";
  // 重置textarea高度
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

    // 先占位
    const entryIndex = state.chat.history.length;
    state.chat.history.push({ role: "assistant", text: "", meta: {} });
    
    const chatContainer = document.getElementById("chat-history");

    // [滚动优化]
    let shouldAutoScroll = true;
    const handleScroll = () => {
        const threshold = 30; 
        const distance = chatContainer.scrollHeight - chatContainer.scrollTop - chatContainer.clientHeight;
        shouldAutoScroll = distance <= threshold;
    };
    chatContainer.addEventListener('scroll', handleScroll);

    const bubble = document.createElement("div");
    bubble.className = "chat-bubble assistant";
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
            } 
            else if (data.type === "sources") {
                // 后端已为每个文档分配了唯一ID，直接使用
                const docs = data.data.map((doc) => ({
                    ...doc, 
                    displayId: doc.id
                }));
                allSources = allSources.concat(docs);
                pendingRenderSources = pendingRenderSources.concat(docs);
            } 
            else if (data.type === "token") {
                if (!data.is_final) updateProcessUI(entryIndex, data);
                if (data.is_final) {
                    currentText += data.content;
                    if (typeof isPendingLanguageFence === 'function' && isPendingLanguageFence(currentText, MARKDOWN_LANGS)) continue;
                    
                    let html = renderMarkdown(currentText, { unwrapLanguages: MARKDOWN_LANGS });
                    html = formatCitationHtml(html);
                    contentDiv.innerHTML = html;
                    renderLatex(contentDiv);
                    if (shouldAutoScroll) {
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }
                }
            } 
            else if (data.type === "final") {
                const final = data.data;
                let finalText = currentText || final.answer || "";
                let html = renderMarkdown(finalText, { unwrapLanguages: MARKDOWN_LANGS });
                html = formatCitationHtml(html);
                contentDiv.innerHTML = html;
                renderLatex(contentDiv);

                // 渲染参考资料卡片
                if (pendingRenderSources && pendingRenderSources.length > 0) {
                    renderSources(bubble, pendingRenderSources, true);
                }

                if (shouldAutoScroll) {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
                
                // 更新历史记录
                state.chat.history[entryIndex].text = finalText;
                if (!state.chat.history[entryIndex].meta) state.chat.history[entryIndex].meta = {};
                state.chat.history[entryIndex].meta.sources = allSources;

                // 高亮被引用的文档卡片
                const usedIds = new Set();
                const regex = /\[(\d+)\]/g;
                let match;
                while ((match = regex.exec(finalText)) !== null) {
                    usedIds.add(parseInt(match[1], 10));
                }

                const refItems = bubble.querySelectorAll(".ref-item");
                refItems.forEach(item => {
                    const id = parseInt(item.id.replace("ref-item-", ""), 10);
                    if (usedIds.has(id)) {
                        item.classList.add("used");
                        item.classList.remove("unused");
                    } else {
                        item.classList.add("unused");
                        item.classList.remove("used");
                    }
                });
                
                const hints = [];
                if (final.dataset_path) hints.push(`Dataset: ${final.dataset_path}`);
                if (final.memory_path) hints.push(`Memory: ${final.memory_path}`);
                state.chat.history[entryIndex].meta.hint = hints.join(" | ");
                
                const procDiv = bubble.querySelector(".process-container");
                if (procDiv) procDiv.classList.add("collapsed");
                setChatStatus("Ready", "ready");
            } 
            else if (data.type === "error") {
                appendChatMessage("system", `Backend Error: ${data.message}`);
                setChatStatus("Error", "error");
            }
          } catch (e) { console.error(e); }
        }
      }
    }
  } catch (err) {
      if (err.name === 'AbortError') return;
      console.error(err);
      appendChatMessage("system", `Network Error: ${err.message}`);
      setChatStatus("Error", "error");
  } finally {
      if (state.chat.controller) {
          state.chat.controller = null;
          setChatRunning(false);
      }
      saveCurrentSession();
      chatContainer.removeEventListener('scroll', handleScroll); // 记得移除监听
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

    if (els.kbBtn) {
        els.kbBtn.onclick = openKBView; 
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
            
            // 增加视觉反馈 (可选)
            els.refreshCollectionsBtn.disabled = true;
            els.refreshCollectionsBtn.innerHTML = '⟳'; // 用一个旋转图标代替

            try {
                await refreshKBFiles(); // 调用已包含同步逻辑的刷新函数
            } finally {
                // 恢复按钮状态
                els.refreshCollectionsBtn.disabled = false;
                els.refreshCollectionsBtn.innerHTML = '↻';
            }
        };
    }

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
    
    // 支持Shift+Enter换行，Enter提交
    if (els.chatInput) {
        els.chatInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleChatSubmit(e);
            }
            // Shift+Enter 时允许默认行为（换行）
        });
        
        // 自动调整textarea高度
        els.chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    }

    if (els.chatNewBtn) els.chatNewBtn.onclick = createNewChatSession;
    if (els.demoToggleBtn) els.demoToggleBtn.onclick = toggleDemoSession;

    if (els.kbBtn) els.kbBtn.onclick = openKBView;
    
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

// [新增] 更新知识库选择器的显示文本
window.updateKbLabel = function(selectEl) {
    const label = document.getElementById('kb-label-text');
    const visualBtn = document.querySelector('.kb-visual-btn');
    if (!label || !visualBtn) return;
    
    // 获取选中的文本
    const selectedText = selectEl.options[selectEl.selectedIndex].text;
    const selectedVal = selectEl.value;

    if (!selectedVal) {
        // 没选时，显示默认
        label.textContent = "Knowledge Base";
        label.style.color = ""; 
        visualBtn.style.background = ""; // 恢复默认背景
    } else {
        // 选中时，显示具体名字
        // 我们可以只显示名字部分，去掉括号里的数量，让它更像 Tag
        // 例如 "wiki_v1 (50)" -> "wiki_v1"
        const cleanName = selectedText.split('(')[0].trim();
        label.textContent = cleanName;
        
        // 可选：选中后给个高亮背景，表示“已激活”
        visualBtn.style.background = "#e0e7ff"; 
        label.style.color = "#2563eb";
    }
};

async function bootstrap() {
  setMode(Modes.BUILDER); 
  resetContextStack(); 
  renderSteps(); 
  updatePipelinePreview(); 
  bindEvents(); 
  updateActionButtons();
  
  // 1. 先加载基础数据
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

  // 2. [修改] 数据加载完后，再恢复历史会话和选中状态
  const savedSessions = localStorage.getItem("ultrarag_sessions");
  if (savedSessions) {
      try {
          state.chat.sessions = JSON.parse(savedSessions);
          state.chat.sessions.sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));
          
          // 恢复上一次活跃的会话
          const lastId = localStorage.getItem("ultrarag_last_active_id");
          if (lastId) {
              const session = state.chat.sessions.find(s => s.id === lastId);
              if (session) {
                  state.chat.currentSessionId = session.id;
                  state.chat.history = cloneDeep(session.messages || []);
                  
                  // [关键] 如果上次有选中的 Pipeline，自动加载它
                  if (session.pipeline) {
                      // 此时 refreshPipelines 已完成，UI 是安全的
                      loadPipeline(session.pipeline); 
                  } else {
                      // 如果没有 pipeline 记录，只设置 label
                      setHeroPipelineLabel(state.selectedPipeline || "");
                  }
                  log(`Restored last session: ${session.title}`);
              }
          }
      } catch (e) {
          console.warn("Failed to load history:", e);
          state.chat.sessions = [];
      }
  } else {
      setHeroPipelineLabel(state.selectedPipeline || "");
  }
}

bootstrap();
