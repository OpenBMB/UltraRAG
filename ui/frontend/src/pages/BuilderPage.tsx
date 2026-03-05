import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import { RefreshCcw } from "lucide-react";
import { dump as dumpYaml, load as parseYaml } from "js-yaml";
import { useNavigate } from "react-router-dom";
import { PipelineCanvas } from "@/features/pipeline/components/PipelineCanvas";
import { usePipelines } from "@/features/pipeline/hooks/usePipelines";
import {
  buildPipeline,
  createPipeline,
  deletePipeline,
  fetchPipelineConfig,
  fetchPipelineParameters,
  fetchServerTools,
  renamePipeline,
  savePipelineParameters,
  savePipelineYaml,
  type ServerToolItem,
} from "@/shared/api/pipelines";
import {
  createPrompt,
  deletePrompt,
  fetchPromptContent,
  fetchPrompts,
  renamePrompt,
  savePromptContent,
} from "@/shared/api/prompts";
import { testAiConnection } from "@/shared/api/ai";
import {
  deletePreviewNode,
  findPreviewNode,
  insertPreviewBranchNode,
  insertPreviewChildNode,
  parsePipelinePreview,
  serializePipelinePreviewToYaml,
  updatePreviewNodeLabel,
  type PipelinePreviewNode,
} from "@/shared/lib/pipelinePreview";
import { renderChatMarkdown } from "@/shared/lib/chatMarkdown";
import { useI18n } from "@/shared/i18n/provider";
import { useThemeStyles } from "@/shared/lib/useThemeStyles";
import "@/pages/builder-page.css";

function yamlFromConfig(config: Record<string, unknown>): string {
  const rawYaml = config._raw_yaml;
  if (typeof rawYaml === "string" && rawYaml.trim()) return rawYaml;
  return JSON.stringify(config, null, 2);
}

function formatTemplate(template: string, values: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (full, key: string) =>
    Object.prototype.hasOwnProperty.call(values, key) ? String(values[key]) : full,
  );
}

function flashHighlight(element: HTMLElement | null): void {
  if (!element) return;
  element.classList.add("ai-apply-highlight");
  window.setTimeout(() => {
    element.classList.remove("ai-apply-highlight");
  }, 1600);
}

type AiRole = "user" | "assistant";

type AiAction = {
  type: string;
  preview?: string;
  content?: string;
  filename?: string;
  path?: string;
  value?: unknown;
};

type AiActionState = {
  status: "applied" | "rejected" | "error";
  message?: string;
};

type AiMessage = {
  id: string;
  role: AiRole;
  content: string;
  actions?: AiAction[];
  actionStates?: Record<number, AiActionState>;
  pending?: boolean;
};

type AiSession = {
  id: string;
  title: string;
  messages: AiMessage[];
  conversationHistory: Array<{ role: AiRole; content: string }>;
  updatedAt: number;
};

type AiSettings = {
  provider: string;
  baseUrl: string;
  apiKey: string;
  model: string;
};

type ParameterEntry = {
  path: string;
  displayPath: string;
  fullPath: string;
  value: unknown;
  type: string;
  serverName: string;
};

type NodePickerMode = "tool" | "branch" | "loop" | "custom";

type NodePickerTarget = {
  parentId: string;
  index: number;
};

type ToolCatalog = {
  order: string[];
  byServer: Record<string, ServerToolItem[]>;
};

const AI_SETTINGS_STORAGE_KEY = "ultrarag_ai_settings";
const AI_HISTORY_STORAGE_KEY = "ultrarag_ai_history";
const AI_PANEL_WIDTH_STORAGE_KEY = "ultrarag_ai_panel_width";
const WORKSPACE_PANE_WIDTH_KEY = "ultrarag_workspace_pane_width";
const DEFAULT_WORKSPACE_PANE_WIDTH = 360;
const MIN_WORKSPACE_PANE_WIDTH = 240;
const MIN_WORKSPACE_CONTENT_WIDTH = 320;
const PIPELINE_PREVIEW_ROOT_ID = "__pipeline_root__";
const AI_PROVIDER_DEFAULT_BASE_URL: Record<string, string> = {
  openai: "https://api.openai.com/v1",
  azure: "https://YOUR_RESOURCE.openai.azure.com",
  anthropic: "https://api.anthropic.com/v1",
  custom: "",
};
const VANILLA_PIPELINE_TEMPLATE: Record<string, unknown> = {
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

function loadAiSettings(): AiSettings {
  const fallback: AiSettings = {
    provider: "openai",
    baseUrl: "https://api.openai.com/v1",
    apiKey: "",
    model: "gpt-5-mini",
  };
  try {
    const raw = localStorage.getItem(AI_SETTINGS_STORAGE_KEY);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    return {
      provider: String(parsed.provider ?? fallback.provider),
      baseUrl: String(parsed.baseUrl ?? fallback.baseUrl),
      apiKey: String(parsed.apiKey ?? fallback.apiKey),
      model: String(parsed.model ?? fallback.model),
    };
  } catch {
    return fallback;
  }
}

function cloneJsonObject<T>(value: T): T {
  if (typeof structuredClone === "function") return structuredClone(value);
  return JSON.parse(JSON.stringify(value)) as T;
}

function escapeRegExp(input: string): string {
  return input.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function setNestedValue(target: Record<string, unknown>, path: string, value: unknown): void {
  const segments = path.split(".").filter(Boolean);
  if (!segments.length) return;
  let cursor: Record<string, unknown> = target;
  for (let i = 0; i < segments.length - 1; i += 1) {
    const key = segments[i];
    const next = cursor[key];
    if (!next || typeof next !== "object" || Array.isArray(next)) {
      cursor[key] = {};
    }
    cursor = cursor[key] as Record<string, unknown>;
  }
  cursor[segments[segments.length - 1]] = value;
}

function flattenParameters(
  payload: Record<string, unknown>,
  prefix = "",
): Array<{ path: string; value: unknown; type: string }> {
  const entries: Array<{ path: string; value: unknown; type: string }> = [];
  Object.keys(payload)
    .sort()
    .forEach((key) => {
      const path = prefix ? `${prefix}.${key}` : key;
      const value = payload[key];
      if (value !== null && typeof value === "object" && !Array.isArray(value)) {
        entries.push(...flattenParameters(value as Record<string, unknown>, path));
        return;
      }
      entries.push({
        path,
        value,
        type: Array.isArray(value) ? "array" : value === null ? "null" : typeof value,
      });
    });
  return entries;
}

function filterServerParameters(
  entries: Array<{ path: string; value: unknown; type: string }>,
  pipelineConfig: Record<string, unknown> | null,
  simplified: boolean,
): Array<{ path: string; value: unknown; type: string; displayPath: string }> {
  if (!simplified) {
    return entries.map((entry) => ({ ...entry, displayPath: entry.path }));
  }

  const serversRecord = (pipelineConfig?.servers ?? {}) as Record<string, unknown>;
  const generationServers = new Set<string>();
  const retrieverServers = new Set<string>();
  Object.entries(serversRecord).forEach(([name, value]) => {
    if (typeof value !== "string") return;
    if (value === "servers/generation" || value.endsWith("/generation")) {
      generationServers.add(name);
      return;
    }
    if (value === "servers/retriever" || value.endsWith("/retriever")) {
      retrieverServers.add(name);
    }
  });

  const kept = entries.filter((entry) => {
    const serverName = entry.path.split(".")[0];
    const escapedServer = escapeRegExp(serverName);
    if (generationServers.has(serverName)) {
      if (entry.path === `${serverName}.backend`) return false;
      if (
        entry.path.startsWith(`${serverName}.backend_configs.`) &&
        !entry.path.startsWith(`${serverName}.backend_configs.openai.`)
      ) {
        return false;
      }
      return [
        new RegExp(`^${escapedServer}\\.backend_configs\\.openai\\.`),
        new RegExp(`^${escapedServer}\\.extra_params\\.`),
        new RegExp(`^${escapedServer}\\.sampling_params\\.`),
        new RegExp(`^${escapedServer}\\.system_prompt$`),
      ].some((pattern) => pattern.test(entry.path));
    }

    if (retrieverServers.has(serverName)) {
      if (entry.path === `${serverName}.backend`) return false;
      if (
        entry.path.startsWith(`${serverName}.backend_configs.`) &&
        !entry.path.startsWith(`${serverName}.backend_configs.openai.`)
      ) {
        return false;
      }
      if (entry.path === `${serverName}.index_backend`) return false;
      if (entry.path.startsWith(`${serverName}.index_backend_configs.`)) return false;
      const skipPattern = new RegExp(
        `^${escapedServer}\\.(batch_size|collection_name|corpus_path|gpu_ids|is_demo|is_multimodal|model_name_or_path)$`,
      );
      if (skipPattern.test(entry.path)) return false;
      return [
        new RegExp(`^${escapedServer}\\.backend_configs\\.openai\\.`),
        new RegExp(`^${escapedServer}\\.query_instruction$`),
        new RegExp(`^${escapedServer}\\.top_k$`),
      ].some((pattern) => pattern.test(entry.path));
    }
    return true;
  });

  return kept.map((entry) => ({
    ...entry,
    displayPath: entry.path.includes(".backend_configs.openai.")
      ? entry.path.replace(".backend_configs.openai.", ".backend_configs.")
      : entry.path,
  }));
}

function makePreviewNodeId(): string {
  return `node_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function findPreviewNodePath(
  nodes: PipelinePreviewNode[],
  targetId: string,
): PipelinePreviewNode[] {
  const walk = (items: PipelinePreviewNode[], path: PipelinePreviewNode[]): PipelinePreviewNode[] => {
    for (const item of items) {
      const nextPath = [...path, item];
      if (item.id === targetId) return nextPath;
      if (item.children.length) {
        const nested = walk(item.children, nextPath);
        if (nested.length) return nested;
      }
    }
    return [];
  };
  return walk(nodes, []);
}

function appendPreviewBranchCase(
  nodes: PipelinePreviewNode[],
  branchNodeId: string,
): PipelinePreviewNode[] {
  const cloned = cloneJsonObject(nodes);
  const walk = (items: PipelinePreviewNode[]): boolean => {
    for (const item of items) {
      if (item.id === branchNodeId && item.kind === "branch") {
        const used = new Set<string>();
        item.children.forEach((child) => {
          if (child.kind === "group" && child.label.trim().toLowerCase() === "router") return;
          const label = child.label.trim();
          if (label.toLowerCase().startsWith("branch:")) {
            const name = label.slice("branch:".length).trim();
            if (name) used.add(name.toLowerCase());
          } else if (label) {
            used.add(label.toLowerCase());
          }
        });
        let index = used.size + 1;
        let candidate = `case${index}`;
        while (used.has(candidate.toLowerCase())) {
          index += 1;
          candidate = `case${index}`;
        }
        item.children.push({
          id: makePreviewNodeId(),
          label: `branch: ${candidate}`,
          kind: "branch",
          children: [],
        });
        return true;
      }
      if (item.children.length && walk(item.children)) return true;
    }
    return false;
  };
  walk(cloned);
  return cloned;
}

function normalizeAiActions(raw: unknown): AiAction[] {
  if (!Array.isArray(raw)) return [];
  const seen = new Set<string>();
  const normalized: AiAction[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const record = item as Record<string, unknown>;
    const action: AiAction = {
      type: String(record.type ?? ""),
      preview: typeof record.preview === "string" ? record.preview : undefined,
      content: typeof record.content === "string" ? record.content : undefined,
      filename: typeof record.filename === "string" ? record.filename : undefined,
      path: typeof record.path === "string" ? record.path : undefined,
      value: record.value,
    };
    const key = JSON.stringify({
      type: action.type,
      filename: action.filename ?? "",
      path: action.path ?? "",
      preview: action.preview ?? action.content ?? "",
      value: action.value ?? null,
    });
    if (seen.has(key)) continue;
    seen.add(key);
    normalized.push(action);
  }
  return normalized;
}

function parseAiActionStates(raw: unknown): Record<number, AiActionState> | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const source = raw as Record<string, unknown>;
  const next: Record<number, AiActionState> = {};
  Object.entries(source).forEach(([key, value]) => {
    const index = Number(key);
    if (!Number.isInteger(index) || !value || typeof value !== "object") return;
    const item = value as Record<string, unknown>;
    const status = item.status;
    if (status !== "applied" && status !== "rejected" && status !== "error") return;
    next[index] = {
      status,
      message: typeof item.message === "string" ? item.message : undefined,
    };
  });
  return Object.keys(next).length ? next : undefined;
}

function deriveAiSessionTitle(messages: AiMessage[], fallback: string): string {
  const firstUser = messages.find((item) => item.role === "user");
  if (!firstUser || !firstUser.content.trim()) return fallback;
  const text = firstUser.content.trim().replace(/\s+/g, " ");
  if (text.length <= 24) return text;
  return `${text.slice(0, 24)}...`;
}

function toAiMessageList(raw: unknown): AiMessage[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((item, index) => {
      if (!item || typeof item !== "object") return null;
      const record = item as Record<string, unknown>;
      const role = record.role === "assistant" ? "assistant" : "user";
      const content = String(record.content ?? "");
      return {
        id: String(record.id ?? `${Date.now()}-${index}-${Math.random().toString(36).slice(2, 6)}`),
        role,
        content,
        actions: normalizeAiActions(record.actions),
        actionStates: parseAiActionStates(record.actionStates),
        pending: Boolean(record.pending),
      } as AiMessage;
    })
    .filter((item): item is AiMessage => Boolean(item));
}

export function BuilderPage() {
  useThemeStyles(true);
  const navigate = useNavigate();
  const { t } = useI18n();

  const { data: pipelines = [], refetch: refetchPipelines } = usePipelines();
  const [selectedPipeline, setSelectedPipeline] = useState<string>("");
  const [pipelineNameDraft, setPipelineNameDraft] = useState("");
  const [yamlText, setYamlText] = useState("");
  const [paramsText, setParamsText] = useState("{}");
  const [parameterData, setParameterData] = useState<Record<string, unknown> | null>(null);
  const [pipelineConfig, setPipelineConfig] = useState<Record<string, unknown> | null>(null);
  const [paramsActiveServer, setParamsActiveServer] = useState("");
  const [paramsExpandedSections, setParamsExpandedSections] = useState<Record<string, boolean>>({});
  const [parameterDrafts, setParameterDrafts] = useState<Record<string, string>>({});
  const [promptFiles, setPromptFiles] = useState<Array<{ name: string; path: string; size: number }>>([]);
  const [selectedPromptPath, setSelectedPromptPath] = useState("");
  const [promptOpenTabs, setPromptOpenTabs] = useState<string[]>([]);
  const [promptTabState, setPromptTabState] = useState<
    Record<string, { content: string; savedContent: string }>
  >({});
  const [promptText, setPromptText] = useState("");
  const [promptSavedText, setPromptSavedText] = useState("");
  const [promptSearch, setPromptSearch] = useState("");
  const [promptCreateDialogOpen, setPromptCreateDialogOpen] = useState(false);
  const [newPromptPathDraft, setNewPromptPathDraft] = useState("my_prompt.jinja");
  const [promptContextMenu, setPromptContextMenu] = useState<{
    open: boolean;
    x: number;
    y: number;
    path: string;
  }>({
    open: false,
    x: 0,
    y: 0,
    path: "",
  });
  const [promptRenameDialogOpen, setPromptRenameDialogOpen] = useState(false);
  const [promptRenameSourcePath, setPromptRenameSourcePath] = useState("");
  const [promptRenameDraft, setPromptRenameDraft] = useState("");
  const [promptDeleteDialogOpen, setPromptDeleteDialogOpen] = useState(false);
  const [promptDeleteTargetPath, setPromptDeleteTargetPath] = useState("");
  const [status, setStatus] = useState("");
  const [saving, setSaving] = useState(false);
  const [buildState, setBuildState] = useState<"idle" | "running" | "success" | "error">("idle");
  const [panelMode, setPanelMode] = useState<"pipeline" | "parameters" | "prompts">("pipeline");
  const [yamlValidation, setYamlValidation] = useState<"idle" | "validating" | "valid" | "invalid">("idle");
  const [activeContextId, setActiveContextId] = useState(PIPELINE_PREVIEW_ROOT_ID);
  const [pipelineDropdownOpen, setPipelineDropdownOpen] = useState(false);
  const [createPipelineDialogOpen, setCreatePipelineDialogOpen] = useState(false);
  const [newPipelineNameDraft, setNewPipelineNameDraft] = useState("NewPipeline");
  const [deletePipelineDialogOpen, setDeletePipelineDialogOpen] = useState(false);
  const [editNodeDialogOpen, setEditNodeDialogOpen] = useState(false);
  const [editingNodeId, setEditingNodeId] = useState("");
  const [editingNodeLabelDraft, setEditingNodeLabelDraft] = useState("");
  const [nodePickerOpen, setNodePickerOpen] = useState(false);
  const [nodePickerTarget, setNodePickerTarget] = useState<NodePickerTarget | null>(null);
  const [nodePickerMode, setNodePickerMode] = useState<NodePickerMode>("tool");
  const [nodePickerServer, setNodePickerServer] = useState("");
  const [nodePickerTool, setNodePickerTool] = useState("");
  const [nodePickerBranchCases, setNodePickerBranchCases] = useState("case1, case2");
  const [nodePickerLoopTimes, setNodePickerLoopTimes] = useState("2");
  const [nodePickerCustom, setNodePickerCustom] = useState("");
  const [nodePickerError, setNodePickerError] = useState("");
  const [nodePickerLoadingTools, setNodePickerLoadingTools] = useState(false);
  const [toolCatalog, setToolCatalog] = useState<ToolCatalog>({ order: [], byServer: {} });
  const [paramsSimplified, setParamsSimplified] = useState(true);
  const [consoleCollapsed, setConsoleCollapsed] = useState(true);
  const [consoleLogs, setConsoleLogs] = useState<string[]>([]);
  const [aiPanelOpen, setAiPanelOpen] = useState(false);
  const [aiView, setAiView] = useState<"home" | "chat">("home");
  const [aiSettingsOpen, setAiSettingsOpen] = useState(false);
  const [aiApiKeyVisible, setAiApiKeyVisible] = useState(false);
  const [aiComposing, setAiComposing] = useState(false);
  const [aiInput, setAiInput] = useState("");
  const [aiMessages, setAiMessages] = useState<AiMessage[]>([]);
  const [aiSessions, setAiSessions] = useState<AiSession[]>([]);
  const [aiCurrentSessionId, setAiCurrentSessionId] = useState("");
  const [aiBusy, setAiBusy] = useState(false);
  const [aiTestingConnection, setAiTestingConnection] = useState(false);
  const [aiSettings, setAiSettings] = useState<AiSettings>(() => loadAiSettings());
  const [aiSettingsStatus, setAiSettingsStatus] = useState("");
  const [aiSettingsStatusKind, setAiSettingsStatusKind] = useState<"" | "success" | "error">("");
  const [aiConnected, setAiConnected] = useState(false);
  const [aiHistoryReady, setAiHistoryReady] = useState(false);
  const yamlEditorRef = useRef<HTMLTextAreaElement | null>(null);
  const yamlGutterRef = useRef<HTMLDivElement | null>(null);
  const promptEditorRef = useRef<HTMLTextAreaElement | null>(null);
  const promptGutterRef = useRef<HTMLDivElement | null>(null);
  const promptContextMenuRef = useRef<HTMLDivElement | null>(null);
  const pipelineDropdownRef = useRef<HTMLDivElement | null>(null);
  const builderSplitRef = useRef<HTMLDivElement | null>(null);
  const builderResizerRef = useRef<HTMLDivElement | null>(null);
  const paramsLayoutRef = useRef<HTMLDivElement | null>(null);
  const paramsResizerRef = useRef<HTMLDivElement | null>(null);
  const promptsLayoutRef = useRef<HTMLDivElement | null>(null);
  const promptsResizerRef = useRef<HTMLDivElement | null>(null);
  const consoleOutputRef = useRef<HTMLPreElement | null>(null);
  const aiMessagesRef = useRef<HTMLDivElement | null>(null);
  const aiInputRef = useRef<HTMLTextAreaElement | null>(null);
  const aiPanelRef = useRef<HTMLDivElement | null>(null);
  const aiPanelResizerRef = useRef<HTMLDivElement | null>(null);
  const preferredPipelineRef = useRef<string>("");
  const aiHistoryLoadedRef = useRef(false);
  const aiRequestControllerRef = useRef<AbortController | null>(null);
  const buildStateTimerRef = useRef<number | null>(null);
  const parameterSectionRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const parameterFieldRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const aiActionBlockRefs = useRef<Record<string, HTMLDivElement | null>>({});

  const pipelinePreview = useMemo(() => parsePipelinePreview(yamlText), [yamlText]);
  const visiblePipelinePreviewError = useMemo(() => {
    if (!pipelinePreview.error) return "";
    return yamlText.trim() ? pipelinePreview.error : "";
  }, [pipelinePreview.error, yamlText]);
  const previewRootNode = useMemo<PipelinePreviewNode>(
    () => ({
      id: PIPELINE_PREVIEW_ROOT_ID,
      label: "root",
      kind: "group",
      children: pipelinePreview.nodes,
    }),
    [pipelinePreview.nodes],
  );
  const activeContextPath = useMemo(() => {
    if (activeContextId === PIPELINE_PREVIEW_ROOT_ID) return [previewRootNode];
    const nestedPath = findPreviewNodePath(previewRootNode.children, activeContextId);
    return nestedPath.length ? [previewRootNode, ...nestedPath] : [previewRootNode];
  }, [activeContextId, previewRootNode]);
  const activeContextNode = useMemo(
    () => activeContextPath[activeContextPath.length - 1] ?? previewRootNode,
    [activeContextPath, previewRootNode],
  );
  const activeCanvasNodes = useMemo(
    () => (activeContextNode.kind === "step" ? [] : activeContextNode.children),
    [activeContextNode],
  );
  const groupedParameterEntries = useMemo(() => {
    if (!parameterData || typeof parameterData !== "object" || Array.isArray(parameterData)) {
      return {} as Record<string, ParameterEntry[]>;
    }
    const flattened = flattenParameters(parameterData).filter(
      (entry) => !/^benchmark(\.|$)/i.test(entry.path),
    );
    const filtered = filterServerParameters(flattened, pipelineConfig, paramsSimplified);
    const grouped: Record<string, ParameterEntry[]> = {};
    filtered.forEach((entry) => {
      const parts = (entry.displayPath || entry.path).split(".");
      const serverName = (parts[0] || "").toUpperCase();
      if (!serverName) return;
      const formatted: ParameterEntry = {
        path: entry.path,
        displayPath: parts.slice(1).join(".") || parts[0],
        fullPath: entry.path,
        value: entry.value,
        type: entry.type,
        serverName,
      };
      if (!grouped[serverName]) grouped[serverName] = [];
      grouped[serverName].push(formatted);
    });
    return grouped;
  }, [parameterData, pipelineConfig, paramsSimplified]);
  const parameterServers = useMemo(
    () => Object.keys(groupedParameterEntries).sort((a, b) => a.localeCompare(b, "en", { sensitivity: "base" })),
    [groupedParameterEntries],
  );
  const syncStatusClass = useMemo(() => {
    if (yamlValidation === "valid") return "synced";
    if (yamlValidation === "validating") return "syncing";
    if (yamlValidation === "invalid") return "error";
    return "modified";
  }, [yamlValidation]);
  const yamlLineNumbers = useMemo(
    () => Array.from({ length: Math.max(1, yamlText.split(/\r?\n/).length) }, (_, idx) => idx + 1),
    [yamlText],
  );
  const promptLineNumbers = useMemo(
    () => Array.from({ length: Math.max(1, promptText.split(/\r?\n/).length) }, (_, idx) => idx + 1),
    [promptText],
  );
  const promptModified = useMemo(
    () => {
      if (!selectedPromptPath) return false;
      const tab = promptTabState[selectedPromptPath];
      if (tab) return tab.content !== tab.savedContent;
      return promptText !== promptSavedText;
    },
    [promptSavedText, promptTabState, promptText, selectedPromptPath],
  );
  const filteredPromptFiles = useMemo(() => {
    const keyword = promptSearch.trim().toLowerCase();
    if (!keyword) return promptFiles;
    return promptFiles.filter((item) =>
      `${item.name} ${item.path}`.toLowerCase().includes(keyword),
    );
  }, [promptFiles, promptSearch]);
  const selectedPipelineMeta = useMemo(
    () => pipelines.find((item) => item.name === selectedPipeline) ?? null,
    [pipelines, selectedPipeline],
  );
  const visibleAiSessions = useMemo(
    () => aiSessions.filter((session) => session.messages.length > 0),
    [aiSessions],
  );
  const nodePickerAvailableTools = useMemo(
    () => (nodePickerServer ? toolCatalog.byServer[nodePickerServer] ?? [] : []),
    [nodePickerServer, toolCatalog.byServer],
  );
  const buildButtonLabel = useMemo(() => {
    if (buildState === "running") return t("builder_build_running", "Building...");
    if (buildState === "success") return t("builder_build_success", "Build success");
    if (buildState === "error") return t("builder_build_failed", "Build failed");
    return t("builder_build", "Build");
  }, [buildState, t]);
  const getWorkspacePaneWidth = useCallback(() => {
    try {
      const stored = Number(localStorage.getItem(WORKSPACE_PANE_WIDTH_KEY));
      if (Number.isFinite(stored) && stored > 0) return stored;
    } catch {
      // ignore storage read failures
    }
    return DEFAULT_WORKSPACE_PANE_WIDTH;
  }, []);
  const setWorkspacePaneWidth = useCallback((width: number, persist = true) => {
    const next = Math.max(MIN_WORKSPACE_PANE_WIDTH, Math.round(width));
    document.documentElement.style.setProperty("--workspace-pane-width", `${next}px`);
    if (!persist) return;
    try {
      localStorage.setItem(WORKSPACE_PANE_WIDTH_KEY, String(next));
    } catch {
      // ignore storage write failures
    }
  }, []);
  const openPromptTab = useCallback((path: string) => {
    const normalized = path.trim();
    if (!normalized) return;
    setPromptOpenTabs((previous) => (previous.includes(normalized) ? previous : [...previous, normalized]));
    setSelectedPromptPath(normalized);
  }, []);

  const closePromptTab = useCallback(
    (path: string) => {
      const normalized = path.trim();
      if (!normalized) return;
      const tab = promptTabState[normalized];
      if (tab && tab.content !== tab.savedContent) {
        const fileName = normalized.split("/").pop() || normalized;
        const confirmed = window.confirm(
          formatTemplate(
            t("builder_prompt_discard_confirm", "Discard unsaved changes for {name}?"),
            { name: fileName },
          ),
        );
        if (!confirmed) return;
      }
      setPromptOpenTabs((previous) => {
        const next = previous.filter((item) => item !== normalized);
        if (selectedPromptPath === normalized) {
          const fallback = next[next.length - 1] ?? "";
          setSelectedPromptPath(fallback);
          if (!fallback) {
            setPromptText("");
            setPromptSavedText("");
          }
        }
        return next;
      });
      setPromptTabState((previous) => {
        if (!Object.prototype.hasOwnProperty.call(previous, normalized)) return previous;
        const next = { ...previous };
        delete next[normalized];
        return next;
      });
    },
    [promptTabState, selectedPromptPath, t],
  );

  const closePromptContextMenu = useCallback(() => {
    setPromptContextMenu((previous) => (previous.open ? { ...previous, open: false } : previous));
  }, []);

  const openPromptContextMenu = useCallback(
    (event: ReactMouseEvent<HTMLElement>, path: string) => {
      event.preventDefault();
      const normalized = path.trim();
      if (!normalized) return;
      const menuWidth = 170;
      const menuHeight = 96;
      const x = Math.min(event.clientX, window.innerWidth - menuWidth - 12);
      const y = Math.min(event.clientY, window.innerHeight - menuHeight - 12);
      setPromptContextMenu({
        open: true,
        x: Math.max(8, x),
        y: Math.max(8, y),
        path: normalized,
      });
    },
    [],
  );

  const openPromptRenameDialog = useCallback(
    (path: string) => {
      const normalized = path.trim();
      if (!normalized) return;
      closePromptContextMenu();
      setPromptRenameSourcePath(normalized);
      setPromptRenameDraft(normalized);
      setPromptRenameDialogOpen(true);
    },
    [closePromptContextMenu],
  );

  const openPromptDeleteDialog = useCallback(
    (path: string) => {
      const normalized = path.trim();
      if (!normalized) return;
      closePromptContextMenu();
      setPromptDeleteTargetPath(normalized);
      setPromptDeleteDialogOpen(true);
    },
    [closePromptContextMenu],
  );

  useEffect(() => {
    if (!parameterServers.length) {
      setParamsActiveServer("");
      return;
    }
    if (parameterServers.includes(paramsActiveServer)) return;
    setParamsActiveServer(parameterServers[0]);
  }, [parameterServers, paramsActiveServer]);

  useEffect(() => {
    setParamsExpandedSections((previous) => {
      const next: Record<string, boolean> = {};
      parameterServers.forEach((serverName) => {
        next[serverName] = previous[serverName] ?? true;
      });
      const changed =
        Object.keys(previous).length !== Object.keys(next).length ||
        Object.entries(next).some(([key, value]) => previous[key] !== value);
      return changed ? next : previous;
    });
  }, [parameterServers]);
  const aiContextHint = useMemo(() => {
    if (panelMode === "pipeline") {
      if (selectedPipeline) {
        return t("builder_ai_context_pipeline_editing", `Editing Pipeline YAML: ${selectedPipeline}`)
          .replace("{pipeline}", selectedPipeline);
      }
      return t("builder_ai_context_pipeline_canvas", "Pipeline canvas");
    }
    if (panelMode === "parameters") {
      if (selectedPipeline) {
        return t("builder_ai_context_parameters_editing", `Editing Parameters for ${selectedPipeline}`)
          .replace("{pipeline}", selectedPipeline);
      }
      return t("builder_ai_context_parameters_no_pipeline", "Parameters panel (no pipeline selected)");
    }
    if (selectedPromptPath) {
      return t("builder_ai_context_prompt_editing", `Editing Prompt: ${selectedPromptPath}`)
        .replace("{file}", selectedPromptPath);
    }
    return t("builder_ai_context_prompt_panel", "Prompt panel");
  }, [panelMode, selectedPipeline, selectedPromptPath, t]);

  const aiStatusText = useMemo(() => {
    if (aiTestingConnection) return t("builder_ai_settings_testing", "Testing connection...");
    if (aiConnected) {
      return t("builder_ai_status_connected", `Connected to ${aiSettings.model}`).replace(
        "{model}",
        aiSettings.model,
      );
    }
    if (aiSettings.apiKey.trim()) {
      return t("builder_ai_status_configured_pending", "Configured - Save & Test to verify");
    }
    return t("builder_ai_status_not_configured", "Not configured");
  }, [aiConnected, aiSettings.apiKey, aiSettings.model, aiTestingConnection, t]);

  const pushConsoleLog = useCallback((message: string) => {
    const content = message.trim();
    if (!content) return;
    const timestamp = new Date().toLocaleTimeString();
    setConsoleLogs((previous) => [...previous.slice(-199), `[${timestamp}] ${content}`]);
  }, []);

  const updateAiPanelOffset = useCallback(() => {
    const panel = aiPanelRef.current;
    if (!panel) return;
    const width = panel.getBoundingClientRect().width || parseInt(window.getComputedStyle(panel).width, 10) || 360;
    const rounded = Math.round(width);
    document.documentElement.style.setProperty("--ai-panel-width", `${rounded}px`);
    try {
      localStorage.setItem(AI_PANEL_WIDTH_STORAGE_KEY, String(rounded));
    } catch {
      // ignore storage failures
    }
  }, []);

  const adjustAiInputHeight = useCallback((target?: HTMLTextAreaElement | null) => {
    const input = target ?? aiInputRef.current;
    if (!input) return;
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
  }, []);

  const refreshToolCatalog = useCallback(async () => {
    setNodePickerLoadingTools(true);
    setNodePickerError("");
    try {
      const tools = await fetchServerTools();
      const grouped: Record<string, ServerToolItem[]> = {};
      tools.forEach((item) => {
        const server = String(item.server ?? "").trim();
        const tool = String(item.tool ?? "").trim();
        if (!server || !tool) return;
        if (!grouped[server]) grouped[server] = [];
        grouped[server].push({ ...item, server, tool });
      });
      Object.keys(grouped).forEach((server) => {
        grouped[server] = grouped[server]
          .slice()
          .sort((a, b) => a.tool.localeCompare(b.tool, "en", { sensitivity: "base" }));
      });
      const order = Object.keys(grouped).sort((a, b) => a.localeCompare(b, "en", { sensitivity: "base" }));
      setToolCatalog({ order, byServer: grouped });
      if (!order.length) {
        setNodePickerServer("");
        setNodePickerTool("");
      } else {
        setNodePickerServer((previous) => (order.includes(previous) ? previous : order[0]));
      }
    } catch (error) {
      setToolCatalog({ order: [], byServer: {} });
      setNodePickerServer("");
      setNodePickerTool("");
      setNodePickerError(error instanceof Error ? error.message : t("builder_node_no_servers", "No Servers"));
    } finally {
      setNodePickerLoadingTools(false);
    }
  }, [t]);

  useEffect(() => {
    preferredPipelineRef.current = (
      localStorage.getItem("ultrarag_react_selected_pipeline") ??
      localStorage.getItem("ultrarag_selected_pipeline") ??
      ""
    ).trim();
  }, []);

  useEffect(() => {
    setWorkspacePaneWidth(getWorkspacePaneWidth(), false);
  }, [getWorkspacePaneWidth, setWorkspacePaneWidth]);

  useEffect(() => {
    const resizer = builderResizerRef.current;
    const container = builderSplitRef.current;
    if (!resizer || !container) return;

    let isResizing = false;
    let startX = 0;
    let startWidth = 0;

    const handlePointerDown = (event: PointerEvent) => {
      isResizing = true;
      startX = event.clientX;
      startWidth = getWorkspacePaneWidth();
      resizer.classList.add("dragging");
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      event.preventDefault();
    };

    const handlePointerMove = (event: PointerEvent) => {
      if (!isResizing) return;
      const dx = event.clientX - startX;
      const maxWidth = Math.max(MIN_WORKSPACE_PANE_WIDTH, container.offsetWidth - MIN_WORKSPACE_CONTENT_WIDTH);
      const next = Math.min(Math.max(startWidth + dx, MIN_WORKSPACE_PANE_WIDTH), maxWidth);
      setWorkspacePaneWidth(next);
    };

    const handlePointerUp = () => {
      if (!isResizing) return;
      isResizing = false;
      resizer.classList.remove("dragging");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    resizer.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);

    return () => {
      resizer.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [getWorkspacePaneWidth, setWorkspacePaneWidth]);

  useEffect(() => {
    const resizer = paramsResizerRef.current;
    const container = paramsLayoutRef.current;
    if (!resizer || !container) return;

    let isResizing = false;
    let startX = 0;
    let startWidth = 0;

    const handlePointerDown = (event: PointerEvent) => {
      isResizing = true;
      startX = event.clientX;
      startWidth = getWorkspacePaneWidth();
      resizer.classList.add("dragging");
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      event.preventDefault();
    };

    const handlePointerMove = (event: PointerEvent) => {
      if (!isResizing) return;
      const dx = event.clientX - startX;
      const maxWidth = Math.max(MIN_WORKSPACE_PANE_WIDTH, container.offsetWidth - MIN_WORKSPACE_CONTENT_WIDTH);
      const next = Math.min(Math.max(startWidth + dx, MIN_WORKSPACE_PANE_WIDTH), maxWidth);
      setWorkspacePaneWidth(next);
    };

    const handlePointerUp = () => {
      if (!isResizing) return;
      isResizing = false;
      resizer.classList.remove("dragging");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    resizer.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);

    return () => {
      resizer.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [getWorkspacePaneWidth, setWorkspacePaneWidth]);

  useEffect(() => {
    const resizer = promptsResizerRef.current;
    const container = promptsLayoutRef.current;
    if (!resizer || !container) return;

    let isResizing = false;
    let startX = 0;
    let startWidth = 0;

    const handlePointerDown = (event: PointerEvent) => {
      isResizing = true;
      startX = event.clientX;
      startWidth = getWorkspacePaneWidth();
      resizer.classList.add("dragging");
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      event.preventDefault();
    };

    const handlePointerMove = (event: PointerEvent) => {
      if (!isResizing) return;
      const dx = event.clientX - startX;
      const maxWidth = Math.max(MIN_WORKSPACE_PANE_WIDTH, container.offsetWidth - MIN_WORKSPACE_CONTENT_WIDTH);
      const next = Math.min(Math.max(startWidth + dx, MIN_WORKSPACE_PANE_WIDTH), maxWidth);
      setWorkspacePaneWidth(next);
    };

    const handlePointerUp = () => {
      if (!isResizing) return;
      isResizing = false;
      resizer.classList.remove("dragging");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };

    resizer.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);

    return () => {
      resizer.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, [getWorkspacePaneWidth, setWorkspacePaneWidth]);

  useEffect(() => {
    return () => {
      if (buildStateTimerRef.current) {
        window.clearTimeout(buildStateTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!pipelines.length) {
      setSelectedPipeline("");
      return;
    }
    if (pipelines.some((pipeline) => pipeline.name === selectedPipeline)) return;
    const preferred = preferredPipelineRef.current;
    const fallback = preferred && pipelines.some((pipeline) => pipeline.name === preferred)
      ? preferred
      : pipelines[0].name;
    setSelectedPipeline(fallback);
  }, [pipelines, selectedPipeline]);

  useEffect(() => {
    if (!selectedPipeline) {
      localStorage.removeItem("ultrarag_react_selected_pipeline");
      return;
    }
    localStorage.setItem("ultrarag_react_selected_pipeline", selectedPipeline);
  }, [selectedPipeline]);

  useEffect(() => {
    setPipelineNameDraft(selectedPipeline);
  }, [selectedPipeline]);

  useEffect(() => {
    if (selectedPipeline) return;
    setPipelineConfig(null);
    setParameterData(null);
    setParamsText("{}");
    setParameterDrafts({});
  }, [selectedPipeline]);

  useEffect(() => {
    if (!selectedPipeline) return;
    let mounted = true;
    const load = async () => {
      try {
        const config = await fetchPipelineConfig(selectedPipeline);
        if (!mounted) return;
        setPipelineConfig(config);
        setYamlText(yamlFromConfig(config));
        setYamlValidation("valid");
        setStatus(`Pipeline "${selectedPipeline}" loaded.`);
      } catch (error) {
        if (!mounted) return;
        setPipelineConfig(null);
        setYamlText("");
        setStatus(error instanceof Error ? error.message : "Failed to load pipeline data");
        return;
      }

      try {
        const params = await fetchPipelineParameters(selectedPipeline);
        if (!mounted) return;
        setParameterData(params);
        setParamsText(JSON.stringify(params, null, 2));
        setParameterDrafts({});
      } catch (error) {
        if (!mounted) return;
        setParameterData({});
        setParamsText("{}");
        setParameterDrafts({});
        const message = error instanceof Error
          ? error.message
          : t("builder_params_not_found", "Parameters not found. Build first.");
        setStatus(message || t("builder_params_not_found", "Parameters not found. Build first."));
      }
    };
    void load();
    return () => {
      mounted = false;
    };
  }, [selectedPipeline, t]);

  useEffect(() => {
    setActiveContextId(PIPELINE_PREVIEW_ROOT_ID);
  }, [selectedPipeline]);

  useEffect(() => {
    if (activeContextId === PIPELINE_PREVIEW_ROOT_ID) return;
    const exists = findPreviewNode([previewRootNode], activeContextId);
    if (!exists) {
      setActiveContextId(PIPELINE_PREVIEW_ROOT_ID);
    }
  }, [activeContextId, previewRootNode]);

  useEffect(() => {
    if (!nodePickerOpen) return;
    void refreshToolCatalog();
  }, [nodePickerOpen, refreshToolCatalog]);

  useEffect(() => {
    if (!nodePickerOpen) return;
    const tools = nodePickerServer ? toolCatalog.byServer[nodePickerServer] ?? [] : [];
    if (!tools.length) {
      if (nodePickerTool) setNodePickerTool("");
      return;
    }
    if (!tools.some((item) => item.tool === nodePickerTool)) {
      setNodePickerTool(tools[0].tool);
    }
  }, [nodePickerOpen, nodePickerServer, nodePickerTool, toolCatalog.byServer]);

  useEffect(() => {
    let mounted = true;
    const loadPrompts = async () => {
      try {
        const files = await fetchPrompts();
        if (!mounted) return;
        setPromptFiles(files);
        if (!files.length) {
          setPromptOpenTabs([]);
          setPromptTabState({});
          setSelectedPromptPath("");
          setPromptText("");
          setPromptSavedText("");
          return;
        }
        openPromptTab(files[0].path);
      } catch (error) {
        if (!mounted) return;
        setStatus(error instanceof Error ? error.message : "Failed to load prompts");
      }
    };
    void loadPrompts();
    return () => {
      mounted = false;
    };
  }, [openPromptTab]);

  useEffect(() => {
    if (!selectedPromptPath) {
      setPromptText("");
      setPromptSavedText("");
      return;
    }
    const draft = promptTabState[selectedPromptPath];
    if (draft) {
      setPromptText(draft.content);
      setPromptSavedText(draft.savedContent);
      return;
    }
    let mounted = true;
    const loadPrompt = async () => {
      try {
        const content = await fetchPromptContent(selectedPromptPath);
        if (mounted) {
          setPromptText(content);
          setPromptSavedText(content);
          setPromptTabState((previous) => ({
            ...previous,
            [selectedPromptPath]: {
              content,
              savedContent: content,
            },
          }));
        }
      } catch (error) {
        if (!mounted) return;
        setStatus(error instanceof Error ? error.message : "Failed to load prompt");
      }
    };
    void loadPrompt();
    return () => {
      mounted = false;
    };
  }, [promptTabState, selectedPromptPath]);

  useEffect(() => {
    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (!target) return;
      if (
        promptContextMenu.open &&
        promptContextMenuRef.current &&
        !promptContextMenuRef.current.contains(target)
      ) {
        closePromptContextMenu();
      }
      if (!pipelineDropdownOpen) return;
      if (pipelineDropdownRef.current && !pipelineDropdownRef.current.contains(target)) {
        setPipelineDropdownOpen(false);
      }
    };
    document.addEventListener("pointerdown", handlePointerDown, true);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown, true);
    };
  }, [closePromptContextMenu, pipelineDropdownOpen, promptContextMenu.open]);

  useEffect(() => {
    if (!status.trim()) return;
    pushConsoleLog(status);
  }, [status, pushConsoleLog]);

  useEffect(() => {
    if (!consoleOutputRef.current || consoleCollapsed) return;
    consoleOutputRef.current.scrollTop = consoleOutputRef.current.scrollHeight;
  }, [consoleCollapsed, consoleLogs]);

  useEffect(() => {
    if (!aiMessagesRef.current) return;
    aiMessagesRef.current.scrollTop = aiMessagesRef.current.scrollHeight;
  }, [aiMessages, aiBusy]);

  useEffect(() => {
    adjustAiInputHeight();
  }, [adjustAiInputHeight, aiInput]);

  useEffect(() => {
    if (!aiPanelOpen || aiView !== "chat") return;
    const timer = window.setTimeout(() => {
      aiInputRef.current?.focus();
    }, 100);
    return () => {
      window.clearTimeout(timer);
    };
  }, [aiPanelOpen, aiView]);

  useEffect(() => {
    try {
      const saved = Number(localStorage.getItem(AI_PANEL_WIDTH_STORAGE_KEY));
      if (!Number.isFinite(saved) || saved <= 0) return;
      const width = Math.min(Math.max(saved, 300), 600);
      if (aiPanelRef.current) {
        aiPanelRef.current.style.width = `${width}px`;
      }
      document.documentElement.style.setProperty("--ai-panel-width", `${width}px`);
    } catch {
      // ignore storage read failures
    }
  }, []);

  useEffect(() => {
    const resizer = aiPanelResizerRef.current;
    const panel = aiPanelRef.current;
    if (!resizer || !panel) return;

    let isResizing = false;
    let startX = 0;
    let startWidth = 0;

    const handlePointerDown = (event: PointerEvent) => {
      isResizing = true;
      startX = event.clientX;
      startWidth = panel.getBoundingClientRect().width || panel.offsetWidth;
      resizer.classList.add("dragging");
      document.body.style.cursor = "ew-resize";
      document.body.style.userSelect = "none";
      event.preventDefault();
    };

    const handlePointerMove = (event: PointerEvent) => {
      if (!isResizing) return;
      const diff = startX - event.clientX;
      const nextWidth = Math.min(Math.max(startWidth + diff, 300), 600);
      panel.style.width = `${nextWidth}px`;
      document.documentElement.style.setProperty("--ai-panel-width", `${nextWidth}px`);
    };

    const handlePointerUp = () => {
      if (!isResizing) return;
      isResizing = false;
      resizer.classList.remove("dragging");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      updateAiPanelOffset();
    };

    resizer.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    return () => {
      resizer.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [updateAiPanelOffset]);

  useEffect(() => {
    if (aiPanelOpen) {
      document.body.classList.add("ai-panel-open");
      updateAiPanelOffset();
      return () => {
        document.body.classList.remove("ai-panel-open");
      };
    }
    document.body.classList.remove("ai-panel-open");
    return undefined;
  }, [aiPanelOpen, updateAiPanelOffset]);

  useEffect(() => {
    try {
      localStorage.setItem(AI_SETTINGS_STORAGE_KEY, JSON.stringify(aiSettings));
    } catch {
      // ignore localStorage write failures
    }
  }, [aiSettings]);

  const createEmptyAiSession = useCallback((): AiSession => {
    const id = `ai_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    return {
      id,
      title: t("builder_ai_session_new", "New Session"),
      messages: [],
      conversationHistory: [],
      updatedAt: Date.now(),
    };
  }, [t]);

  useEffect(() => {
    if (aiHistoryLoadedRef.current) return;
    aiHistoryLoadedRef.current = true;
    try {
      const saved = localStorage.getItem(AI_HISTORY_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved) as {
          sessions?: unknown[];
          currentSessionId?: string;
          messages?: unknown[];
        };
        const parsedSessions = Array.isArray(parsed.sessions)
          ? parsed.sessions
              .map((rawSession, index) => {
                if (!rawSession || typeof rawSession !== "object") return null;
                const record = rawSession as Record<string, unknown>;
                const messages = toAiMessageList(record.messages);
                const fallbackTitle = t("builder_ai_session_new", "New Session");
                const title = String(record.title ?? deriveAiSessionTitle(messages, fallbackTitle));
                return {
                  id: String(record.id ?? `ai_${Date.now()}_${index}`),
                  title,
                  messages,
                  conversationHistory: messages.map((item) => ({
                    role: item.role,
                    content: item.content,
                  })),
                  updatedAt: Number(record.updatedAt ?? Date.now()) || Date.now(),
                } as AiSession;
              })
              .filter((item): item is AiSession => Boolean(item))
          : [];

        let normalizedSessions = parsedSessions;
        if (!normalizedSessions.length && Array.isArray(parsed.messages)) {
          const fallbackMessages = toAiMessageList(parsed.messages);
          normalizedSessions = [
            {
              ...createEmptyAiSession(),
              messages: fallbackMessages,
              conversationHistory: fallbackMessages.map((item) => ({
                role: item.role,
                content: item.content,
              })),
              title: deriveAiSessionTitle(
                fallbackMessages,
                t("builder_ai_session_new", "New Session"),
              ),
            },
          ];
        }
        if (!normalizedSessions.length) {
          normalizedSessions = [createEmptyAiSession()];
        }

        const preferredId = String(parsed.currentSessionId ?? "");
        const currentSession =
          normalizedSessions.find((item) => item.id === preferredId) ?? normalizedSessions[0];
        setAiSessions(normalizedSessions);
        setAiCurrentSessionId(currentSession.id);
        setAiMessages(currentSession.messages);
        setAiHistoryReady(true);
        return;
      }
    } catch {
      // ignore parse errors and fallback to fresh session
    }

    const initial = createEmptyAiSession();
    setAiSessions([initial]);
    setAiCurrentSessionId(initial.id);
    setAiMessages([]);
    setAiHistoryReady(true);
  }, [createEmptyAiSession, t]);

  useEffect(() => {
    if (!aiHistoryReady || !aiCurrentSessionId) return;
    setAiSessions((previous) => {
      const index = previous.findIndex((item) => item.id === aiCurrentSessionId);
      if (index === -1) return previous;
      const fallbackTitle = t("builder_ai_session_new", "New Session");
      const persistedMessages = aiMessages.filter((item) => !item.pending);
      const updated = {
        ...previous[index],
        messages: persistedMessages,
        conversationHistory: persistedMessages.map((item) => ({ role: item.role, content: item.content })),
        updatedAt: Date.now(),
        title: deriveAiSessionTitle(persistedMessages, fallbackTitle),
      };
      const next = [...previous];
      next[index] = updated;
      return next;
    });
  }, [aiCurrentSessionId, aiHistoryReady, aiMessages, t]);

  useEffect(() => {
    if (!aiHistoryReady) return;
    try {
      localStorage.setItem(
        AI_HISTORY_STORAGE_KEY,
        JSON.stringify({
          sessions: aiSessions,
          currentSessionId: aiCurrentSessionId,
        }),
      );
    } catch {
      // ignore localStorage write failures
    }
  }, [aiCurrentSessionId, aiHistoryReady, aiSessions]);

  useEffect(() => {
    return () => {
      aiRequestControllerRef.current?.abort();
      aiRequestControllerRef.current = null;
    };
  }, []);

  const addAiMessage = useCallback((role: AiRole, content: string, actions: AiAction[] = [], pending = false) => {
    const next: AiMessage = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      role,
      content,
      actions: role === "assistant" && !pending ? normalizeAiActions(actions) : undefined,
      actionStates: undefined,
      pending,
    };
    setAiMessages((previous) => [...previous, next]);
    return next;
  }, []);

  const updateAiMessage = useCallback(
    (
      messageId: string,
      next: Partial<Pick<AiMessage, "content" | "actions" | "actionStates" | "pending">>,
    ) => {
      setAiMessages((previous) =>
        previous.map((message) => {
          if (message.id !== messageId) return message;
          return {
            ...message,
            ...next,
            actions:
              typeof next.actions !== "undefined"
                ? normalizeAiActions(next.actions)
                : message.actions,
            pending: typeof next.pending === "boolean" ? next.pending : message.pending,
          };
        }),
      );
    },
    [],
  );

  const removeAiMessage = useCallback((messageId: string) => {
    setAiMessages((previous) => previous.filter((item) => item.id !== messageId));
  }, []);

  const switchAiSession = useCallback((sessionId: string) => {
    aiRequestControllerRef.current?.abort();
    const target = aiSessions.find((session) => session.id === sessionId);
    if (!target) return;
    setAiCurrentSessionId(target.id);
    setAiMessages(target.messages);
    setAiPanelOpen(true);
    setAiView("chat");
    setAiSettingsOpen(false);
  }, [aiSessions]);

  const handleNewAiSession = useCallback(() => {
    aiRequestControllerRef.current?.abort();
    const current = aiSessions.find((session) => session.id === aiCurrentSessionId);
    if (current && current.messages.length === 0) {
      setAiPanelOpen(true);
      setAiView("chat");
      setAiSettingsOpen(false);
      return;
    }
    const session = createEmptyAiSession();
    setAiSessions((previous) => [session, ...previous]);
    setAiCurrentSessionId(session.id);
    setAiMessages([]);
    setAiPanelOpen(true);
    setAiView("chat");
    setAiSettingsOpen(false);
  }, [aiCurrentSessionId, aiSessions, createEmptyAiSession]);

  const deleteAiSession = useCallback((sessionId?: string) => {
    const targetId = sessionId || aiCurrentSessionId;
    if (!targetId) return;
    aiRequestControllerRef.current?.abort();
    const confirmed = window.confirm(
      t("ai_session_delete_confirm", "Are you sure you want to delete this session?"),
    );
    if (!confirmed) return;
    setAiSessions((previous) => {
      const filtered = previous.filter((session) => session.id !== targetId);
      if (!filtered.length) {
        const fallback = createEmptyAiSession();
        setAiCurrentSessionId(fallback.id);
        setAiMessages([]);
        return [fallback];
      }
      const nextCurrent = filtered.find((session) => session.id === aiCurrentSessionId) ?? filtered[0];
      setAiCurrentSessionId(nextCurrent.id);
      setAiMessages(nextCurrent.messages);
      return filtered;
    });
    setAiView("home");
  }, [aiCurrentSessionId, createEmptyAiSession, t]);

  const deleteAllAiSessions = useCallback(() => {
    const nonEmptySessions = aiSessions.filter((session) => session.messages.length > 0);
    if (!nonEmptySessions.length) return;
    const confirmed = window.confirm(
      formatTemplate(
        t("ai_session_delete_all_confirm", "Are you sure you want to delete all {count} session(s)?"),
        { count: nonEmptySessions.length },
      ),
    );
    if (!confirmed) return;
    aiRequestControllerRef.current?.abort();
    const fallback = createEmptyAiSession();
    setAiSessions([fallback]);
    setAiCurrentSessionId(fallback.id);
    setAiMessages([]);
    setAiView("home");
  }, [aiSessions, createEmptyAiSession, t]);

  const setAiActionState = useCallback(
    (messageId: string, actionIndex: number, state: AiActionState) => {
      setAiMessages((previous) =>
        previous.map((message) => {
          if (message.id !== messageId) return message;
          return {
            ...message,
            actionStates: {
              ...(message.actionStates ?? {}),
              [actionIndex]: state,
            },
          };
        }),
      );
    },
    [],
  );

  const applyPipelineModification = useCallback(
    async (action: AiAction): Promise<boolean> => {
      if (!action.content) return false;
      setPanelMode("pipeline");
      setYamlText(action.content);
      setYamlValidation("idle");
      if (selectedPipeline) {
        await savePipelineYaml(selectedPipeline, action.content);
        await refetchPipelines();
      }
      window.setTimeout(() => {
        flashHighlight(yamlEditorRef.current);
      }, 0);
      const detail = selectedPipeline
        ? formatTemplate(t("builder_ai_pipeline_updated_message", "Replaced YAML for {name}"), {
            name: selectedPipeline,
          })
        : t("builder_ai_pipeline_updated_message_default", "AI updated current YAML");
      setStatus(detail);
      pushConsoleLog(detail);
      return true;
    },
    [pushConsoleLog, refetchPipelines, selectedPipeline, t],
  );

  const applyPromptModification = useCallback(
    async (action: AiAction): Promise<boolean> => {
      if (!action.content) return false;
      setPanelMode("prompts");
      const requestedPath =
        (action.filename ?? "").trim() ||
        selectedPromptPath ||
        promptFiles[0]?.path ||
        "chat/new_prompt.jinja";
      let targetPath =
        promptFiles.find((item) => item.path === requestedPath || item.name === requestedPath)?.path ??
        requestedPath;
      const exists = promptFiles.some((item) => item.path === targetPath);
      if (!exists) {
        await createPrompt(targetPath, "");
      }
      await savePromptContent(targetPath, action.content);
      const files = await fetchPrompts();
      setPromptFiles(files);
      const refreshedTarget = files.find((item) => item.path === targetPath || item.name === targetPath);
      if (refreshedTarget) {
        targetPath = refreshedTarget.path;
      } else if (!files.some((item) => item.path === targetPath)) {
        targetPath = files[0]?.path ?? targetPath;
      }
      openPromptTab(targetPath);
      setPromptTabState((previous) => ({
        ...previous,
        [targetPath]: {
          content: action.content ?? "",
          savedContent: action.content ?? "",
        },
      }));
      setPromptText(action.content);
      setPromptSavedText(action.content);
      window.setTimeout(() => {
        flashHighlight(promptEditorRef.current);
      }, 0);
      const detail = formatTemplate(t("builder_ai_prompt_updated_message", "Updated {name}"), {
        name: targetPath,
      });
      setStatus(detail);
      pushConsoleLog(detail);
      return true;
    },
    [openPromptTab, promptFiles, pushConsoleLog, selectedPromptPath, t],
  );

  const applyParameterModification = useCallback(
    async (action: AiAction): Promise<boolean> => {
      if (!action.path || typeof action.path !== "string") return false;
      const normalizedPath = action.path.trim();
      if (!normalizedPath) return false;
      let base: Record<string, unknown> =
        parameterData && typeof parameterData === "object" && !Array.isArray(parameterData)
          ? cloneJsonObject(parameterData)
          : {};
      if (!Object.keys(base).length && selectedPipeline) {
        try {
          const remote = await fetchPipelineParameters(selectedPipeline);
          if (remote && typeof remote === "object" && !Array.isArray(remote)) {
            base = cloneJsonObject(remote);
          }
        } catch (error) {
          pushConsoleLog(
            formatTemplate(t("builder_ai_params_load_failed", "Failed to load parameters: {error}"), {
              error: error instanceof Error ? error.message : t("builder_ai_error_generic", "Request failed. Please try again."),
            }),
          );
        }
      }
      setNestedValue(base, normalizedPath, action.value);
      setParameterData(base);
      setParamsText(JSON.stringify(base, null, 2));
      setParameterDrafts({});
      setPanelMode("parameters");
      const serverName = normalizedPath.split(".")[0]?.toUpperCase();
      if (serverName) {
        setParamsActiveServer(serverName);
        setParamsExpandedSections((previous) => ({ ...previous, [serverName]: true }));
      }
      if (selectedPipeline) {
        await savePipelineParameters(selectedPipeline, base);
      }
      window.setTimeout(() => {
        const field = parameterFieldRefs.current[normalizedPath];
        if (field) {
          field.scrollIntoView({ behavior: "smooth", block: "nearest" });
          flashHighlight(field);
          return;
        }
        flashHighlight(document.getElementById("parameter-form"));
      }, 0);
      const detail = formatTemplate(t("builder_ai_parameter_updated_message", "Set {path}"), {
        path: normalizedPath,
      });
      setStatus(detail);
      pushConsoleLog(detail);
      return true;
    },
    [parameterData, pushConsoleLog, selectedPipeline, t],
  );

  const handleApplyAiAction = useCallback(
    async (messageId: string, actionIndex: number, action: AiAction) => {
      try {
        if (action.type === "modify_pipeline" && panelMode === "parameters") {
          const ignored = t("builder_ai_pipeline_change_ignored_hint", "You are editing parameters. Ask explicitly to update the pipeline YAML or switch to Pipeline view.");
          setAiActionState(messageId, actionIndex, { status: "error", message: ignored });
          setStatus(t("builder_ai_pipeline_change_ignored", "Pipeline change ignored"));
          pushConsoleLog(ignored);
          return;
        }

        let success = false;
        if (action.type === "modify_pipeline") {
          success = await applyPipelineModification(action);
        } else if (action.type === "modify_prompt") {
          success = await applyPromptModification(action);
        } else if (action.type === "modify_parameter") {
          success = await applyParameterModification(action);
        } else {
          throw new Error(
            formatTemplate(t("builder_ai_action_unknown", "Unknown action type: {type}"), {
              type: action.type || "unknown",
            }),
          );
        }

        if (success) {
          setAiActionState(messageId, actionIndex, { status: "applied" });
          pushConsoleLog(t("builder_ai_action_applied_log", "AI modification applied successfully."));
          window.setTimeout(() => {
            flashHighlight(aiActionBlockRefs.current[`${messageId}-${actionIndex}`] ?? null);
          }, 0);
        }
      } catch (error) {
        const errorText =
          error instanceof Error ? error.message : t("builder_ai_error_generic", "Request failed. Please try again.");
        setAiActionState(messageId, actionIndex, { status: "error", message: errorText });
        pushConsoleLog(
          formatTemplate(t("builder_ai_action_apply_failed", "Failed to apply modification: {error}"), {
            error: errorText,
          }),
        );
      }
    },
    [
      applyParameterModification,
      applyPipelineModification,
      applyPromptModification,
      panelMode,
      pushConsoleLog,
      setAiActionState,
      t,
    ],
  );

  const handleRejectAiAction = useCallback(
    (messageId: string, actionIndex: number) => {
      setAiActionState(messageId, actionIndex, { status: "rejected" });
    },
    [setAiActionState],
  );

  const saveAndTestAiSettings = async () => {
    if (!aiSettings.apiKey.trim()) {
      setAiSettingsStatus(t("builder_ai_settings_api_key_required", "Please enter an API key"));
      setAiSettingsStatusKind("error");
      return;
    }
    setAiTestingConnection(true);
    setAiSettingsStatusKind("");
    setAiSettingsStatus(t("builder_ai_settings_testing", "Testing connection..."));
    try {
      const result = await testAiConnection(aiSettings);
      if (result.success) {
        setAiConnected(true);
        setAiSettingsStatusKind("success");
        setAiSettingsStatus(
          t("builder_ai_settings_connection_success", `Connection successful! Model: ${result.model ?? aiSettings.model}`).replace(
            "{model}",
            result.model ?? aiSettings.model,
          ),
        );
        pushConsoleLog(
          t("builder_ai_settings_connection_success", `Connection successful! Model: ${result.model ?? aiSettings.model}`).replace(
            "{model}",
            result.model ?? aiSettings.model,
          ),
        );
      } else {
        setAiConnected(false);
        setAiSettingsStatusKind("error");
        const errorText = result.error || t("builder_ai_error_generic", "Request failed. Please try again.");
        setAiSettingsStatus(
          t("builder_ai_settings_connection_failed", `Connection failed: ${errorText}`).replace(
            "{error}",
            errorText,
          ),
        );
      }
    } catch (error) {
      setAiConnected(false);
      setAiSettingsStatusKind("error");
      const errorText = error instanceof Error ? error.message : t("builder_ai_error_generic", "Request failed. Please try again.");
      setAiSettingsStatus(
        t("builder_ai_settings_connection_failed", `Connection failed: ${errorText}`).replace(
          "{error}",
          errorText,
        ),
      );
    } finally {
      setAiTestingConnection(false);
    }
  };

  const buildAiContext = useCallback(() => {
    const context: Record<string, unknown> = {
      currentMode: panelMode,
      selectedPipeline,
      isBuilt: Boolean(selectedPipelineMeta?.is_ready),
    };
    if (panelMode === "pipeline") {
      context.pipelineYaml = yamlText;
    } else if (panelMode === "parameters") {
      context.parameters = parameterData ?? {};
    } else {
      context.currentPromptFile = selectedPromptPath || "";
      context.promptContent = promptText;
    }
    return context;
  }, [panelMode, parameterData, promptText, selectedPipeline, selectedPipelineMeta?.is_ready, selectedPromptPath, yamlText]);

  const handleSendAiMessage = async (rawInput?: string) => {
    if (aiBusy) {
      aiRequestControllerRef.current?.abort();
      return;
    }

    const message = (rawInput ?? aiInput).trim();
    if (!message) return;
    if (!aiCurrentSessionId) {
      const fallback = createEmptyAiSession();
      setAiSessions((previous) => [fallback, ...previous]);
      setAiCurrentSessionId(fallback.id);
      setAiMessages([]);
    }
    setAiPanelOpen(true);
    setAiView("chat");
    setAiSettingsOpen(false);
    setAiInput("");

    if (!aiSettings.apiKey.trim()) {
      addAiMessage(
        "assistant",
        t(
          "builder_ai_settings_api_key_required",
          "Please enter an API key",
        ),
      );
      return;
    }

    const requestMessages = [
      ...aiMessages
        .filter((item) => !item.pending)
        .map((item) => ({ role: item.role, content: item.content })),
      { role: "user" as const, content: message },
    ];

    addAiMessage("user", message);
    const placeholder = addAiMessage("assistant", "", [], true);
    setAiBusy(true);
    const controller = new AbortController();
    aiRequestControllerRef.current = controller;
    let accumulated = "";
    let finalActions: AiAction[] = [];

    try {
      const response = await fetch("/api/ai/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          settings: aiSettings,
          messages: requestMessages,
          context: buildAiContext(),
          stream: true,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(response.statusText || t("builder_ai_error_request_failed", "Request failed"));
      }

      const contentType = (response.headers.get("content-type") || "").toLowerCase();
      if (!contentType.includes("text/event-stream")) {
        const result = (await response.json()) as {
          content?: string;
          message?: string;
          answer?: string;
          actions?: unknown[];
          error?: string;
        };
        if (result.error) throw new Error(String(result.error));
        const content =
          result.content ||
          result.message ||
          result.answer ||
          t("builder_ai_no_response", "No response");
        finalActions = normalizeAiActions(result.actions);
        updateAiMessage(placeholder.id, {
          content,
          actions: finalActions,
          actionStates: undefined,
          pending: false,
        });
        setAiConnected(true);
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error(t("builder_ai_error_request_failed", "Request failed"));
      }
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() ?? "";

        for (const chunk of chunks) {
          const lines = chunk
            .split("\n")
            .map((line) => line.trim())
            .filter(Boolean);
          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const jsonText = line.slice(5).trim();
            if (!jsonText) continue;
            let payload: Record<string, unknown>;
            try {
              payload = JSON.parse(jsonText) as Record<string, unknown>;
            } catch {
              continue;
            }
            const type = String(payload.type ?? "");
            if (type === "token") {
              const token = typeof payload.content === "string" ? payload.content : "";
              if (!token) continue;
              accumulated += token;
              updateAiMessage(placeholder.id, { content: accumulated, pending: true });
            } else if (type === "final") {
              if (typeof payload.content === "string") {
                accumulated = payload.content;
              }
              finalActions = normalizeAiActions(payload.actions);
            } else if (type === "error") {
              throw new Error(
                typeof payload.message === "string"
                  ? payload.message
                  : t("builder_ai_error_generic", "Request failed. Please try again."),
              );
            }
          }
        }
      }

      if (!accumulated.trim()) {
        accumulated = t("builder_ai_no_response", "No response");
      }
      updateAiMessage(placeholder.id, {
        content: accumulated,
        actions: finalActions,
        actionStates: undefined,
        pending: false,
      });
      setAiConnected(true);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        if (accumulated.trim()) {
          updateAiMessage(placeholder.id, {
            content: `${accumulated}\n\n*(Stopped)*`,
            actions: finalActions,
            actionStates: undefined,
            pending: false,
          });
        } else {
          removeAiMessage(placeholder.id);
        }
        pushConsoleLog(t("builder_ai_stopped", "Generation stopped."));
        return;
      }
      setAiConnected(false);
      const errorText = error instanceof Error ? error.message : t("builder_ai_error_generic", "Request failed. Please try again.");
      updateAiMessage(placeholder.id, {
        content: t("builder_ai_error_with_message", `Request failed: ${errorText}`).replace("{error}", errorText),
        actions: [],
        actionStates: undefined,
        pending: false,
      });
      pushConsoleLog(t("builder_ai_error_with_message", `Request failed: ${errorText}`).replace("{error}", errorText));
    } finally {
      setAiBusy(false);
      aiRequestControllerRef.current = null;
    }
  };

  const handleAiStarterChip = useCallback(
    (prompt: string) => {
      setAiPanelOpen(true);
      setAiView("chat");
      setAiSettingsOpen(false);
      setAiInput(prompt);
      window.setTimeout(() => {
        if (!aiInputRef.current) return;
        aiInputRef.current.focus();
        adjustAiInputHeight(aiInputRef.current);
      }, 0);
    },
    [adjustAiInputHeight],
  );

  const mutatePreviewNodes = useCallback(
    (mutation: (tree: PipelinePreviewNode[]) => PipelinePreviewNode[]) => {
      if (pipelinePreview.error) return;
      const nextTree = mutation([cloneJsonObject(previewRootNode)]);
      const nextRoot = nextTree.find((node) => node.id === PIPELINE_PREVIEW_ROOT_ID) ?? nextTree[0];
      if (!nextRoot) return;
      const pipelineOnlyYaml = serializePipelinePreviewToYaml(nextRoot.children);
      let nextPipeline: unknown[] = [];
      try {
        const parsedPipeline = parseYaml(pipelineOnlyYaml) as Record<string, unknown>;
        if (parsedPipeline && typeof parsedPipeline === "object" && !Array.isArray(parsedPipeline)) {
          nextPipeline = Array.isArray(parsedPipeline.pipeline)
            ? cloneJsonObject(parsedPipeline.pipeline)
            : [];
        }
      } catch {
        nextPipeline = [];
      }

      let merged = {} as Record<string, unknown>;
      try {
        const parsedCurrent = parseYaml(yamlText);
        if (parsedCurrent && typeof parsedCurrent === "object" && !Array.isArray(parsedCurrent)) {
          merged = cloneJsonObject(parsedCurrent as Record<string, unknown>);
        }
      } catch {
        merged = {};
      }

      if (!Object.keys(merged).length && pipelineConfig && typeof pipelineConfig === "object" && !Array.isArray(pipelineConfig)) {
        merged = cloneJsonObject(pipelineConfig);
      }

      merged.pipeline = nextPipeline;
      if (Object.prototype.hasOwnProperty.call(merged, "steps")) {
        delete merged.steps;
      }
      if (!merged.servers && pipelineConfig?.servers) {
        merged.servers = cloneJsonObject(pipelineConfig.servers as Record<string, unknown>);
      }

      const ordered: Record<string, unknown> = {};
      if (merged.servers) ordered.servers = merged.servers;
      ordered.pipeline = merged.pipeline;
      Object.entries(merged).forEach(([key, value]) => {
        if (key === "servers" || key === "pipeline" || key === "steps") return;
        ordered[key] = value;
      });

      const nextYaml = dumpYaml(ordered, {
        lineWidth: 120,
        noRefs: true,
        sortKeys: false,
      }).trimEnd();
      setPipelineConfig(ordered);
      setYamlText(nextYaml);
      setYamlValidation("idle");
    },
    [pipelinePreview.error, pipelineConfig, previewRootNode, yamlText],
  );

  const handleCanvasInsertNode = useCallback((parentId: string, index: number) => {
    setNodePickerTarget({ parentId, index });
    setNodePickerMode("tool");
    setNodePickerServer("");
    setNodePickerTool("");
    setNodePickerBranchCases("case1, case2");
    setNodePickerLoopTimes("2");
    setNodePickerCustom("");
    setNodePickerError("");
    setNodePickerOpen(true);
  }, []);

  const handleNodePickerConfirm = useCallback(() => {
    if (!nodePickerTarget) {
      setNodePickerOpen(false);
      return;
    }
    const { parentId, index } = nodePickerTarget;
    try {
      if (nodePickerMode === "tool") {
        const server = nodePickerServer.trim();
        const tool = nodePickerTool.trim();
        if (!server || !tool) {
          throw new Error(t("builder_node_select_tool_error", "Please select a tool"));
        }
        mutatePreviewNodes((tree) =>
          insertPreviewChildNode(tree, parentId, index, "step", `${server}.${tool}`),
        );
        setActiveContextId(parentId);
      } else if (nodePickerMode === "loop") {
        const times = Math.max(1, Number(nodePickerLoopTimes) || 1);
        const loopNodeId = makePreviewNodeId();
        mutatePreviewNodes((tree) =>
          insertPreviewChildNode(tree, parentId, index, "loop", `loop x${times}`, loopNodeId),
        );
        setActiveContextId(loopNodeId);
      } else if (nodePickerMode === "branch") {
        const parsedCases = nodePickerBranchCases
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean);
        const branchNodeId = makePreviewNodeId();
        const routerNodeId = makePreviewNodeId();
        mutatePreviewNodes((tree) =>
          insertPreviewBranchNode(tree, parentId, index, parsedCases, "branch", branchNodeId, routerNodeId),
        );
        setActiveContextId(routerNodeId);
      } else {
        const raw = nodePickerCustom.trim();
        if (!raw) {
          throw new Error(t("builder_node_custom_empty_error", "Custom value cannot be empty"));
        }
        let label = raw;
        if ((raw.startsWith("{") && raw.endsWith("}")) || (raw.startsWith("[") && raw.endsWith("]"))) {
          try {
            const parsed = JSON.parse(raw) as unknown;
            if (typeof parsed === "string" && parsed.trim()) {
              label = parsed.trim();
            } else if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
              const firstKey = Object.keys(parsed as Record<string, unknown>)[0];
              if (firstKey) label = firstKey;
            }
          } catch {
            // keep raw text when not valid json
          }
        }
        mutatePreviewNodes((tree) => insertPreviewChildNode(tree, parentId, index, "step", label));
        setActiveContextId(parentId);
      }
      setNodePickerError("");
      setNodePickerOpen(false);
      setNodePickerTarget(null);
    } catch (error) {
      setNodePickerError(error instanceof Error ? error.message : t("builder_node_select_tool_error", "Select a tool"));
    }
  }, [
    mutatePreviewNodes,
    nodePickerBranchCases,
    nodePickerCustom,
    nodePickerLoopTimes,
    nodePickerMode,
    nodePickerServer,
    nodePickerTarget,
    nodePickerTool,
    t,
  ]);

  const handleCanvasAddBranchCase = useCallback(
    (branchNodeId: string) => {
      mutatePreviewNodes((tree) => appendPreviewBranchCase(tree, branchNodeId));
    },
    [mutatePreviewNodes],
  );

  const handleCanvasActivateContext = useCallback((contextId: string) => {
    setActiveContextId(contextId);
  }, []);

  const handleCanvasEditNode = useCallback(
    (nodeId: string) => {
      const target = findPreviewNode([previewRootNode], nodeId);
      if (!target) return;
      setEditingNodeId(nodeId);
      setEditingNodeLabelDraft(target.label);
      setEditNodeDialogOpen(true);
    },
    [previewRootNode],
  );

  const handleCanvasDeleteNode = useCallback(
    (nodeId: string) => {
      mutatePreviewNodes((tree) => deletePreviewNode(tree, nodeId));
    },
    [mutatePreviewNodes],
  );

  const handleConfirmNodeEdit = useCallback(() => {
    const targetId = editingNodeId.trim();
    const label = editingNodeLabelDraft.trim();
    if (!targetId || !label) return;
    mutatePreviewNodes((tree) => updatePreviewNodeLabel(tree, targetId, label));
    setEditNodeDialogOpen(false);
    setEditingNodeId("");
    setEditingNodeLabelDraft("");
  }, [editingNodeId, editingNodeLabelDraft, mutatePreviewNodes]);

  const handleCreatePipeline = async (rawName: string) => {
    const name = rawName.trim();
    if (!name) {
      setStatus(t("builder_pipeline_name_required", "Pipeline name is required."));
      return;
    }
    if (!/^[a-zA-Z_][a-zA-Z0-9_-]*$/.test(name)) {
      setStatus(
        t(
          "builder_invalid_pipeline_name_full",
          "Pipeline name is invalid. Use letters/numbers/_/-, and start with a letter or underscore.",
        ),
      );
      return;
    }
    setSaving(true);
    setStatus("");
    try {
      const templatePipeline: unknown[] = Array.isArray(VANILLA_PIPELINE_TEMPLATE.pipeline)
        ? (cloneJsonObject(VANILLA_PIPELINE_TEMPLATE.pipeline) as unknown[])
        : [];
      await createPipeline(name, templatePipeline);
      const defaultYaml = dumpYaml(VANILLA_PIPELINE_TEMPLATE, {
        lineWidth: 120,
        noRefs: true,
        sortKeys: false,
      }).trimEnd();
      await savePipelineYaml(name, defaultYaml);
      await refetchPipelines();
      setSelectedPipeline(name);
      setPipelineNameDraft(name);
      setYamlText(defaultYaml);
      setYamlValidation("valid");
      setCreatePipelineDialogOpen(false);
      setNewPipelineNameDraft("NewPipeline");
      setStatus(formatTemplate(t("builder_pipeline_created", `Pipeline "{name}" created.`), { name }));
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to create pipeline");
    } finally {
      setSaving(false);
    }
  };

  const handleDeletePipeline = async () => {
    if (!selectedPipeline) return;
    setSaving(true);
    setStatus("");
    try {
      await deletePipeline(selectedPipeline);
      await refetchPipelines();
      setStatus(`Pipeline "${selectedPipeline}" deleted.`);
      setDeletePipelineDialogOpen(false);
      setSelectedPipeline("");
      setYamlText("");
      setPipelineConfig(null);
      setParameterData(null);
      setParamsText("{}");
      setParameterDrafts({});
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to delete pipeline");
    } finally {
      setSaving(false);
    }
  };

  const handleSavePipeline = async () => {
    if (!selectedPipeline) return;
    setSaving(true);
    setStatus("");
    try {
      const draftName = pipelineNameDraft.trim();
      let targetName = selectedPipeline;
      if (draftName && draftName !== selectedPipeline) {
        await renamePipeline(selectedPipeline, draftName);
        targetName = draftName;
        setSelectedPipeline(draftName);
      }
      await savePipelineYaml(targetName, yamlText);
      setYamlValidation("valid");
      setStatus(`Pipeline "${targetName}" saved.`);
      await refetchPipelines();
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to save pipeline");
    } finally {
      setSaving(false);
    }
  };

  const handleBuildPipeline = async () => {
    if (!selectedPipeline) return;
    if (buildStateTimerRef.current) {
      window.clearTimeout(buildStateTimerRef.current);
      buildStateTimerRef.current = null;
    }
    setConsoleCollapsed(false);
    setSaving(true);
    setBuildState("running");
    setStatus("");
    try {
      await buildPipeline(selectedPipeline);
      await refetchPipelines();
      setYamlValidation("valid");
      setStatus(`Pipeline "${selectedPipeline}" built successfully.`);
      setBuildState("success");
      try {
        const params = await fetchPipelineParameters(selectedPipeline);
        setParameterData(params);
        setParamsText(JSON.stringify(params, null, 2));
        setParameterDrafts({});
      } catch (error) {
        setParameterData({});
        setParamsText("{}");
        setParameterDrafts({});
        const message = error instanceof Error
          ? error.message
          : t("builder_params_not_found", "Parameters not found. Build first.");
        pushConsoleLog(message);
      }
      setPanelMode("parameters");
      setConsoleCollapsed(true);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to build pipeline");
      setBuildState("error");
    } finally {
      setSaving(false);
      buildStateTimerRef.current = window.setTimeout(() => {
        setBuildState("idle");
        buildStateTimerRef.current = null;
      }, 1600);
    }
  };

  const handleSaveParams = async () => {
    if (!selectedPipeline) return;
    setSaving(true);
    setStatus("");
    try {
      const payload = parameterData
        ? cloneJsonObject(parameterData)
        : (JSON.parse(paramsText) as Record<string, unknown>);
      await savePipelineParameters(selectedPipeline, payload);
      setStatus(t("builder_params_saved", "Parameters saved."));
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to save parameters");
    } finally {
      setSaving(false);
    }
  };

  const handleReloadParams = async () => {
    if (!selectedPipeline) {
      setStatus(t("builder_no_pipeline_selected", "No Pipeline Selected"));
      return;
    }
    setSaving(true);
    try {
      const params = await fetchPipelineParameters(selectedPipeline);
      setParameterData(params);
      setParamsText(JSON.stringify(params, null, 2));
      setParameterDrafts({});
      setStatus(t("builder_params_reloaded", "Parameters reloaded."));
    } catch (error) {
      setParameterData({});
      setParamsText("{}");
      setParameterDrafts({});
      setStatus(error instanceof Error ? error.message : t("builder_params_not_found", "Parameters not found. Build first."));
    } finally {
      setSaving(false);
    }
  };

  const handleSavePrompt = async () => {
    if (!selectedPromptPath) return;
    setSaving(true);
    setStatus("");
    try {
      await savePromptContent(selectedPromptPath, promptText);
      setPromptSavedText(promptText);
      setPromptTabState((previous) => ({
        ...previous,
        [selectedPromptPath]: {
          content: promptText,
          savedContent: promptText,
        },
      }));
      setStatus(`Prompt "${selectedPromptPath}" saved.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to save prompt");
    } finally {
      setSaving(false);
    }
  };

  const refreshPromptFiles = async (nextSelectedPath?: string) => {
    const files = await fetchPrompts();
    setPromptFiles(files);
    const validPaths = new Set(files.map((item) => item.path));
    setPromptOpenTabs((previous) => previous.filter((path) => validPaths.has(path)));
    setPromptTabState((previous) => {
      const next: Record<string, { content: string; savedContent: string }> = {};
      Object.entries(previous).forEach(([path, value]) => {
        if (validPaths.has(path)) next[path] = value;
      });
      return next;
    });
    if (!files.length) {
      setSelectedPromptPath("");
      setPromptText("");
      setPromptSavedText("");
      return;
    }
    if (nextSelectedPath) {
      const matchedPath = files.find((item) =>
        item.path === nextSelectedPath || item.name === nextSelectedPath,
      )?.path;
      if (matchedPath) {
        openPromptTab(matchedPath);
        return;
      }
    }
    if (selectedPromptPath && validPaths.has(selectedPromptPath)) {
      return;
    }
    openPromptTab(files[0].path);
  };

  const handleCreatePrompt = async () => {
    const normalizedInput = newPromptPathDraft.trim();
    if (!normalizedInput) return;
    const normalized =
      normalizedInput.endsWith(".jinja") || normalizedInput.endsWith(".jinja2")
        ? normalizedInput
        : `${normalizedInput}.jinja`;
    if (!normalized) return;
    setSaving(true);
    setStatus("");
    try {
      await createPrompt(normalized, "");
      await refreshPromptFiles(normalized);
      setPromptCreateDialogOpen(false);
      setNewPromptPathDraft("my_prompt.jinja");
      setStatus(`Prompt "${normalized}" created.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Failed to create prompt");
    } finally {
      setSaving(false);
    }
  };

  const handleRenamePrompt = async () => {
    const sourcePath = promptRenameSourcePath.trim();
    const nextPath = promptRenameDraft.trim();
    if (!sourcePath) return;
    if (!nextPath) {
      setStatus(t("builder_prompt_rename_required", "Prompt filename cannot be empty."));
      return;
    }
    if (sourcePath === nextPath) {
      setPromptRenameDialogOpen(false);
      setPromptRenameSourcePath("");
      return;
    }
    setSaving(true);
    setStatus("");
    try {
      await renamePrompt(sourcePath, nextPath);
      await refreshPromptFiles(nextPath);
      setPromptRenameDialogOpen(false);
      setPromptRenameSourcePath("");
      setPromptRenameDraft("");
      setStatus(
        formatTemplate(t("builder_prompt_renamed", "Renamed to: {name}"), {
          name: nextPath,
        }),
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : t("common_unknown_error", "Unknown error");
      setStatus(
        formatTemplate(t("builder_prompt_rename_failed", "Failed to rename prompt: {error}"), {
          error: message,
        }),
      );
    } finally {
      setSaving(false);
    }
  };

  const handleDeletePrompt = async (targetPath?: string) => {
    const path = (targetPath ?? promptDeleteTargetPath ?? selectedPromptPath).trim();
    if (!path) return;
    setSaving(true);
    setStatus("");
    try {
      await deletePrompt(path);
      await refreshPromptFiles();
      setPromptDeleteDialogOpen(false);
      setPromptDeleteTargetPath("");
      setStatus(
        formatTemplate(t("builder_prompt_deleted", "Deleted: {path}"), {
          path,
        }),
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : t("common_unknown_error", "Unknown error");
      setStatus(
        formatTemplate(t("builder_prompt_delete_failed", "Failed to delete prompt: {error}"), {
          error: message,
        }),
      );
    } finally {
      setSaving(false);
    }
  };

  const togglePipelineDropdown = () => {
    setPipelineDropdownOpen((previous) => !previous);
  };

  const selectPipelineFromMenu = (name: string) => {
    setSelectedPipeline(name);
    setPipelineDropdownOpen(false);
  };

  const openChatView = () => {
    const params = new URLSearchParams(window.location.search);
    if (selectedPipeline) {
      params.set("pipeline", selectedPipeline);
    } else {
      params.delete("pipeline");
    }
    const query = params.toString();
    navigate(query ? `/chat?${query}` : "/chat");
  };

  const handleYamlScroll = () => {
    if (yamlEditorRef.current && yamlGutterRef.current) {
      yamlGutterRef.current.scrollTop = yamlEditorRef.current.scrollTop;
    }
  };

  const handlePromptScroll = () => {
    if (promptEditorRef.current && promptGutterRef.current) {
      promptGutterRef.current.scrollTop = promptEditorRef.current.scrollTop;
    }
  };

  const parseParameterInput = (input: string, entryType: string): unknown => {
    let value: unknown = input;
    if (entryType === "number") {
      value = Number(input);
    } else if (entryType === "boolean") {
      value = input.trim().toLowerCase() === "true";
    } else if (entryType === "array" || entryType === "object") {
      try {
        value = JSON.parse(input);
      } catch {
        value = input;
      }
    }
    return value;
  };

  const getParameterDisplayValue = useCallback(
    (entry: ParameterEntry): string => {
      const draft = parameterDrafts[entry.fullPath];
      if (typeof draft === "string") return draft;
      if (entry.type === "array" || entry.type === "object") {
        if (entry.value === undefined) return "";
        try {
          const serialized = JSON.stringify(entry.value, null, 2);
          return typeof serialized === "string" ? serialized : "";
        } catch {
          return String(entry.value ?? "");
        }
      }
      return String(entry.value ?? "");
    },
    [parameterDrafts],
  );

  const handleParameterFieldChange = (entry: ParameterEntry, rawValue: string) => {
    setParameterDrafts((previous) => ({
      ...previous,
      [entry.fullPath]: rawValue,
    }));
    setStatus("");
  };

  const handleParameterFieldCommit = (entry: ParameterEntry, rawValue: string) => {
    setParameterData((previous) => {
      const base =
        previous && typeof previous === "object" && !Array.isArray(previous)
          ? cloneJsonObject(previous)
          : {};
      const parsed = parseParameterInput(rawValue, entry.type);
      setNestedValue(base, entry.fullPath, parsed);
      setParamsText(JSON.stringify(base, null, 2));
      return base;
    });
    setParameterDrafts((previous) => {
      if (!Object.prototype.hasOwnProperty.call(previous, entry.fullPath)) return previous;
      const next = { ...previous };
      delete next[entry.fullPath];
      return next;
    });
  };

  const scrollToParameterServer = (serverName: string) => {
    setParamsActiveServer(serverName);
    setParamsExpandedSections((previous) => ({ ...previous, [serverName]: true }));
    parameterSectionRefs.current[serverName]?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const toggleParameterSection = (serverName: string) => {
    setParamsExpandedSections((previous) => ({
      ...previous,
      [serverName]: !(previous[serverName] ?? true),
    }));
  };

  return (
    <>
      <nav className="navbar fixed-top builder-navbar">
        <div className="navbar-container">
          <div className="navbar-left">
            <a href="#pipeline-form" className="navbar-logo-link" id="builder-logo-link">
              <img src="/theme/ultrarag.svg" alt="UltraRAG" className="navbar-logo-img" />
            </a>
          </div>
          <div className="navbar-right">
            <span className="pipeline-label text-muted small" id="hero-selected-pipeline">
              {selectedPipeline || t("builder_no_pipeline_selected", "No Pipeline Selected")}
            </span>
            {selectedPipeline ? (
              <span
                className={`badge ms-2 ${selectedPipelineMeta?.is_ready ? "bg-success-subtle text-success" : "bg-warning-subtle text-warning"}`}
              >
                {selectedPipelineMeta?.is_ready
                  ? t("builder_pipeline_built_title", "Built")
                  : t("builder_build", "Build")}
              </span>
            ) : null}
            <div className="vr mx-2" />
            <button
              type="button"
              className={`btn btn-sm btn-ghost text-primary ${aiPanelOpen ? "active" : ""}`}
              id="navbar-ai-btn"
              onClick={() => {
                setAiPanelOpen((previous) => {
                  const next = !previous;
                  if (next) setAiView("home");
                  return next;
                });
                setAiSettingsOpen(false);
              }}
              title={t("builder_ai_assistant", "AI Assistant")}
              aria-pressed={aiPanelOpen}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z" />
                <circle cx="7.5" cy="14.5" r="1.5" />
                <circle cx="16.5" cy="14.5" r="1.5" />
              </svg>
              <span className="ms-1 fw-semibold">{t("builder_ai_assistant", "AI Assistant")}</span>
            </button>
          </div>
        </div>
      </nav>

      <main className="canvas-main" id="pipeline-form">
        <div className="workspace-layout">
          <aside className="workspace-sidebar">
            <nav className="workspace-nav">
              <div className="workspace-nav-top">
                <button
                  type="button"
                  className={`workspace-nav-btn ${panelMode === "pipeline" ? "active" : ""}`}
                  onClick={() => setPanelMode("pipeline")}
                  title={t("builder_nav_pipeline_yaml", "Pipeline YAML")}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="16 18 22 12 16 6" />
                    <polyline points="8 6 2 12 8 18" />
                  </svg>
                  <span className="nav-label">{t("builder_nav_pipeline", "Pipeline")}</span>
                </button>
                <button
                  type="button"
                  className={`workspace-nav-btn ${panelMode === "parameters" ? "active" : ""}`}
                  onClick={() => setPanelMode("parameters")}
                  title={t("builder_nav_parameters", "Parameters")}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="3" />
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
                  </svg>
                  <span className="nav-label">{t("builder_nav_params", "Params")}</span>
                </button>
                <button
                  type="button"
                  className={`workspace-nav-btn ${panelMode === "prompts" ? "active" : ""}`}
                  onClick={() => setPanelMode("prompts")}
                  title={t("builder_nav_prompt_templates", "Prompt Templates")}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                    <line x1="16" y1="13" x2="8" y2="13" />
                    <line x1="16" y1="17" x2="8" y2="17" />
                    <polyline points="10 9 9 9 8 9" />
                  </svg>
                  <span className="nav-label">{t("builder_nav_prompts", "Prompts")}</span>
                </button>
              </div>
              <div className="workspace-nav-spacer" />
              <div className="workspace-nav-divider" aria-hidden="true" />
              <button
                type="button"
                className="workspace-nav-btn"
                id="workspace-chat-btn"
                onClick={openChatView}
                title={t("builder_nav_chat", "Chat")}
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
                <span className="nav-label">{t("builder_nav_chat", "Chat")}</span>
              </button>
            </nav>
          </aside>

          <div className="workspace-content">
            <div className={`workspace-panel ${panelMode === "pipeline" ? "" : "d-none"}`} id="panel-pipeline">
              <div className="canvas-split-layout" ref={builderSplitRef}>
                <div className="canvas-left-panel">
                  <div className="canvas-panel-header">
                    <div className="panel-toolbar">
                      <div className="dropdown" ref={pipelineDropdownRef}>
                        <button
                          className={`btn btn-sm btn-outline-secondary dropdown-toggle ${pipelineDropdownOpen ? "show" : ""}`}
                          type="button"
                          id="pipelineDropdownBtn"
                          aria-expanded={pipelineDropdownOpen}
                          onClick={togglePipelineDropdown}
                        >
                          {selectedPipeline || t("builder_select_pipeline", "Select Pipeline")}
                        </button>
                        <ul
                          className={`dropdown-menu shadow-sm border-0 ${pipelineDropdownOpen ? "show" : ""}`}
                          id="pipeline-menu"
                        >
                          {pipelines.length ? (
                            pipelines
                              .slice()
                              .sort((a, b) => a.name.localeCompare(b.name, "en", { sensitivity: "base" }))
                              .map((pipeline) => (
                                <li key={pipeline.name}>
                                  <button
                                    type="button"
                                    className="dropdown-item small pipeline-menu-item d-flex align-items-center justify-content-between gap-2"
                                    onClick={() => selectPipelineFromMenu(pipeline.name)}
                                  >
                                    <span>{pipeline.name}</span>
                                    {pipeline.is_ready ? (
                                      <span
                                        className="pipeline-ready-dot"
                                        title={t("builder_pipeline_built_title", "Built")}
                                      />
                                    ) : null}
                                  </button>
                                </li>
                              ))
                          ) : (
                            <li>
                              <span className="dropdown-item text-muted small">
                                {t("builder_no_pipelines", "No pipelines")}
                              </span>
                            </li>
                          )}
                        </ul>
                      </div>
                    </div>
                  </div>
                  <div className="canvas-body">
                    <div
                      id="flow-canvas"
                      className="flow-canvas"
                      aria-label={t("builder_pipeline_canvas", "Pipeline canvas")}
                    >
                      {visiblePipelinePreviewError ? (
                        <p className="react-builder-preview-error px-3 py-2">{visiblePipelinePreviewError}</p>
                      ) : (
                        <PipelineCanvas
                          nodes={activeCanvasNodes}
                          rootId={activeContextNode.id}
                          activeContextId={activeContextId}
                          onInsertNode={handleCanvasInsertNode}
                          onAddBranchCase={handleCanvasAddBranchCase}
                          onActivateContext={handleCanvasActivateContext}
                          onEditNode={handleCanvasEditNode}
                          onDeleteNode={handleCanvasDeleteNode}
                        />
                      )}
                    </div>
                    <div id="context-controls" className="context-controls">
                      <div className="context-breadcrumb d-flex flex-wrap gap-2 align-items-center">
                        {activeContextPath.map((contextNode, index) => {
                          const isLast = index === activeContextPath.length - 1;
                          const normalized = contextNode.label.trim().toLowerCase();
                          const label = index === 0
                            ? t("builder_context_root", "Root")
                            : contextNode.kind === "loop"
                              ? t("builder_context_loop", "Loop")
                              : contextNode.kind === "branch" && normalized.startsWith("branch:")
                                ? `${t("builder_context_case_short", "Case")}: ${contextNode.label.replace(/^branch:\s*/i, "")}`
                                : contextNode.kind === "branch"
                                  ? t("builder_context_branch_short", "Branch")
                                  : contextNode.kind === "group" && normalized === "router"
                                    ? t("builder_context_router", "Router")
                                    : contextNode.label;
                          return (
                            <button
                              key={`${contextNode.id}-${index}`}
                              type="button"
                              className={`btn btn-sm rounded-pill ${isLast ? "btn-dark" : "btn-light border"}`}
                              onClick={() => setActiveContextId(contextNode.id)}
                            >
                              {label}
                            </button>
                          );
                        })}
                        {activeContextId !== PIPELINE_PREVIEW_ROOT_ID ? (
                          <button
                            type="button"
                            className="btn btn-sm btn-link text-danger text-decoration-none"
                            onClick={() => setActiveContextId(PIPELINE_PREVIEW_ROOT_ID)}
                          >
                            {t("builder_exit_context", "Exit Context")}
                          </button>
                        ) : null}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="canvas-resizer" id="builder-resizer" ref={builderResizerRef} />

                <div className="canvas-right-panel">
                  <div className="canvas-panel-header editor-header">
                    <div className="panel-title-group">
                      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="16 18 22 12 16 6" />
                        <polyline points="8 6 2 12 8 18" />
                      </svg>
                      <input
                        type="text"
                        id="pipeline-name"
                        className="pipeline-name-input"
                        value={pipelineNameDraft}
                        onChange={(event) => setPipelineNameDraft(event.target.value)}
                        placeholder={t("builder_pipeline_name_placeholder", "Pipeline name...")}
                      />
                    </div>
                    <div className="panel-toolbar">
                      <button
                        type="button"
                        id="new-pipeline-btn"
                        className="btn btn-sm builder-action-btn"
                        disabled={saving}
                        onClick={() => {
                          setNewPipelineNameDraft("NewPipeline");
                          setCreatePipelineDialogOpen(true);
                        }}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <line x1="12" y1="5" x2="12" y2="19" />
                          <line x1="5" y1="12" x2="19" y2="12" />
                        </svg>
                        <span>{t("common_new", "New")}</span>
                      </button>
                      <button
                        type="button"
                        id="save-pipeline"
                        className="btn btn-sm builder-action-btn"
                        disabled={saving || !selectedPipeline}
                        onClick={() => void handleSavePipeline()}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
                          <polyline points="17 21 17 13 7 13 7 21" />
                          <polyline points="7 3 7 8 15 8" />
                        </svg>
                        <span>{t("common_save", "Save")}</span>
                      </button>
                      <button
                        type="button"
                        id="build-pipeline"
                        className="btn btn-sm builder-action-btn"
                        disabled={saving || buildState === "running" || !selectedPipeline}
                        onClick={() => void handleBuildPipeline()}
                      >
                        {buildState === "running" ? (
                          <span className="spinner-border spinner-border-sm" aria-hidden="true" />
                        ) : (
                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                          </svg>
                        )}
                        <span>{buildButtonLabel}</span>
                      </button>
                      <button
                        type="button"
                        id="delete-pipeline"
                        className="btn btn-sm builder-action-btn builder-action-btn--danger"
                        disabled={saving || !selectedPipeline}
                        onClick={() => setDeletePipelineDialogOpen(true)}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="3 6 5 6 21 6" />
                          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                        </svg>
                        <span>{t("common_delete", "Delete")}</span>
                      </button>
                      <span
                        id="yaml-sync-status"
                        className={`sync-status ${syncStatusClass}`}
                        title={
                          yamlValidation === "valid"
                            ? t("builder_yaml_synced_title", "Synced with canvas")
                            : yamlValidation === "validating"
                              ? t("builder_yaml_syncing_title", "Syncing...")
                              : yamlValidation === "invalid"
                                ? t("builder_yaml_error_title", "Parse error")
                                : t("builder_yaml_modified_title", "Editor modified")
                        }
                      >
                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          width="14"
                          height="14"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <polyline points="20 6 9 17 4 12" />
                        </svg>
                      </span>
                    </div>
                  </div>

                  <div className="yaml-editor-container">
                    <div className="yaml-editor-wrapper">
                      <div className="yaml-gutter" id="yaml-line-numbers" ref={yamlGutterRef}>
                        {yamlLineNumbers.map((lineNumber) => (
                          <div key={lineNumber} className="line-number">
                            {lineNumber}
                          </div>
                        ))}
                      </div>
                      <textarea
                        id="yaml-editor"
                        className="yaml-editor"
                        ref={yamlEditorRef}
                        spellCheck={false}
                        value={yamlText}
                        onScroll={handleYamlScroll}
                        onChange={(event) => {
                          setYamlText(event.target.value);
                          setYamlValidation("idle");
                        }}
                        onKeyDown={(event) => {
                          const isSaveShortcut =
                            (event.key === "s" || event.key === "S") && (event.ctrlKey || event.metaKey);
                          if (isSaveShortcut) {
                            event.preventDefault();
                            if (!saving && selectedPipeline) {
                              void handleSavePipeline();
                            }
                            return;
                          }
                          if (event.key === "Tab") {
                            event.preventDefault();
                            const target = event.currentTarget;
                            const start = target.selectionStart;
                            const end = target.selectionEnd;
                            const nextValue = `${yamlText.slice(0, start)}  ${yamlText.slice(end)}`;
                            setYamlText(nextValue);
                            setYamlValidation("idle");
                            window.requestAnimationFrame(() => {
                              target.selectionStart = target.selectionEnd = start + 2;
                            });
                          }
                        }}
                        placeholder={t("builder_yaml_placeholder", "# Write your pipeline YAML here...")}
                      />
                    </div>
                    {visiblePipelinePreviewError ? (
                      <div id="yaml-error-bar" className="yaml-error-bar">
                        <span>{visiblePipelinePreviewError}</span>
                      </div>
                    ) : null}
                  </div>
                </div>
              </div>
            </div>

            <div className={`workspace-panel ${panelMode === "parameters" ? "" : "d-none"}`} id="panel-parameters">
              <div className="params-layout" ref={paramsLayoutRef}>
                <aside className="params-sidebar">
                  <div className="params-sidebar-header">
                    <span>{t("builder_params_servers", "Servers")}</span>
                    <button
                      type="button"
                      id="params-refresh-btn"
                      className="btn btn-sm btn-ghost"
                      title={t("builder_params_reload", "Reload Parameters")}
                      disabled={saving || !selectedPipeline}
                      onClick={() => void handleReloadParams()}
                    >
                      <RefreshCcw size={14} />
                    </button>
                  </div>
                  <nav id="parameter-nav" className="params-nav">
                    {parameterServers.length ? (
                      parameterServers.map((serverName) => (
                        <button
                          key={serverName}
                          type="button"
                          className={`parameter-nav-item ${paramsActiveServer === serverName ? "active" : ""}`}
                          onClick={() => scrollToParameterServer(serverName)}
                        >
                          <span className="nav-item-name">{serverName}</span>
                          <span className="nav-item-count">
                            {groupedParameterEntries[serverName]?.length ?? 0}
                          </span>
                        </button>
                      ))
                    ) : (
                      <div className="params-empty">
                        <p>{t("builder_params_not_found", "Parameters not found.")}</p>
                      </div>
                    )}
                  </nav>
                </aside>

                <div className="workspace-split-resizer" id="params-resizer" ref={paramsResizerRef} />

                <main className="params-main">
                  <div className="params-main-header canvas-panel-header">
                    <div className="panel-title-group">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="18"
                        height="18"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <circle cx="12" cy="12" r="3" />
                        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
                      </svg>
                      <span className="panel-title">{t("builder_params_panel_title", "Parameters")}</span>
                    </div>
                    <div className="panel-toolbar">
                      <label className="params-toggle-label" title={t("builder_params_toggle_title", "Toggle display mode")}>
                        <span className="params-toggle-text">{t("builder_params_simplified", "Simplified")}</span>
                        <input
                          type="checkbox"
                          id="params-simplified-toggle"
                          className="params-toggle-checkbox"
                          checked={paramsSimplified}
                          onChange={(event) => setParamsSimplified(event.target.checked)}
                        />
                        <span className="params-toggle-switch" />
                      </label>
                      <button
                        type="button"
                        id="params-save-btn"
                        className="btn btn-sm builder-action-btn"
                        disabled={saving || !selectedPipeline}
                        onClick={() => void handleSaveParams()}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
                          <polyline points="17 21 17 13 7 13 7 21" />
                          <polyline points="7 3 7 8 15 8" />
                        </svg>
                        <span>{t("common_save", "Save")}</span>
                      </button>
                    </div>
                  </div>
                  <div className="params-main-body">
                    <p className="parameter-hint">
                      {t(
                        "builder_params_hint",
                        "Select a server from the left to configure its parameters, or expand all sections below.",
                      )}
                    </p>
                    {!selectedPipeline || !selectedPipelineMeta?.is_ready ? (
                      <div id="params-empty" className="params-empty">
                        <p
                          dangerouslySetInnerHTML={{
                            __html: t(
                              "builder_params_empty",
                              "Please <strong>Build</strong> a pipeline first to configure parameters.",
                            ),
                          }}
                        />
                      </div>
                    ) : !parameterServers.length ? (
                      <div id="params-empty" className="params-empty">
                        <p>{t("builder_params_not_found", "Parameters not found.")}</p>
                      </div>
                    ) : (
                      <div id="parameter-form" className="params-sections">
                        {parameterServers.map((serverName) => {
                          const entries = groupedParameterEntries[serverName] ?? [];
                          const expanded = paramsExpandedSections[serverName] ?? true;
                          return (
                            <div
                              key={serverName}
                              id={`param-section-${serverName}`}
                              className={`parameter-section ${expanded ? "expanded" : ""}`}
                              ref={(node) => {
                                parameterSectionRefs.current[serverName] = node;
                              }}
                            >
                              <div
                                className="parameter-section-header"
                                onClick={() => toggleParameterSection(serverName)}
                              >
                                <div className="section-header-left">
                                  <svg className="section-chevron" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <polyline points="6 9 12 15 18 9" />
                                  </svg>
                                  <span className="section-title">{serverName}</span>
                                  <span className="section-badge">{entries.length} params</span>
                                </div>
                              </div>
                              <div className="parameter-section-content">
                                <div className="parameter-grid">
                                  {entries.map((entry) => {
                                    const isComplex = entry.type === "array" || entry.type === "object";
                                    const displayValue = getParameterDisplayValue(entry);
                                    return (
                                      <div
                                        key={entry.fullPath}
                                        className={`parameter-field ${isComplex ? "full-width" : ""}`}
                                        data-parameter-path={entry.fullPath}
                                        ref={(node) => {
                                          parameterFieldRefs.current[entry.fullPath] = node;
                                        }}
                                      >
                                        <label className="parameter-label" title={entry.fullPath}>
                                          {entry.displayPath}
                                        </label>
                                        {isComplex ? (
                                          <textarea
                                            className="parameter-input"
                                            rows={4}
                                            value={displayValue}
                                            onChange={(event) =>
                                              handleParameterFieldChange(entry, event.target.value)
                                            }
                                            onBlur={(event) =>
                                              handleParameterFieldCommit(entry, event.target.value)
                                            }
                                          />
                                        ) : (
                                          <input
                                            className="parameter-input"
                                            type="text"
                                            value={displayValue}
                                            onChange={(event) =>
                                              handleParameterFieldChange(entry, event.target.value)
                                            }
                                            onBlur={(event) =>
                                              handleParameterFieldCommit(entry, event.target.value)
                                            }
                                          />
                                        )}
                                      </div>
                                    );
                                  })}
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </main>
              </div>
            </div>

            <div className={`workspace-panel ${panelMode === "prompts" ? "" : "d-none"}`} id="panel-prompts">
              <div className="prompts-layout" ref={promptsLayoutRef}>
                <aside className="prompts-sidebar">
                  <div className="prompts-sidebar-header">
                    <span>{t("builder_prompt_files", "Prompt Files")}</span>
                  </div>
                  <nav id="prompt-list" className="prompts-list">
                    {filteredPromptFiles.map((prompt) => (
                      <button
                        key={prompt.path}
                        type="button"
                        className={`prompt-item ${selectedPromptPath === prompt.path ? "active" : ""}`}
                        onClick={() => openPromptTab(prompt.path)}
                        onContextMenu={(event) => openPromptContextMenu(event, prompt.path)}
                        title={prompt.path}
                      >
                        <span className="prompt-item-icon" aria-hidden="true">
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="16"
                            height="16"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                            <polyline points="14 2 14 8 20 8" />
                            <line x1="16" y1="13" x2="8" y2="13" />
                            <line x1="16" y1="17" x2="8" y2="17" />
                          </svg>
                        </span>
                        <span className="prompt-item-name">{prompt.name || prompt.path}</span>
                      </button>
                    ))}
                  </nav>
                </aside>

                <div className="workspace-split-resizer" id="prompts-resizer" ref={promptsResizerRef} />

                <main className="prompts-main">
                  <div className="prompts-editor-header canvas-panel-header">
                    <div className="panel-title-group">
                      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                        <line x1="16" y1="13" x2="8" y2="13" />
                        <line x1="16" y1="17" x2="8" y2="17" />
                        <polyline points="10 9 9 9 8 9" />
                      </svg>
                      <span className="panel-title">{t("builder_nav_prompts", "Prompts")}</span>
                      <span
                        id="prompt-modified"
                        className={`prompt-modified-badge ${promptModified ? "" : "d-none"}`}
                      >
                        {t("builder_prompt_modified", "Modified")}
                      </span>
                    </div>
                    <div className="panel-toolbar prompt-toolbar">
                      <input
                        type="text"
                        id="prompt-search"
                        className="form-control form-control-sm prompt-search-input"
                        placeholder={t("builder_prompt_search_placeholder", "Search prompts...")}
                        value={promptSearch}
                        onChange={(event) => setPromptSearch(event.target.value)}
                      />
                      <button
                        type="button"
                        id="prompt-new-btn"
                        className="btn btn-sm builder-action-btn"
                        disabled={saving}
                        onClick={() => {
                          setPromptCreateDialogOpen(true);
                          setNewPromptPathDraft("my_prompt.jinja");
                        }}
                        title={t("builder_new_prompt", "New Prompt")}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <line x1="12" y1="5" x2="12" y2="19" />
                          <line x1="5" y1="12" x2="19" y2="12" />
                        </svg>
                        <span>{t("common_new", "New")}</span>
                      </button>
                      <button
                        type="button"
                        id="prompt-save-btn"
                        className="btn btn-sm builder-action-btn"
                        disabled={saving || !selectedPromptPath}
                        onClick={() => void handleSavePrompt()}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
                          <polyline points="17 21 17 13 7 13 7 21" />
                          <polyline points="7 3 7 8 15 8" />
                        </svg>
                        <span>{t("common_save", "Save")}</span>
                      </button>
                      <button
                        type="button"
                        id="prompt-delete-btn"
                        className="btn btn-sm builder-action-btn builder-action-btn--danger"
                        disabled={saving || !selectedPromptPath}
                        onClick={() => {
                          if (!selectedPromptPath) return;
                          openPromptDeleteDialog(selectedPromptPath);
                        }}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="3 6 5 6 21 6" />
                          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                        </svg>
                        <span>{t("common_delete", "Delete")}</span>
                      </button>
                    </div>
                  </div>
                  <div className={`prompt-tabs ${promptOpenTabs.length ? "" : "d-none"}`} id="prompt-tabs">
                    {promptOpenTabs.map((path) => {
                      const file = promptFiles.find((item) => item.path === path);
                      const pathSegments = path.split("/");
                      const name = file?.name || pathSegments[pathSegments.length - 1] || path;
                      const tab = promptTabState[path];
                      const unsaved = Boolean(tab && tab.content !== tab.savedContent);
                      return (
                        <div
                          key={path}
                          className={`prompt-tab ${selectedPromptPath === path ? "active" : ""} ${unsaved ? "unsaved" : ""}`}
                          onClick={() => openPromptTab(path)}
                          onContextMenu={(event) => openPromptContextMenu(event, path)}
                          role="button"
                          tabIndex={0}
                          onKeyDown={(event) => {
                            if (event.key === "Enter" || event.key === " ") {
                              event.preventDefault();
                              openPromptTab(path);
                            }
                          }}
                        >
                          <span className="prompt-tab-name">{name}</span>
                          <button
                            type="button"
                            className="prompt-tab-close"
                            title={t("common_close", "Close")}
                            onClick={(event) => {
                              event.stopPropagation();
                              closePromptTab(path);
                            }}
                          >
                            ×
                          </button>
                        </div>
                      );
                    })}
                  </div>

                  <div className="prompts-editor-container">
                    <div id="prompt-empty" className={`prompts-empty ${selectedPromptPath ? "d-none" : ""}`}>
                      <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                        <line x1="16" y1="13" x2="8" y2="13" />
                        <line x1="16" y1="17" x2="8" y2="17" />
                        <polyline points="10 9 9 9 8 9" />
                      </svg>
                      <p>{t("builder_prompt_empty", "Select a prompt file from the list or create a new one.")}</p>
                    </div>
                    <div className={`prompt-editor-wrapper ${selectedPromptPath ? "" : "d-none"}`} id="prompt-editor-wrapper">
                      <div className="prompt-gutter" id="prompt-line-numbers" ref={promptGutterRef}>
                        {promptLineNumbers.map((lineNumber) => (
                          <div key={lineNumber} className="line-number">
                            {lineNumber}
                          </div>
                        ))}
                      </div>
                      <textarea
                        id="prompt-editor"
                        className="prompt-editor"
                        ref={promptEditorRef}
                        spellCheck={false}
                        value={promptText}
                        onScroll={handlePromptScroll}
                        onChange={(event) => {
                          const nextValue = event.target.value;
                          setPromptText(nextValue);
                          if (selectedPromptPath) {
                            setPromptTabState((previous) => {
                              const current = previous[selectedPromptPath] ?? {
                                content: "",
                                savedContent: promptSavedText,
                              };
                              if (current.content === nextValue) return previous;
                              return {
                                ...previous,
                                [selectedPromptPath]: {
                                  ...current,
                                  content: nextValue,
                                },
                              };
                            });
                          }
                        }}
                        onKeyDown={(event) => {
                          const isSaveShortcut =
                            (event.key === "s" || event.key === "S") && (event.ctrlKey || event.metaKey);
                          if (isSaveShortcut) {
                            event.preventDefault();
                            if (!saving && selectedPromptPath) {
                              void handleSavePrompt();
                            }
                            return;
                          }
                          if (event.key === "Tab") {
                            event.preventDefault();
                            const target = event.currentTarget;
                            const start = target.selectionStart;
                            const end = target.selectionEnd;
                            const nextValue = `${promptText.slice(0, start)}  ${promptText.slice(end)}`;
                            setPromptText(nextValue);
                            if (selectedPromptPath) {
                              setPromptTabState((previous) => {
                                const current = previous[selectedPromptPath] ?? {
                                  content: "",
                                  savedContent: promptSavedText,
                                };
                                if (current.content === nextValue) return previous;
                                return {
                                  ...previous,
                                  [selectedPromptPath]: {
                                    ...current,
                                    content: nextValue,
                                  },
                                };
                              });
                            }
                            window.requestAnimationFrame(() => {
                              target.selectionStart = target.selectionEnd = start + 2;
                            });
                          }
                        }}
                        placeholder={t("builder_prompt_template_placeholder", "# Jinja Template...")}
                      />
                    </div>
                  </div>
                </main>
              </div>
            </div>
          </div>
        </div>

        <div className={`canvas-console ${consoleCollapsed ? "collapsed" : ""}`} id="canvas-console">
          <div
            className="console-header"
            id="console-toggle"
            onClick={() => setConsoleCollapsed((previous) => !previous)}
          >
            <div className="console-title">
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="4 17 10 11 4 5" />
                <line x1="12" y1="19" x2="20" y2="19" />
              </svg>
              <span>{t("builder_console_title", "Console")}</span>
            </div>
            <button
              type="button"
              className="console-toggle-btn"
              title={t("builder_console_toggle", "Toggle Console")}
              onClick={(event) => {
                event.stopPropagation();
                setConsoleCollapsed((previous) => !previous);
              }}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="chevron-icon"
              >
                <polyline points="18 15 12 9 6 15" />
              </svg>
            </button>
          </div>
          <pre id="log" className="console-output" ref={consoleOutputRef}>
            {consoleLogs.join("\n")}
          </pre>
        </div>

        <div className="ai-assistant-container" id="ai-assistant-container">
          <div
            className={`ai-assistant-panel ${aiPanelOpen ? "open" : ""}`}
            id="ai-assistant-panel"
            ref={aiPanelRef}
          >
            <div className="ai-panel-resizer" id="ai-panel-resizer" ref={aiPanelResizerRef} />
            <div className="ai-panel-content">
              <div className="ai-panel-header">
                <div className="ai-panel-title">
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z" />
                    <circle cx="7.5" cy="14.5" r="1.5" />
                    <circle cx="16.5" cy="14.5" r="1.5" />
                  </svg>
                  <span>{t("builder_ai_assistant", "AI Assistant")}</span>
                </div>
                <div className="ai-panel-actions">
                  <button
                    className={`ai-panel-btn ${aiView === "home" ? "d-none" : ""}`}
                    type="button"
                    id="ai-back-btn"
                    title={t("builder_ai_back", "Back to list")}
                    onClick={() => setAiView("home")}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="15 18 9 12 15 6" />
                    </svg>
                  </button>
                  <button
                    className="ai-panel-btn"
                    type="button"
                    id="ai-settings-btn"
                    title={t("settings", "Settings")}
                    onClick={() => setAiSettingsOpen(true)}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <circle cx="12" cy="12" r="3" />
                      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
                    </svg>
                  </button>
                  <button
                    className="ai-panel-btn"
                    type="button"
                    id="ai-close-btn"
                    title={t("common_close", "Close")}
                    onClick={() => {
                      setAiPanelOpen(false);
                      setAiSettingsOpen(false);
                    }}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="18" y1="6" x2="6" y2="18" />
                      <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                  </button>
                </div>
              </div>

              <div
                className="ai-connection-status"
                id="ai-connection-status"
                onClick={() => setAiSettingsOpen(true)}
                style={{ cursor: "pointer" }}
              >
                <div className="ai-status-left">
                  <span className={`ai-status-dot ${aiTestingConnection ? "connecting" : aiConnected ? "connected" : "disconnected"}`} />
                  <span className="ai-status-text" id="ai-status-text">{aiStatusText}</span>
                </div>
                <div className="ai-context-hint" id="ai-context-hint">{aiContextHint}</div>
              </div>

              <div className={`ai-view ${aiView === "home" ? "active" : "d-none"}`} id="ai-home-view">
                <div className="ai-session-bar">
                  <div className="ai-session-title">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <rect x="3" y="4" width="18" height="18" rx="2" />
                      <path d="M16 2v4" />
                      <path d="M8 2v4" />
                      <path d="M3 10h18" />
                    </svg>
                    <span>{t("builder_ai_recent", "Recent")}</span>
                  </div>
                  <div className="ai-session-actions">
                    <button
                      className="ai-session-btn"
                      type="button"
                      id="ai-session-new"
                      title={t("builder_ai_new_session", "New session")}
                      onClick={handleNewAiSession}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="12" y1="5" x2="12" y2="19" />
                        <line x1="5" y1="12" x2="19" y2="12" />
                      </svg>
                    </button>
                    <button
                      className="ai-session-btn"
                      type="button"
                      id="ai-session-delete"
                      title={t("builder_ai_delete_session", "Delete session")}
                      disabled={!visibleAiSessions.length}
                      onClick={deleteAllAiSessions}
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="3 6 5 6 21 6" />
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                      </svg>
                    </button>
                  </div>
                </div>

                <div className="ai-session-list" id="ai-session-list">
                  {visibleAiSessions.length ? (
                    visibleAiSessions
                      .map((session) => (
                        <div
                          key={session.id}
                          className={`ai-session-item ${session.id === aiCurrentSessionId ? "active" : ""}`}
                        >
                          <button
                            type="button"
                            className="ai-session-content"
                            onClick={() => switchAiSession(session.id)}
                          >
                            <div className="ai-session-title-text">
                              {session.title || t("builder_ai_session_new", "New Session")}
                            </div>
                            <div className="ai-session-meta">
                              {new Date(session.updatedAt).toLocaleString()}
                            </div>
                          </button>
                          <button
                            type="button"
                            className="ai-session-delete-btn"
                            title={t("builder_ai_delete_session", "Delete session")}
                            onClick={(event) => {
                              event.stopPropagation();
                              deleteAiSession(session.id);
                            }}
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <polyline points="3 6 5 6 21 6" />
                              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                            </svg>
                          </button>
                        </div>
                      ))
                  ) : (
                    <div className="ai-session-empty">
                      {t("builder_ai_sessions_empty", "No sessions yet")}
                    </div>
                  )}
                </div>
              </div>

              <div className={`ai-view ${aiView === "chat" ? "active" : "d-none"}`} id="ai-chat-view">
                <div className="ai-messages" id="ai-messages" ref={aiMessagesRef}>
                  {aiMessages.length ? (
                    aiMessages.map((message) => (
                      <div key={message.id} className={`ai-message ${message.role}`}>
                        <div className="ai-message-avatar">
                          {message.role === "assistant" ? (
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z" />
                            </svg>
                          ) : (
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                              <circle cx="12" cy="7" r="4" />
                            </svg>
                          )}
                        </div>
                        <div className="ai-message-content">
                          {message.pending ? (
                            <div className="ai-thinking">
                              <span className="ai-thinking-dot" />
                              <span className="ai-thinking-dot" />
                              <span className="ai-thinking-dot" />
                            </div>
                          ) : (
                            <div dangerouslySetInnerHTML={{ __html: renderChatMarkdown(message.content) }} />
                          )}
                          {!message.pending && message.role === "assistant" && (message.actions?.length ?? 0) > 0
                            ? message.actions?.map((action, actionIndex) => {
                                const state = message.actionStates?.[actionIndex];
                                const preview =
                                  action.preview ||
                                  action.content ||
                                  (action.path
                                    ? `${action.path}: ${JSON.stringify(action.value ?? "")}`
                                    : "");
                                const actionLabel =
                                  action.type === "modify_pipeline"
                                    ? t("builder_ai_action_modify_pipeline", "Pipeline Modification")
                                    : action.type === "modify_prompt"
                                      ? t("builder_ai_action_modify_prompt", "Prompt Modification")
                                      : action.type === "modify_parameter"
                                        ? t("builder_ai_action_modify_parameter", "Parameter Change")
                                        : t("builder_ai_action_modify_generic", "Modification");
                                return (
                                  <div
                                    key={`${message.id}-action-${actionIndex}`}
                                    className={`ai-action-block ${
                                      state?.status === "applied"
                                        ? "applied"
                                        : state?.status === "rejected"
                                          ? "rejected"
                                          : state?.status === "error"
                                            ? "error"
                                            : ""
                                    }`}
                                    ref={(node) => {
                                      aiActionBlockRefs.current[`${message.id}-${actionIndex}`] = node;
                                    }}
                                  >
                                    <div className="ai-action-header">
                                      <span className="ai-action-type">
                                        {action.type === "modify_pipeline" ? (
                                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <polyline points="16 18 22 12 16 6" />
                                            <polyline points="8 6 2 12 8 18" />
                                          </svg>
                                        ) : action.type === "modify_prompt" ? (
                                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                            <polyline points="14 2 14 8 20 8" />
                                          </svg>
                                        ) : (
                                          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                            <circle cx="12" cy="12" r="3" />
                                          </svg>
                                        )}
                                        {actionLabel}
                                      </span>
                                      <div className="ai-action-buttons">
                                        {state ? (
                                          <span
                                            style={{
                                              color:
                                                state.status === "applied"
                                                  ? "#22c55e"
                                                  : state.status === "rejected"
                                                    ? "#94a3b8"
                                                    : "#ef4444",
                                              fontSize: "0.75rem",
                                            }}
                                          >
                                            {state.status === "applied"
                                              ? t("builder_ai_action_applied", "Applied")
                                              : state.status === "rejected"
                                                ? t("builder_ai_action_rejected", "Rejected")
                                                : state.message ||
                                                  t("builder_ai_action_apply_failed", "Failed to apply")}
                                          </span>
                                        ) : (
                                          <>
                                            <button
                                              className="ai-action-btn apply"
                                              type="button"
                                              onClick={() =>
                                                void handleApplyAiAction(message.id, actionIndex, action)
                                              }
                                            >
                                              {t("common_apply", "Apply")}
                                            </button>
                                            <button
                                              className="ai-action-btn reject"
                                              type="button"
                                              onClick={() => handleRejectAiAction(message.id, actionIndex)}
                                            >
                                              {t("common_reject", "Reject")}
                                            </button>
                                          </>
                                        )}
                                      </div>
                                    </div>
                                    <pre className="ai-action-preview">{preview}</pre>
                                  </div>
                                );
                              })
                            : null}
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="ai-welcome">
                      <div className="ai-welcome-icon">
                        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73c-.6-.34-1-.99-1-1.73a2 2 0 0 1 2-2z" />
                          <circle cx="7.5" cy="14.5" r="1.5" />
                          <circle cx="16.5" cy="14.5" r="1.5" />
                        </svg>
                      </div>
                      <h4>{t("builder_ai_welcome_title", "UltraRAG AI Assistant")}</h4>
                      <p>{t("builder_ai_welcome_desc", "I can help you build pipelines, configure parameters, and edit prompts.")}</p>
                      <p className="ai-welcome-hint">{t("builder_ai_welcome_hint", "Click the settings icon to configure your API connection.")}</p>
                      <div className="ai-starter-chips">
                        <button
                          type="button"
                          className="ai-starter-chip"
                          onClick={() =>
                            handleAiStarterChip(
                              t(
                                "builder_ai_prompt_pipeline_adjustment",
                                "Update the current RAG pipeline to include a citation module, ensuring the final output displays source references for fact-checking purposes.",
                              ),
                            )
                          }
                        >
                          <span className="ai-starter-chip-icon">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <circle cx="12" cy="12" r="3" />
                              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
                            </svg>
                          </span>
                          <span className="ai-starter-chip-text">
                            {t("builder_ai_chip_pipeline_adjustment", "Pipeline Adjustment")}
                          </span>
                        </button>
                        <button
                          type="button"
                          className="ai-starter-chip"
                          onClick={() =>
                            handleAiStarterChip(
                              t(
                                "builder_ai_prompt_prompt_adaptation",
                                "Optimize the system prompt for the [Insert Domain, e.g., Medical/Legal] domain. Please refine the instructions to ensure the generated responses strictly adhere to professional terminology and logical accuracy suitable for this field.",
                              ),
                            )
                          }
                        >
                          <span className="ai-starter-chip-icon">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M12 20h9" />
                              <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z" />
                            </svg>
                          </span>
                          <span className="ai-starter-chip-text">
                            {t("builder_ai_chip_prompt_adaptation", "Prompt Adaptation")}
                          </span>
                        </button>
                        <button
                          type="button"
                          className="ai-starter-chip"
                          onClick={() =>
                            handleAiStarterChip(
                              t(
                                "builder_ai_prompt_parameter_settings",
                                "Reconfigure the generation backend. Switch the backend type to OpenAI, set the model name to [Insert Model Name, e.g., Llama-3-70B], and update the API endpoint to port [Insert Port, e.g., 8000].",
                              ),
                            )
                          }
                        >
                          <span className="ai-starter-chip-icon">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <line x1="4" y1="21" x2="4" y2="14" />
                              <line x1="4" y1="10" x2="4" y2="3" />
                              <line x1="12" y1="21" x2="12" y2="12" />
                              <line x1="12" y1="8" x2="12" y2="3" />
                              <line x1="20" y1="21" x2="20" y2="16" />
                              <line x1="20" y1="12" x2="20" y2="3" />
                              <line x1="1" y1="14" x2="7" y2="14" />
                              <line x1="9" y1="8" x2="15" y2="8" />
                              <line x1="17" y1="16" x2="23" y2="16" />
                            </svg>
                          </span>
                          <span className="ai-starter-chip-text">
                            {t("builder_ai_chip_parameter_settings", "Parameter Settings")}
                          </span>
                        </button>
                        <button
                          type="button"
                          className="ai-starter-chip"
                          onClick={() =>
                            handleAiStarterChip(
                              t(
                                "builder_ai_prompt_freeform_tuning",
                                "I want to redesign my RAG workflow based on this article/paper: [Insert Link]. Please analyze its core methodologies and assist me in constructing a similar pipeline architecture.",
                              ),
                            )
                          }
                        >
                          <span className="ai-starter-chip-icon">
                            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5" />
                              <path d="M9 18h6" />
                              <path d="M10 22h4" />
                            </svg>
                          </span>
                          <span className="ai-starter-chip-text">
                            {t("builder_ai_chip_freeform_tuning", "Free-form Tuning")}
                          </span>
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                <div className="ai-input-area">
                  <div className="ai-input-wrapper">
                    <textarea
                      id="ai-input"
                      className="ai-input"
                      ref={aiInputRef}
                      rows={1}
                      value={aiInput}
                      disabled={aiBusy}
                      placeholder={t("builder_ai_input_placeholder", "Ask me anything about your pipeline...")}
                      onCompositionStart={() => setAiComposing(true)}
                      onCompositionEnd={() => setAiComposing(false)}
                      onChange={(event) => {
                        setAiInput(event.target.value);
                        adjustAiInputHeight(event.currentTarget);
                      }}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" && !event.shiftKey && !aiComposing) {
                          event.preventDefault();
                          void handleSendAiMessage();
                        }
                      }}
                    />
                    <button
                      className={`ai-send-btn btn-send ${aiBusy ? "stop" : ""} ${aiBusy || aiInput.trim() ? "active" : ""}`}
                      id="ai-send-btn"
                      type="button"
                      title={aiBusy ? t("builder_ai_stop", "Stop") : t("builder_ai_send", "Send")}
                      disabled={!aiBusy && !aiInput.trim()}
                      onClick={() => {
                        if (aiBusy) {
                          aiRequestControllerRef.current?.abort();
                          return;
                        }
                        void handleSendAiMessage();
                      }}
                    >
                      <span className="send-icon-wrapper" id="ai-send-icon">
                        {aiBusy ? (
                          <span className="icon-stop" />
                        ) : (
                          <svg className="icon-send" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <line x1="12" y1="19" x2="12" y2="5" />
                            <polyline points="5 12 12 5 19 12" />
                          </svg>
                        )}
                      </span>
                    </button>
                  </div>
                  <div className="ai-input-hint">
                    <span>{t("builder_ai_input_hint", "Press Enter to send, Shift+Enter for new line")}</span>
                  </div>
                </div>
              </div>

              <div className={`ai-settings-panel ${aiSettingsOpen ? "open" : ""}`} id="ai-settings-panel">
                <div className="ai-settings-header">
                  <h4>{t("builder_ai_settings_title", "AI Settings")}</h4>
                  <button
                    className="ai-panel-btn"
                    id="ai-settings-close"
                    type="button"
                    title={t("common_close", "Close")}
                    onClick={() => setAiSettingsOpen(false)}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="18" y1="6" x2="6" y2="18" />
                      <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                  </button>
                </div>
                <div className="ai-settings-body">
                  <div className="ai-settings-group">
                    <label className="ai-settings-label">{t("builder_ai_settings_provider", "Provider")}</label>
                    <select
                      id="ai-provider"
                      className="ai-settings-input"
                      value={aiSettings.provider}
                      onChange={(event) => {
                        const provider = event.target.value;
                        const baseUrl = AI_PROVIDER_DEFAULT_BASE_URL[provider] ?? "";
                        setAiSettings((previous) => ({
                          ...previous,
                          provider,
                          baseUrl: baseUrl || previous.baseUrl,
                        }));
                      }}
                    >
                      <option value="openai">{t("builder_ai_provider_openai", "OpenAI")}</option>
                      <option value="azure">{t("builder_ai_provider_azure", "Azure OpenAI")}</option>
                      <option value="anthropic">{t("builder_ai_provider_anthropic", "Anthropic")}</option>
                      <option value="custom">{t("builder_ai_provider_custom", "Custom API")}</option>
                    </select>
                  </div>
                  <div className="ai-settings-group">
                    <label className="ai-settings-label">{t("builder_ai_settings_base_url", "API Base URL")}</label>
                    <input
                      type="text"
                      id="ai-base-url"
                      className="ai-settings-input"
                      value={aiSettings.baseUrl}
                      placeholder="https://api.openai.com/v1"
                      onChange={(event) => setAiSettings((previous) => ({ ...previous, baseUrl: event.target.value }))}
                    />
                  </div>
                  <div className="ai-settings-group">
                    <label className="ai-settings-label">{t("builder_ai_settings_api_key", "API Key")}</label>
                    <div className="ai-settings-input-wrapper">
                      <input
                        type={aiApiKeyVisible ? "text" : "password"}
                        id="ai-api-key"
                        className="ai-settings-input"
                        value={aiSettings.apiKey}
                        placeholder="sk-..."
                        onChange={(event) => setAiSettings((previous) => ({ ...previous, apiKey: event.target.value }))}
                      />
                      <button
                        className="ai-toggle-visibility"
                        id="ai-toggle-key"
                        type="button"
                        title={t("builder_ai_toggle_visibility", "Toggle visibility")}
                        onClick={() => setAiApiKeyVisible((previous) => !previous)}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                          <circle cx="12" cy="12" r="3" />
                        </svg>
                      </button>
                    </div>
                  </div>
                  <div className="ai-settings-group">
                    <label className="ai-settings-label">{t("builder_ai_settings_model", "Model")}</label>
                    <input
                      type="text"
                      id="ai-model"
                      className="ai-settings-input"
                      value={aiSettings.model}
                      placeholder="gpt-5-mini"
                      onChange={(event) => setAiSettings((previous) => ({ ...previous, model: event.target.value }))}
                    />
                  </div>
                  <div className="ai-settings-actions">
                    <button
                      className="btn btn-sm btn-dark"
                      type="button"
                      id="ai-save-settings"
                      disabled={aiBusy || aiTestingConnection}
                      onClick={() => void saveAndTestAiSettings()}
                    >
                      {t("builder_ai_settings_save_test", "Save & Test")}
                    </button>
                  </div>
                  <div
                    className={`ai-settings-status ${aiSettingsStatusKind}`.trim()}
                    id="ai-settings-status"
                    style={aiSettingsStatus ? { display: "block" } : undefined}
                  >
                    {aiSettingsStatus}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

      </main>
      <div
        ref={promptContextMenuRef}
        className={`prompt-context-menu ${promptContextMenu.open ? "" : "d-none"}`.trim()}
        style={
          promptContextMenu.open
            ? {
                left: `${promptContextMenu.x}px`,
                top: `${promptContextMenu.y}px`,
              }
            : undefined
        }
        onClick={(event) => event.stopPropagation()}
      >
        <button
          type="button"
          className="prompt-context-item"
          onClick={() => openPromptRenameDialog(promptContextMenu.path)}
          disabled={!promptContextMenu.path}
        >
          {t("common_rename", "Rename")}
        </button>
        <button
          type="button"
          className="prompt-context-item text-danger"
          onClick={() => openPromptDeleteDialog(promptContextMenu.path)}
          disabled={!promptContextMenu.path}
        >
          {t("common_delete", "Delete")}
        </button>
      </div>
      {nodePickerOpen ? (
        <div
          className="node-picker-modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-labelledby="node-picker-title"
          onClick={() => {
            setNodePickerOpen(false);
            setNodePickerError("");
            setNodePickerTarget(null);
          }}
        >
          <div className="node-picker-modal" onClick={(event) => event.stopPropagation()}>
            <div className="node-picker-header">
              <h5 id="node-picker-title">
                {t("builder_node_add_title", "Add Node")}
              </h5>
              <button
                type="button"
                className="node-picker-close"
                aria-label={t("common_close", "Close")}
                onClick={() => {
                  setNodePickerOpen(false);
                  setNodePickerError("");
                  setNodePickerTarget(null);
                }}
              >
                ×
              </button>
            </div>
            <div className="node-picker-body">
              <div className="node-picker-tabs" role="tablist" aria-label={t("builder_node_add_title", "Add Node")}>
                {(
                  [
                    ["tool", t("builder_node_tab_tool", "Tool")],
                    ["branch", t("builder_node_tab_branch", "Branch")],
                    ["loop", t("builder_node_tab_loop", "Loop")],
                    ["custom", t("builder_node_tab_custom", "Custom")],
                  ] as Array<[NodePickerMode, string]>
                ).map(([mode, label]) => (
                  <button
                    key={mode}
                    type="button"
                    className={`node-picker-tab ${nodePickerMode === mode ? "active" : ""}`}
                    onClick={() => {
                      setNodePickerMode(mode);
                      setNodePickerError("");
                    }}
                  >
                    {label}
                  </button>
                ))}
              </div>
              {nodePickerMode === "tool" ? (
                <div className="node-picker-panel">
                  <label className="node-picker-label">{t("builder_node_label_server", "Server")}</label>
                  <select
                    className="node-picker-input"
                    value={nodePickerServer}
                    onChange={(event) => {
                      setNodePickerServer(event.target.value);
                      setNodePickerError("");
                    }}
                    disabled={nodePickerLoadingTools || !toolCatalog.order.length}
                  >
                    {toolCatalog.order.length ? (
                      toolCatalog.order.map((server) => (
                        <option key={server} value={server}>
                          {server}
                        </option>
                      ))
                    ) : (
                      <option value="">
                        {nodePickerLoadingTools
                          ? t("common_loading", "Loading...")
                          : t("builder_node_no_servers", "No Servers")}
                      </option>
                    )}
                  </select>
                  <label className="node-picker-label">{t("builder_node_label_function", "Function")}</label>
                  <select
                    className="node-picker-input"
                    value={nodePickerTool}
                    onChange={(event) => {
                      setNodePickerTool(event.target.value);
                      setNodePickerError("");
                    }}
                    disabled={nodePickerLoadingTools || !nodePickerAvailableTools.length}
                  >
                    {nodePickerAvailableTools.length ? (
                      nodePickerAvailableTools.map((item) => (
                        <option key={`${item.server}.${item.tool}`} value={item.tool}>
                          {item.tool}
                        </option>
                      ))
                    ) : (
                      <option value="">
                        {nodePickerServer
                          ? t("builder_node_no_tools", "No tools")
                          : t("builder_node_select_server", "Select Server")}
                      </option>
                    )}
                  </select>
                </div>
              ) : null}
              {nodePickerMode === "branch" ? (
                <div className="node-picker-panel">
                  <label className="node-picker-label">{t("builder_node_label_cases", "Cases")}</label>
                  <input
                    className="node-picker-input"
                    type="text"
                    value={nodePickerBranchCases}
                    onChange={(event) => {
                      setNodePickerBranchCases(event.target.value);
                      setNodePickerError("");
                    }}
                  />
                </div>
              ) : null}
              {nodePickerMode === "loop" ? (
                <div className="node-picker-panel">
                  <label className="node-picker-label">{t("builder_node_label_iterations", "Iterations")}</label>
                  <input
                    className="node-picker-input"
                    type="number"
                    min={1}
                    value={nodePickerLoopTimes}
                    onChange={(event) => {
                      setNodePickerLoopTimes(event.target.value);
                      setNodePickerError("");
                    }}
                  />
                </div>
              ) : null}
              {nodePickerMode === "custom" ? (
                <div className="node-picker-panel">
                  <label className="node-picker-label">{t("builder_node_label_json_config", "JSON Config")}</label>
                  <textarea
                    className="node-picker-input node-picker-textarea"
                    rows={4}
                    value={nodePickerCustom}
                    onChange={(event) => {
                      setNodePickerCustom(event.target.value);
                      setNodePickerError("");
                    }}
                  />
                </div>
              ) : null}
              {nodePickerError ? (
                <div className="node-picker-error">{nodePickerError}</div>
              ) : null}
            </div>
            <div className="node-picker-footer">
              <button
                type="button"
                className="node-picker-confirm"
                onClick={handleNodePickerConfirm}
                disabled={nodePickerLoadingTools}
              >
                {t("builder_node_confirm_add", "Confirm Add")}
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {promptRenameDialogOpen ? (
        <div
          className="builder-inline-modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-labelledby="builder-rename-prompt-title"
          onClick={() => {
            if (saving) return;
            setPromptRenameDialogOpen(false);
            setPromptRenameSourcePath("");
          }}
        >
          <div className="builder-inline-modal" onClick={(event) => event.stopPropagation()}>
            <h4 id="builder-rename-prompt-title">
              {t("builder_prompt_rename_title", "Rename Prompt")}
            </h4>
            <p>{t("builder_prompt_rename_prompt", "Enter the new file name:")}</p>
            <input
              className="builder-inline-modal-input"
              type="text"
              value={promptRenameDraft}
              onChange={(event) => setPromptRenameDraft(event.target.value)}
              placeholder={t("builder_prompt_rename_placeholder", "e.g. my_prompt.jinja")}
              autoFocus
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  if (!saving) void handleRenamePrompt();
                } else if (event.key === "Escape") {
                  event.preventDefault();
                  if (!saving) {
                    setPromptRenameDialogOpen(false);
                    setPromptRenameSourcePath("");
                  }
                }
              }}
            />
            <div className="builder-inline-modal-actions">
              <button
                type="button"
                className="btn btn-sm btn-light border"
                disabled={saving}
                onClick={() => {
                  setPromptRenameDialogOpen(false);
                  setPromptRenameSourcePath("");
                }}
              >
                {t("common_cancel", "Cancel")}
              </button>
              <button
                type="button"
                className="btn btn-sm btn-dark"
                disabled={saving}
                onClick={() => void handleRenamePrompt()}
              >
                {t("common_confirm", "Confirm")}
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {promptDeleteDialogOpen ? (
        <div
          className="builder-inline-modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-labelledby="builder-delete-prompt-title"
          onClick={() => {
            if (saving) return;
            setPromptDeleteDialogOpen(false);
            setPromptDeleteTargetPath("");
          }}
        >
          <div className="builder-inline-modal" onClick={(event) => event.stopPropagation()}>
            <h4 id="builder-delete-prompt-title">
              {t("builder_prompt_delete_title", "Delete Prompt")}
            </h4>
            <p>
              {formatTemplate(t("builder_prompt_delete_confirm", "Delete \"{name}\"?"), {
                name: promptDeleteTargetPath,
              })}
            </p>
            <div className="builder-inline-modal-actions">
              <button
                type="button"
                className="btn btn-sm btn-light border"
                disabled={saving}
                onClick={() => {
                  setPromptDeleteDialogOpen(false);
                  setPromptDeleteTargetPath("");
                }}
              >
                {t("common_cancel", "Cancel")}
              </button>
              <button
                type="button"
                className="btn btn-sm btn-danger"
                disabled={saving}
                onClick={() => void handleDeletePrompt(promptDeleteTargetPath)}
              >
                {t("common_delete", "Delete")}
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {promptCreateDialogOpen ? (
        <div
          className="builder-inline-modal-overlay prompt-create-modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-labelledby="builder-create-prompt-title"
          onClick={() => {
            if (saving) return;
            setPromptCreateDialogOpen(false);
          }}
        >
          <div className="builder-inline-modal prompt-create-modal" onClick={(event) => event.stopPropagation()}>
            <div className="prompt-create-modal-icon" aria-hidden="true">
              <svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 20h9" />
                <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" />
              </svg>
            </div>
            <h4 id="builder-create-prompt-title">
              {t("builder_new_prompt", "New Prompt")}
            </h4>
            <p>
              {t("builder_prompt_new_prompt", "Please enter prompt filename:")}
            </p>
            <input
              className="builder-inline-modal-input prompt-create-input"
              type="text"
              value={newPromptPathDraft}
              onChange={(event) => setNewPromptPathDraft(event.target.value)}
              placeholder={t("builder_prompt_new_placeholder", "e.g. my_prompt.jinja")}
              autoFocus
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  if (!saving) void handleCreatePrompt();
                } else if (event.key === "Escape") {
                  event.preventDefault();
                  if (!saving) setPromptCreateDialogOpen(false);
                }
              }}
            />
            <div className="builder-inline-modal-actions">
              <button
                type="button"
                className="btn btn-sm btn-light border"
                disabled={saving}
                onClick={() => setPromptCreateDialogOpen(false)}
              >
                {t("common_cancel", "Cancel")}
              </button>
              <button
                type="button"
                className="btn btn-sm btn-dark"
                disabled={saving}
                onClick={() => void handleCreatePrompt()}
              >
                {t("common_confirm", "Confirm")}
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {createPipelineDialogOpen ? (
        <div
          className="builder-inline-modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-labelledby="builder-create-pipeline-title"
          onClick={() => {
            if (saving) return;
            setCreatePipelineDialogOpen(false);
          }}
        >
          <div className="builder-inline-modal" onClick={(event) => event.stopPropagation()}>
            <h4 id="builder-create-pipeline-title">
              {t("builder_new_pipeline", "New Pipeline")}
            </h4>
            <p>
              {t("builder_new_pipeline_prompt", "Please enter a new pipeline name:")}
            </p>
            <input
              className="builder-inline-modal-input"
              type="text"
              value={newPipelineNameDraft}
              onChange={(event) => setNewPipelineNameDraft(event.target.value)}
              autoFocus
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  if (!saving) void handleCreatePipeline(newPipelineNameDraft);
                } else if (event.key === "Escape") {
                  event.preventDefault();
                  if (!saving) setCreatePipelineDialogOpen(false);
                }
              }}
            />
            <div className="builder-inline-modal-actions">
              <button
                type="button"
                className="btn btn-sm btn-light border"
                disabled={saving}
                onClick={() => setCreatePipelineDialogOpen(false)}
              >
                {t("common_cancel", "Cancel")}
              </button>
              <button
                type="button"
                className="btn btn-sm btn-dark"
                disabled={saving}
                onClick={() => void handleCreatePipeline(newPipelineNameDraft)}
              >
                {t("common_create", "Create")}
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {editNodeDialogOpen ? (
        <div
          className="builder-inline-modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-labelledby="builder-edit-node-title"
          onClick={() => {
            if (saving) return;
            setEditNodeDialogOpen(false);
          }}
        >
          <div className="builder-inline-modal" onClick={(event) => event.stopPropagation()}>
            <h4 id="builder-edit-node-title">{t("common_edit", "Edit")}</h4>
            <p>{t("builder_node_label", "Node content")}</p>
            <input
              className="builder-inline-modal-input"
              type="text"
              value={editingNodeLabelDraft}
              onChange={(event) => setEditingNodeLabelDraft(event.target.value)}
              autoFocus
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  handleConfirmNodeEdit();
                } else if (event.key === "Escape") {
                  event.preventDefault();
                  setEditNodeDialogOpen(false);
                }
              }}
            />
            <div className="builder-inline-modal-actions">
              <button
                type="button"
                className="btn btn-sm btn-light border"
                onClick={() => setEditNodeDialogOpen(false)}
              >
                {t("common_cancel", "Cancel")}
              </button>
              <button
                type="button"
                className="btn btn-sm btn-dark"
                onClick={handleConfirmNodeEdit}
              >
                {t("common_save", "Save")}
              </button>
            </div>
          </div>
        </div>
      ) : null}
      {deletePipelineDialogOpen && selectedPipeline ? (
        <div
          className="builder-inline-modal-overlay"
          role="dialog"
          aria-modal="true"
          aria-labelledby="builder-delete-pipeline-title"
          onClick={() => {
            if (saving) return;
            setDeletePipelineDialogOpen(false);
          }}
        >
          <div className="builder-inline-modal" onClick={(event) => event.stopPropagation()}>
            <h4 id="builder-delete-pipeline-title">
              {t("builder_delete_pipeline_title", "Delete Pipeline")}
            </h4>
            <p>
              {formatTemplate(
                t("builder_delete_pipeline_confirm", `Delete pipeline "{name}"?`),
                { name: selectedPipeline },
              )}
            </p>
            <div className="builder-inline-modal-actions">
              <button
                type="button"
                className="btn btn-sm btn-light border"
                disabled={saving}
                onClick={() => setDeletePipelineDialogOpen(false)}
              >
                {t("common_cancel", "Cancel")}
              </button>
              <button
                type="button"
                className="btn btn-sm btn-danger"
                disabled={saving}
                onClick={() => void handleDeletePipeline()}
              >
                {t("common_delete", "Delete")}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
