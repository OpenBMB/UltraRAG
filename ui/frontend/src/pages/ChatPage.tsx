import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent, MouseEvent } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import {
  HardDrive,
  LogOut,
  Settings,
  Upload,
} from "lucide-react";
import { useAuthMe } from "@/features/auth/hooks/useAuthMe";
import { useAuthMutations } from "@/features/auth/hooks/useAuthMutations";
import { usePipelines } from "@/features/pipeline/hooks/usePipelines";
import { CHAT_SESSIONS_QUERY_KEY, useChatSessions } from "@/features/chat/hooks/useChatSessions";
import {
  clearChatSessions,
  deleteChatSession,
  exportChatDocx,
  fetchChatSession,
  renameChatSession,
  upsertChatSession,
} from "@/shared/api/chat";
import {
  clearPipelineChatHistory,
  startPipelineDemoSession,
  startPipelineBackgroundChat,
  stopPipelineChatGeneration,
} from "@/shared/api/pipelines";
import {
  clearKbStaging,
  clearMemoryVectors,
  deleteKbFile,
  fetchKbConfig,
  fetchKbFiles,
  fetchKbTaskStatus,
  fetchKbVisibility,
  inspectKbFolder,
  listKbVisibilityUsers,
  runKbTask,
  saveKbConfig,
  saveKbVisibility,
  syncMemoryToKb,
  uploadKbFiles,
  type KbVisibilityPayload,
  type KbFileEntry,
  type KbFilesResponse,
} from "@/shared/api/kb";
import { fetchMemory, saveMemory } from "@/shared/api/memory";
import {
  clearCompletedBackgroundTasks,
  fetchBackgroundTask,
  deleteBackgroundTask,
  fetchBackgroundTasks,
  type BackgroundTask,
} from "@/shared/api/backgroundTasks";
import { streamPipelineChat } from "@/shared/api/streaming";
import type { ChatMessage, ChatSession, PipelineItem, SourceDoc } from "@/shared/api/types";
import { normalizeSourceDocs, renderChatMarkdown } from "@/shared/lib/chatMarkdown";
import { createClientId } from "@/shared/lib/id";
import { useThemeStyles } from "@/shared/lib/useThemeStyles";
import { AuthDialog } from "@/features/auth/components/AuthDialog";
import { AccountSettingsDialog } from "@/features/auth/components/AccountSettingsDialog";
import { useI18n } from "@/shared/i18n/provider";
import "@/pages/chat-page.css";
import "@/pages/chat-mobile.css";

function createSessionId(): string {
  return createClientId();
}

function readStoredEngineMap(): Record<string, string> {
  try {
    const raw = localStorage.getItem("ultrarag_react_active_engines") ?? "{}";
    const parsed = JSON.parse(raw) as Record<string, string>;
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

type ChunkConfigState = {
  chunk_backend: string;
  tokenizer_or_token_counter: string;
  chunk_size: number;
  use_title: boolean;
};

type IndexConfigState = {
  api_key: string;
  base_url: string;
  model_name: string;
};

type ThinkingStepEvent = {
  type: "step_start" | "step_end";
  name: string;
  tokens?: string;
  output?: string;
  timestamp?: number;
};

type ThinkingStepView = {
  name: string;
  tokens: string;
  output: string;
  completed: boolean;
};

type CitationSource = SourceDoc & {
  displayId?: number;
};

type CitationPayload = {
  text: string;
  sources: CitationSource[];
  idMap: Record<number, number>;
};

const CHUNK_CONFIG_STORAGE_KEY = "ultrarag_chunk_config";
const INDEX_CONFIG_STORAGE_KEY = "ultrarag_index_config";
const MOBILE_VIEWPORT_QUERY = "(max-width: 992px)";

function matchesMobileViewport(): boolean {
  if (typeof window === "undefined") return false;
  return window.matchMedia(MOBILE_VIEWPORT_QUERY).matches;
}

const DEFAULT_CHUNK_CONFIG: ChunkConfigState = {
  chunk_backend: "sentence",
  tokenizer_or_token_counter: "character",
  chunk_size: 512,
  use_title: true,
};

const DEFAULT_INDEX_CONFIG: IndexConfigState = {
  api_key: "",
  base_url: "https://api.openai.com/v1",
  model_name: "text-embedding-3-small",
};

function readStoredJson<T>(storageKey: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(storageKey);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as T;
    if (!parsed || typeof parsed !== "object") return fallback;
    return { ...fallback, ...parsed };
  } catch {
    return fallback;
  }
}

function saveStoredJson(storageKey: string, payload: unknown): void {
  try {
    localStorage.setItem(storageKey, JSON.stringify(payload));
  } catch {
    // Ignore quota/storage errors and keep app behavior.
  }
}

function normalizeKbName(value: unknown): string {
  if (value === null || value === undefined) return "";
  let text = String(value).trim();
  try {
    text = text.normalize("NFKC");
  } catch {
    // Fallback for environments without normalize.
  }
  return text.toLowerCase();
}

function getCollectionDisplayName(collection?: Pick<KbFileEntry, "name" | "display_name"> | null): string {
  if (!collection) return "";
  return collection.display_name || collection.name || "";
}

function findMatchingCollection(collections: KbFileEntry[], inputName: string): KbFileEntry | null {
  const normalizedInput = normalizeKbName(inputName);
  if (!normalizedInput) return null;
  return (
    collections.find((collection) => {
      const displayName = normalizeKbName(collection.display_name || collection.name);
      const rawName = normalizeKbName(collection.name);
      return normalizedInput === displayName || normalizedInput === rawName;
    }) ?? null
  );
}

function getKbEntryPath(entry: KbFileEntry): string {
  return String(entry.path || `${entry.category}/${entry.name}`);
}

function collectKbPaths(payload: KbFilesResponse | null | undefined): Set<string> {
  const paths = new Set<string>();
  if (!payload) return paths;
  for (const item of [...payload.raw, ...payload.corpus, ...payload.chunks]) {
    paths.add(getKbEntryPath(item));
  }
  return paths;
}

function deriveSessionTitle(messages: ChatMessage[]): string {
  const firstUserText = messages.find((message) => message.role === "user")?.text?.trim() ?? "";
  if (!firstUserText) return "New Chat";
  return firstUserText.length > 20 ? `${firstUserText.slice(0, 20)}...` : firstUserText;
}

function hasMessageContent(messages: ChatMessage[]): boolean {
  return messages.some((message) => {
    if (typeof message.text === "string" && message.text.trim() !== "") return true;
    if (!message.meta || typeof message.meta !== "object") return false;
    return Object.keys(message.meta as Record<string, unknown>).length > 0;
  });
}

function hasSessionContent(session: ChatSession | undefined): boolean {
  if (!session) return false;
  return hasMessageContent(normalizeSessionMessages(session));
}

function normalizeSessionMessages(session: ChatSession | undefined): ChatMessage[] {
  if (!session?.messages || !Array.isArray(session.messages)) return [];
  return session.messages
    .filter((message): message is ChatMessage => message.role === "user" || message.role === "assistant")
    .map((message) => ({
      role: message.role,
      text: String(message.text ?? ""),
      meta: message.meta ?? {},
      timestamp: message.timestamp,
    }));
}

function formatClockTime(epochSeconds?: number): string {
  if (!epochSeconds) return "";
  const milliseconds = epochSeconds < 10_000_000_000 ? epochSeconds * 1000 : epochSeconds;
  return new Date(milliseconds).toLocaleTimeString();
}

function kbLabel(item: KbFileEntry): string {
  return item.display_name || item.name;
}

function formatTemplate(template: string, params: Record<string, string | number>): string {
  let result = template;
  Object.entries(params).forEach(([key, value]) => {
    result = result.replaceAll(`{${key}}`, String(value));
  });
  return result;
}

function normalizeMemoryUserId(rawUserId: string): string {
  return rawUserId.replace(/[^A-Za-z0-9_-]/g, "_") || "default";
}

function getCurrentUserMemoryCollectionNameSet(rawUserId: string): Set<string> {
  const normalized = normalizeMemoryUserId(rawUserId);
  return new Set([`user_${normalized}`.toLowerCase(), `user_${normalized}_memory`.toLowerCase()]);
}

function filterCurrentUserMemoryCollections(
  collections: KbFileEntry[],
  rawUserId: string,
): KbFileEntry[] {
  const candidates = getCurrentUserMemoryCollectionNameSet(rawUserId);
  return collections.filter((collection) =>
    candidates.has(String(collection.name || "").toLowerCase()),
  );
}

function isInternalMemoryCollection(name: string): boolean {
  const normalized = String(name || "").trim().toLowerCase();
  if (!normalized) return false;
  return /^user_[a-z0-9_-]+(?:_memory)?$/.test(normalized);
}

function filterVisibleKbCollections(collections: KbFileEntry[]): KbFileEntry[] {
  return collections.filter((collection) => !isInternalMemoryCollection(collection.name));
}

function getKbInitial(name = ""): string {
  const initial = name.trim().charAt(0);
  return initial ? initial.toUpperCase() : "?";
}

function pickKbColors(seed: string): { bg: string; text: string } {
  const palette = [
    { bg: "#dbeafe", text: "#1e40af" },
    { bg: "#dcfce7", text: "#166534" },
    { bg: "#fef3c7", text: "#92400e" },
    { bg: "#fce7f3", text: "#9d174d" },
    { bg: "#e0e7ff", text: "#3730a3" },
  ];
  const hash = Array.from(seed).reduce((acc, char) => acc + char.charCodeAt(0), 0);
  return palette[hash % palette.length];
}

function appendSources(current: SourceDoc[], incoming: unknown): SourceDoc[] {
  const merged = normalizeSourceDocs([...current, ...normalizeSourceDocs(incoming)]);
  return merged;
}

function readMessageSources(message: ChatMessage): SourceDoc[] {
  if (!message.meta || typeof message.meta !== "object") return [];
  return normalizeSourceDocs((message.meta as Record<string, unknown>).sources);
}

function normalizeStepOutput(raw: unknown): string {
  if (raw === null || raw === undefined) return "";
  if (typeof raw === "string") return raw;
  try {
    return JSON.stringify(raw);
  } catch {
    return String(raw);
  }
}

function normalizeThinkingSteps(raw: unknown): ThinkingStepEvent[] {
  if (!Array.isArray(raw)) return [];
  const normalized: ThinkingStepEvent[] = [];
  raw.forEach((item) => {
    if (!item || typeof item !== "object") return;
    const candidate = item as Record<string, unknown>;
    const type = candidate.type;
    if (type !== "step_start" && type !== "step_end") return;
    const name = String(candidate.name ?? "").trim();
    if (!name) return;
    const tokens = candidate.tokens === undefined ? "" : String(candidate.tokens ?? "");
    const output = normalizeStepOutput(candidate.output);
    const timestamp =
      typeof candidate.timestamp === "number" && Number.isFinite(candidate.timestamp)
        ? candidate.timestamp
        : undefined;
    normalized.push({
      type,
      name,
      tokens: tokens || undefined,
      output: output || undefined,
      timestamp,
    });
  });
  return normalized;
}

function readMessageThinkingSteps(message: ChatMessage): ThinkingStepEvent[] {
  if (!message.meta || typeof message.meta !== "object") return [];
  return normalizeThinkingSteps((message.meta as Record<string, unknown>).steps);
}

function mergeThinkingSteps(steps: ThinkingStepEvent[]): ThinkingStepView[] {
  if (!steps.length) return [];
  const merged: ThinkingStepView[] = [];
  const openStepIndexesByName = new Map<string, number[]>();
  steps.forEach((step) => {
    if (step.type === "step_start") {
      merged.push({
        name: step.name,
        tokens: step.tokens || "",
        output: "",
        completed: false,
      });
      const indices = openStepIndexesByName.get(step.name) ?? [];
      indices.push(merged.length - 1);
      openStepIndexesByName.set(step.name, indices);
      return;
    }

    const indices = openStepIndexesByName.get(step.name);
    const targetIndex = indices && indices.length ? indices.pop() : undefined;
    if (indices && indices.length === 0) {
      openStepIndexesByName.delete(step.name);
    }

    if (typeof targetIndex === "number") {
      const existing = merged[targetIndex];
      existing.completed = true;
      if (step.output) {
        existing.output = step.output;
      }
      return;
    }

    merged.push({
      name: step.name,
      tokens: "",
      output: step.output || "",
      completed: true,
    });
  });
  return merged;
}

function renderUserTextAsHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;")
    .replaceAll("\n", "<br>");
}

const SOURCE_DETAIL_EMPTY_HTML =
  '<div class="text-muted small text-center mt-5">Select a reference to view details</div>';
const CHAT_SCROLL_FOLLOW_THRESHOLD = 48;

function cleanPdfText(text: string): string {
  if (!text) return "";
  let cleaned = text.replace(/\r\n/g, "\n");
  cleaned = cleaned.replace(/([a-zA-Z])-\n([a-zA-Z])/g, "$1$2");
  cleaned = cleaned.replace(/\n\s*\n/g, "___PARAGRAPH_BREAK___");
  cleaned = cleaned.replace(/\n/g, " ");
  cleaned = cleaned.replace(/  +/g, " ");
  return cleaned.replace(/___PARAGRAPH_BREAK___/g, "\n\n").trim();
}

function escapeDetailHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderSourceDetailHtml(content: string): string {
  const rawText = content || "No content available.";
  let cleanedText = cleanPdfText(rawText);

  const bibkeyMatch = cleanedText.match(/^bibkey:\s*\S+\s+([\s\S]*)/i);
  if (bibkeyMatch) {
    cleanedText = bibkeyMatch[1].trim();
  }

  const titleMatch = cleanedText.match(/^Title:\s*(.+?)(?:\n|Content:)/i);
  const contentMatch = cleanedText.match(/Content:\s*([\s\S]*)/i);
  if (titleMatch && contentMatch) {
    const docTitle = titleMatch[1].trim();
    const docContent = contentMatch[1].trim();
    return `<div class="source-doc-title">${escapeDetailHtml(docTitle)}</div>${renderChatMarkdown(
      docContent || "No content available.",
    )}`;
  }

  return renderChatMarkdown(cleanedText || "No content available.");
}

function truncateRefPreview(text: string, maxLength = 110): string {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) return "";
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, maxLength)}...`;
}

function getReferencePreviewText(source: CitationSource): string {
  let content = String(source.content ?? "");
  const bibkeyMatch = content.match(/^bibkey:\s*\S+\s+([\s\S]*)/i);
  if (bibkeyMatch) {
    content = bibkeyMatch[1].trim();
  }

  const titleMatch = content.match(/^Title:\s*(.+?)(?:\n|Content:)/i);
  const contentMatch = content.match(/Content:\s*([\s\S]*)/i);

  let rawPreview = "";
  if (titleMatch && contentMatch) {
    const docTitle = titleMatch[1].trim();
    const firstLine = String(contentMatch[1] ?? "").trim().split("\n")[0]?.trim() ?? "";
    rawPreview = firstLine ? `${docTitle}: ${firstLine}` : docTitle;
  } else if (contentMatch) {
    rawPreview = String(contentMatch[1] ?? "").trim().split("\n")[0]?.trim() ?? "";
  } else {
    rawPreview = content.split("\n")[0]?.trim() ?? "";
  }

  const fallback = String(source.title ?? "").trim();
  return truncateRefPreview(rawPreview || fallback || `Source ${getSourceDisplayId(source)}`);
}

function extractCitationIds(text: string): Set<number> {
  const ids = new Set<number>();
  const regex = /\[(\d+)\]/g;
  let match: RegExpExecArray | null = regex.exec(text);
  while (match) {
    const parsed = Number(match[1]);
    if (Number.isFinite(parsed)) {
      ids.add(parsed);
    }
    match = regex.exec(text);
  }
  return ids;
}

function getSourceDisplayId(source: Pick<CitationSource, "id" | "displayId">): number {
  const candidate = Number(source.displayId);
  if (Number.isFinite(candidate)) return candidate;
  return Number(source.id);
}

function renumberCitations(text: string, sources: SourceDoc[]): CitationPayload {
  const normalizedText = String(text || "");
  const sourceList: CitationSource[] = sources.map((source) => ({ ...source }));
  if (!normalizedText) return { text: normalizedText, sources: sourceList, idMap: {} };

  const regex = /\[(\d+)\]/g;
  const idMap: Record<number, number> = {};
  let counter = 1;
  let match: RegExpExecArray | null = regex.exec(normalizedText);

  // Build original id -> new sequential id by first appearance.
  while (match) {
    const originalId = Number.parseInt(match[1], 10);
    if (Number.isFinite(originalId) && idMap[originalId] === undefined) {
      idMap[originalId] = counter;
      counter += 1;
    }
    match = regex.exec(normalizedText);
  }

  // No citations in answer; keep text/sources untouched.
  if (counter === 1) {
    return { text: normalizedText, sources: sourceList, idMap };
  }

  const renumberedText = normalizedText.replace(/\[(\d+)\]/g, (raw, idText: string) => {
    const originalId = Number.parseInt(idText, 10);
    if (!Number.isFinite(originalId)) return raw;
    const mapped = idMap[originalId];
    return mapped ? `[${mapped}]` : raw;
  });

  const renumberedSources = sourceList.map((source) => {
    const originalId = getSourceDisplayId(source);
    const mapped = idMap[originalId];
    if (mapped !== undefined) {
      return { ...source, displayId: mapped };
    }
    const fallbackId = counter;
    counter += 1;
    idMap[originalId] = fallbackId;
    return { ...source, displayId: fallbackId };
  });

  return {
    text: renumberedText,
    sources: renumberedSources,
    idMap,
  };
}

function buildCitationPayload(message: ChatMessage): CitationPayload {
  return renumberCitations(message.text, readMessageSources(message));
}

function normalizeExportTitle(questionText: string): string {
  const normalized = questionText.replace(/\s+/g, " ").trim().replace(/^#+\s*/, "");
  return normalized || "Chat Export";
}

function getDownloadTimestamp(): string {
  const now = new Date();
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(
    now.getHours(),
  )}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
}

function buildDownloadFileName(questionText: string, extension = "md"): string {
  const baseTitle = normalizeExportTitle(questionText);
  const safeTitle = baseTitle
    .replace(/[\\/:*?"<>|]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^\.+|\.+$/g, "")
    .slice(0, 80);
  const finalTitle = safeTitle || "chat-export";
  const ext = String(extension || "md").replace(/^\.+/, "") || "md";
  return `${finalTitle}-${getDownloadTimestamp()}.${ext}`;
}

function buildDownloadMarkdown(text: string, sources: CitationSource[], questionText: string): string {
  const answerText = String(text || "");
  const exportTitle = normalizeExportTitle(questionText);
  const sourceMap = new Map<number, CitationSource>();
  sources.forEach((source) => {
    const sourceId = getSourceDisplayId(source);
    if (!Number.isFinite(sourceId) || sourceMap.has(sourceId)) return;
    sourceMap.set(sourceId, source);
  });
  const usedIds = Array.from(extractCitationIds(answerText));
  const usedSources = usedIds
    .map((id) => sourceMap.get(id))
    .filter((item): item is CitationSource => Boolean(item));
  const extraSources = Array.from(sourceMap.values()).filter(
    (source) => !usedIds.includes(getSourceDisplayId(source)),
  );
  const allSources = [...usedSources, ...extraSources];

  const lines: string[] = [`# ${exportTitle}`, "", "## Answer", "", answerText.trim() || "_No content._"];
  if (allSources.length) {
    lines.push("", "## References", "");
    allSources.forEach((source) => {
      const displayId = getSourceDisplayId(source);
      lines.push(`### [${displayId}] ${source.title || `Source ${displayId}`}`);
      lines.push("");
      lines.push(source.content || "_Reference content unavailable._");
      lines.push("");
    });
  }
  return `${lines.join("\n").trimEnd()}\n`;
}

function triggerDownloadBlob(blob: Blob, filename: string): void {
  const downloadUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = downloadUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 0);
}

function decodeCopyPayload(payload: string): string {
  try {
    return decodeURIComponent(payload);
  } catch {
    return payload;
  }
}

function taskResultText(task: BackgroundTask | undefined): string {
  if (!task) return "";
  if (typeof task.result === "string") return task.result;
  if (typeof task.result_preview === "string") return task.result_preview;
  return "";
}

export function ChatPage() {
  useThemeStyles(true);

  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { locale, setLocale, t } = useI18n();
  const { data: authInfo } = useAuthMe();
  const loggedIn = Boolean(authInfo?.logged_in);
  const isAdmin = Boolean(authInfo?.is_admin);
  const userId = authInfo?.user_id ?? "default";
  const displayUserName = authInfo?.nickname || authInfo?.username || authInfo?.user_id || "default";
  const {
    loginMutation,
    registerMutation,
    logoutMutation,
    nicknameMutation,
    changePasswordMutation,
    modelSettingsMutation,
  } = useAuthMutations();
  const { data: pipelines = [], isFetched: pipelinesFetched } = usePipelines();
  const { data: persistedSessions = [] } = useChatSessions(loggedIn);

  const [authOpen, setAuthOpen] = useState(false);
  const [authError, setAuthError] = useState<string>("");
  const [accountOpen, setAccountOpen] = useState(false);
  const [accountDialogMode, setAccountDialogMode] = useState<"account" | "model">("account");
  const [accountStatus, setAccountStatus] = useState("");

  const [selectedPipeline, setSelectedPipeline] = useState<string>(() => {
    try {
      const urlPipeline = (new URLSearchParams(window.location.search).get("pipeline") ?? "").trim();
      if (urlPipeline) return urlPipeline;
      return (localStorage.getItem("ultrarag_react_selected_pipeline") ?? "").trim();
    } catch {
      return "";
    }
  });
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [activeView, setActiveView] = useState<"chat" | "kb" | "explore" | "memory">("chat");
  const [selectedCollectionName, setSelectedCollectionName] = useState("");
  const [backgroundMode, setBackgroundMode] = useState(false);
  const [exportingMessageKey, setExportingMessageKey] = useState<string>("");
  const [copiedMessageKey, setCopiedMessageKey] = useState<string>("");
  const [downloadedMessageKey, setDownloadedMessageKey] = useState<string>("");
  const [inputComposing, setInputComposing] = useState(false);
  const [sending, setSending] = useState(false);
  const [chatStatus, setChatStatus] = useState<"ready" | "warn" | "error" | "offline">("offline");
  const [, setStatusText] = useState("Offline");
  const [isMobileViewport, setIsMobileViewport] = useState<boolean>(() => matchesMobileViewport());
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [settingsMenuOpen, setSettingsMenuOpen] = useState(false);
  const [languageSubmenuOpen, setLanguageSubmenuOpen] = useState(false);
  const [kbDropdownOpen, setKbDropdownOpen] = useState(false);
  const [pipelineDropdownOpen, setPipelineDropdownOpen] = useState(false);
  const [expandedUnusedRefs, setExpandedUnusedRefs] = useState<Record<number, boolean>>({});
  const [expandedThinkingByMessage, setExpandedThinkingByMessage] = useState<Record<string, boolean>>({});
  const [activeReferenceKey, setActiveReferenceKey] = useState("");
  const [detailPanelOpen, setDetailPanelOpen] = useState(false);
  const [detailSource, setDetailSource] = useState<SourceDoc | null>(null);
  const [detailSourceDisplayId, setDetailSourceDisplayId] = useState<number | null>(null);
  const [backgroundPanelOpen, setBackgroundPanelOpen] = useState(false);
  const [backgroundDetailTaskId, setBackgroundDetailTaskId] = useState("");
  const [kbData, setKbData] = useState<KbFilesResponse>({
    raw: [],
    corpus: [],
    chunks: [],
    index: [],
    db_status: "unknown",
  });
  const [kbLoading, setKbLoading] = useState(false);
  const [kbMessage, setKbMessage] = useState("");
  const [kbConfigBusy, setKbConfigBusy] = useState(false);
  const [kbConfigUri, setKbConfigUri] = useState("");
  const [kbConfigToken, setKbConfigToken] = useState("");
  const [kbChunkConfig, setKbChunkConfig] = useState<ChunkConfigState>(() =>
    readStoredJson(CHUNK_CONFIG_STORAGE_KEY, DEFAULT_CHUNK_CONFIG),
  );
  const [kbChunkConfigDraft, setKbChunkConfigDraft] = useState<ChunkConfigState>(() =>
    readStoredJson(CHUNK_CONFIG_STORAGE_KEY, DEFAULT_CHUNK_CONFIG),
  );
  const [kbIndexConfig, setKbIndexConfig] = useState<IndexConfigState>(() =>
    readStoredJson(INDEX_CONFIG_STORAGE_KEY, DEFAULT_INDEX_CONFIG),
  );
  const [kbIndexConfigDraft, setKbIndexConfigDraft] = useState<IndexConfigState>(() =>
    readStoredJson(INDEX_CONFIG_STORAGE_KEY, DEFAULT_INDEX_CONFIG),
  );
  const [kbRunTargetFile, setKbRunTargetFile] = useState("");
  const [kbRunCollectionName, setKbRunCollectionName] = useState("");
  const [kbRunIndexMode, setKbRunIndexMode] = useState<"new" | "append" | "overwrite">("new");
  const [kbTaskStatusVisible, setKbTaskStatusVisible] = useState(false);
  const [kbTaskStatusMessage, setKbTaskStatusMessage] = useState("Processing...");
  const [kbTaskProgressVisible, setKbTaskProgressVisible] = useState(false);
  const [kbTaskProgress, setKbTaskProgress] = useState(0);
  const [kbVisibilityCollection, setKbVisibilityCollection] = useState<KbFileEntry | null>(null);
  const [kbVisibilityData, setKbVisibilityData] = useState<KbVisibilityPayload | null>(null);
  const [kbVisibilityUsers, setKbVisibilityUsers] = useState<string[]>([]);
  const [kbImportModalOpen, setKbImportModalOpen] = useState(false);
  const [kbMilvusModalOpen, setKbMilvusModalOpen] = useState(false);
  const [kbIndexChoiceModalOpen, setKbIndexChoiceModalOpen] = useState(false);
  const [exportFormatModalOpen, setExportFormatModalOpen] = useState(false);
  const [kbIndexChoiceMessage, setKbIndexChoiceMessage] = useState("");
  const [confirmModalOpen, setConfirmModalOpen] = useState(false);
  const [confirmModalTitle, setConfirmModalTitle] = useState("");
  const [confirmModalMessage, setConfirmModalMessage] = useState("");
  const [confirmModalType, setConfirmModalType] = useState<
    "info" | "warning" | "confirm" | "success" | "error"
  >("confirm");
  const [confirmModalDanger, setConfirmModalDanger] = useState(false);
  const [confirmModalHideCancel, setConfirmModalHideCancel] = useState(false);
  const [confirmModalConfirmText, setConfirmModalConfirmText] = useState("");
  const [confirmModalCancelText, setConfirmModalCancelText] = useState("");
  const [chatSessionContextMenu, setChatSessionContextMenu] = useState<{
    open: boolean;
    x: number;
    y: number;
    sessionId: string;
  }>({
    open: false,
    x: 0,
    y: 0,
    sessionId: "",
  });
  const [chatRenameModalOpen, setChatRenameModalOpen] = useState(false);
  const [chatRenameTargetSessionId, setChatRenameTargetSessionId] = useState("");
  const [chatRenameDraft, setChatRenameDraft] = useState("");
  const [kbChunkConfigModalOpen, setKbChunkConfigModalOpen] = useState(false);
  const [kbIndexConfigModalOpen, setKbIndexConfigModalOpen] = useState(false);
  const [kbConfigModalOpen, setKbConfigModalOpen] = useState(false);
  const [kbVisibilityModalOpen, setKbVisibilityModalOpen] = useState(false);
  const [kbFolderDetailModalOpen, setKbFolderDetailModalOpen] = useState(false);
  const [kbInspectFiles, setKbInspectFiles] = useState<Array<{ name: string; size: number }>>([]);
  const [kbInspectTitle, setKbInspectTitle] = useState("");
  const [kbInspectLoading, setKbInspectLoading] = useState(false);
  const [kbInspectHasHiddenOnly, setKbInspectHasHiddenOnly] = useState(false);
  const [memoryContent, setMemoryContent] = useState("");
  const [memoryMessage, setMemoryMessage] = useState("");
  const [memoryBusy, setMemoryBusy] = useState(false);
  const [memoryStatusText, setMemoryStatusText] = useState("Ready");
  const [memoryStatusState, setMemoryStatusState] = useState<
    "ready" | "loading" | "saving" | "success" | "error"
  >("ready");
  const [memoryKbCollections, setMemoryKbCollections] = useState<KbFileEntry[]>([]);
  const [memoryKbError, setMemoryKbError] = useState("");
  const [backgroundTasks, setBackgroundTasks] = useState<BackgroundTask[]>([]);
  const [taskDetailMap, setTaskDetailMap] = useState<Record<string, BackgroundTask>>({});

  const [guestSessions, setGuestSessions] = useState<ChatSession[]>([]);

  const activeEngineRef = useRef<Record<string, string>>(readStoredEngineMap());
  const abortControllerRef = useRef<AbortController | null>(null);
  const chatHistoryRef = useRef<HTMLDivElement | null>(null);
  const shouldAutoScrollRef = useRef(true);
  const chatScrollTopRef = useRef(0);
  const thinkingBodyRef = useRef<HTMLDivElement | null>(null);
  const thinkingShouldAutoScrollRef = useRef(true);
  const thinkingScrollTopRef = useRef(0);
  const chatSidebarRef = useRef<HTMLElement | null>(null);
  const settingsMenuRef = useRef<HTMLDivElement | null>(null);
  const kbDropdownRef = useRef<HTMLDivElement | null>(null);
  const pipelineDropdownRef = useRef<HTMLDivElement | null>(null);
  const detailPanelRef = useRef<HTMLElement | null>(null);
  const detailContentRef = useRef<HTMLDivElement | null>(null);
  const kbUploadInputRef = useRef<HTMLInputElement | null>(null);
  const kbFileSnapshotRef = useRef<Set<string>>(new Set());
  const kbTaskTimerRef = useRef<number | null>(null);
  const kbIndexChoiceResolveRef = useRef<((value: "append" | "overwrite" | null) => void) | null>(null);
  const exportFormatResolveRef = useRef<((value: "markdown" | "docx" | null) => void) | null>(null);
  const confirmModalResolveRef = useRef<((value: boolean) => void) | null>(null);
  const kbImportDialogRef = useRef<HTMLDialogElement | null>(null);
  const kbMilvusDialogRef = useRef<HTMLDialogElement | null>(null);
  const kbIndexChoiceDialogRef = useRef<HTMLDialogElement | null>(null);
  const exportFormatDialogRef = useRef<HTMLDialogElement | null>(null);
  const confirmDialogRef = useRef<HTMLDialogElement | null>(null);
  const chatRenameDialogRef = useRef<HTMLDialogElement | null>(null);
  const chatRenameInputRef = useRef<HTMLInputElement | null>(null);
  const chatSessionContextMenuRef = useRef<HTMLDivElement | null>(null);
  const kbChunkConfigDialogRef = useRef<HTMLDialogElement | null>(null);
  const kbIndexConfigDialogRef = useRef<HTMLDialogElement | null>(null);
  const kbConfigDialogRef = useRef<HTMLDialogElement | null>(null);
  const kbVisibilityDialogRef = useRef<HTMLDialogElement | null>(null);
  const kbFolderDetailDialogRef = useRef<HTMLDialogElement | null>(null);
  const backgroundDetailDialogRef = useRef<HTMLDialogElement | null>(null);

  const sessions = loggedIn ? persistedSessions : guestSessions;
  const visibleSessions = useMemo(
    () => sessions.filter((session) => hasSessionContent(session)),
    [sessions],
  );

  const readyPipelines = useMemo(
    () =>
      pipelines
        .filter((pipeline) => pipeline.is_ready)
        .slice()
        .sort((left, right) => left.name.localeCompare(right.name, "en", { sensitivity: "base" })),
    [pipelines],
  );
  const selectedPipelineInfo = useMemo<PipelineItem | undefined>(
    () => readyPipelines.find((pipeline) => pipeline.name === selectedPipeline),
    [readyPipelines, selectedPipeline],
  );
  const userAvatarText = useMemo(
    () => (displayUserName.trim().charAt(0) || "D").toUpperCase(),
    [displayUserName],
  );
  const visibleKbCollections = useMemo(
    () => filterVisibleKbCollections(kbData.index ?? []),
    [kbData.index],
  );
  const selectedCollection = useMemo(
    () => visibleKbCollections.find((collection) => collection.name === selectedCollectionName),
    [selectedCollectionName, visibleKbCollections],
  );
  const kbConnectionClass = useMemo(() => {
    const status = String(kbData.db_status ?? "disconnected").toLowerCase();
    if (status === "connected") return "connected";
    if (status === "connecting") return "connecting";
    return "disconnected";
  }, [kbData.db_status]);
  const kbConnectionText = useMemo(() => {
    const status = String(kbData.db_status ?? "disconnected").toLowerCase();
    if (status === "connected") return t("kb_connected", "Connected");
    if (status === "connecting") return t("kb_connecting", "Connecting");
    return t("kb_disconnected", "Disconnected");
  }, [kbData.db_status, t]);
  const kbUriDisplay = useMemo(() => {
    const config =
      kbData.db_config && typeof kbData.db_config === "object"
        ? (kbData.db_config as Record<string, unknown>)
        : {};
    const endpoint =
      (config.uri as string | undefined) ??
      (config.endpoint as string | undefined) ??
      ((config.milvus as Record<string, unknown> | undefined)?.uri as string | undefined);
    return endpoint || "—";
  }, [kbData.db_config]);
  const kbVisibilityCandidateUsers = useMemo(() => {
    const merged = new Set<string>([
      ...kbVisibilityUsers,
      ...(kbVisibilityData?.visible_users ?? []),
    ]);
    return [...merged].sort((left, right) => left.localeCompare(right, "en", { sensitivity: "base" }));
  }, [kbVisibilityData?.visible_users, kbVisibilityUsers]);
  const kbVisibilityOwnerHint = useMemo(() => {
    const owner = String(kbVisibilityData?.owner_user_id ?? "").trim();
    if (!owner) return "";
    return formatTemplate(t("kb_visibility_owner_hint", "Owner: {owner}"), { owner });
  }, [kbVisibilityData?.owner_user_id, t]);
  const kbVisibilityReadonlyHint = useMemo(() => {
    if (kbVisibilityData?.can_manage) return "";
    const owner = String(kbVisibilityData?.owner_user_id ?? "").trim();
    if (owner && !loggedIn) {
      return formatTemplate(
        t("kb_visibility_readonly_login_hint", "You are not logged in. Please sign in as owner ({owner}) before editing."),
        { owner },
      );
    }
    if (owner) {
      return formatTemplate(
        t("kb_visibility_readonly_owner_hint", "You do not have permission to modify this collection. Current owner: {owner}."),
        { owner },
      );
    }
    return t("kb_visibility_readonly_hint", "You do not have permission to modify this collection.");
  }, [kbVisibilityData?.can_manage, kbVisibilityData?.owner_user_id, loggedIn, t]);
  const chatStatusBadgeClass = useMemo(() => {
    if (chatStatus === "ready") return "bg-success text-white";
    if (chatStatus === "error") return "bg-danger text-white";
    return "bg-warning text-dark";
  }, [chatStatus]);
  const demoLoading = useMemo(
    () => chatStatus !== "ready" && chatStatus !== "error",
    [chatStatus],
  );
  const displayStatusText = useMemo(() => {
    if (chatStatus === "ready") {
      return t("status_ready", locale === "zh" ? "就绪" : "Ready");
    }
    if (chatStatus === "error") {
      return locale === "zh" ? "失败" : "Failed";
    }
    return locale === "zh" ? "加载中" : "Loading";
  }, [chatStatus, locale, t]);
  const greetingHeadline = useMemo(() => {
    const nickname = (authInfo?.nickname ?? "").trim();
    if (!nickname) {
      return t("greeting_full", "Hi. What shall we explore today?");
    }
    return t("greeting_full_with_nickname", "Hi, {nickname}. What shall we explore today?").replace(
      "{nickname}",
      nickname,
    );
  }, [authInfo?.nickname, t]);
  const sourceDetailTitle = useMemo(
    () => (detailSource ? `Reference [${detailSourceDisplayId ?? detailSource.id}]` : "Reference Detail"),
    [detailSource, detailSourceDisplayId],
  );
  const sourceDetailHtml = useMemo(
    () => (detailSource ? renderSourceDetailHtml(detailSource.content) : SOURCE_DETAIL_EMPTY_HTML),
    [detailSource],
  );
  const runningBackgroundCount = useMemo(
    () => backgroundTasks.filter((task) => task.status === "running").length,
    [backgroundTasks],
  );
  const showBackgroundFab = activeView === "chat" && backgroundTasks.length > 0;
  const backgroundDetailTask = useMemo(() => {
    if (!backgroundDetailTaskId) return null;
    return taskDetailMap[backgroundDetailTaskId] ?? backgroundTasks.find((task) => task.task_id === backgroundDetailTaskId) ?? null;
  }, [backgroundDetailTaskId, taskDetailMap, backgroundTasks]);

  useEffect(() => {
    const storedSidebarCollapsed = localStorage.getItem("ultrarag_sidebar_collapsed");
    if (storedSidebarCollapsed === "true") {
      setSidebarCollapsed(true);
    }
  }, []);

  useEffect(() => {
    if (!settingsMenuOpen) {
      setLanguageSubmenuOpen(false);
    }
  }, [settingsMenuOpen]);

  useEffect(() => {
    if (!pipelinesFetched) return;
    if (!readyPipelines.length) {
      return;
    }
    if (!selectedPipeline) {
      setSelectedPipeline(readyPipelines[0].name);
      return;
    }
    const exists = readyPipelines.some((pipeline) => pipeline.name === selectedPipeline);
    if (exists) return;
    setSelectedPipeline(readyPipelines[0].name);
  }, [pipelinesFetched, readyPipelines, selectedPipeline]);

  useEffect(() => {
    if (!selectedPipeline) {
      localStorage.removeItem("ultrarag_react_selected_pipeline");
      return;
    }
    localStorage.setItem("ultrarag_react_selected_pipeline", selectedPipeline);
  }, [selectedPipeline]);

  useEffect(() => {
    localStorage.setItem("ultrarag_sidebar_collapsed", String(sidebarCollapsed));
  }, [sidebarCollapsed]);

  useEffect(() => {
    if (!visibleSessions.length) {
      if (!activeSessionId) {
        setMessages([]);
      }
      return;
    }
    if (!activeSessionId) {
      setMessages([]);
      return;
    }

    const targetSession = visibleSessions.find((session) => session.id === activeSessionId);
    if (targetSession) {
      setMessages(normalizeSessionMessages(targetSession));
      return;
    }

    const existsInRawSessions = sessions.some((session) => session.id === activeSessionId);
    if (existsInRawSessions) {
      setActiveSessionId("");
      setMessages([]);
    }
  }, [activeSessionId, sessions, visibleSessions]);

  useEffect(() => {
    shouldAutoScrollRef.current = true;
    chatScrollTopRef.current = 0;
    thinkingBodyRef.current = null;
    thinkingShouldAutoScrollRef.current = true;
    thinkingScrollTopRef.current = 0;
    setExpandedUnusedRefs({});
    setExpandedThinkingByMessage({});
    setActiveReferenceKey("");
    setDetailSource(null);
    setDetailSourceDisplayId(null);
    setDetailPanelOpen(false);
  }, [activeSessionId]);

  useEffect(() => {
    const panel = chatHistoryRef.current;
    if (!panel) return;
    if (!shouldAutoScrollRef.current) return;
    panel.scrollTop = panel.scrollHeight;
    chatScrollTopRef.current = panel.scrollTop;
  }, [messages]);

  useEffect(() => {
    const body = thinkingBodyRef.current;
    if (!body) return;
    if (!sending) return;
    if (!thinkingShouldAutoScrollRef.current) return;
    body.scrollTop = body.scrollHeight;
    thinkingScrollTopRef.current = body.scrollTop;
  }, [messages, sending]);

  useEffect(() => {
    if (!detailPanelOpen) return;
    if (detailContentRef.current) {
      detailContentRef.current.scrollTop = 0;
    }
  }, [detailPanelOpen, sourceDetailHtml]);

  useEffect(() => {
    const dialog = kbImportDialogRef.current;
    if (!dialog) return;
    if (kbImportModalOpen) {
      if (!dialog.open) {
        dialog.showModal();
      }
      return;
    }
    if (dialog.open) {
      dialog.close();
    }
  }, [kbImportModalOpen]);

  useEffect(() => {
    const dialog = kbMilvusDialogRef.current;
    if (!dialog) return;
    if (kbMilvusModalOpen) {
      if (!dialog.open) {
        dialog.showModal();
      }
      return;
    }
    if (dialog.open) {
      dialog.close();
    }
  }, [kbMilvusModalOpen]);

  useEffect(() => {
    const dialog = kbIndexChoiceDialogRef.current;
    if (!dialog) return;
    if (kbIndexChoiceModalOpen) {
      if (!dialog.open) {
        dialog.showModal();
      }
      return;
    }
    if (dialog.open) {
      dialog.close();
    }
  }, [kbIndexChoiceModalOpen]);

  useEffect(() => {
    const dialog = exportFormatDialogRef.current;
    if (!dialog) return;
    if (exportFormatModalOpen) {
      if (!dialog.open) {
        dialog.showModal();
      }
      return;
    }
    if (dialog.open) {
      dialog.close();
    }
  }, [exportFormatModalOpen]);

  useEffect(() => {
    const dialog = confirmDialogRef.current;
    if (!dialog) return;
    if (confirmModalOpen) {
      if (!dialog.open) {
        dialog.showModal();
      }
      return;
    }
    if (dialog.open) {
      dialog.close();
    }
  }, [confirmModalOpen]);

  useEffect(() => {
    const dialog = chatRenameDialogRef.current;
    if (!dialog) return;
    if (chatRenameModalOpen) {
      if (!dialog.open) {
        dialog.showModal();
      }
      return;
    }
    if (dialog.open) {
      dialog.close();
    }
  }, [chatRenameModalOpen]);

  useEffect(() => {
    if (!chatRenameModalOpen) return;
    const timer = window.setTimeout(() => {
      chatRenameInputRef.current?.focus();
      chatRenameInputRef.current?.select();
    }, 0);
    return () => {
      window.clearTimeout(timer);
    };
  }, [chatRenameModalOpen]);

  useEffect(() => {
    const dialog = kbChunkConfigDialogRef.current;
    if (!dialog) return;
    if (kbChunkConfigModalOpen) {
      if (!dialog.open) {
        dialog.showModal();
      }
      return;
    }
    if (dialog.open) {
      dialog.close();
    }
  }, [kbChunkConfigModalOpen]);

  useEffect(() => {
    const dialog = kbIndexConfigDialogRef.current;
    if (!dialog) return;
    if (kbIndexConfigModalOpen) {
      if (!dialog.open) {
        dialog.showModal();
      }
      return;
    }
    if (dialog.open) {
      dialog.close();
    }
  }, [kbIndexConfigModalOpen]);

  useEffect(() => {
    const dialog = kbConfigDialogRef.current;
    if (!dialog) return;
    if (kbConfigModalOpen) {
      if (!dialog.open) {
        dialog.showModal();
      }
      return;
    }
    if (dialog.open) {
      dialog.close();
    }
  }, [kbConfigModalOpen]);

  useEffect(() => {
    const dialog = kbVisibilityDialogRef.current;
    if (!dialog) return;
    if (kbVisibilityModalOpen) {
      if (!dialog.open) {
        dialog.showModal();
      }
      return;
    }
    if (dialog.open) {
      dialog.close();
    }
  }, [kbVisibilityModalOpen]);

  useEffect(() => {
    const dialog = kbFolderDetailDialogRef.current;
    if (!dialog) return;
    if (kbFolderDetailModalOpen) {
      if (!dialog.open) {
        dialog.showModal();
      }
      return;
    }
    if (dialog.open) {
      dialog.close();
    }
  }, [kbFolderDetailModalOpen]);

  useEffect(() => {
    const dialog = backgroundDetailDialogRef.current;
    if (!dialog) return;
    if (backgroundDetailTaskId) {
      if (!dialog.open) {
        dialog.showModal();
      }
      return;
    }
    if (dialog.open) {
      dialog.close();
    }
  }, [backgroundDetailTaskId]);

  useEffect(() => {
    const dialog = backgroundDetailDialogRef.current;
    if (!dialog) return;
    const handleClose = () => {
      setBackgroundDetailTaskId("");
    };
    dialog.addEventListener("close", handleClose);
    return () => {
      dialog.removeEventListener("close", handleClose);
    };
  }, []);

  useEffect(
    () => () => {
      if (kbTaskTimerRef.current !== null) {
        window.clearInterval(kbTaskTimerRef.current);
      }
      if (kbIndexChoiceResolveRef.current) {
        kbIndexChoiceResolveRef.current(null);
        kbIndexChoiceResolveRef.current = null;
      }
      if (confirmModalResolveRef.current) {
        confirmModalResolveRef.current(false);
        confirmModalResolveRef.current = null;
      }
    },
    [],
  );

  useEffect(() => {
    const media = window.matchMedia(MOBILE_VIEWPORT_QUERY);
    const syncViewport = () => {
      setIsMobileViewport(media.matches);
    };
    syncViewport();
    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", syncViewport);
      return () => {
        media.removeEventListener("change", syncViewport);
      };
    }
    media.addListener(syncViewport);
    return () => {
      media.removeListener(syncViewport);
    };
  }, []);

  useEffect(() => {
    if (isMobileViewport) return;
    setMobileSidebarOpen(false);
  }, [isMobileViewport]);

  useEffect(() => {
    const handleDocumentPointerDown = (event: Event) => {
      const target = event.target as Node | null;
      if (!target) return;
      const targetElement = target instanceof Element ? target : null;
      if (
        chatSessionContextMenu.open &&
        chatSessionContextMenuRef.current &&
        !chatSessionContextMenuRef.current.contains(target)
      ) {
        setChatSessionContextMenu((previous) => (previous.open ? { ...previous, open: false } : previous));
      }
      if (settingsMenuRef.current && !settingsMenuRef.current.contains(target)) {
        setSettingsMenuOpen(false);
        setLanguageSubmenuOpen(false);
      }
      if (kbDropdownRef.current && !kbDropdownRef.current.contains(target)) {
        setKbDropdownOpen(false);
      }
      if (pipelineDropdownRef.current && !pipelineDropdownRef.current.contains(target)) {
        setPipelineDropdownOpen(false);
      }
      if (
        isMobileViewport &&
        mobileSidebarOpen &&
        chatSidebarRef.current &&
        !chatSidebarRef.current.contains(target) &&
        !targetElement?.closest("[data-mobile-sidebar-toggle='true']")
      ) {
        setMobileSidebarOpen(false);
      }
      const keepDetailOpen =
        !!targetElement?.closest(".citation-link") ||
        !!targetElement?.closest(".ref-item") ||
        !!targetElement?.closest(".reference-container");
      if (
        detailPanelOpen &&
        detailPanelRef.current &&
        !detailPanelRef.current.contains(target) &&
        !keepDetailOpen
      ) {
        setDetailPanelOpen(false);
        setDetailSourceDisplayId(null);
      }
    };
    document.addEventListener("mousedown", handleDocumentPointerDown);
    return () => {
      document.removeEventListener("mousedown", handleDocumentPointerDown);
    };
  }, [chatSessionContextMenu.open, detailPanelOpen, isMobileViewport, mobileSidebarOpen]);

  const loadKbData = useCallback(async (): Promise<KbFilesResponse | null> => {
    setKbLoading(true);
    setKbMessage("");
    try {
      const payload = await fetchKbFiles();
      setKbData(payload);
      const visibleCollections = filterVisibleKbCollections(payload.index ?? []);
      if (!kbRunCollectionName && visibleCollections.length) {
        setKbRunCollectionName(visibleCollections[0].name);
      }
      return payload;
    } catch (error) {
      setKbMessage(error instanceof Error ? error.message : "Failed to load KB files");
      return null;
    } finally {
      setKbLoading(false);
    }
  }, [kbRunCollectionName]);

  const loadKbConfigData = useCallback(async () => {
    setKbConfigBusy(true);
    try {
      const config = await fetchKbConfig();
      const milvus =
        config.milvus && typeof config.milvus === "object"
          ? (config.milvus as Record<string, unknown>)
          : {};
      setKbConfigUri(String(milvus.uri ?? ""));
      setKbConfigToken(String(milvus.token ?? ""));
    } catch (error) {
      setKbMessage(error instanceof Error ? error.message : "Failed to load KB config");
    } finally {
      setKbConfigBusy(false);
    }
  }, []);

  const setMemoryStatus = useCallback(
    (text: string, state: "ready" | "loading" | "saving" | "success" | "error" = "ready") => {
      setMemoryStatusText(text);
      setMemoryStatusState(state);
    },
    [],
  );

  const refreshMemoryKbCards = useCallback(async () => {
    try {
      const payload = await fetchKbFiles();
      setMemoryKbError("");
      setMemoryKbCollections(filterCurrentUserMemoryCollections(payload.index ?? [], userId));
    } catch (error) {
      const message = error instanceof Error ? error.message : t("common_unknown_error", "Unknown error");
      setMemoryKbCollections([]);
      setMemoryKbError(message);
    }
  }, [t, userId]);

  const loadMemoryData = useCallback(async () => {
    setMemoryBusy(true);
    setMemoryMessage("");
    setMemoryStatus(t("memory_loading", "Loading memory..."), "loading");
    try {
      const payload = await fetchMemory(userId);
      setMemoryContent(payload.content ?? "");
      setMemoryStatus(t("memory_loaded", "Memory loaded"), "ready");
    } catch (error) {
      setMemoryMessage(error instanceof Error ? error.message : "Failed to load memory");
      setMemoryStatus(t("memory_load_failed", "Load failed"), "error");
    } finally {
      setMemoryBusy(false);
    }
  }, [setMemoryStatus, t, userId]);

  const loadBackgroundTaskData = useCallback(async () => {
    const tasks = await fetchBackgroundTasks(50);
    setBackgroundTasks(tasks);
  }, []);

  useEffect(() => {
    if (!selectedCollectionName) return;
    if (visibleKbCollections.some((collection) => collection.name === selectedCollectionName)) return;
    setSelectedCollectionName("");
  }, [selectedCollectionName, visibleKbCollections]);

  useEffect(() => {
    // Keep KB backend status warm even when user stays in Chat view.
    void loadKbData();
    void loadKbConfigData();
  }, [loadKbConfigData, loadKbData, loggedIn]);

  useEffect(() => {
    if (activeView === "kb") {
      void loadKbData();
      void loadKbConfigData();
    } else if (activeView === "explore") {
      setMemoryMessage("");
    } else if (activeView === "memory") {
      void refreshMemoryKbCards();
      void loadMemoryData();
    }
  }, [activeView, loadKbConfigData, loadKbData, loadMemoryData, refreshMemoryKbCards]);

  useEffect(() => {
    void loadBackgroundTaskData();
  }, [loadBackgroundTaskData]);

  useEffect(() => {
    if (!backgroundTasks.some((task) => task.status === "running")) return;
    const timer = window.setInterval(() => {
      void loadBackgroundTaskData();
    }, 3000);
    return () => {
      window.clearInterval(timer);
    };
  }, [backgroundTasks, loadBackgroundTaskData]);

  const saveEngines = useCallback(() => {
    localStorage.setItem("ultrarag_react_active_engines", JSON.stringify(activeEngineRef.current));
  }, []);

  const syncGuestSessionMessages = (sessionId: string, nextMessages: ChatMessage[]) => {
    setGuestSessions((previous) => {
      const existing = previous.find((session) => session.id === sessionId);
      const remains = previous.filter((session) => session.id !== sessionId);
      if (!hasMessageContent(nextMessages)) {
        return remains;
      }
      return [
        {
          ...(existing ?? {}),
          id: sessionId,
          title: deriveSessionTitle(nextMessages),
          pipeline: selectedPipeline || existing?.pipeline || null,
          timestamp: Date.now(),
          messages: nextMessages,
        },
        ...remains,
      ];
    });
  };

  const persistSession = async (sessionId: string, nextMessages: ChatMessage[]) => {
    if (!hasMessageContent(nextMessages)) {
      if (!loggedIn) {
        syncGuestSessionMessages(sessionId, nextMessages);
      }
      return;
    }
    if (!loggedIn) {
      syncGuestSessionMessages(sessionId, nextMessages);
      return;
    }
    await upsertChatSession({
      id: sessionId,
      title: deriveSessionTitle(nextMessages),
      pipeline: selectedPipeline || null,
      messages: nextMessages,
      timestamp: Date.now(),
    });
    await queryClient.invalidateQueries({ queryKey: CHAT_SESSIONS_QUERY_KEY });
  };

  const ensureEngineSession = useCallback(async (): Promise<string> => {
    if (!selectedPipeline) {
      throw new Error("Please select a pipeline before chatting.");
    }
    setChatStatus("warn");
    setStatusText("Initializing...");
    const cachedSessionId = activeEngineRef.current[selectedPipeline];
    if (cachedSessionId) {
      try {
        // Reuse existing engine session if possible; backend get_or_create is idempotent.
        await startPipelineDemoSession(selectedPipeline, cachedSessionId);
        setChatStatus("ready");
        setStatusText("Ready");
        return cachedSessionId;
      } catch {
        delete activeEngineRef.current[selectedPipeline];
        saveEngines();
      }
    }

    const nextSessionId = createSessionId();
    await startPipelineDemoSession(selectedPipeline, nextSessionId);
    activeEngineRef.current[selectedPipeline] = nextSessionId;
    saveEngines();
    setChatStatus("ready");
    setStatusText("Ready");
    return nextSessionId;
  }, [saveEngines, selectedPipeline]);

  useEffect(() => {
    if (!selectedPipeline) {
      setChatStatus("offline");
      setStatusText("Engine Offline");
      return;
    }
    let cancelled = false;
    const warmupEngine = async () => {
      try {
        await ensureEngineSession();
        if (cancelled) return;
        setChatStatus("ready");
        setStatusText("Engine Ready");
      } catch {
        if (cancelled) return;
        setChatStatus("error");
        setStatusText("Engine Error");
      }
    };
    void warmupEngine();
    return () => {
      cancelled = true;
    };
  }, [ensureEngineSession, selectedPipeline]);

  const handleNewChat = async () => {
    setChatSessionContextMenu((previous) => (previous.open ? { ...previous, open: false } : previous));
    const previousSessionId = activeSessionId;
    const previousMessages = messages;
    const newSessionId = createSessionId();
    setActiveSessionId(newSessionId);
    setMessages([]);
    setMobileSidebarOpen(false);
    if (previousSessionId && hasMessageContent(previousMessages)) {
      void persistSession(previousSessionId, previousMessages);
    }
  };

  const handleDeleteSession = async (sessionId: string) => {
    setChatSessionContextMenu((previous) => (previous.open ? { ...previous, open: false } : previous));
    const confirmed = await showConfirmModal({
      title: t("chat_delete_title", "Delete Chat"),
      message: t("chat_delete_confirm", "Delete this chat session?"),
      type: "warning",
      confirmText: t("common_delete", "Delete"),
      danger: true,
    });
    if (!confirmed) return;

    if (loggedIn) {
      await deleteChatSession(sessionId);
      await queryClient.invalidateQueries({ queryKey: CHAT_SESSIONS_QUERY_KEY });
      if (sessionId === activeSessionId) {
        setActiveSessionId("");
        setMessages([]);
      }
      return;
    }
    setGuestSessions((previous) => previous.filter((session) => session.id !== sessionId));
    if (sessionId === activeSessionId) {
      const fallback = visibleSessions.find((session) => session.id !== sessionId);
      if (fallback) {
        setActiveSessionId(fallback.id);
        setMessages(normalizeSessionMessages(fallback));
      } else {
        setActiveSessionId(createSessionId());
        setMessages([]);
      }
    }
  };

  const openChatSessionContextMenu = (event: MouseEvent<HTMLElement>, sessionId: string) => {
    event.preventDefault();
    const menuWidth = 160;
    const menuHeight = 96;
    const x = Math.min(event.clientX, window.innerWidth - menuWidth - 12);
    const y = Math.min(event.clientY, window.innerHeight - menuHeight - 12);
    setChatSessionContextMenu({
      open: true,
      x: Math.max(8, x),
      y: Math.max(8, y),
      sessionId,
    });
  };

  const openRenameSessionDialog = (sessionId: string) => {
    setChatSessionContextMenu((previous) => (previous.open ? { ...previous, open: false } : previous));
    const current = sessions.find((item) => item.id === sessionId);
    setChatRenameTargetSessionId(sessionId);
    setChatRenameDraft(current?.title || t("new_chat", "New Chat"));
    setChatRenameModalOpen(true);
  };

  const submitRenameSession = async () => {
    const sessionId = chatRenameTargetSessionId;
    if (!sessionId) return;
    const current = sessions.find((item) => item.id === sessionId);
    const trimmed = chatRenameDraft.trim();
    if (!trimmed) {
      await showAlertModal({
        title: t("chat_rename_title", "Rename Chat"),
        message: t("chat_rename_required", "Name cannot be empty."),
        type: "warning",
      });
      return;
    }
    if (trimmed === current?.title) {
      setChatRenameModalOpen(false);
      setChatRenameTargetSessionId("");
      setChatRenameDraft("");
      return;
    }

    if (loggedIn) {
      await renameChatSession(sessionId, trimmed);
      await queryClient.invalidateQueries({ queryKey: CHAT_SESSIONS_QUERY_KEY });
    } else {
      setGuestSessions((previous) =>
        previous.map((session) => (session.id === sessionId ? { ...session, title: trimmed } : session)),
      );
    }
    setChatRenameModalOpen(false);
    setChatRenameTargetSessionId("");
    setChatRenameDraft("");
  };

  const handleClearSessions = async () => {
    if (!visibleSessions.length && !sessions.length) return;
    const confirmed = await showConfirmModal({
      title: t("chat_delete_all_title", "Clear Chat History"),
      message: t("chat_delete_all_confirm", "Delete all chat sessions?"),
      type: "warning",
      confirmText: t("chat_delete_all_action", "Delete All"),
      danger: true,
    });
    if (!confirmed) return;

    const runningSessionId = selectedPipeline ? activeEngineRef.current[selectedPipeline] : undefined;
    if (runningSessionId) {
      try {
        await clearPipelineChatHistory(runningSessionId);
      } catch {
        // ignore clear-history failures
      }
    }
    if (loggedIn) {
      await clearChatSessions();
      await queryClient.invalidateQueries({ queryKey: CHAT_SESSIONS_QUERY_KEY });
      setMessages([]);
      setActiveSessionId(createSessionId());
      return;
    }
    setGuestSessions([]);
    setActiveSessionId(createSessionId());
    setMessages([]);
  };

  const handleOpenSession = async (sessionId: string) => {
    setChatSessionContextMenu((previous) => (previous.open ? { ...previous, open: false } : previous));
    setActiveView("chat");
    setActiveSessionId(sessionId);
    setMobileSidebarOpen(false);
    if (!loggedIn) {
      const localSession = guestSessions.find((session) => session.id === sessionId);
      setMessages(normalizeSessionMessages(localSession));
      return;
    }
    const session = await fetchChatSession(sessionId);
    setMessages(normalizeSessionMessages(session));
  };

  const stopSending = async () => {
    const sessionId = selectedPipeline ? activeEngineRef.current[selectedPipeline] : undefined;
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (sessionId) {
      try {
        await stopPipelineChatGeneration(sessionId);
      } catch {
        // no-op
      }
    }
    setSending(false);
    setChatStatus("ready");
    setStatusText("Ready");
  };

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter") return;
    if (event.shiftKey) return;
    if (inputComposing || event.nativeEvent.isComposing) return;
    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  };

  const handleSend = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (sending) {
      await stopSending();
      return;
    }
    const question = input.trim();
    if (!question) return;
    if (!selectedPipelineInfo) {
      setChatStatus("error");
      setStatusText("Params Missing");
      return;
    }

    if (backgroundMode) {
      setInput("");
      setSending(true);
      setChatStatus("warn");
      setStatusText("Submitting...");
      try {
        const engineSessionId = await ensureEngineSession();
        await startPipelineBackgroundChat(selectedPipelineInfo.name, {
          question,
          sessionId: engineSessionId,
          userId,
          collectionName: selectedCollectionName || undefined,
        });
        setBackgroundMode(false);
        await loadBackgroundTaskData();
        await showAlertModal({
          title: t("bg_task_submitted_title", locale === "zh" ? "任务已提交" : "Task Submitted"),
          message: t(
            "bg_task_submitted_message",
            locale === "zh"
              ? "问题已提交到后台，处理完成后会通知你。"
              : "Question sent to background, you will be notified when complete.",
          ),
          type: "info",
        });
        setChatStatus("ready");
        setStatusText("Ready");
      } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to send to background";
        await showAlertModal({
          title: t("status_error", "Error"),
          message,
          type: "error",
        });
        setChatStatus("error");
        setStatusText("Error");
      } finally {
        setSending(false);
      }
      return;
    }

    const sessionId = activeSessionId || createSessionId();
    if (!activeSessionId) {
      setActiveSessionId(sessionId);
    }

    // New streaming turn should follow bottom by default unless user scrolls up.
    shouldAutoScrollRef.current = true;
    thinkingShouldAutoScrollRef.current = true;

    const userMessage: ChatMessage = {
      role: "user",
      text: question,
      timestamp: new Date().toISOString(),
    };
    const historyBeforeAssistant = [...messages, userMessage];
    const assistantPlaceholder: ChatMessage = {
      role: "assistant",
      text: "",
      timestamp: new Date().toISOString(),
    };
    const optimisticMessages = [...historyBeforeAssistant, assistantPlaceholder];

    setInput("");
    setMessages(optimisticMessages);
    setSending(true);
    setChatStatus("warn");
    setStatusText("Thinking...");

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const engineSessionId = await ensureEngineSession();

      let assistantText = "";
      let assistantSources: SourceDoc[] = [];
      let assistantSteps: ThinkingStepEvent[] = [];

      const patchAssistantMessage = (updater: (message: ChatMessage) => ChatMessage) => {
        setMessages((previous) => {
          if (!previous.length) return previous;
          const next = [...previous];
          next[next.length - 1] = updater(next[next.length - 1]);
          return next;
        });
      };

      const buildAssistantMeta = (currentMeta: ChatMessage["meta"]) => {
        const nextMeta =
          currentMeta && typeof currentMeta === "object"
            ? { ...(currentMeta as Record<string, unknown>) }
            : {};
        if (assistantSources.length) {
          nextMeta.sources = assistantSources;
        } else {
          delete nextMeta.sources;
        }
        if (assistantSteps.length) {
          nextMeta.steps = assistantSteps;
        } else {
          delete nextMeta.steps;
        }
        return Object.keys(nextMeta).length ? nextMeta : undefined;
      };

      for await (const streamEvent of streamPipelineChat({
        pipelineName: selectedPipelineInfo.name,
        question,
        sessionId: engineSessionId,
        chatSessionId: sessionId,
        history: historyBeforeAssistant.map((message) => ({
          role: message.role,
          text: message.text,
        })),
        userId,
        collectionName: selectedCollectionName || undefined,
        signal: controller.signal,
      })) {
        if (streamEvent.type === "step_start" || streamEvent.type === "step_end") {
          const rawEvent = streamEvent as Record<string, unknown>;
          const stepName = String(rawEvent.name ?? "").trim();
          if (!stepName) continue;
          const stepEntry: ThinkingStepEvent = {
            type: streamEvent.type,
            name: stepName,
            timestamp: Date.now(),
          };
          if (streamEvent.type === "step_end") {
            const output = normalizeStepOutput(rawEvent.output);
            if (output) {
              stepEntry.output = output;
            }
          }
          assistantSteps = [...assistantSteps, stepEntry];
          patchAssistantMessage((current) => ({
            ...current,
            meta: buildAssistantMeta(current.meta),
          }));
        } else if (streamEvent.type === "token" && streamEvent.is_final === false) {
          const tokenText = streamEvent.content ?? "";
          if (tokenText) {
            for (let i = assistantSteps.length - 1; i >= 0; i -= 1) {
              const candidate = assistantSteps[i];
              if (candidate.type !== "step_start") continue;
              const updated = {
                ...candidate,
                tokens: `${candidate.tokens ?? ""}${tokenText}`,
              };
              assistantSteps = [
                ...assistantSteps.slice(0, i),
                updated,
                ...assistantSteps.slice(i + 1),
              ];
              break;
            }
            patchAssistantMessage((current) => ({
              ...current,
              meta: buildAssistantMeta(current.meta),
            }));
          }
        } else if (streamEvent.type === "token" && streamEvent.is_final !== false) {
          assistantText += streamEvent.content ?? "";
          patchAssistantMessage((current) => ({
            ...current,
            text: assistantText,
            meta: buildAssistantMeta(current.meta),
          }));
        } else if (streamEvent.type === "final") {
          assistantText = streamEvent.data?.answer ?? assistantText;
          patchAssistantMessage((current) => ({
            ...current,
            text: assistantText,
            meta: buildAssistantMeta(current.meta),
          }));
        } else if (streamEvent.type === "sources") {
          assistantSources = appendSources(
            assistantSources,
            (streamEvent as { data?: unknown }).data,
          );
          patchAssistantMessage((current) => ({
            ...current,
            meta: buildAssistantMeta(current.meta),
          }));
        } else if (streamEvent.type === "error") {
          throw new Error(streamEvent.message || "Pipeline stream failed");
        }
      }

      const finalMessages = [
        ...historyBeforeAssistant,
        {
          ...assistantPlaceholder,
          text: assistantText || "(empty)",
          meta:
            assistantSources.length || assistantSteps.length
              ? {
                  ...(assistantSources.length ? { sources: assistantSources } : {}),
                  ...(assistantSteps.length ? { steps: assistantSteps } : {}),
                }
              : undefined,
        },
      ];
      setMessages(finalMessages);
      await persistSession(sessionId, finalMessages);
      setChatStatus("ready");
      setStatusText("Ready");
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setMessages((previous) => {
        const next = [...previous];
        if (next.length && next[next.length - 1].role === "assistant" && !next[next.length - 1].text.trim()) {
          next[next.length - 1] = { ...next[next.length - 1], text: `Error: ${message}` };
        }
        return next;
      });
      setChatStatus("error");
      setStatusText("Error");
    } finally {
      abortControllerRef.current = null;
      setSending(false);
    }
  };

  const handleLogin = async (payload: { username: string; password: string }) => {
    setAuthError("");
    try {
      await loginMutation.mutateAsync(payload);
    } catch (error) {
      setAuthError(error instanceof Error ? error.message : "Login failed");
      throw error;
    }
  };

  const handleRegister = async (payload: { username: string; password: string }) => {
    setAuthError("");
    try {
      await registerMutation.mutateAsync(payload);
    } catch (error) {
      setAuthError(error instanceof Error ? error.message : "Register failed");
      throw error;
    }
  };

  const handleLogout = async () => {
    await logoutMutation.mutateAsync();
    setMessages([]);
    setActiveSessionId("");
    setAccountOpen(false);
  };

  const handleSaveNickname = async (nickname: string) => {
    setAccountStatus("");
    try {
      await nicknameMutation.mutateAsync(nickname);
      setAccountStatus("昵称已保存。");
    } catch (error) {
      setAccountStatus(error instanceof Error ? error.message : "保存昵称失败");
      throw error;
    }
  };

  const handleChangePassword = async (payload: { current_password: string; new_password: string }) => {
    setAccountStatus("");
    try {
      await changePasswordMutation.mutateAsync(payload);
      setAccountStatus("密码已更新。");
    } catch (error) {
      setAccountStatus(error instanceof Error ? error.message : "修改密码失败");
      throw error;
    }
  };

  const handleSaveModelSettings = async (payload: {
    retriever?: { api_key?: string; base_url?: string; model_name?: string };
    generation?: { api_key?: string; base_url?: string; model_name?: string };
  }) => {
    setAccountStatus("");
    try {
      await modelSettingsMutation.mutateAsync(payload);
      setAccountStatus("模型设置已保存。");
    } catch (error) {
      setAccountStatus(error instanceof Error ? error.message : "保存模型设置失败");
      throw error;
    }
  };

  const handleClearModelSettings = async () => {
    setAccountStatus("");
    try {
      await modelSettingsMutation.mutateAsync({});
      setAccountStatus("模型设置已清空。");
    } catch (error) {
      setAccountStatus(error instanceof Error ? error.message : "清空模型设置失败");
      throw error;
    }
  };

  const openBuilderView = () => {
    const params = new URLSearchParams(window.location.search);
    if (selectedPipeline) {
      params.set("pipeline", selectedPipeline);
    } else {
      params.delete("pipeline");
    }
    const query = params.toString();
    navigate(query ? `/settings?${query}` : "/settings");
  };

  const toggleKbDropdown = () => {
    setKbDropdownOpen((previous) => !previous);
  };

  const toggleMobileSidebar = () => {
    setMobileSidebarOpen((previous) => !previous);
  };

  const selectKbCollection = (collectionName: string) => {
    setSelectedCollectionName(collectionName);
    setKbDropdownOpen(false);
  };

  const togglePipelineDropdown = () => {
    setPipelineDropdownOpen((previous) => !previous);
  };

  const selectPipelineItem = (pipelineName: string) => {
    const pipelineChanged = pipelineName !== selectedPipeline;
    setSelectedPipeline(pipelineName);
    setPipelineDropdownOpen(false);
    if (pipelineChanged) {
      setActiveView("chat");
      void handleNewChat();
    }
  };

  const handleCopyMessage = async (text: string, messageKey: string) => {
    if (!text.trim()) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageKey(messageKey);
      window.setTimeout(() => {
        setCopiedMessageKey((previous) => (previous === messageKey ? "" : previous));
      }, 2000);
    } catch {
      // Ignore clipboard errors to keep chat flow stable.
    }
  };

  const switchView = (view: "chat" | "kb" | "explore" | "memory") => {
    setActiveView(view);
    setMobileSidebarOpen(false);
    setKbDropdownOpen(false);
    setSettingsMenuOpen(false);
    setPipelineDropdownOpen(false);
    setChatSessionContextMenu((previous) => (previous.open ? { ...previous, open: false } : previous));
    setChatRenameModalOpen(false);
    setExportFormatModalOpen(false);
    if (exportFormatResolveRef.current) {
      exportFormatResolveRef.current(null);
      exportFormatResolveRef.current = null;
    }
    if (view !== "kb") {
      setKbImportModalOpen(false);
      setKbMilvusModalOpen(false);
      closeKbIndexChoiceModal(null);
      closeConfirmModal(false);
      setKbChunkConfigModalOpen(false);
      setKbIndexConfigModalOpen(false);
      setKbConfigModalOpen(false);
      setKbVisibilityModalOpen(false);
      setKbFolderDetailModalOpen(false);
      resetKbTaskStatus();
    }
    if (view !== "chat") {
      setDetailPanelOpen(false);
      setDetailSourceDisplayId(null);
      setBackgroundPanelOpen(false);
      setBackgroundDetailTaskId("");
    }
  };

  const openKbImportModal = () => {
    void (async () => {
      const payload = await loadKbData();
      kbFileSnapshotRef.current = collectKbPaths(payload ?? kbData);
      setKbImportModalOpen(true);
    })();
  };

  const closeKbImportModal = () => {
    setKbImportModalOpen(false);
    setKbMilvusModalOpen(false);
    closeKbIndexChoiceModal(null);
    closeConfirmModal(false);
    setKbChunkConfigModalOpen(false);
    setKbIndexConfigModalOpen(false);
    setKbFolderDetailModalOpen(false);
    resetKbTaskStatus();
    void loadKbData();
  };

  const openKbConfigModal = () => {
    void loadKbConfigData();
    setKbConfigModalOpen(true);
  };

  const closeKbConfigModal = () => {
    setKbConfigModalOpen(false);
  };

  const closeKbVisibilityModal = () => {
    setKbVisibilityModalOpen(false);
    setKbVisibilityCollection(null);
    setKbVisibilityData(null);
  };

  const closeKbMilvusModal = () => {
    setKbMilvusModalOpen(false);
    closeKbIndexChoiceModal(null);
    closeConfirmModal(false);
  };

  const closeKbIndexChoiceModal = (result: "append" | "overwrite" | null = null) => {
    setKbIndexChoiceModalOpen(false);
    const resolver = kbIndexChoiceResolveRef.current;
    kbIndexChoiceResolveRef.current = null;
    if (resolver) {
      resolver(result);
    }
  };

  const closeExportFormatModal = (result: "markdown" | "docx" | null = null) => {
    setExportFormatModalOpen(false);
    const resolver = exportFormatResolveRef.current;
    exportFormatResolveRef.current = null;
    if (resolver) {
      resolver(result);
    }
  };

  const showKbIndexChoiceModal = (message: string): Promise<"append" | "overwrite" | null> =>
    new Promise((resolve) => {
      if (kbIndexChoiceResolveRef.current) {
        kbIndexChoiceResolveRef.current(null);
      }
      kbIndexChoiceResolveRef.current = resolve;
      setKbIndexChoiceMessage(message);
      setKbIndexChoiceModalOpen(true);
    });

  const showExportFormatModal = (): Promise<"markdown" | "docx" | null> =>
    new Promise((resolve) => {
      if (exportFormatResolveRef.current) {
        exportFormatResolveRef.current(null);
      }
      exportFormatResolveRef.current = resolve;
      setExportFormatModalOpen(true);
    });

  const closeConfirmModal = (result: boolean) => {
    setConfirmModalOpen(false);
    const resolver = confirmModalResolveRef.current;
    confirmModalResolveRef.current = null;
    if (resolver) {
      resolver(result);
    }
  };

  const showConfirmModal = (options: {
    title: string;
    message: string;
    type?: "info" | "warning" | "confirm" | "success" | "error";
    confirmText?: string;
    cancelText?: string;
    danger?: boolean;
    hideCancel?: boolean;
  }): Promise<boolean> =>
    new Promise((resolve) => {
      if (confirmModalResolveRef.current) {
        confirmModalResolveRef.current(false);
      }
      confirmModalResolveRef.current = resolve;
      setConfirmModalTitle(options.title);
      setConfirmModalMessage(options.message);
      setConfirmModalType(options.type ?? "confirm");
      setConfirmModalDanger(Boolean(options.danger));
      setConfirmModalHideCancel(Boolean(options.hideCancel));
      setConfirmModalConfirmText(options.confirmText ?? t("common_confirm", "Confirm"));
      setConfirmModalCancelText(options.cancelText ?? t("common_cancel", "Cancel"));
      setConfirmModalOpen(true);
    });

  const showAlertModal = async (options: {
    title: string;
    message: string;
    type?: "info" | "warning" | "confirm" | "success" | "error";
    confirmText?: string;
    danger?: boolean;
  }): Promise<void> => {
    await showConfirmModal({
      ...options,
      hideCancel: true,
      confirmText: options.confirmText ?? t("common_ok", "OK"),
      cancelText: "",
    });
  };

  const closeKbChunkConfigModal = () => {
    setKbChunkConfigModalOpen(false);
  };

  const closeKbIndexConfigModal = () => {
    setKbIndexConfigModalOpen(false);
  };

  const closeKbFolderDetailModal = () => {
    setKbFolderDetailModalOpen(false);
    setKbInspectFiles([]);
    setKbInspectTitle("");
    setKbInspectLoading(false);
    setKbInspectHasHiddenOnly(false);
  };

  const resetKbTaskStatus = () => {
    if (kbTaskTimerRef.current !== null) {
      window.clearInterval(kbTaskTimerRef.current);
      kbTaskTimerRef.current = null;
    }
    setKbTaskStatusVisible(false);
    setKbTaskProgressVisible(false);
    setKbTaskProgress(0);
    setKbTaskStatusMessage("Processing...");
  };

  const showKbTaskStatus = (message: string) => {
    setKbTaskStatusVisible(true);
    setKbTaskStatusMessage(message);
    setKbTaskProgressVisible(false);
    setKbTaskProgress(0);
  };

  const handleKbUpload = async (files: FileList | File[] | null): Promise<boolean> => {
    const fileList = Array.isArray(files) ? files : files ? Array.from(files) : [];
    if (!fileList.length) return false;
    setKbMessage("");
    showKbTaskStatus(t("kb_uploading", "Uploading..."));
    try {
      await uploadKbFiles(fileList);
      await loadKbData();
      resetKbTaskStatus();
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : "Upload failed";
      await showAlertModal({
        title: t("status_error", "Error"),
        message,
        type: "error",
      });
      resetKbTaskStatus();
      return false;
    }
  };

  const handleOpenChunkConfigModal = () => {
    setKbChunkConfigDraft({ ...kbChunkConfig });
    setKbChunkConfigModalOpen(true);
  };

  const handleSaveChunkConfig = async () => {
    if (!Number.isFinite(kbChunkConfigDraft.chunk_size) || kbChunkConfigDraft.chunk_size <= 0) {
      await showAlertModal({
        title: t("kb_validation_error", "Validation Error"),
        message: t("kb_chunk_size_invalid", "Chunk size must be a positive number"),
        type: "warning",
      });
      return;
    }
    const next = {
      ...kbChunkConfigDraft,
      chunk_size: Math.floor(kbChunkConfigDraft.chunk_size),
    };
    setKbChunkConfig(next);
    saveStoredJson(CHUNK_CONFIG_STORAGE_KEY, next);
    setKbChunkConfigModalOpen(false);
  };

  const handleOpenIndexConfigModal = () => {
    setKbIndexConfigDraft({ ...kbIndexConfig });
    setKbIndexConfigModalOpen(true);
  };

  const handleSaveIndexConfig = async () => {
    if (!kbIndexConfigDraft.base_url.trim()) {
      await showAlertModal({
        title: t("kb_validation_error", "Validation Error"),
        message: t("kb_base_url_required", "Base URL is required"),
        type: "warning",
      });
      return;
    }
    if (!kbIndexConfigDraft.model_name.trim()) {
      await showAlertModal({
        title: t("kb_validation_error", "Validation Error"),
        message: t("kb_model_name_required", "Model Name is required"),
        type: "warning",
      });
      return;
    }
    const next = {
      api_key: kbIndexConfigDraft.api_key.trim(),
      base_url: kbIndexConfigDraft.base_url.trim(),
      model_name: kbIndexConfigDraft.model_name.trim(),
    };
    setKbIndexConfig(next);
    saveStoredJson(INDEX_CONFIG_STORAGE_KEY, next);
    setKbIndexConfigModalOpen(false);
  };

  const handleOpenFolderDetail = async (item: KbFileEntry) => {
    const normalizedCategory = item.category === "collection" || item.category === "index" ? null : item.category;
    if (!normalizedCategory || !["raw", "corpus", "chunks"].includes(normalizedCategory)) {
      return;
    }
    setKbInspectTitle(kbLabel(item));
    setKbInspectFiles([]);
    setKbInspectLoading(true);
    setKbInspectHasHiddenOnly(false);
    setKbFolderDetailModalOpen(true);
    try {
      const payload = await inspectKbFolder(
        normalizedCategory as "raw" | "corpus" | "chunks",
        item.name,
      );
      const visibleFiles = (payload.files ?? []).filter((file) => !String(file.name || "").startsWith("_"));
      setKbInspectFiles(visibleFiles);
      setKbInspectHasHiddenOnly((payload.files ?? []).length > 0 && visibleFiles.length === 0);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to inspect folder";
      await showAlertModal({
        title: t("kb_load_failed_title", "Load Failed"),
        message,
        type: "error",
      });
    } finally {
      setKbInspectLoading(false);
    }
  };

  const pollKbTaskStatus = (taskId: string) => {
    if (kbTaskTimerRef.current !== null) {
      window.clearInterval(kbTaskTimerRef.current);
      kbTaskTimerRef.current = null;
    }
    let progress = 0;
    let busy = false;
    setKbTaskProgressVisible(true);
    setKbTaskProgress(0);
    kbTaskTimerRef.current = window.setInterval(() => {
      if (busy) return;
      busy = true;
      void (async () => {
        try {
          if (progress < 95) {
            const increment = Math.max(0.5, (95 - progress) * 0.08);
            progress = Math.min(95, progress + increment);
            setKbTaskProgress(progress);
          }
          const payload = await fetchKbTaskStatus(taskId);
          const status = String(payload.status ?? "running").toLowerCase();
          if (status === "success" || status === "completed") {
            if (kbTaskTimerRef.current !== null) {
              window.clearInterval(kbTaskTimerRef.current);
              kbTaskTimerRef.current = null;
            }
            setKbTaskProgress(100);
            window.setTimeout(() => {
              resetKbTaskStatus();
              void loadKbData();
            }, 400);
            return;
          }
          if (status === "failed" || status === "error") {
            if (kbTaskTimerRef.current !== null) {
              window.clearInterval(kbTaskTimerRef.current);
              kbTaskTimerRef.current = null;
            }
            resetKbTaskStatus();
            const errorText = String(
              (payload as Record<string, unknown>).error ||
                t("common_unknown_error", "Unknown error"),
            );
            const message = formatTemplate(t("kb_task_failed_message", "Task Failed: {error}"), { error: errorText });
            await showAlertModal({
              title: t("kb_task_failed_title", "Task Failed"),
              message,
              type: "error",
            });
          }
        } finally {
          busy = false;
        }
      })();
    }, 800);
  };

  const runKbPipelineTask = async (
    pipelineName: "build_text_corpus" | "corpus_chunk" | "milvus_index",
    filePath: string,
    extraPayload: Partial<{
      collection_name: string;
      index_mode: "new" | "append" | "overwrite";
      chunk_backend: string;
      tokenizer_or_token_counter: string;
      chunk_size: number;
      use_title: boolean;
      emb_api_key: string;
      emb_base_url: string;
      emb_model_name: string;
    }> = {},
  ) => {
    showKbTaskStatus(
      formatTemplate(t("kb_running_task", "Running {task}..."), { task: pipelineName }),
    );
    try {
      const submit = await runKbTask({
        pipeline_name: pipelineName,
        target_file: filePath,
        ...extraPayload,
      });
      if (!submit.task_id) {
        throw new Error(t("kb_task_start_failed", "Task start failed"));
      }
      pollKbTaskStatus(submit.task_id);
    } catch (error) {
      resetKbTaskStatus();
      const message = error instanceof Error ? error.message : t("kb_task_start_failed", "Task start failed");
      await showAlertModal({
        title: t("kb_task_error_title", "Task Error"),
        message,
        type: "error",
      });
    }
  };

  const handleKBAction = (item: KbFileEntry, pipelineName: "build_text_corpus" | "corpus_chunk" | "milvus_index") => {
    const filePath = getKbEntryPath(item);
    setKbRunTargetFile(filePath);
    if (pipelineName === "milvus_index") {
      const fileName = filePath.split("/").pop() ?? "collection";
      const fallbackName = fileName.replace(/\.jsonl$/i, "").replaceAll(".", "_");
      if (kbRunIndexMode === "new" || !kbRunCollectionName.trim()) {
        setKbRunCollectionName(fallbackName);
      }
      if (kbRunIndexMode !== "new" && !visibleKbCollections.some((collection) => collection.name === kbRunCollectionName)) {
        setKbRunCollectionName(visibleKbCollections[0]?.name ?? fallbackName);
      }
      setKbMilvusModalOpen(true);
      return;
    }
    if (pipelineName === "corpus_chunk") {
      void runKbPipelineTask(pipelineName, filePath, {
        chunk_backend: kbChunkConfig.chunk_backend,
        tokenizer_or_token_counter: kbChunkConfig.tokenizer_or_token_counter,
        chunk_size: kbChunkConfig.chunk_size,
        use_title: kbChunkConfig.use_title,
      });
      return;
    }
    void runKbPipelineTask(pipelineName, filePath);
  };

  const renderKbPipelineList = (
    entries: KbFileEntry[],
    pipelineName: "build_text_corpus" | "corpus_chunk" | "milvus_index",
    actionLabel: string,
  ) => {
    if (!entries.length) {
      return <div className="text-muted small text-center mt-5 opacity-50">{t("kb_empty", "Empty")}</div>;
    }
    return entries
      .slice()
      .sort((left, right) => (Number(right.mtime ?? 0) - Number(left.mtime ?? 0)))
      .map((entry) => {
        const isFolder = entry.type === "folder";
        const displayText = kbLabel(entry);
        const tooltipText =
          entry.display_name && entry.display_name !== entry.name
            ? `${entry.display_name}\n(${entry.name})`
            : entry.name;
        const sizeStr = `${(Number(entry.size ?? 0) / 1024).toFixed(1)} KB`;
        const metaText =
          isFolder && Number(entry.file_count ?? 0) > 0
            ? `${entry.file_count} ${t("kb_files", "files")} · ${sizeStr}`
            : sizeStr;
        const pathKey = getKbEntryPath(entry);
        const isNew = !kbFileSnapshotRef.current.has(pathKey);
        return (
          <div key={`${entry.category}-${entry.name}-${pathKey}`} className={`file-item ${isNew ? "new-upload" : ""}`.trim()}>
            <div
              className="file-item-inner"
              onClick={() => {
                if (isFolder) {
                  void handleOpenFolderDetail(entry);
                }
              }}
            >
              <div className="file-icon-wrapper">
                {isFolder ? (
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="text-blue-500"
                  >
                    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
                  </svg>
                ) : (
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="text-gray-400"
                  >
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                    <line x1="16" y1="13" x2="8" y2="13" />
                    <line x1="16" y1="17" x2="8" y2="17" />
                    <polyline points="10 9 9 9 8 9" />
                  </svg>
                )}
              </div>
              <div className="file-info-wrapper">
                <div className="file-title" title={tooltipText}>
                  {displayText}
                </div>
                <div className="file-meta">{metaText}</div>
              </div>
              <div className="file-actions">
                <button
                  type="button"
                  className="btn btn-sm btn-light border ms-auto flex-shrink-0"
                  style={{ fontSize: "0.75rem" }}
                  onClick={(event) => {
                    event.stopPropagation();
                    handleKBAction(entry, pipelineName);
                  }}
                >
                  {actionLabel}
                </button>
                {entry.category !== "collection" ? (
                  <button
                    type="button"
                    className="btn btn-sm text-danger ms-2 btn-icon-only flex-shrink-0"
                    title={t("kb_delete", "Delete")}
                    onClick={(event) => {
                      event.stopPropagation();
                      void handleDeleteKbItem(entry);
                    }}
                  >
                    ×
                  </button>
                ) : null}
              </div>
            </div>
          </div>
        );
      });
  };

  const handleConfirmIndexTask = async () => {
    if (!kbRunTargetFile.trim()) {
      await showAlertModal({
        title: t("kb_task_error_title", "Task Error"),
        message: t("kb_task_start_failed", "Task start failed"),
        type: "error",
      });
      return;
    }
    const selectedCollectionName = kbRunCollectionName.trim();
    if (kbRunIndexMode === "new") {
      if (!selectedCollectionName) {
        await showAlertModal({
          title: t("kb_validation_error", "Validation Error"),
          message: t("kb_collection_name_required", "Knowledge base name is required"),
          type: "warning",
        });
        return;
      }
      const matched = findMatchingCollection(visibleKbCollections, selectedCollectionName);
      if (matched) {
        const displayName = getCollectionDisplayName(matched);
        const suffix =
          displayName && displayName !== selectedCollectionName
            ? locale === "zh"
              ? `（别名“${displayName}”）`
              : ` as "${displayName}"`
            : "";
        const choice = await showKbIndexChoiceModal(
          formatTemplate(
            t(
              "kb_collection_exists",
              'Knowledge base name "{inputName}" already exists{displayName}. Choose "Append" to add data or "Overwrite" to drop and rebuild.',
            ),
            { inputName: selectedCollectionName, displayName: suffix },
          ),
        );
        if (!choice) return;
        setKbRunIndexMode(choice);
        setKbRunCollectionName(matched.name);
        const label = getCollectionDisplayName(matched) || matched.name;
        const confirmed =
          choice === "append"
            ? await showConfirmModal({
                title: t("kb_confirm_append_title", "Append Confirmation"),
                message: formatTemplate(
                  t("kb_confirm_append", 'Append data to the existing knowledge base "{label}"?'),
                  { label },
                ),
                type: "info",
                confirmText: t("kb_continue", "Continue"),
              })
            : await showConfirmModal({
                title: t("kb_confirm_overwrite_title", "Overwrite Confirmation"),
                message: formatTemplate(
                  t(
                    "kb_confirm_overwrite",
                    'Overwrite the existing knowledge base "{label}"? This will drop and rebuild the knowledge base.',
                  ),
                  { label },
                ),
                type: "warning",
                danger: true,
                confirmText: t("kb_mode_overwrite", "Overwrite"),
              });
        if (!confirmed) return;
        setKbMilvusModalOpen(false);
        void runKbPipelineTask("milvus_index", kbRunTargetFile.trim(), {
          collection_name: matched.name,
          index_mode: choice,
          emb_api_key: kbIndexConfig.api_key,
          emb_base_url: kbIndexConfig.base_url,
          emb_model_name: kbIndexConfig.model_name,
        });
        return;
      }
      const confirmedNew = await showConfirmModal({
        title: t("kb_confirm_new_title", "Create Knowledge Base"),
        message: formatTemplate(t("kb_confirm_new", 'Create a new knowledge base named "{label}"?'), {
          label: selectedCollectionName,
        }),
        type: "info",
        confirmText: t("kb_create", "Create"),
      });
      if (!confirmedNew) return;
    } else {
      if (!visibleKbCollections.length) {
        await showAlertModal({
          title: t("kb_no_collections_title", "No Knowledge Bases"),
          message: t("kb_no_collections_message", 'No existing knowledge bases found. Use "New" mode to create one.'),
          type: "warning",
        });
        return;
      }
      if (!selectedCollectionName) {
        await showAlertModal({
          title: t("kb_select_collection_title", "Selection Required"),
          message: t("kb_select_collection_message", "Please select a knowledge base."),
          type: "warning",
        });
        return;
      }
      const selectedCollection =
        visibleKbCollections.find((collection) => collection.name === selectedCollectionName) ?? null;
      const label = getCollectionDisplayName(selectedCollection) || selectedCollectionName;
      const confirmed =
        kbRunIndexMode === "append"
          ? await showConfirmModal({
              title: t("kb_confirm_append_title", "Append Confirmation"),
              message: formatTemplate(
                t("kb_confirm_append", 'Append data to the existing knowledge base "{label}"?'),
                { label },
              ),
              type: "info",
              confirmText: t("kb_continue", "Continue"),
            })
          : await showConfirmModal({
              title: t("kb_confirm_overwrite_title", "Overwrite Confirmation"),
              message: formatTemplate(
                t(
                  "kb_confirm_overwrite",
                  'Overwrite the existing knowledge base "{label}"? This will drop and rebuild the knowledge base.',
                ),
                { label },
              ),
              type: "warning",
              danger: true,
              confirmText: t("kb_mode_overwrite", "Overwrite"),
            });
      if (!confirmed) return;
    }
    setKbMilvusModalOpen(false);
    void runKbPipelineTask("milvus_index", kbRunTargetFile.trim(), {
      collection_name: selectedCollectionName,
      index_mode: kbRunIndexMode,
      emb_api_key: kbIndexConfig.api_key,
      emb_base_url: kbIndexConfig.base_url,
      emb_model_name: kbIndexConfig.model_name,
    });
  };

  const handleDeleteKbItem = async (item: KbFileEntry) => {
    const category = item.category === "collection" ? "collection" : item.category;
    const action =
      category === "collection"
        ? t("kb_delete_collection_action", "drop this knowledge base")
        : t("kb_delete_file_action", "delete this file");
    const label = category === "collection" ? kbLabel(item) : item.name;
    const confirmed = await showConfirmModal({
      title: t("kb_delete_confirm_title", "Delete Confirmation"),
      message: formatTemplate(
        t("kb_delete_confirm_message", "Permanently {action} ({name})?"),
        { action, name: label },
      ),
      type: "warning",
      danger: true,
      confirmText: t("common_delete", "Delete"),
    });
    if (!confirmed) return;
    try {
      await deleteKbFile(category, item.name);
      await loadKbData();
    } catch (error) {
      const message = error instanceof Error ? error.message : "Delete failed";
      await showAlertModal({
        title: t("status_error", "Error"),
        message,
        type: "error",
      });
    }
  };

  const handleSaveKbConfig = async (): Promise<boolean> => {
    if (!kbConfigUri.trim()) {
      await showAlertModal({
        title: t("kb_validation_error", "Validation Error"),
        message: t("kb_uri_required", "Milvus URI is required."),
        type: "warning",
      });
      return false;
    }
    setKbConfigBusy(true);
    try {
      const currentConfig = await fetchKbConfig();
      const merged: Record<string, unknown> = { ...currentConfig };
      const milvus =
        currentConfig.milvus && typeof currentConfig.milvus === "object"
          ? { ...(currentConfig.milvus as Record<string, unknown>) }
          : {};
      milvus.uri = kbConfigUri.trim();
      milvus.token = kbConfigToken.trim();
      merged.milvus = milvus;
      await saveKbConfig(merged);
      await loadKbData();
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to save KB config";
      await showAlertModal({
        title: t("status_error", "Error"),
        message,
        type: "error",
      });
      return false;
    } finally {
      setKbConfigBusy(false);
    }
  };

  const handleClearKbStaging = async () => {
    const confirmed = await showConfirmModal({
      title: t("kb_clear_staging_title", "Clear Staging Area"),
      message: t("kb_clear_staging_prompt", "Are you sure you want to clear ALL temporary files (Raw, Corpus, Chunks)?"),
      type: "warning",
      danger: true,
      confirmText: t("kb_clear_all", "Clear All"),
    });
    if (!confirmed) return;
    try {
      const result = await clearKbStaging();
      const counts =
        result.deleted_counts && typeof result.deleted_counts === "object"
          ? (result.deleted_counts as Record<string, unknown>)
          : {};
      const raw = Number(counts.raw ?? 0);
      const corpus = Number(counts.corpus ?? 0);
      const chunks = Number(counts.chunks ?? 0);
      const total = Number(result.total_deleted ?? raw + corpus + chunks);
      let message = formatTemplate(
        t(
          "kb_deleted_summary",
          "Deleted:\n- Raw: {raw} items\n- Corpus: {corpus} items\n- Chunks: {chunks} items\n\nTotal: {total} items",
        ),
        { raw, corpus, chunks, total },
      );
      const errors = Array.isArray(result.errors) ? result.errors.map((item) => String(item)).filter(Boolean) : [];
      if (errors.length > 0) {
        message += `\n\n${formatTemplate(t("kb_note_errors", "Note: Some errors occurred:\n{errors}"), {
          errors: errors.slice(0, 3).join("\n"),
        })}`;
        if (errors.length > 3) {
          message += `\n${formatTemplate(t("kb_more_errors", "... and {count} more errors"), {
            count: errors.length - 3,
          })}`;
        }
      }
      await showAlertModal({
        title: t("kb_staging_cleared_title", "Staging Area Cleared"),
        message,
        type: "success",
      });
      await loadKbData();
      setKbInspectFiles([]);
      setKbInspectTitle("");
    } catch (error) {
      const message =
        error instanceof Error
          ? `${t("kb_clear_error", "Clear error: ")}${error.message}`
          : t("kb_clear_failed", "Clear failed: ");
      await showAlertModal({
        title: t("status_error", "Error"),
        message,
        type: "error",
      });
    }
  };

  const handleOpenVisibility = async (item: KbFileEntry) => {
    if (item.category !== "collection" && item.category !== "index") return;
    try {
      const [detail, usersPayload] = await Promise.all([
        fetchKbVisibility(item.name),
        listKbVisibilityUsers().catch(() => ({ users: [] })),
      ]);
      setKbVisibilityCollection(item);
      setKbVisibilityData(detail);
      setKbVisibilityUsers(usersPayload.users ?? []);
      setKbVisibilityModalOpen(true);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to load visibility";
      await showAlertModal({
        title: t("kb_visibility_load_failed_title", "Failed to load visibility"),
        message,
        type: "error",
      });
      setKbVisibilityModalOpen(false);
    }
  };

  const handleSaveVisibility = async (): Promise<boolean> => {
    if (!kbVisibilityCollection || !kbVisibilityData) return false;
    try {
      const next = await saveKbVisibility(kbVisibilityCollection.name, {
        visibility: kbVisibilityData.visibility,
        visible_users:
          kbVisibilityData.visibility === "shared" ? kbVisibilityData.visible_users ?? [] : [],
      });
      setKbVisibilityData(next);
      await loadKbData();
      return true;
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to save visibility";
      await showAlertModal({
        title: t("kb_visibility_save_failed_title", "Failed to save visibility"),
        message,
        type: "error",
      });
      return false;
    }
  };

  const handleVisibilityModeChange = (mode: "private" | "public" | "shared") => {
    setKbVisibilityData((previous) => {
      if (!previous) return previous;
      return {
        ...previous,
        visibility: mode,
        visible_users: mode === "shared" ? previous.visible_users ?? [] : [],
      };
    });
  };

  const toggleVisibilityUser = (username: string) => {
    setKbVisibilityData((previous) => {
      if (!previous) return previous;
      const current = new Set(previous.visible_users ?? []);
      if (current.has(username)) {
        current.delete(username);
      } else {
        current.add(username);
      }
      return {
        ...previous,
        visible_users: [...current],
      };
    });
  };

  const handleSaveMemory = async () => {
    setMemoryBusy(true);
    setMemoryMessage("");
    setMemoryStatus(t("memory_saving", "Saving memory..."), "saving");
    try {
      await saveMemory(memoryContent, userId);
      setMemoryStatus(t("memory_saved", "Memory saved"), "success");
    } catch (error) {
      setMemoryStatus(t("memory_save_failed", "Save failed"), "error");
      setMemoryMessage(error instanceof Error ? error.message : t("common_unknown_error", "Unknown error"));
    } finally {
      setMemoryBusy(false);
    }
  };

  const pollMemorySyncTask = useCallback(
    async (taskId: string, collectionName = "") => {
      const startedAt = Date.now();
      while (true) {
        if (Date.now() - startedAt > 5 * 60 * 1000) {
          throw new Error(t("memory_sync_timeout", "Sync timed out. Please try again later."));
        }
        await new Promise((resolve) => window.setTimeout(resolve, 1200));
        const task = await fetchKbTaskStatus(taskId);
        const status = String(task.status ?? "running");
        if (status === "success") {
          await Promise.allSettled([loadKbData(), refreshMemoryKbCards()]);
          const result = task.result && typeof task.result === "object" ? (task.result as Record<string, unknown>) : {};
          const name = collectionName || String(result.collection_name ?? "");
          setMemoryStatus(t("memory_sync_success", "Sync completed"), "success");
          if (name) {
            setMemoryMessage(
              formatTemplate(
                t("memory_sync_success_message", "Memory synced into collection: {collection}"),
                { collection: name },
              ),
            );
          }
          return;
        }
        if (status === "failed") {
          throw new Error(String(task.error ?? t("memory_sync_failed", "Sync failed")));
        }
        setMemoryStatus(t("memory_sync_running", "Syncing memory to knowledge base..."), "loading");
      }
    },
    [loadKbData, refreshMemoryKbCards, setMemoryStatus, t],
  );

  const handleSyncMemory = async () => {
    setMemoryBusy(true);
    setMemoryMessage("");
    setMemoryStatus(t("memory_sync_submitting", "Submitting sync task..."), "loading");
    try {
      const submit = await syncMemoryToKb("append");
      setMemoryStatus(t("memory_sync_running", "Syncing memory to knowledge base..."), "loading");
      await pollMemorySyncTask(submit.task_id);
    } catch (error) {
      setMemoryStatus(t("memory_sync_failed", "Sync failed"), "error");
      setMemoryMessage(error instanceof Error ? error.message : t("common_unknown_error", "Unknown error"));
    } finally {
      setMemoryBusy(false);
    }
  };

  const handleClearMemoryVectors = async () => {
    const confirmed = await showConfirmModal({
      title: t("memory_clear_vectors_confirm_title", "Clear Working Memory Vectors"),
      message: t(
        "memory_clear_vectors_confirm_message",
        "Clear all vectors in the current user's working-memory collection and delete this user's working-memory files? (Global MEMORY is kept.)",
      ),
      type: "warning",
      danger: true,
      confirmText: t("memory_clear_vectors_action", "Clear Memory Vectors"),
    });
    if (!confirmed) return;
    setMemoryBusy(true);
    setMemoryMessage("");
    setMemoryStatus(t("memory_clear_vectors_running", "Clearing working memory vectors..."), "loading");
    try {
      const result = await clearMemoryVectors();
      await Promise.allSettled([loadKbData(), refreshMemoryKbCards()]);
      setMemoryStatus(t("memory_clear_vectors_success", "Working memory vectors cleared"), "success");
      setMemoryMessage(
        formatTemplate(
          t(
            "memory_clear_vectors_success_message",
            "Collection cleared: {collection}, deleted vectors: {count}, deleted working-memory files: {files}",
          ),
          {
            collection: result.collection_name ?? "",
            count: Number(result.cleared_count ?? 0),
            files: 0,
          },
        ),
      );
    } catch (error) {
      setMemoryStatus(t("memory_clear_vectors_failed", "Clear failed"), "error");
      setMemoryMessage(error instanceof Error ? error.message : t("common_unknown_error", "Unknown error"));
    } finally {
      setMemoryBusy(false);
    }
  };

  const handleDeleteBackgroundTask = async (taskId: string) => {
    await deleteBackgroundTask(taskId);
    if (backgroundDetailTaskId === taskId) {
      setBackgroundDetailTaskId("");
    }
    setTaskDetailMap((previous) => {
      const next = { ...previous };
      delete next[taskId];
      return next;
    });
    await loadBackgroundTaskData();
  };

  const handleClearCompletedTasks = async () => {
    await clearCompletedBackgroundTasks();
    setTaskDetailMap({});
    await loadBackgroundTaskData();
  };

  const handleLoadBackgroundTaskDetail = async (taskId: string) => {
    try {
      const detail = await fetchBackgroundTask(taskId);
      setTaskDetailMap((previous) => ({ ...previous, [taskId]: detail }));
    } catch {
      // Keep task panel usable even when detail request fails.
    }
  };

  const handleLoadTaskResultToChat = async (task: BackgroundTask, openNewSession: boolean) => {
    const result = taskResultText(task);
    if (!result.trim()) return;
    const userText = task.full_question || task.question || "Background Question";
    const mergedMessages: ChatMessage[] = [
      {
        role: "user",
        text: userText,
        timestamp: new Date().toISOString(),
      },
      {
        role: "assistant",
        text: result,
        timestamp: new Date().toISOString(),
        meta: task.sources ? { sources: task.sources } : undefined,
      },
    ];

    let targetSessionId = activeSessionId;
    if (openNewSession || !targetSessionId) {
      targetSessionId = createSessionId();
      setActiveSessionId(targetSessionId);
      if (!loggedIn) {
        setGuestSessions((previous) => [
          {
            id: targetSessionId,
            title: deriveSessionTitle(mergedMessages),
            pipeline: selectedPipeline || null,
            timestamp: Date.now(),
            messages: mergedMessages,
          },
          ...previous,
        ]);
      }
    }

    setMessages(mergedMessages);
    await persistSession(targetSessionId, mergedMessages);
    setActiveView("chat");
  };

  const openSourceDetail = useCallback(
    (messageIndex: number, sourceDisplayId: number) => {
      const message = messages[messageIndex];
      if (!message) return;
      const citationPayload = buildCitationPayload(message);
      const source = citationPayload.sources.find(
        (item) => getSourceDisplayId(item) === sourceDisplayId,
      );
      if (!source) return;
      setActiveReferenceKey(`${messageIndex}:${sourceDisplayId}`);
      setDetailSource(source);
      setDetailSourceDisplayId(sourceDisplayId);
      setDetailPanelOpen(true);
    },
    [messages],
  );

  const toggleBackgroundPanel = () => {
    if (activeView !== "chat") return;
    setBackgroundPanelOpen((previous) => {
      const next = !previous;
      if (next) {
        void loadBackgroundTaskData();
      }
      return next;
    });
  };

  const openBackgroundTaskDetail = (taskId: string) => {
    setBackgroundDetailTaskId(taskId);
    void handleLoadBackgroundTaskDetail(taskId);
  };

  const closeBackgroundTaskDetail = () => {
    setBackgroundDetailTaskId("");
  };

  const handleCopyTaskResult = async () => {
    if (!backgroundDetailTask || typeof backgroundDetailTask.result !== "string") return;
    try {
      await navigator.clipboard.writeText(backgroundDetailTask.result);
    } catch {
      // Ignore clipboard errors.
    }
  };

  const handleLoadTaskResultFromModal = async (openNewSession: boolean) => {
    if (!backgroundDetailTask) return;
    await handleLoadTaskResultToChat(backgroundDetailTask, openNewSession);
    setBackgroundPanelOpen(false);
    closeBackgroundTaskDetail();
  };

  const handleMessageContentClick = async (event: MouseEvent<HTMLElement>) => {
    const target = event.target as HTMLElement | null;
    const copyTrigger = target?.closest<HTMLElement>(
      ".code-block-copy[data-copy-text], .table-copy-btn[data-copy-text]",
    );
    if (copyTrigger) {
      event.preventDefault();
      event.stopPropagation();
      const payload = copyTrigger.getAttribute("data-copy-text") ?? "";
      if (!payload) return;
      try {
        await navigator.clipboard.writeText(decodeCopyPayload(payload));
        copyTrigger.classList.add("copied");
        window.setTimeout(() => {
          copyTrigger.classList.remove("copied");
        }, 1400);
      } catch {
        // Ignore clipboard errors to keep chat flow stable.
      }
      return;
    }
    const citation = target?.closest<HTMLElement>(
      ".citation-link[data-source-id], .react-chat-citation[data-source-id]",
    );
    if (!citation) return;
    const sourceId = citation.dataset.sourceId;
    if (!sourceId) return;
    const parsedSourceId = Number.parseInt(sourceId, 10);
    if (!Number.isFinite(parsedSourceId)) return;
    const bubble = citation.closest<HTMLElement>(".chat-bubble");
    if (!bubble) return;
    const messageIdx = Number.parseInt(bubble.dataset.messageIdx ?? "", 10);
    if (Number.isFinite(messageIdx)) {
      setExpandedUnusedRefs((previous) =>
        previous[messageIdx] ? previous : { ...previous, [messageIdx]: true },
      );
      openSourceDetail(messageIdx, parsedSourceId);
    }
  };

  const handleChatHistoryScroll = () => {
    const panel = chatHistoryRef.current;
    if (!panel) return;
    const currentTop = panel.scrollTop;
    const previousTop = chatScrollTopRef.current;
    const distanceToBottom = panel.scrollHeight - currentTop - panel.clientHeight;
    if (currentTop < previousTop - 1) {
      shouldAutoScrollRef.current = false;
    } else if (distanceToBottom <= CHAT_SCROLL_FOLLOW_THRESHOLD) {
      shouldAutoScrollRef.current = true;
    }
    chatScrollTopRef.current = currentTop;
  };

  const handleThinkingBodyScroll = () => {
    const body = thinkingBodyRef.current;
    if (!body) return;
    const currentTop = body.scrollTop;
    const previousTop = thinkingScrollTopRef.current;
    const distanceToBottom = body.scrollHeight - currentTop - body.clientHeight;
    if (currentTop < previousTop - 1) {
      thinkingShouldAutoScrollRef.current = false;
    } else if (distanceToBottom <= CHAT_SCROLL_FOLLOW_THRESHOLD) {
      thinkingShouldAutoScrollRef.current = true;
    }
    thinkingScrollTopRef.current = currentTop;
  };

  const bindThinkingBodyRef = (node: HTMLDivElement | null, isStreamingAssistant: boolean) => {
    if (!isStreamingAssistant) return;
    thinkingBodyRef.current = node;
    if (!node) return;
    if (thinkingShouldAutoScrollRef.current) {
      node.scrollTop = node.scrollHeight;
    }
    thinkingScrollTopRef.current = node.scrollTop;
  };

  const handleExportDocx = async (message: ChatMessage, messageIndex: number) => {
    const messageKey = `${message.role}-${messageIndex}-${message.timestamp ?? "msg"}`;
    const citationPayload = buildCitationPayload(message);
    let question = "";
    for (let index = messageIndex - 1; index >= 0; index -= 1) {
      if (messages[index].role === "user") {
        question = messages[index].text;
        break;
      }
    }

    const selectedFormat = await showExportFormatModal();
    if (!selectedFormat) return;

    setExportingMessageKey(messageKey);
    try {
      if (selectedFormat === "docx") {
        const { blob, filename } = await exportChatDocx({
          text: citationPayload.text,
          question,
          sources: citationPayload.sources,
        });
        triggerDownloadBlob(blob, filename || buildDownloadFileName(question, "docx"));
      } else {
        const markdown = buildDownloadMarkdown(citationPayload.text, citationPayload.sources, question);
        const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
        triggerDownloadBlob(blob, buildDownloadFileName(question, "md"));
      }
      setDownloadedMessageKey(messageKey);
      window.setTimeout(() => {
        setDownloadedMessageKey((previous) => (previous === messageKey ? "" : previous));
      }, 2000);
    } catch (error) {
      const messageText = error instanceof Error ? error.message : t("common_unknown_error", "Unknown error");
      await showAlertModal({
        title: t("chat_export_docx_failed_title", "Export Failed"),
        message: formatTemplate(t("chat_export_docx_failed_message", "Failed to export DOCX: {error}"), {
          error: messageText,
        }),
        type: "error",
      });
    } finally {
      setExportingMessageKey("");
    }
  };

  return (
    <>
      <section
        id="chat-view"
        className="view-overlay"
        data-viewport={isMobileViewport ? "mobile" : "desktop"}
        data-sidebar-open={isMobileViewport && mobileSidebarOpen ? "true" : "false"}
      >
      <main className="chat-layout">
        <aside
          ref={chatSidebarRef}
          className={`chat-sidebar ${mobileSidebarOpen ? "show" : ""} ${
            !isMobileViewport && sidebarCollapsed ? "collapsed" : ""
          }`}
        >
          <header className="sidebar-header">
            <div className="sidebar-toggle-wrapper">
              <button
                type="button"
                id="chat-logo-btn"
                className="sidebar-logo sidebar-logo-btn"
                onClick={() => {
                  switchView("chat");
                  void handleNewChat();
                }}
              >
                <img src="/theme/ultrarag.svg" alt="UltraRAG" className="sidebar-logo-img" />
              </button>
              <button
                type="button"
                id="sidebar-toggle-btn"
                className="btn-sidebar-toggle"
                title={t("chat_sidebar_toggle_title", "Toggle Sidebar")}
                onClick={() => setSidebarCollapsed((previous) => !previous)}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                  <line x1="9" y1="3" x2="9" y2="21" />
                </svg>
              </button>
            </div>

            <button
              type="button"
              id="chat-new-btn"
              className="btn-nav"
              onClick={() => {
                switchView("chat");
                void handleNewChat();
              }}
            >
              <span className="icon">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" width="20" height="20">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10" />
                </svg>
              </span>
              <span className="btn-text">{t("new_chat", "New Chat")}</span>
            </button>

            <button
              type="button"
              id="kb-btn"
              className={`btn-nav ${activeView === "kb" ? "active" : ""}`}
              onClick={() => switchView("kb")}
            >
              <span className="icon">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor" width="20" height="20">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
                </svg>
              </span>
              <span className="btn-text">{t("knowledge_base", "Knowledge Base")}</span>
            </button>

            <button
              type="button"
              id="explore-btn"
              className={`btn-nav ${activeView === "explore" || activeView === "memory" ? "active" : ""}`}
              onClick={() => switchView("explore")}
            >
              <span className="icon">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                  <polyline points="3.3 7 12 12 20.7 7" />
                  <line x1="12" y1="22" x2="12" y2="12" />
                </svg>
              </span>
              <span className="btn-text">{t("explore", "Explore")}</span>
            </button>
          </header>

          <section className="sidebar-list">
            <div className="chat-session-bar">
              <div className="chat-session-bar-title">
                <span>{t("recent", "Recent")}</span>
              </div>
              <div className="chat-session-bar-actions">
                <button
                  className="chat-session-bar-btn"
                  id="clear-all-chats"
                  title={t("chat_clear_all_title", "Clear all chats")}
                  onClick={() => void handleClearSessions()}
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
                    <polyline points="3 6 5 6 21 6" />
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                  </svg>
                </button>
              </div>
            </div>
            <div className="session-list" id="chat-session-list">
              {visibleSessions.length ? (
                visibleSessions.map((session) => (
                  <div
                    key={session.id}
                    role="button"
                    tabIndex={0}
                    className={`chat-session-item ${
                      activeView === "chat" && activeSessionId === session.id ? "active" : ""
                    }`}
                    onClick={() => void handleOpenSession(session.id)}
                    onContextMenu={(event) => {
                      openChatSessionContextMenu(event, session.id);
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        void handleOpenSession(session.id);
                      }
                    }}
                  >
                    <div
                      className="chat-session-content"
                      title={t("common_rename", "Rename")}
                      onDoubleClick={(event) => {
                        event.stopPropagation();
                        openRenameSessionDialog(session.id);
                      }}
                    >
                      <span className="chat-session-title">{session.title || t("new_chat", "New Chat")}</span>
                    </div>
                    <button
                      type="button"
                      className="chat-session-delete-btn"
                      onClick={(event) => {
                        event.stopPropagation();
                        void handleDeleteSession(session.id);
                      }}
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
                        <polyline points="3 6 5 6 21 6" />
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                      </svg>
                    </button>
                  </div>
                ))
              ) : (
                <div className="text-muted small" style={{ paddingLeft: "24px" }}>
                  {t("chat_no_history", "No history")}
                </div>
              )}
            </div>
          </section>

          <footer className="sidebar-footer">
            <div
              className={`settings-menu ${settingsMenuOpen ? "open" : ""}`}
              id="settings-menu"
              ref={settingsMenuRef}
            >
              <button
                type="button"
                className="settings-trigger btn-nav"
                id="settings-menu-trigger"
                onClick={() =>
                  setSettingsMenuOpen((previous) => {
                    const next = !previous;
                    if (!next) {
                      setLanguageSubmenuOpen(false);
                    }
                    return next;
                  })
                }
              >
                <span className="settings-user-avatar" id="settings-user-avatar">
                  {userAvatarText}
                </span>
                <span className="btn-text settings-user-label" id="settings-user-label">
                  {displayUserName}
                </span>
              </button>

              <div className="settings-dropdown" id="settings-dropdown">
                <button
                  type="button"
                  className="settings-item settings-user-entry"
                  id="settings-user-entry"
                  onClick={() => {
                    setSettingsMenuOpen(false);
                    if (loggedIn) {
                      setAccountStatus("");
                      setAccountDialogMode("account");
                      setAccountOpen(true);
                    } else {
                      setAuthOpen(true);
                    }
                  }}
                >
                  <span className="item-icon">
                    <span className="settings-user-avatar settings-user-avatar-sm">{userAvatarText}</span>
                  </span>
                  <div className="item-text">
                    <span id="settings-user-entry-text">{displayUserName}</span>
                  </div>
                </button>

                <div className="settings-divider" role="separator" />

                {isAdmin ? (
                  <button
                    type="button"
                    className="settings-item"
                    id="settings-builder"
                    onClick={() => {
                      setSettingsMenuOpen(false);
                      openBuilderView();
                    }}
                  >
                    <span className="item-icon">
                      <Settings size={18} />
                    </span>
                    <div className="item-text">{t("builder", "Builder")}</div>
                  </button>
                ) : null}

                <button
                  type="button"
                  className="settings-item"
                  id="settings-model"
                  onClick={() => {
                    setSettingsMenuOpen(false);
                    if (loggedIn) {
                      setAccountStatus("");
                      setAccountDialogMode("model");
                      setAccountOpen(true);
                    } else {
                      setAuthOpen(true);
                    }
                  }}
                >
                  <span className="item-icon">
                    <HardDrive size={18} />
                  </span>
                  <div className="item-text">{t("auth_model_settings_title", "Model Settings")}</div>
                </button>

                <div
                  className={`settings-item settings-item-has-submenu ${languageSubmenuOpen ? "submenu-open" : ""}`}
                  id="settings-language"
                  tabIndex={0}
                  onMouseEnter={() => setLanguageSubmenuOpen(true)}
                  onClick={(event) => {
                    event.stopPropagation();
                    setLanguageSubmenuOpen((previous) => !previous);
                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      setLanguageSubmenuOpen((previous) => !previous);
                    } else if (event.key === "Escape") {
                      event.preventDefault();
                      setLanguageSubmenuOpen(false);
                    }
                  }}
                >
                  <div className="settings-item-main">
                    <span className="item-icon">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="18"
                        height="18"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.8"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <circle cx="12" cy="12" r="10" />
                        <line x1="2" y1="12" x2="22" y2="12" />
                        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                      </svg>
                    </span>
                    <div className="item-text">{t("language", "Language")}</div>
                    <span className="submenu-caret">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1.8"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <polyline points="9 18 15 12 9 6" />
                      </svg>
                    </span>
                  </div>
                  <div
                    className="settings-submenu"
                    id="settings-language-submenu"
                    onMouseEnter={() => setLanguageSubmenuOpen(true)}
                    onClick={(event) => event.stopPropagation()}
                  >
                    <button
                      type="button"
                      className={`settings-submenu-item ${locale === "zh" ? "active" : ""}`}
                      onClick={(event) => {
                        event.stopPropagation();
                        setLocale("zh");
                        setLanguageSubmenuOpen(false);
                        setSettingsMenuOpen(false);
                      }}
                    >
                      <span>中文</span>
                    </button>
                    <button
                      type="button"
                      className={`settings-submenu-item ${locale === "en" ? "active" : ""}`}
                      onClick={(event) => {
                        event.stopPropagation();
                        setLocale("en");
                        setLanguageSubmenuOpen(false);
                        setSettingsMenuOpen(false);
                      }}
                    >
                      <span>English</span>
                    </button>
                  </div>
                </div>

                <div className="settings-divider" role="separator" />

                {loggedIn ? (
                  <button
                    type="button"
                    className="settings-item"
                    id="settings-logout"
                    onClick={() => {
                      setSettingsMenuOpen(false);
                      void handleLogout();
                    }}
                  >
                    <span className="item-icon">
                      <LogOut size={18} />
                    </span>
                    <div className="item-text">{t("auth_logout_action", "Logout")}</div>
                  </button>
                ) : (
                  <button
                    type="button"
                    className="settings-item"
                    id="settings-login"
                    onClick={() => {
                      setSettingsMenuOpen(false);
                      setAuthOpen(true);
                    }}
                  >
                    <span className="item-icon">
                      <LogOut size={18} />
                    </span>
                    <div className="item-text">{t("auth_login_title", "Login")}</div>
                  </button>
                )}
              </div>
            </div>
          </footer>
        </aside>

        <button
          type="button"
          className="chat-mobile-backdrop"
          tabIndex={isMobileViewport && mobileSidebarOpen ? 0 : -1}
          aria-label={t("common_close", "Close")}
          onClick={(event) => {
            event.currentTarget.blur();
            setMobileSidebarOpen(false);
          }}
        />

        <section className="chat-main" id="chat-main-view">
          {activeView === "chat" ? (
          <header className="view-header border-bottom-0 justify-content-start gap-3">
            <div className="d-md-none">
              <button
                type="button"
                className="btn btn-icon chat-mobile-menu-btn"
                id="chat-mobile-menu"
                data-mobile-sidebar-toggle="true"
                onClick={toggleMobileSidebar}
              >
                ☰
              </button>
            </div>

            <div className="pipeline-selector-wrapper dropdown" ref={pipelineDropdownRef}>
              <button
                className={`btn btn-pipeline-select dropdown-toggle ${
                  pipelineDropdownOpen ? "show" : ""
                }`}
                type="button"
                id="chatPipelineDropdown"
                aria-expanded={pipelineDropdownOpen}
                onClick={togglePipelineDropdown}
              >
                <span className="fw-bold fs-6" id="chat-pipeline-label">
                  {selectedPipeline || t("select_pipeline", "Select Pipeline")}
                </span>
                <span className="ms-1 small" style={{ color: "#8f8f8f" }}>
                  UltraRAG
                </span>
              </button>
              <ul
                className={`dropdown-menu shadow-lg border-0 rounded-4 mt-2 p-2 ${
                  pipelineDropdownOpen ? "show" : ""
                }`}
                id="chat-pipeline-menu"
                style={{ minWidth: "240px" }}
              >
                {readyPipelines.length ? (
                  readyPipelines.map((pipeline) => (
                    <li key={pipeline.name}>
                      <button
                        type="button"
                        className={`dropdown-item small pipeline-menu-item d-flex align-items-center justify-content-between gap-2 ${
                          selectedPipeline === pipeline.name ? "active" : ""
                        }`}
                        onClick={() => selectPipelineItem(pipeline.name)}
                      >
                        <span>{pipeline.name}</span>
                        {selectedPipeline === pipeline.name ? (
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="16"
                            height="16"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2.4"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <polyline points="20 6 9 17 4 12" />
                          </svg>
                        ) : null}
                      </button>
                    </li>
                  ))
                ) : (
                  <li>
                    <span className="dropdown-item text-muted small">
                      {locale === "zh" ? "暂无可用 Pipeline（请先 Build）" : "No built pipelines (build first)"}
                    </span>
                  </li>
                )}
              </ul>
            </div>

            <div className="ms-auto d-flex align-items-center gap-2">
              <button
                id="demo-toggle-btn"
                type="button"
                className={`btn btn-sm btn-outline-secondary rounded-pill px-3 fw-bold ${
                  demoLoading ? "" : "d-none"
                }`}
                style={{ fontSize: "0.75rem" }}
                disabled={demoLoading}
                aria-hidden={!demoLoading}
              >
                {demoLoading ? "Connecting..." : "Loading..."}
              </button>
              <span
                id="chat-status"
                className={`badge rounded-pill border ${chatStatusBadgeClass}`}
              >
                {displayStatusText}
              </span>
            </div>
          </header>
          ) : null}

          {activeView === "chat" ? (
            <div className={`chat-container ${messages.length === 0 ? "empty-state" : ""}`}>
              <div
                className="chat-scroll-area"
                id="chat-history"
                ref={chatHistoryRef}
                onScroll={handleChatHistoryScroll}
              >
                {messages.length === 0 ? (
                  <div className="empty-state-wrapper fade-in-up">
                    <div className="greeting-section">
                      <div className="greeting-text">
                        <span className="greeting-gradient">{greetingHeadline}</span>
                      </div>
                    </div>
                  </div>
                ) : null}
                {messages.map((message, index) => {
                  const citationPayload = message.role === "assistant" ? buildCitationPayload(message) : null;
                  const sources = citationPayload?.sources ?? readMessageSources(message);
                  const renderedMessageText = citationPayload?.text ?? message.text;
                  const thinkingSteps = readMessageThinkingSteps(message);
                  const mergedThinkingSteps = mergeThinkingSteps(thinkingSteps);
                  const citedIds = extractCitationIds(renderedMessageText);
                  const citedSources = sources.filter((source) => citedIds.has(getSourceDisplayId(source)));
                  const otherSources = sources.filter((source) => !citedIds.has(getSourceDisplayId(source)));
                  const orderedUsedSources = [...(citedSources.length ? citedSources : sources)].sort(
                    (left, right) => getSourceDisplayId(left) - getSourceDisplayId(right),
                  );
                  const orderedOtherSources = [...otherSources].sort(
                    (left, right) => getSourceDisplayId(left) - getSourceDisplayId(right),
                  );
                  const unusedExpanded = Boolean(expandedUnusedRefs[index]);
                  const messageKey = `${message.role}-${index}-${message.timestamp ?? "msg"}`;
                  const exporting = exportingMessageKey === messageKey;
                  const isStreamingAssistant =
                    message.role === "assistant" && sending && index === messages.length - 1;
                  const isThinkingExpanded =
                    expandedThinkingByMessage[messageKey] ?? isStreamingAssistant;
                  const renderedAssistantHtml =
                    message.role === "assistant"
                      ? renderChatMarkdown(renderedMessageText, {
                          enhanceBlocks: true,
                          copyButtonLabel: t("chat_copy_button_title", "Copy"),
                          copyCodeButtonAriaLabel: t("chat_copy_code_button_title", "Copy code"),
                          copyTableButtonAriaLabel: t("chat_copy_table_button_title", "Copy table"),
                          enableInteractiveControls: !isStreamingAssistant,
                        })
                      : "";
                  return (
                    <article
                      key={messageKey}
                      className={`chat-bubble ${message.role} fade-in-up`}
                      data-message-idx={index}
                    >
                      {message.role === "assistant" && mergedThinkingSteps.length ? (
                        <div className={`process-container ${isThinkingExpanded ? "" : "collapsed"}`.trim()}>
                          <div
                            className="process-header"
                            role="button"
                            tabIndex={0}
                            onClick={() =>
                              setExpandedThinkingByMessage((previous) => ({
                                ...previous,
                                [messageKey]: !isThinkingExpanded,
                              }))
                            }
                            onKeyDown={(event) => {
                              if (event.key === "Enter" || event.key === " ") {
                                event.preventDefault();
                                setExpandedThinkingByMessage((previous) => ({
                                  ...previous,
                                  [messageKey]: !isThinkingExpanded,
                                }));
                              }
                            }}
                          >
                            <span>{t("chat_show_thinking", "显示思考")}</span>
                            <svg
                              className="process-chevron"
                              width="16"
                              height="16"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="2.5"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            >
                              <polyline points="6 9 12 15 18 9" />
                            </svg>
                          </div>
                          <div
                            className="process-body"
                            ref={isStreamingAssistant ? (node) => bindThinkingBodyRef(node, isStreamingAssistant) : undefined}
                            onScroll={isStreamingAssistant ? handleThinkingBodyScroll : undefined}
                          >
                            {mergedThinkingSteps.map((step, stepIndex) => (
                              <div
                                key={`${messageKey}-${step.name}-${stepIndex}`}
                                className="process-step"
                                data-step-name={step.name}
                              >
                                <div className="step-title">
                                  {isStreamingAssistant && !step.completed ? (
                                    <span className="step-spinner" />
                                  ) : null}
                                  <span>{step.name}</span>
                                </div>
                                {step.tokens ? (
                                  <div className="step-content-stream">
                                    <span>{step.tokens}</span>
                                  </div>
                                ) : null}
                                {step.output ? <div className="step-details">{step.output}</div> : null}
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : null}
                      <div
                        className="msg-content"
                        onClick={message.role === "assistant" ? handleMessageContentClick : undefined}
                      >
                        {message.role === "assistant" ? (
                          <div
                            dangerouslySetInnerHTML={{
                              __html: renderedAssistantHtml,
                            }}
                          />
                        ) : (
                          <div
                            dangerouslySetInnerHTML={{
                              __html: renderUserTextAsHtml(message.text),
                            }}
                          />
                        )}
                      </div>
                      {message.role === "assistant" && !isStreamingAssistant && message.text.trim() ? (
                        <div className="chat-copy-row">
                          <button
                            type="button"
                            className={`chat-copy-btn ${copiedMessageKey === messageKey ? "copied" : ""}`.trim()}
                            title={t("chat_copy_button_title", "Copy")}
                            aria-label={t("chat_copy_button_title", "Copy")}
                            onClick={() => void handleCopyMessage(renderedMessageText, messageKey)}
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
                              <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                            </svg>
                          </button>
                          <button
                            type="button"
                            className={`chat-download-btn ${
                              exporting || downloadedMessageKey === messageKey ? "downloaded" : ""
                            }`}
                            title={t("chat_export_button_title", "Export")}
                            aria-label={t("chat_export_button_title", "Export")}
                            disabled={exporting}
                            onClick={() => void handleExportDocx(message, index)}
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
                              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                              <polyline points="7 10 12 15 17 10" />
                              <line x1="12" y1="15" x2="12" y2="3" />
                            </svg>
                          </button>
                        </div>
                      ) : null}
                      {!isStreamingAssistant && sources.length ? (
                        <div className="reference-container">
                          <div className="ref-header">
                            {t("chat_cited_references", "Cited References")} ({citedSources.length || sources.length})
                          </div>
                          <div className="ref-list">
                            {orderedUsedSources.map((source) => {
                              const sourceDisplayId = getSourceDisplayId(source);
                              const previewText = getReferencePreviewText(source);
                              return (
                                <div
                                  key={`source-${source.id}-${sourceDisplayId}`}
                                  role="button"
                                  tabIndex={0}
                                  className={`ref-item used ${
                                    activeReferenceKey === `${index}:${sourceDisplayId}`
                                      ? "active-highlight"
                                      : ""
                                  }`.trim()}
                                  data-source-id={String(sourceDisplayId)}
                                  title={previewText}
                                  onClick={() => openSourceDetail(index, sourceDisplayId)}
                                  onKeyDown={(event) => {
                                    if (event.key === "Enter" || event.key === " ") {
                                      event.preventDefault();
                                      openSourceDetail(index, sourceDisplayId);
                                    }
                                  }}
                                >
                                  <span className="ref-id">[{sourceDisplayId}]</span>
                                  <span className="ref-title">{previewText}</span>
                                </div>
                              );
                            })}
                          </div>
                          {orderedOtherSources.length ? (
                            <div className={`unused-refs-section ${unusedExpanded ? "" : "collapsed"}`.trim()}>
                              <div
                                className="ref-header unused-header"
                                role="button"
                                tabIndex={0}
                                onClick={() =>
                                  setExpandedUnusedRefs((previous) => ({
                                    ...previous,
                                    [index]: !unusedExpanded,
                                  }))
                                }
                                onKeyDown={(event) => {
                                  if (event.key === "Enter" || event.key === " ") {
                                    event.preventDefault();
                                    setExpandedUnusedRefs((previous) => ({
                                      ...previous,
                                      [index]: !unusedExpanded,
                                    }));
                                  }
                                }}
                              >
                                <span>{t("chat_other_retrieved", "Other Retrieved")} ({orderedOtherSources.length})</span>
                                <span className="toggle-icon">▶</span>
                              </div>
                              <div className="ref-list unused-list">
                                {orderedOtherSources.map((source) => {
                                  const sourceDisplayId = getSourceDisplayId(source);
                                  const previewText = getReferencePreviewText(source);
                                  return (
                                    <div
                                      key={`unused-source-${source.id}-${sourceDisplayId}`}
                                      role="button"
                                      tabIndex={0}
                                      className={`ref-item unused ${
                                        activeReferenceKey === `${index}:${sourceDisplayId}`
                                          ? "active-highlight"
                                          : ""
                                      }`.trim()}
                                      data-source-id={String(sourceDisplayId)}
                                      title={previewText}
                                      onClick={() => openSourceDetail(index, sourceDisplayId)}
                                      onKeyDown={(event) => {
                                        if (event.key === "Enter" || event.key === " ") {
                                          event.preventDefault();
                                          openSourceDetail(index, sourceDisplayId);
                                        }
                                      }}
                                    >
                                      <span className="ref-id">[{sourceDisplayId}]</span>
                                      <span className="ref-title">{previewText}</span>
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          ) : null}
                        </div>
                      ) : null}
                    </article>
                  );
                })}
              </div>

              <div className="chat-input-wrapper">
                <form id="chat-form" className="chat-input-container shadow-sm" onSubmit={handleSend}>
                  <textarea
                    id="chat-input"
                    value={input}
                    onChange={(event) => setInput(event.target.value)}
                    onKeyDown={handleComposerKeyDown}
                    onCompositionStart={() => setInputComposing(true)}
                    onCompositionEnd={() => setInputComposing(false)}
                    placeholder={t("placeholder_chat_input", "Ask UltraRAG")}
                    rows={1}
                    className="form-control chat-input"
                  />
                  <div className="actions-row">
                    <div className="left-actions">
                      <div className={`kb-dropdown-wrapper ${kbDropdownOpen ? "open" : ""}`} ref={kbDropdownRef}>
                        <button
                          type="button"
                          className={`kb-dropdown-trigger ${selectedCollectionName ? "active" : ""}`}
                          id="kb-dropdown-trigger"
                          onClick={toggleKbDropdown}
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
                            className="kb-icon-svg"
                          >
                            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                          </svg>
                          <span id="kb-label-text">
                            {selectedCollection ? kbLabel(selectedCollection) : t("knowledge_base", "Knowledge Base")}
                          </span>
                          <span
                            id="kb-clear-btn"
                            className="kb-clear-btn"
                            style={{
                              display: selectedCollectionName ? "inline-flex" : "none",
                              marginLeft: "4px",
                              color: "#999",
                              cursor: "pointer",
                              alignItems: "center",
                            }}
                            onClick={(event) => {
                              event.stopPropagation();
                              selectKbCollection("");
                            }}
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
                              <line x1="18" y1="6" x2="6" y2="18" />
                              <line x1="6" y1="6" x2="18" y2="18" />
                            </svg>
                          </span>
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="12"
                            height="12"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            className="kb-chevron"
                          >
                            <polyline points="6 9 12 15 18 9" />
                          </svg>
                        </button>
                        <div className="kb-dropdown-menu" id="kb-dropdown-menu">
                          {visibleKbCollections.map((collection) => (
                            <div
                              key={collection.name}
                              role="button"
                              tabIndex={0}
                              className={`kb-dropdown-item ${
                                selectedCollectionName === collection.name ? "selected" : ""
                              }`}
                              onClick={() => selectKbCollection(collection.name)}
                              onKeyDown={(event) => {
                                if (event.key === "Enter" || event.key === " ") {
                                  event.preventDefault();
                                  selectKbCollection(collection.name);
                                }
                              }}
                            >
                              <span className="kb-item-check">✓</span>
                              <span className="kb-item-text">{kbLabel(collection)}</span>
                            </div>
                          ))}
                        </div>
                        <select
                          id="chat-collection-select"
                          className="kb-select-hidden"
                          value={selectedCollectionName}
                          onChange={(event) => selectKbCollection(event.target.value)}
                        >
                          <option value="">{t("no_knowledge_base", "No Knowledge Base")}</option>
                          {visibleKbCollections.map((collection) => (
                            <option key={collection.name} value={collection.name}>
                              {kbLabel(collection)}
                            </option>
                          ))}
                        </select>
                      </div>

                      <button
                        type="button"
                        className={`bg-mode-toggle ${backgroundMode ? "active" : ""}`}
                        id="bg-mode-toggle"
                        title={t("background", "Background")}
                        onClick={() => setBackgroundMode((previous) => !previous)}
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
                          className="bg-icon-svg"
                        >
                          <line x1="22" y1="2" x2="11" y2="13" />
                          <polygon points="22 2 15 22 11 13 2 9 22 2" />
                        </svg>
                        <span id="bg-label-text">{t("background", "Background")}</span>
                      </button>
                    </div>

                    <div className="right-actions">
                      <button
                        className={`btn btn-send ${sending ? "stop" : ""}`}
                        type="submit"
                        id="chat-send"
                        title={
                          sending
                            ? t("chat_stop_button_title", "Stop")
                            : t("chat_send_button_title", "Send")
                        }
                      >
                        <span id="chat-send-icon" className="send-icon-wrapper">
                          {sending ? (
                            <span className="icon-stop" />
                          ) : (
                            <svg
                              className="icon-send"
                              xmlns="http://www.w3.org/2000/svg"
                              width="20"
                              height="20"
                              viewBox="0 0 24 24"
                              fill="none"
                              stroke="currentColor"
                              strokeWidth="2"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            >
                              <line x1="12" y1="19" x2="12" y2="5" />
                              <polyline points="5 12 12 5 19 12" />
                            </svg>
                          )}
                        </span>
                      </button>
                    </div>
                  </div>
                </form>
              </div>
            </div>
          ) : null}

          {activeView === "kb" ? (
            <div className="kb-main" id="kb-main-view">
              <header className="view-header border-bottom px-4 py-3 d-flex justify-content-between align-items-center bg-white sticky-top">
                <div className="subview-header-main d-flex align-items-center gap-2">
                  <button
                    type="button"
                    className="btn btn-icon chat-mobile-menu-btn d-md-none"
                    data-mobile-sidebar-toggle="true"
                    aria-label={t("chat_sidebar_toggle_title", "Toggle Sidebar")}
                    onClick={toggleMobileSidebar}
                  >
                    ☰
                  </button>
                  <h3 className="m-0 fs-5 fw-bold text-dark subview-header-title">
                    {t("kb_collections", "Knowledge Base")}
                  </h3>
                </div>
                <div className="kb-header-actions d-flex gap-2 align-items-center flex-wrap justify-content-end">
                  <div className="kb-connection-group">
                    <button
                      type="button"
                      className="kb-conn-chip"
                      id="db-connection-chip"
                      title={`${t("kb_endpoint", "Endpoint")}: ${kbUriDisplay}`}
                      onClick={openKbConfigModal}
                    >
                      <span id="db-connection-status" className={`kb-conn-dot ${kbConnectionClass}`} />
                      <span id="db-connection-text" className="kb-conn-text">
                        {kbConnectionText}
                      </span>
                    </button>
                    <small className="kb-conn-uri" id="db-uri-display">
                      {kbUriDisplay}
                    </small>
                  </div>

                  <button
                    type="button"
                    className="btn btn-outline-secondary btn-sm"
                    disabled={kbConfigBusy}
                    onClick={openKbConfigModal}
                  >
                    {t("kb_configure_db", "Configure DB")}
                  </button>

                  <button
                    type="button"
                    className="btn btn-dark btn-sm px-3 d-flex align-items-center gap-2 mb-0"
                    onClick={openKbImportModal}
                  >
                    <Upload size={16} />
                    <span>{t("kb_new_collection", "New Knowledge Base")}</span>
                  </button>
                </div>
              </header>

              <div className="kb-wrapper p-4 bg-light">
                {kbMessage ? <p className="react-side-panel-note mb-3">{kbMessage}</p> : null}
                {kbLoading ? (
                  <p className="react-side-panel-note mb-3">{t("kb_connecting", "Connecting")}</p>
                ) : null}

                <div id="bookshelf-grid" className="bookshelf-grid">
                  {visibleKbCollections.length === 0 ? (
                    <div className="col-12 text-center py-5 text-muted" style={{ gridColumn: "1 / -1" }}>
                      <div style={{ fontSize: "3rem", marginBottom: "1rem", opacity: 0.3 }}>📚</div>
                      <h5>{t("kb_library_empty_title", "No collections yet")}</h5>
                      <p>{t("kb_library_empty_hint", "Create a knowledge base to get started.")}</p>
                    </div>
                  ) : (
                    visibleKbCollections
                      .slice()
                      .sort((left, right) =>
                        (left.display_name || left.name || "").localeCompare(
                          right.display_name || right.name || "",
                          "en",
                          { sensitivity: "base" },
                        ),
                      )
                      .map((collection) => {
                        const displayName = kbLabel(collection);
                        const countText =
                          collection.count !== undefined
                            ? `${collection.count} ${t("kb_vectors", "vectors")}`
                            : t("kb_ready", "Ready");
                        const colors = pickKbColors(displayName || collection.name || "collection");
                        return (
                          <div
                            key={`collection-card-${collection.name}`}
                            role="button"
                            tabIndex={0}
                            className="collection-card kb-card"
                            onClick={() => void handleOpenVisibility(collection)}
                            onKeyDown={(event) => {
                              if (event.key === "Enter" || event.key === " ") {
                                event.preventDefault();
                                void handleOpenVisibility(collection);
                              }
                            }}
                          >
                            <div className="kb-card-main">
                              <div
                                className="kb-icon-box"
                                style={{ backgroundColor: colors.bg, color: colors.text }}
                              >
                                {getKbInitial(displayName || collection.name || "C")}
                              </div>
                              <div className="kb-info-box">
                                <div className="kb-card-title" title={displayName}>
                                  {displayName}
                                </div>
                                <div className="kb-meta-count">{countText}</div>
                              </div>
                              <button
                                type="button"
                                className="btn-delete-book"
                                title={t("kb_delete_collection", "Delete collection")}
                                onClick={(event) => {
                                  event.stopPropagation();
                                  void handleDeleteKbItem({ ...collection, category: "collection" });
                                }}
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
                                  <polyline points="3 6 5 6 21 6" />
                                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                                </svg>
                              </button>
                            </div>
                          </div>
                        );
                      })
                  )}
                </div>

              </div>
            </div>
          ) : null}

          {activeView === "explore" ? (
            <div className="explore-main" id="explore-main-view">
              <header className="view-header border-bottom px-4 py-3 d-flex justify-content-between align-items-center bg-white sticky-top">
                <div className="subview-header-main d-flex align-items-center gap-2">
                  <button
                    type="button"
                    className="btn btn-icon chat-mobile-menu-btn d-md-none"
                    data-mobile-sidebar-toggle="true"
                    aria-label={t("chat_sidebar_toggle_title", "Toggle Sidebar")}
                    onClick={toggleMobileSidebar}
                  >
                    ☰
                  </button>
                  <h3 className="m-0 fs-5 fw-bold text-dark subview-header-title">
                    {t("explore_title", "Explore")}
                  </h3>
                </div>
              </header>
              <div className="explore-wrapper p-4 bg-light">
                <p className="explore-subtitle mb-3">
                  {t("explore_subtitle", "Browse UltraRAG feature extensions here.")}
                </p>
                <div className="explore-grid">
                  <button
                    type="button"
                    id="explore-feature-memory"
                    className="explore-feature-card"
                    onClick={() => switchView("memory")}
                  >
                    <div className="explore-feature-icon">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        strokeWidth="1.5"
                        stroke="currentColor"
                        width="20"
                        height="20"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M9.5 3a3.5 3.5 0 0 0-3.5 3.5V8a2.5 2.5 0 0 0 0 5v1.5A3.5 3.5 0 0 0 9.5 18H10v1a2 2 0 1 0 4 0v-1h.5a3.5 3.5 0 0 0 3.5-3.5V13a2.5 2.5 0 0 0 0-5V6.5A3.5 3.5 0 0 0 14.5 3h-5z"
                        />
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v12" />
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 9h2.5" />
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 12h3" />
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 15h2.5" />
                      </svg>
                    </div>
                    <div className="explore-feature-content">
                      <div className="explore-feature-title">
                        {t("explore_feature_memory_title", "Memory")}
                      </div>
                      <div className="explore-feature-desc">
                        {t(
                          "explore_feature_memory_desc",
                          "Manage global memory and sync working memory to KB.",
                        )}
                      </div>
                    </div>
                    <span className="explore-feature-arrow" aria-hidden="true">
                      →
                    </span>
                  </button>

                  <div className="explore-feature-card explore-feature-card-muted">
                    <div className="explore-feature-icon">
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        strokeWidth="1.5"
                        stroke="currentColor"
                        width="20"
                        height="20"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v12m6-6H6" />
                      </svg>
                    </div>
                    <div className="explore-feature-content">
                      <div className="explore-feature-title">
                        {t("explore_feature_more_title", "More Features")}
                      </div>
                      <div className="explore-feature-desc">
                        {t(
                          "explore_feature_more_desc",
                          "More capabilities will be available here in the future.",
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : null}

          {activeView === "memory" ? (
            <div className="memory-main" id="memory-main-view">
              <header className="view-header border-bottom px-4 py-3 d-flex justify-content-between align-items-center bg-white sticky-top">
                <div className="subview-header-main d-flex align-items-center gap-2">
                  <button
                    type="button"
                    className="btn btn-icon chat-mobile-menu-btn d-md-none"
                    data-mobile-sidebar-toggle="true"
                    aria-label={t("chat_sidebar_toggle_title", "Toggle Sidebar")}
                    onClick={toggleMobileSidebar}
                  >
                    ☰
                  </button>
                  <h3 className="m-0 fs-5 fw-bold text-dark subview-header-title">{t("memory", "Memory")}</h3>
                </div>
                <div className="d-flex align-items-center gap-2">
                  <span
                    id="memory-status"
                    data-state={memoryStatusState}
                    className="badge rounded-pill bg-light text-dark border"
                  >
                    {memoryStatusText}
                  </span>
                </div>
              </header>

              <div className="memory-wrapper p-4 bg-light">
                <section className="memory-panel memory-panel-global">
                  <div className="memory-panel-header">
                    <div className="memory-panel-header-top">
                      <h4 className="memory-panel-title">{t("memory_global_title", "Global Memory")}</h4>
                      <div className="memory-panel-actions">
                        <button
                          type="button"
                          className="btn btn-dark btn-sm px-3 d-flex align-items-center gap-2"
                          id="memory-save-btn"
                          disabled={memoryBusy}
                          onClick={() => void handleSaveMemory()}
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
                            <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
                            <polyline points="17 21 17 13 7 13 7 21" />
                            <polyline points="7 3 7 8 15 8" />
                          </svg>
                          <span>{t("common_save", "Save")}</span>
                        </button>
                      </div>
                    </div>
                  </div>
                  <textarea
                    id="memory-editor"
                    className="memory-editor"
                    spellCheck={false}
                    placeholder={t("memory_editor_placeholder", "# MEMORY\n\nWrite persistent notes here.")}
                    value={memoryContent}
                    onChange={(event) => setMemoryContent(event.target.value)}
                  />
                </section>

                <section className="memory-panel memory-panel-workkb">
                  <div className="memory-panel-header">
                    <div className="memory-panel-header-top">
                      <h4 className="memory-panel-title">
                        {t("memory_working_kb_title", "Working Memory")}
                      </h4>
                      <div className="memory-panel-actions">
                        <button
                          type="button"
                          className="btn btn-outline-secondary btn-sm px-3 d-flex align-items-center gap-2"
                          id="memory-sync-btn"
                          disabled={memoryBusy}
                          onClick={() => void handleSyncMemory()}
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
                            <polyline points="23 4 23 10 17 10" />
                            <polyline points="1 20 1 14 7 14" />
                            <path d="M3.51 9a9 9 0 0 1 14.13-3.36L23 10M1 14l5.36 4.36A9 9 0 0 0 20.49 15" />
                          </svg>
                          <span>{t("memory_sync_to_kb", "Sync to KB")}</span>
                        </button>
                        <button
                          type="button"
                          className="btn btn-outline-danger btn-sm px-3 d-flex align-items-center gap-2"
                          id="memory-clear-btn"
                          disabled={memoryBusy}
                          onClick={() => void handleClearMemoryVectors()}
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
                            <polyline points="3 6 5 6 21 6" />
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                          </svg>
                          <span>{t("memory_clear_vectors_action", "Clear Memory Vectors")}</span>
                        </button>
                      </div>
                    </div>
                    <p className="memory-panel-subtitle">
                      {t(
                        "memory_working_kb_hint",
                        "Internal working-memory collection cards for current user.",
                      )}
                    </p>
                  </div>

                  <div id="memory-kb-cards" className="memory-kb-cards">
                    {memoryKbError ? (
                      <div className="memory-kb-empty">
                        <div className="memory-kb-empty-title">{t("status_error", "Error")}</div>
                        <div className="memory-kb-empty-desc">{memoryKbError}</div>
                      </div>
                    ) : memoryKbCollections.length ? (
                      memoryKbCollections
                        .slice()
                        .sort((left, right) =>
                          (left.display_name || left.name || "").localeCompare(
                            right.display_name || right.name || "",
                            "en",
                            { sensitivity: "base" },
                          ),
                        )
                        .map((collection) => (
                          <div key={collection.name} className="memory-kb-card">
                            <div className="memory-kb-card-title" title={kbLabel(collection)}>
                              {kbLabel(collection)}
                            </div>
                            <div className="memory-kb-card-meta">
                              <span>
                                {collection.count !== undefined
                                  ? `${collection.count} ${t("kb_vectors", "vectors")}`
                                  : t("kb_ready", "Ready")}
                              </span>
                              <code>{collection.name}</code>
                            </div>
                          </div>
                        ))
                    ) : (
                      <div className="memory-kb-empty">
                        <div className="memory-kb-empty-title">
                          {t("memory_working_kb_empty_title", "No working-memory collection yet")}
                        </div>
                        <div className="memory-kb-empty-desc">
                          {formatTemplate(
                            t(
                              "memory_working_kb_empty_desc",
                              "No indexed working memory for this user. Click \"Sync to KB\" to create: {collection}",
                            ),
                            { collection: `user_${normalizeMemoryUserId(userId)}` },
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </section>
                {memoryMessage ? <p className="react-side-panel-note">{memoryMessage}</p> : null}
              </div>
            </div>
          ) : null}
        </section>

        <aside
          className={`chat-detail-sidebar ${detailPanelOpen ? "show" : ""}`}
          id="source-detail-panel"
          ref={detailPanelRef}
        >
          <div className="detail-header">
            <h3 className="detail-title">{sourceDetailTitle}</h3>
            <button
              type="button"
              className="btn btn-icon btn-sm"
              onClick={() => {
                setDetailPanelOpen(false);
                setDetailSourceDisplayId(null);
              }}
              aria-label={t("common_close", "Close")}
            >
              ✕
            </button>
          </div>
          <div
            className="detail-content"
            id="source-detail-content"
            ref={detailContentRef}
            dangerouslySetInnerHTML={{ __html: sourceDetailHtml }}
          />
        </aside>
      </main>
      </section>

      <div
        id="background-tasks-panel"
        className={`bg-tasks-panel ${backgroundPanelOpen && activeView === "chat" ? "" : "d-none"}`.trim()}
      >
        <div className="bg-tasks-header">
          <h6>
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
              className="me-2"
            >
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
            <span>{t("bg_tasks_title", "Background Tasks")}</span>
          </h6>
          <div className="d-flex gap-1 align-items-center">
            <button
              type="button"
              title={t("bg_tasks_refresh", "Refresh")}
              aria-label={t("bg_tasks_refresh", "Refresh")}
              onClick={() => void loadBackgroundTaskData()}
            >
              ↻
            </button>
            <button
              type="button"
              title={t("bg_tasks_clear_completed", "Clear completed")}
              onClick={() => void handleClearCompletedTasks()}
            >
              {t("bg_tasks_clear", "Clear")}
            </button>
            <button
              type="button"
              className="bg-panel-close-btn"
              title={t("bg_tasks_close", "Close")}
              aria-label={t("bg_tasks_close", "Close")}
              onClick={toggleBackgroundPanel}
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
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        </div>
        <div className="bg-tasks-body" id="bg-tasks-list">
          {backgroundTasks.length ? (
            backgroundTasks.map((task) => {
              const statusText =
                task.status === "running"
                  ? t("bg_task_running", "Running")
                  : task.status === "completed"
                    ? t("bg_task_completed", "Completed")
                    : t("bg_task_failed", "Failed");
              const taskQuestion = task.full_question || task.question || "Background Task";
              return (
                <div
                  key={task.task_id}
                  className={`bg-task-item ${task.status}`.trim()}
                  role="button"
                  tabIndex={0}
                  onClick={() => openBackgroundTaskDetail(task.task_id)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      openBackgroundTaskDetail(task.task_id);
                    }
                  }}
                >
                  <div className="bg-task-header">
                    <div className="bg-task-question">{taskQuestion}</div>
                    <span className={`bg-task-status ${task.status}`.trim()}>{statusText}</span>
                  </div>
                  <div className="bg-task-meta">
                    <span>{task.pipeline_name}</span>
                    <span>{formatClockTime(task.created_at)}</span>
                  </div>
                  {task.status === "completed" && task.result_preview ? (
                    <div className="bg-task-preview">{task.result_preview}</div>
                  ) : null}
                </div>
              );
            })
          ) : (
            <div className="text-muted text-center py-4 small">{t("bg_tasks_empty", "No background tasks")}</div>
          )}
        </div>
      </div>

      <button
        id="bg-tasks-fab"
        type="button"
        className={`bg-tasks-fab ${showBackgroundFab ? "" : "d-none"}`.trim()}
        title={t("bg_tasks_title", "Background Tasks")}
        aria-label={t("bg_tasks_title", "Background Tasks")}
        onClick={toggleBackgroundPanel}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="bg-icon-svg"
        >
          <line x1="22" y1="2" x2="11" y2="13" />
          <polygon points="22 2 15 22 11 13 2 9 22 2" />
        </svg>
        <span id="bg-tasks-count" className={`bg-tasks-count ${runningBackgroundCount > 0 ? "" : "d-none"}`.trim()}>
          {runningBackgroundCount}
        </span>
      </button>

      <div
        ref={chatSessionContextMenuRef}
        className={`chat-session-context-menu ${chatSessionContextMenu.open ? "" : "d-none"}`.trim()}
        style={
          chatSessionContextMenu.open
            ? {
                left: `${chatSessionContextMenu.x}px`,
                top: `${chatSessionContextMenu.y}px`,
              }
            : undefined
        }
        onClick={(event) => event.stopPropagation()}
      >
        <button
          type="button"
          className="chat-session-context-item"
          onClick={() => openRenameSessionDialog(chatSessionContextMenu.sessionId)}
          disabled={!chatSessionContextMenu.sessionId}
        >
          <span>{t("common_rename", "Rename")}</span>
        </button>
        <button
          type="button"
          className="chat-session-context-item text-danger"
          onClick={() => void handleDeleteSession(chatSessionContextMenu.sessionId)}
          disabled={!chatSessionContextMenu.sessionId}
        >
          <span>{t("common_delete", "Delete")}</span>
        </button>
      </div>

      <dialog
        id="chat-rename-modal"
        className="unified-modal"
        ref={chatRenameDialogRef}
        onClose={() => {
          setChatRenameModalOpen(false);
          setChatRenameTargetSessionId("");
        }}
      >
        <div className="unified-modal-content">
          <div className="unified-modal-icon info">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="28"
              height="28"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M12 20h9" />
              <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" />
            </svg>
          </div>
          <h4 className="unified-modal-title">{t("chat_rename_title", "Rename Chat")}</h4>
          <p className="unified-modal-message">{t("chat_rename_prompt", "Enter a new name for this chat:")}</p>
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void submitRenameSession();
            }}
          >
            <input
              ref={chatRenameInputRef}
              className="form-control mb-3"
              type="text"
              value={chatRenameDraft}
              onChange={(event) => setChatRenameDraft(event.target.value)}
              placeholder={t("chat_rename_placeholder", "e.g., My important conversation")}
            />
            <div className="unified-modal-actions">
              <button
                type="button"
                className="btn unified-modal-btn unified-modal-btn-secondary"
                onClick={() => {
                  setChatRenameModalOpen(false);
                  setChatRenameTargetSessionId("");
                }}
              >
                {t("common_cancel", "Cancel")}
              </button>
              <button type="submit" className="btn unified-modal-btn unified-modal-btn-primary">
                {t("common_confirm", "Confirm")}
              </button>
            </div>
          </form>
        </div>
      </dialog>

      <dialog
        id="import-modal"
        ref={kbImportDialogRef}
        onClose={closeKbImportModal}
        style={{
          width: "90vw",
          maxWidth: "1200px",
          height: "85vh",
          border: "none",
          borderRadius: "16px",
          padding: 0,
          boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.25)",
        }}
      >
        <div className="d-flex flex-column h-100">
          <div className="modal-header px-4 py-3 border-bottom d-flex justify-content-between align-items-center bg-white">
            <div>
              <h5 className="m-0 fw-bold">{t("kb_import_knowledge", "Import Knowledge")}</h5>
              <small className="text-muted">
                {t("kb_import_subtitle", "Process raw documents into vector index.")}
              </small>
            </div>
            <div className="d-flex gap-2 flex-wrap justify-content-end">
              <button
                type="button"
                className="btn btn-light btn-sm text-danger"
                title={t("kb_clear_temp_files", "Clear all temporary files")}
                onClick={() => void handleClearKbStaging()}
              >
                {t("kb_delete_all", "Delete All")}
              </button>
              <button type="button" className="btn-close" onClick={closeKbImportModal} />
            </div>
          </div>

          <div className="modal-body bg-light p-4 overflow-hidden d-flex flex-column">
            <div
              id="task-status-bar"
              className={`status-bar ${kbTaskStatusVisible ? "" : "hidden"} mb-3 mx-auto shadow-sm`.trim()}
            >
              <span id="task-spinner" className={`spinner ${kbTaskProgressVisible ? "hidden" : ""}`.trim()} />
              <div
                id="task-progress-wrapper"
                className={`progress-ring-wrapper ${kbTaskProgressVisible ? "" : "hidden"}`.trim()}
              >
                <svg className="progress-ring" width="24" height="24">
                  <circle
                    className="progress-ring__circle-bg"
                    stroke="#dbeafe"
                    strokeWidth="3"
                    fill="transparent"
                    r="9"
                    cx="12"
                    cy="12"
                  />
                  <circle
                    className="progress-ring__circle"
                    id="task-progress-circle"
                    stroke="#1d4ed8"
                    strokeWidth="3"
                    fill="transparent"
                    r="9"
                    cx="12"
                    cy="12"
                    style={{
                      strokeDasharray: `${2 * Math.PI * 9}`,
                      strokeDashoffset: `${2 * Math.PI * 9 - (Math.max(0, Math.min(100, kbTaskProgress)) / 100) * 2 * Math.PI * 9}`,
                    }}
                  />
                </svg>
                <span id="task-progress-text" className="progress-text">
                  {Math.round(kbTaskProgress)}%
                </span>
              </div>
              <span id="task-msg">{kbTaskStatusMessage}</span>
            </div>

            <div className="kb-pipeline-grid flex-grow-1">
              <div className="pipeline-card">
                <div className="card-head d-flex justify-content-between align-items-center">
                  <div className="head-title d-flex align-items-center">
                    <span className="step-badge rounded-circle d-flex align-items-center justify-content-center">1</span>
                    <span className="fw-bold text-dark small">{t("kb_raw_files", "Raw Files")}</span>
                  </div>
                  <input
                    type="file"
                    id="file-upload"
                    ref={kbUploadInputRef}
                    multiple
                    hidden
                    onChange={(event) => {
                      void handleKbUpload(event.target.files);
                      event.currentTarget.value = "";
                    }}
                  />
                  <button
                    type="button"
                    className="btn btn-dark btn-sm py-0 px-2"
                    style={{ fontSize: "0.8rem" }}
                    onClick={() => kbUploadInputRef.current?.click()}
                  >
                    {t("kb_upload", "Upload")}
                  </button>
                </div>
                <div className="card-body-scroll" id="list-raw">
                  {renderKbPipelineList(kbData.raw, "build_text_corpus", t("kb_action_parse", "Parse"))}
                </div>
              </div>

              <div className="pipeline-card">
                <div className="card-head d-flex justify-content-between align-items-center">
                  <div className="head-title d-flex align-items-center">
                    <span className="step-badge rounded-circle d-flex align-items-center justify-content-center">2</span>
                    <span className="fw-bold text-dark small">{t("kb_corpus", "Corpus")}</span>
                  </div>
                  <button
                    type="button"
                    className="btn btn-light border btn-sm py-0 px-2 text-muted d-flex align-items-center gap-1"
                    style={{ fontSize: "0.75rem" }}
                    onClick={handleOpenChunkConfigModal}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="12"
                      height="12"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <circle cx="12" cy="12" r="3" />
                      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83a2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33a1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2a2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0a2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2a2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83a2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2a2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0a2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2a2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
                    </svg>
                    <span>{t("settings", "Settings")}</span>
                  </button>
                </div>
                <div className="card-body-scroll" id="list-corpus">
                  {renderKbPipelineList(kbData.corpus, "corpus_chunk", t("kb_action_chunk", "Chunk"))}
                </div>
              </div>

              <div className="pipeline-card">
                <div className="card-head d-flex justify-content-between align-items-center">
                  <div className="head-title d-flex align-items-center">
                    <span className="step-badge rounded-circle d-flex align-items-center justify-content-center">3</span>
                    <span className="fw-bold text-dark small">{t("kb_chunks", "Chunks")}</span>
                  </div>
                  <button
                    type="button"
                    className="btn btn-light border btn-sm py-0 px-2 text-muted d-flex align-items-center gap-1"
                    style={{ fontSize: "0.75rem" }}
                    onClick={handleOpenIndexConfigModal}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="12"
                      height="12"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <circle cx="12" cy="12" r="3" />
                      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83a2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33a1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2a2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0a2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2a2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83a2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2a2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0a2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2a2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
                    </svg>
                    <span>{t("settings", "Settings")}</span>
                  </button>
                </div>
                <div className="card-body-scroll" id="list-chunks">
                  {renderKbPipelineList(kbData.chunks, "milvus_index", t("kb_action_index", "Index"))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </dialog>

      <dialog
        id="milvus-dialog"
        className="apple-modal"
        ref={kbMilvusDialogRef}
        onClose={closeKbMilvusModal}
      >
        <h5 className="mb-3 fw-bold">{t("kb_build_vector_index", "Build Vector Index")}</h5>

        <div className="alert alert-light border mb-3 p-2 small">
          <span>{t("kb_target", "Target")}</span>:{" "}
          <strong id="modal-target-db">{kbUriDisplay || t("kb_not_configured", "Not configured")}</strong>
        </div>

        <div className="mb-3">
          <label className="form-label small text-muted text-uppercase fw-bold" id="idx-collection-label">
            {kbRunIndexMode === "new"
              ? t("kb_collection_name", "Collection Name")
              : t("kb_existing_collection", "Existing Knowledge Base")}
          </label>
          <input
            type="text"
            id="idx-collection"
            className={`form-control ${kbRunIndexMode === "new" ? "" : "d-none"}`.trim()}
            placeholder="e.g. wiki_v1"
            value={kbRunCollectionName}
            onChange={(event) => setKbRunCollectionName(event.target.value)}
          />
          <select
            id="idx-collection-select"
            className={`form-select ${kbRunIndexMode === "new" ? "d-none" : ""}`.trim()}
            value={kbRunCollectionName}
            onChange={(event) => setKbRunCollectionName(event.target.value)}
          >
            <option value="" disabled>
              {t("kb_select_collection_message", "Please select a knowledge base.")}
            </option>
            {visibleKbCollections
              .slice()
              .sort((left, right) =>
                (left.display_name || left.name || "").localeCompare(
                  right.display_name || right.name || "",
                  "en",
                  { sensitivity: "base" },
                ),
              )
              .map((collection) => (
                <option key={collection.name} value={collection.name}>
                  {kbLabel(collection)}
                </option>
              ))}
          </select>
        </div>

        <div className="mb-4">
          <label className="form-label small text-muted text-uppercase fw-bold">{t("kb_mode", "Mode")}</label>
          <select
            id="idx-mode"
            className="form-select"
            value={kbRunIndexMode}
            onChange={(event) => {
              const nextMode = event.target.value as "new" | "append" | "overwrite";
              setKbRunIndexMode(nextMode);
              if (nextMode !== "new" && !kbRunCollectionName) {
                setKbRunCollectionName(visibleKbCollections[0]?.name ?? "");
              }
            }}
          >
            <option value="new">{t("kb_mode_new", "New")}</option>
            <option value="append">{t("kb_mode_append", "Append")}</option>
            <option value="overwrite">{t("kb_mode_overwrite", "Overwrite")}</option>
          </select>
        </div>

        <div className="modal-actions d-flex justify-content-end gap-2">
          <button type="button" className="btn btn-light border btn-sm" onClick={closeKbMilvusModal}>
            {t("kb_cancel", "Cancel")}
          </button>
          <button
            type="button"
            className="btn btn-dark btn-sm"
            onClick={() => {
              void handleConfirmIndexTask();
            }}
          >
            {t("kb_start_indexing", "Start Indexing")}
          </button>
        </div>
      </dialog>

      <dialog
        id="unified-confirm-modal"
        className="unified-modal"
        ref={confirmDialogRef}
        onClose={() => closeConfirmModal(false)}
      >
        <div className="unified-modal-content">
          <div className={`unified-modal-icon ${confirmModalType}`}>
            {confirmModalType === "warning" ? (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="28"
                height="28"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3l-8.47-14.14a2 2 0 0 0-3.42 0z" />
                <line x1="12" y1="9" x2="12" y2="13" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
            ) : confirmModalType === "success" ? (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="28"
                height="28"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M20 6 9 17l-5-5" />
              </svg>
            ) : confirmModalType === "error" ? (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="28"
                height="28"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
            ) : confirmModalType === "info" ? (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="28"
                height="28"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="16" x2="12" y2="12" />
                <line x1="12" y1="8" x2="12.01" y2="8" />
              </svg>
            ) : (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="28"
                height="28"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <circle cx="12" cy="12" r="10" />
                <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
            )}
          </div>
          <h4 className="unified-modal-title">{confirmModalTitle || t("modal_confirm_title", "Confirm")}</h4>
          <p className="unified-modal-message">{confirmModalMessage}</p>
          <div className="unified-modal-actions">
            {confirmModalHideCancel ? null : (
              <button
                type="button"
                className="btn unified-modal-btn unified-modal-btn-secondary"
                onClick={() => closeConfirmModal(false)}
              >
                {confirmModalCancelText || t("common_cancel", "Cancel")}
              </button>
            )}
            <button
              type="button"
              className={`btn unified-modal-btn ${confirmModalDanger ? "unified-modal-btn-danger" : "unified-modal-btn-primary"}`}
              onClick={() => closeConfirmModal(true)}
            >
              {confirmModalConfirmText || t("common_confirm", "Confirm")}
            </button>
          </div>
        </div>
      </dialog>

      <dialog
        id="kb-index-choice-modal"
        className="unified-modal unified-modal--export-choice"
        ref={kbIndexChoiceDialogRef}
        onClose={() => closeKbIndexChoiceModal(null)}
      >
        <div className="unified-modal-content">
          <div className="unified-modal-icon warning">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="28"
              height="28"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3l-8.47-14.14a2 2 0 0 0-3.42 0z" />
              <line x1="12" y1="9" x2="12" y2="13" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
          </div>
          <h4 className="unified-modal-title">{t("kb_name_exists_title", "Name Already Exists")}</h4>
          <p className="unified-modal-message">{kbIndexChoiceMessage}</p>
          <div className="unified-modal-actions unified-modal-actions--export-choice">
            <button
              type="button"
              className="btn unified-modal-btn unified-modal-btn-option unified-modal-btn-primary"
              onClick={() => closeKbIndexChoiceModal("append")}
            >
              <span className="choice-main">{t("kb_mode_append", "Append")}</span>
            </button>
            <button
              type="button"
              className="btn unified-modal-btn unified-modal-btn-option unified-modal-btn-secondary"
              onClick={() => closeKbIndexChoiceModal("overwrite")}
            >
              <span className="choice-main">{t("kb_mode_overwrite", "Overwrite")}</span>
            </button>
            <button
              type="button"
              className="btn unified-modal-btn unified-modal-btn-cancel"
              onClick={() => closeKbIndexChoiceModal(null)}
            >
              {t("common_cancel", "Cancel")}
            </button>
          </div>
        </div>
      </dialog>

      <dialog
        id="chat-export-format-modal"
        className="unified-modal unified-modal--export-choice"
        ref={exportFormatDialogRef}
        onClose={() => closeExportFormatModal(null)}
      >
        <div className="unified-modal-content">
          <div className="unified-modal-icon info">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="28"
              height="28"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
          </div>
          <h4 className="unified-modal-title">{t("chat_export_format_title", "Export Chat")}</h4>
          <p className="unified-modal-message">{t("chat_export_format_message", "Choose an export format:")}</p>
          <div className="unified-modal-actions unified-modal-actions--export-choice">
            <button
              type="button"
              className="btn unified-modal-btn unified-modal-btn-option unified-modal-btn-primary"
              onClick={() => closeExportFormatModal("markdown")}
            >
              <span className="choice-main">{t("chat_export_format_markdown", "Markdown (.md)")}</span>
            </button>
            <button
              type="button"
              className="btn unified-modal-btn unified-modal-btn-option unified-modal-btn-secondary"
              onClick={() => closeExportFormatModal("docx")}
            >
              <span className="choice-main">{t("chat_export_format_docx", "Word Document (.docx)")}</span>
            </button>
            <button
              type="button"
              className="btn unified-modal-btn unified-modal-btn-cancel"
              onClick={() => closeExportFormatModal(null)}
            >
              {t("common_cancel", "Cancel")}
            </button>
          </div>
        </div>
      </dialog>

      <dialog
        id="folder-detail-modal"
        className="apple-modal"
        ref={kbFolderDetailDialogRef}
        onClose={closeKbFolderDetailModal}
      >
        <h5 className="mb-3 fw-bold" id="folder-detail-title">
          {kbInspectTitle || t("kb_folder_contents", "Folder Contents")}
        </h5>
        <div id="folder-detail-list" className="folder-list-container" style={{ maxHeight: "300px", overflowY: "auto" }}>
          {kbInspectLoading ? (
            <div className="text-center text-muted p-3">Loading...</div>
          ) : kbInspectFiles.length > 0 ? (
            kbInspectFiles.map((file) => (
              <div key={`${file.name}-${file.size}`} className="folder-file-row">
                <span className="file-row-icon">📄</span>
                <span className="file-row-name text-truncate">{file.name}</span>
                <span className="text-muted ms-auto" style={{ fontSize: "0.75rem" }}>
                  {(Number(file.size ?? 0) / 1024).toFixed(1)} KB
                </span>
              </div>
            ))
          ) : kbInspectHasHiddenOnly ? (
            <div className="text-center text-muted small mt-3">{t("kb_empty_no_visible", "Empty (No visible files)")}</div>
          ) : (
            <div className="text-center text-muted small mt-3">{t("kb_empty_folder", "Empty Folder")}</div>
          )}
        </div>
        <div className="modal-actions d-flex justify-content-end mt-3">
          <button type="button" className="btn btn-dark btn-sm" onClick={closeKbFolderDetailModal}>
            {t("kb_close", "Close")}
          </button>
        </div>
      </dialog>

      <dialog
        id="chunk-config-modal"
        ref={kbChunkConfigDialogRef}
        onClose={closeKbChunkConfigModal}
        style={{
          margin: "auto",
          border: "none",
          borderRadius: "12px",
          padding: 0,
          boxShadow: "0 10px 25px rgba(0,0,0,0.2)",
          overflow: "visible",
          maxWidth: "90vw",
        }}
      >
        <div className="card" style={{ width: "380px", border: "none" }}>
          <div className="card-header bg-white border-bottom pt-3 pb-2 d-flex justify-content-between align-items-center">
            <h6 className="card-title m-0 fw-bold">{t("kb_chunk_configuration", "Chunk Configuration")}</h6>
            <button type="button" className="btn-close small" onClick={closeKbChunkConfigModal} />
          </div>
          <div className="card-body">
            <div className="mb-3">
              <label className="form-label small fw-bold text-muted">{t("kb_chunk_backend", "Chunk Backend")}</label>
              <select
                className="form-select form-select-sm"
                value={kbChunkConfigDraft.chunk_backend}
                onChange={(event) =>
                  setKbChunkConfigDraft((previous) => ({ ...previous, chunk_backend: event.target.value }))
                }
              >
                <option value="token">token</option>
                <option value="sentence">sentence</option>
                <option value="recursive">recursive</option>
              </select>
            </div>

            <div className="mb-3">
              <label className="form-label small fw-bold text-muted">
                {t("kb_tokenizer_counter", "Tokenizer / Counter")}
              </label>
              <select
                className="form-select form-select-sm"
                value={kbChunkConfigDraft.tokenizer_or_token_counter}
                onChange={(event) =>
                  setKbChunkConfigDraft((previous) => ({
                    ...previous,
                    tokenizer_or_token_counter: event.target.value,
                  }))
                }
              >
                <option value="gpt2">gpt2</option>
                <option value="character">character</option>
                <option value="word">word</option>
              </select>
            </div>

            <div className="mb-3">
              <label className="form-label small fw-bold text-muted">{t("kb_chunk_size", "Chunk Size")}</label>
              <div className="input-group input-group-sm">
                <input
                  type="number"
                  min={1}
                  className="form-control"
                  value={kbChunkConfigDraft.chunk_size}
                  onChange={(event) =>
                    setKbChunkConfigDraft((previous) => ({
                      ...previous,
                      chunk_size: Number(event.target.value || 0),
                    }))
                  }
                />
              </div>
            </div>

            <div className="mb-1">
              <label className="form-label small fw-bold text-muted">{t("kb_use_title", "Use Title")}</label>
              <select
                className="form-select form-select-sm"
                value={kbChunkConfigDraft.use_title ? "true" : "false"}
                onChange={(event) =>
                  setKbChunkConfigDraft((previous) => ({
                    ...previous,
                    use_title: event.target.value === "true",
                  }))
                }
              >
                <option value="true">True</option>
                <option value="false">False</option>
              </select>
              <div className="form-text text-xs">
                {t("kb_use_title_help", "Prepend document title to each chunk.")}
              </div>
            </div>
          </div>
          <div className="card-footer bg-light border-top d-flex justify-content-end gap-2 py-2">
            <button type="button" className="btn btn-sm btn-light border" onClick={closeKbChunkConfigModal}>
              {t("kb_cancel", "Cancel")}
            </button>
            <button
              type="button"
              className="btn btn-sm btn-dark"
              onClick={() => {
                void handleSaveChunkConfig();
              }}
            >
              {t("kb_save_config", "Save Config")}
            </button>
          </div>
        </div>
      </dialog>

      <dialog
        id="index-config-modal"
        ref={kbIndexConfigDialogRef}
        onClose={closeKbIndexConfigModal}
        style={{
          margin: "auto",
          border: "none",
          borderRadius: "12px",
          padding: 0,
          boxShadow: "0 10px 25px rgba(0,0,0,0.2)",
          overflow: "visible",
          maxWidth: "90vw",
        }}
      >
        <div className="card" style={{ width: "420px", border: "none" }}>
          <div className="card-header bg-white border-bottom pt-3 pb-2 d-flex justify-content-between align-items-center">
            <h6 className="card-title m-0 fw-bold">
              {t("kb_embedding_configuration", "Embedding Configuration")}
            </h6>
            <button type="button" className="btn-close small" onClick={closeKbIndexConfigModal} />
          </div>
          <div className="card-body">
            <p className="text-muted small mb-3">
              {t(
                "kb_embedding_desc",
                "Configure the embedding model for vector indexing. Uses OpenAI-compatible API.",
              )}
            </p>
            <div className="mb-3">
              <label className="form-label small fw-bold text-muted">{t("kb_api_key", "API Key")}</label>
              <input
                type="password"
                className="form-control form-control-sm"
                placeholder="sk-..."
                value={kbIndexConfigDraft.api_key}
                onChange={(event) =>
                  setKbIndexConfigDraft((previous) => ({ ...previous, api_key: event.target.value }))
                }
              />
              <div className="form-text text-xs">
                {t("kb_api_key_help", "Your OpenAI or compatible API key.")}
              </div>
            </div>

            <div className="mb-3">
              <label className="form-label small fw-bold text-muted">{t("kb_base_url", "Base URL")}</label>
              <input
                type="text"
                className="form-control form-control-sm"
                placeholder="https://api.openai.com/v1"
                value={kbIndexConfigDraft.base_url}
                onChange={(event) =>
                  setKbIndexConfigDraft((previous) => ({ ...previous, base_url: event.target.value }))
                }
              />
              <div className="form-text text-xs">
                {t("kb_base_url_help", "API endpoint. Change for Azure, local models, etc.")}
              </div>
            </div>

            <div className="mb-1">
              <label className="form-label small fw-bold text-muted">{t("kb_model_name", "Model Name")}</label>
              <input
                type="text"
                className="form-control form-control-sm"
                placeholder="text-embedding-3-small"
                value={kbIndexConfigDraft.model_name}
                onChange={(event) =>
                  setKbIndexConfigDraft((previous) => ({ ...previous, model_name: event.target.value }))
                }
              />
              <div className="form-text text-xs">
                {t("kb_model_name_help", "e.g. text-embedding-3-small, text-embedding-ada-002")}
              </div>
            </div>
          </div>
          <div className="card-footer bg-light border-top d-flex justify-content-end gap-2 py-2">
            <button type="button" className="btn btn-sm btn-light border" onClick={closeKbIndexConfigModal}>
              {t("kb_cancel", "Cancel")}
            </button>
            <button
              type="button"
              className="btn btn-sm btn-dark"
              onClick={() => {
                void handleSaveIndexConfig();
              }}
            >
              {t("kb_save_config", "Save Config")}
            </button>
          </div>
        </div>
      </dialog>

      <dialog
        id="db-config-modal"
        className="apple-modal"
        ref={kbConfigDialogRef}
        onClose={closeKbConfigModal}
      >
        <h5 className="mb-3 fw-bold">{t("kb_database_settings", "Database Settings")}</h5>

        <div className="mb-3">
          <label className="form-label small text-muted text-uppercase fw-bold">
            {t("kb_milvus_uri", "Milvus URI")}
          </label>
          <input
            type="text"
            className="form-control"
            value={kbConfigUri}
            onChange={(event) => setKbConfigUri(event.target.value)}
            placeholder="path/to/local.db OR http://localhost:19530"
            disabled={kbConfigBusy}
          />
          <div className="form-text">
            {t("kb_milvus_uri_help", "Supports local file paths or remote HTTP/TCP connections.")}
          </div>
        </div>

        <div className="mb-4">
          <label className="form-label small text-muted text-uppercase fw-bold">
            {t("kb_token_optional", "Token (Optional)")}
          </label>
          <input
            type="password"
            className="form-control"
            value={kbConfigToken}
            onChange={(event) => setKbConfigToken(event.target.value)}
            placeholder="API Key or user:password"
            disabled={kbConfigBusy}
          />
        </div>

        <div className="modal-actions d-flex justify-content-end gap-2">
          <button type="button" className="btn btn-light border btn-sm" onClick={closeKbConfigModal}>
            {t("kb_cancel", "Cancel")}
          </button>
          <button
            type="button"
            className="btn btn-dark btn-sm"
            disabled={kbConfigBusy}
            onClick={() => {
              void (async () => {
                const saved = await handleSaveKbConfig();
                if (saved) {
                  closeKbConfigModal();
                }
              })();
            }}
          >
            {kbConfigBusy ? t("common_saving", "Saving...") : t("kb_save_connect", "Save & Connect")}
          </button>
        </div>
      </dialog>

      <dialog
        id="kb-visibility-modal"
        className="apple-modal"
        ref={kbVisibilityDialogRef}
        onClose={closeKbVisibilityModal}
      >
        <h5 className="mb-3 fw-bold">{t("kb_visibility_title", "Knowledge Base Visibility")}</h5>
        <div id="kb-visibility-collection-name" className="small text-muted mb-3">
          {kbVisibilityCollection
            ? formatTemplate(t("kb_visibility_collection_hint", "Knowledge Base: {name}"), {
                name: kbLabel(kbVisibilityCollection),
              })
            : ""}
        </div>

        <div className="mb-3">
          <label className="form-label small text-muted text-uppercase fw-bold">
            {t("kb_visibility_mode_label", "Visibility")}
          </label>
          <select
            id="kb-visibility-mode"
            className="form-select"
            value={kbVisibilityData?.visibility ?? "private"}
            onChange={(event) => handleVisibilityModeChange(event.target.value as "private" | "public" | "shared")}
            disabled={!kbVisibilityData?.can_manage}
          >
            <option value="private">{t("kb_visibility_mode_private", "Private (owner only)")}</option>
            <option value="public">{t("kb_visibility_mode_public", "Public (all users)")}</option>
            <option value="shared">{t("kb_visibility_mode_shared", "Shared with selected users")}</option>
          </select>
          <div id="kb-visibility-owner-hint" className="form-text">
            {kbVisibilityOwnerHint}
          </div>
        </div>

        <div
          id="kb-visibility-users-wrap"
          className={`mb-3 ${kbVisibilityData?.visibility === "shared" ? "" : "d-none"}`.trim()}
        >
          <label className="form-label small text-muted text-uppercase fw-bold">
            {t("kb_visibility_users_label", "Visible Users")}
          </label>
          <div id="kb-visibility-users" className="kb-visibility-users">
            {kbVisibilityCandidateUsers.map((username) => {
              const checked = (kbVisibilityData?.visible_users ?? []).includes(username);
              const disabled = !kbVisibilityData?.can_manage;
              return (
                <label
                  key={username}
                  className={`kb-visibility-user-item ${checked ? "is-selected" : ""} ${disabled ? "is-disabled" : ""}`.trim()}
                >
                  <input
                    type="checkbox"
                    className="form-check-input"
                    checked={checked}
                    disabled={disabled}
                    onChange={() => toggleVisibilityUser(username)}
                  />
                  <span className="kb-visibility-user-name">{username}</span>
                </label>
              );
            })}
          </div>
          <div id="kb-visibility-users-empty" className={`form-text ${kbVisibilityCandidateUsers.length ? "d-none" : ""}`.trim()}>
            {t("kb_visibility_users_empty", "No shareable users found")}
          </div>
        </div>

        <div
          id="kb-visibility-readonly-hint"
          className={`form-text text-warning ${kbVisibilityData?.can_manage ? "d-none" : ""}`.trim()}
        >
          {kbVisibilityReadonlyHint}
        </div>

        <div className="d-flex justify-content-end gap-2 mt-4">
          <button type="button" className="btn btn-light border btn-sm" onClick={closeKbVisibilityModal}>
            {t("kb_cancel", "Cancel")}
          </button>
          <button
            id="kb-visibility-save-btn"
            type="button"
            className="btn btn-dark btn-sm"
            disabled={!kbVisibilityData?.can_manage}
            onClick={() => {
              void (async () => {
                const saved = await handleSaveVisibility();
                if (saved) {
                  closeKbVisibilityModal();
                  await showAlertModal({
                    title: t("kb_visibility_title", "Knowledge Base Visibility"),
                    message: t("kb_visibility_saved", "Visibility settings saved."),
                    type: "success",
                  });
                }
              })();
            }}
          >
            {t("common_save", "Save")}
          </button>
        </div>
      </dialog>

      <dialog id="bg-task-detail-modal" className="bg-task-detail-modal" ref={backgroundDetailDialogRef}>
        {backgroundDetailTask ? (
          <>
            <div className="bg-task-detail-header">
              <div>
                <span className={`bg-task-status ${backgroundDetailTask.status}`.trim()}>
                  {backgroundDetailTask.status === "running"
                    ? t("bg_task_running", "Running")
                    : backgroundDetailTask.status === "completed"
                      ? t("bg_task_completed", "Completed")
                      : t("bg_task_failed", "Failed")}
                </span>
                <div className="text-muted">{backgroundDetailTask.pipeline_name}</div>
              </div>
              <button
                type="button"
                className="bg-modal-close-btn"
                onClick={closeBackgroundTaskDetail}
                aria-label={t("bg_tasks_close", "Close")}
              >
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
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
            <div className="bg-task-detail-body">
              <div className="bg-task-detail-question">
                <strong>{t("bg_task_question", "Question")}</strong>
                {backgroundDetailTask.full_question || backgroundDetailTask.question || "Background Task"}
              </div>
              {backgroundDetailTask.status === "completed" ? (
                <div
                  className="bg-task-detail-answer"
                  dangerouslySetInnerHTML={{
                    __html: renderChatMarkdown(String(backgroundDetailTask.result || "")),
                  }}
                />
              ) : backgroundDetailTask.status === "failed" ? (
                <div
                  className="bg-task-detail-question"
                  style={{
                    background: "rgba(239, 68, 68, 0.06)",
                    border: "1px solid rgba(239, 68, 68, 0.15)",
                  }}
                >
                  <strong style={{ color: "#ef4444" }}>{t("bg_task_error", "Error")}</strong>
                  {backgroundDetailTask.error || t("bg_task_unknown_error", "Unknown error")}
                </div>
              ) : (
                <div style={{ textAlign: "center", padding: "40px 20px" }}>
                  <div
                    className="spinner-border"
                    style={{ color: "#3b82f6", width: "2rem", height: "2rem" }}
                    role="status"
                  >
                    <span className="visually-hidden">{t("bg_task_loading_details", "Loading task details...")}</span>
                  </div>
                  <div style={{ marginTop: "16px", color: "var(--text-secondary)", fontSize: "0.9rem" }}>
                    {t("bg_task_processing", "Processing your request...")}
                  </div>
                </div>
              )}
            </div>
            <div className="bg-task-detail-actions d-flex gap-2 flex-wrap align-items-center p-4 pt-0">
              {backgroundDetailTask.status === "completed" ? (
                <>
                  <button type="button" className="btn btn-primary" onClick={() => void handleCopyTaskResult()}>
                    {t("bg_task_copy_result", "Copy Result")}
                  </button>
                  <button
                    type="button"
                    className="btn btn-outline-secondary"
                    onClick={() => void handleLoadTaskResultFromModal(false)}
                  >
                    {t("bg_task_load_current", "Load to Current Chat")}
                  </button>
                  <button
                    type="button"
                    className="btn btn-outline-secondary"
                    onClick={() => void handleLoadTaskResultFromModal(true)}
                  >
                    {t("bg_task_load_new", "Load to New Chat")}
                  </button>
                </>
              ) : null}
              <button
                type="button"
                className="btn btn-outline-danger ms-auto"
                onClick={() => void handleDeleteBackgroundTask(backgroundDetailTask.task_id)}
              >
                {t("bg_task_delete", "Delete")}
              </button>
            </div>
          </>
        ) : (
          <div style={{ padding: "32px", textAlign: "center" }}>
            <div className="spinner-border" style={{ color: "#3b82f6", width: "2.5rem", height: "2.5rem" }} role="status">
              <span className="visually-hidden">{t("bg_task_loading_details", "Loading task details...")}</span>
            </div>
            <div style={{ marginTop: "16px", color: "var(--text-secondary)" }}>
              {t("bg_task_loading_details", "Loading task details...")}
            </div>
          </div>
        )}
      </dialog>

      <AuthDialog
        open={authOpen}
        onOpenChange={setAuthOpen}
        onLogin={handleLogin}
        onRegister={handleRegister}
        isSubmitting={loginMutation.isPending || registerMutation.isPending}
        errorMessage={authError}
      />
      <AccountSettingsDialog
        key={`${authInfo?.user_id ?? "guest"}-${authInfo?.nickname ?? ""}-${accountDialogMode}-${accountOpen ? "1" : "0"}`}
        open={accountOpen}
        onOpenChange={setAccountOpen}
        mode={accountDialogMode}
        userLabel={displayUserName}
        initialNickname={authInfo?.nickname}
        initialModelSettings={authInfo?.model_settings}
        isSubmitting={
          accountDialogMode === "model"
            ? modelSettingsMutation.isPending
            : nicknameMutation.isPending || changePasswordMutation.isPending
        }
        statusMessage={accountStatus}
        onSaveNickname={handleSaveNickname}
        onChangePassword={handleChangePassword}
        onSaveModelSettings={handleSaveModelSettings}
        onClearModelSettings={handleClearModelSettings}
      />
    </>
  );
}
