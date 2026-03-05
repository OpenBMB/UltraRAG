import DOMPurify from "dompurify";
import hljs from "highlight.js";
import katex from "katex";
import { Marked } from "marked";
import type { SourceDoc } from "@/shared/api/types";

type RenderChatMarkdownOptions = {
  enhanceBlocks?: boolean;
  copyButtonLabel?: string;
  copyCodeButtonAriaLabel?: string;
  copyTableButtonAriaLabel?: string;
  enableInteractiveControls?: boolean;
};

function escapeHtml(raw: string): string {
  return raw
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function encodeCopyPayload(raw: string): string {
  try {
    return encodeURIComponent(raw);
  } catch {
    return raw;
  }
}

function buildCopyButton(
  options: RenderChatMarkdownOptions,
  className: "code-block-copy" | "table-copy-btn",
  copyText: string,
  ariaFallback: string,
): string {
  if (options.enableInteractiveControls !== true) return "";
  const tooltip = escapeHtml(options.copyButtonLabel ?? "Copy");
  const ariaLabel = escapeHtml(
    className === "code-block-copy"
      ? options.copyCodeButtonAriaLabel ?? ariaFallback
      : options.copyTableButtonAriaLabel ?? ariaFallback,
  );
  return `
<button type="button" class="${className}" data-copy-text="${encodeCopyPayload(copyText)}" data-tooltip="${tooltip}" aria-label="${ariaLabel}">
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
  </svg>
</button>`.trim();
}

function renderCodeBlock(code: string, language: string | undefined, options: RenderChatMarkdownOptions): string {
  const lang = (language ?? "").trim().split(/\s+/)[0]?.toLowerCase() ?? "";
  const safeLang = lang.replace(/[^a-z0-9_-]/gi, "") || "text";
  try {
    const highlighted =
      lang && hljs.getLanguage(lang)
        ? hljs.highlight(code, { language: lang, ignoreIllegals: true }).value
        : hljs.highlightAuto(code).value;
    if (options.enhanceBlocks !== true) {
      return `<pre><code class="hljs language-${safeLang}">${highlighted}</code></pre>`;
    }
    const langLabel = escapeHtml(safeLang);
    const copyButton = buildCopyButton(options, "code-block-copy", code, "Copy code");
    return `<div class="code-block-wrapper"><div class="code-block-header"><span class="code-block-lang">${langLabel}</span>${copyButton}</div><pre><code class="hljs language-${safeLang}">${highlighted}</code></pre></div>`;
  } catch {
    if (options.enhanceBlocks !== true) {
      return `<pre><code class="hljs language-${safeLang}">${escapeHtml(code)}</code></pre>`;
    }
    const langLabel = escapeHtml(safeLang);
    const copyButton = buildCopyButton(options, "code-block-copy", code, "Copy code");
    return `<div class="code-block-wrapper"><div class="code-block-header"><span class="code-block-lang">${langLabel}</span>${copyButton}</div><pre><code class="hljs language-${safeLang}">${escapeHtml(code)}</code></pre></div>`;
  }
}

function renderMath(expression: string, displayMode: boolean): string {
  const trimmed = expression.trim();
  if (!trimmed) return "";
  try {
    return katex.renderToString(trimmed, {
      throwOnError: false,
      strict: "ignore",
      displayMode,
      output: "htmlAndMathml",
    });
  } catch {
    return displayMode ? `$$${trimmed}$$` : `$${trimmed}$`;
  }
}

function injectMath(raw: string): string {
  const withBlockMath = raw.replace(/\$\$([\s\S]+?)\$\$/g, (_match, expr: string) => {
    return `\n${renderMath(expr, true)}\n`;
  });
  return withBlockMath.replace(
    /(^|[^$])\$(?!\$)([^$\n]+?)\$(?!\$)/g,
    (_match, prefix: string, expr: string) => `${prefix}${renderMath(expr, false)}`,
  );
}

function isTableSeparatorLine(line: string): boolean {
  return /^\s*\|?[-: ]+\|[-:| ]*\|?\s*$/.test(line);
}

function looksLikeTableHeader(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed) return false;
  if (!trimmed.includes("|")) return false;
  if (/^\|?[-:| ]+\|?$/.test(trimmed)) return false;
  return true;
}

function normalizeTableBlocks(raw: string): string {
  const lines = raw.split(/\r?\n/);
  for (let index = 1; index < lines.length; index += 1) {
    if (!isTableSeparatorLine(lines[index])) continue;
    const headerIndex = index - 1;
    if (headerIndex < 0 || !looksLikeTableHeader(lines[headerIndex])) continue;
    const previousLine = lines[headerIndex - 1] ?? "";
    if (previousLine.trim() !== "") {
      lines.splice(headerIndex, 0, "");
      index += 1;
    }
  }
  return lines.join("\n");
}

function createMarkdown(options: RenderChatMarkdownOptions): Marked {
  return new Marked({
    async: false,
    gfm: true,
    breaks: true,
    renderer: {
      code({ text, lang }) {
        return renderCodeBlock(text, lang, options);
      },
      link({ href, title, tokens }) {
        const label = this.parser.parseInline(tokens);
        const safeHref = escapeHtml(href ?? "#");
        const titleAttr = title ? ` title="${escapeHtml(title)}"` : "";
        return `<a href="${safeHref}" target="_blank" rel="noreferrer noopener"${titleAttr}>${label}</a>`;
      },
    },
  });
}

function htmlToPlainText(html: string): string {
  return html
    .replace(/<\/(th|td)>/gi, "\t")
    .replace(/<\/tr>/gi, "\n")
    .replace(/<[^>]+>/g, "")
    .replace(/\t+\n/g, "\n")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function wrapTables(html: string, options: RenderChatMarkdownOptions): string {
  if (options.enhanceBlocks !== true) return html;
  return html.replace(/<table[\s\S]*?<\/table>/g, (tableHtml) => {
    const copyButton = buildCopyButton(options, "table-copy-btn", htmlToPlainText(tableHtml), "Copy table");
    return `<div class="table-block-wrapper">${copyButton}<div class="table-scroll">${tableHtml}</div></div>`;
  });
}

export function renderChatMarkdown(rawText: string, options: RenderChatMarkdownOptions = {}): string {
  const markdown = createMarkdown(options);
  const normalizedMarkdown = normalizeTableBlocks(injectMath(rawText));
  const html = markdown.parse(normalizedMarkdown, { async: false }) as string;
  const withTables = wrapTables(html, options);
  const withCitations = withTables.replace(
    /\[(\d+)\]/g,
    '<span class="citation-link" data-source-id="$1">[$1]</span>',
  );
  return DOMPurify.sanitize(withCitations, {
    ADD_ATTR: ["class", "target", "rel", "aria-label", "data-source-id", "data-copy-text", "data-tooltip"],
  });
}

function toSourceDoc(raw: unknown): SourceDoc | null {
  if (!raw || typeof raw !== "object") return null;
  const record = raw as Record<string, unknown>;
  const id = Number(record.id);
  if (!Number.isFinite(id)) return null;
  const displayIdRaw = Number(record.displayId);
  const displayId = Number.isFinite(displayIdRaw) ? displayIdRaw : undefined;
  const title = String(record.title ?? `Source ${id}`);
  const content = String(record.content ?? "");
  return { id, displayId, title, content };
}

export function normalizeSourceDocs(raw: unknown): SourceDoc[] {
  if (!Array.isArray(raw)) return [];
  const map = new Map<number, SourceDoc>();
  for (const item of raw) {
    const doc = toSourceDoc(item);
    if (!doc) continue;
    map.set(doc.id, doc);
  }
  return [...map.values()].sort((left, right) => left.id - right.id);
}
