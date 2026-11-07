import { memo, useMemo } from "react";
import { renderChatMarkdown } from "@/shared/lib/chatMarkdown";

type MessageMarkdownProps = {
  text: string;
  streaming: boolean;
  copyLabel: string;
  copyCodeLabel: string;
  copyTableLabel: string;
};

/** Keep completed answers stable while the composer or another answer updates. */
export const MessageMarkdown = memo(function MessageMarkdown({
  text, streaming, copyLabel, copyCodeLabel, copyTableLabel,
}: MessageMarkdownProps) {
  const html = useMemo(() => renderChatMarkdown(text, {
    enhanceBlocks: true,
    copyButtonLabel: copyLabel,
    copyCodeButtonAriaLabel: copyCodeLabel,
    copyTableButtonAriaLabel: copyTableLabel,
    enableInteractiveControls: !streaming,
  }), [text, streaming, copyLabel, copyCodeLabel, copyTableLabel]);
  return <div dangerouslySetInnerHTML={{ __html: html }} />;
});

export const SimpleMarkdown = memo(function SimpleMarkdown({ text }: { text: string }) {
  const html = useMemo(() => renderChatMarkdown(text), [text]);
  return <div dangerouslySetInnerHTML={{ __html: html }} />;
});

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


export const SourceDetailContent = memo(function SourceDetailContent({ content }: { content: string }) {
  const html = useMemo(() => renderSourceDetailHtml(content), [content]);
  return <div dangerouslySetInnerHTML={{ __html: html }} />;
});
