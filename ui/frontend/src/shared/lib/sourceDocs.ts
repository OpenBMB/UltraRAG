import type { SourceDoc } from "@/shared/api/types";

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
