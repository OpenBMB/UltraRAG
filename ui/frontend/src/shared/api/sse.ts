/** Decode JSON SSE frames independently of network packet and UTF-8 boundaries. */
export async function* readJsonEvents(body: ReadableStream<Uint8Array>): AsyncGenerator<unknown> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let dataLines: string[] = [];
  let ended = false;

  function parseData(): unknown {
    const data = dataLines.join("\n");
    dataLines = [];
    if (!data || data === "[DONE]") return undefined;
    try {
      return JSON.parse(data);
    } catch {
      // A malformed event must not discard valid frames arriving after it.
      return undefined;
    }
  }

  try {
    while (true) {
      const { done, value } = await reader.read();
      ended = done;
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let lineEnd: number;
      while ((lineEnd = buffer.search(/[\r\n]/)) !== -1) {
        const carriageReturn = buffer[lineEnd] === "\r";
        // Wait for the following byte before deciding whether this is CRLF.
        if (carriageReturn && lineEnd === buffer.length - 1 && !done) break;
        const line = buffer.slice(0, lineEnd);
        const length = carriageReturn && buffer[lineEnd + 1] === "\n" ? 2 : 1;
        buffer = buffer.slice(lineEnd + length);
        if (!line) {
          const event = parseData();
          if (event !== undefined) yield event;
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).replace(/^ /, ""));
        }
      }
      if (done) {
        // Some HTTP backends close immediately after their final JSON frame.
        if (buffer.startsWith("data:")) dataLines.push(buffer.slice(5).replace(/^ /, ""));
        const event = parseData();
        if (event !== undefined) yield event;
        return;
      }
    }
  } finally {
    try {
      if (!ended) await reader.cancel();
    } finally {
      reader.releaseLock();
    }
  }
}
