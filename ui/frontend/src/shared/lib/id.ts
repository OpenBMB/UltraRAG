function toHex(byte: number): string {
  return byte.toString(16).padStart(2, "0");
}

export function createClientId(): string {
  const cryptoApi = globalThis.crypto;

  if (cryptoApi && typeof cryptoApi.randomUUID === "function") {
    return cryptoApi.randomUUID().replace(/-/g, "");
  }

  if (cryptoApi && typeof cryptoApi.getRandomValues === "function") {
    const bytes = new Uint8Array(16);
    cryptoApi.getRandomValues(bytes);
    return Array.from(bytes, toHex).join("");
  }

  const timePart = Date.now().toString(16).padStart(12, "0");
  const randomPart = Array.from({ length: 20 }, () => Math.floor(Math.random() * 16).toString(16)).join("");
  return `${timePart}${randomPart}`.slice(0, 32);
}
