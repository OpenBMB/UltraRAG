/** Preferences are optional: blocked or full storage must not break the UI. */
export function readStorage(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

export function writeStorage(key: string, value: string): void {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Keep the current in-memory state when persistence is unavailable.
  }
}

export function removeStorage(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch {
    // Storage access can be disabled by browser privacy settings.
  }
}
