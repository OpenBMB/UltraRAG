import assert from "node:assert/strict";
import test from "node:test";
import { readStorage, removeStorage, writeStorage } from "../src/shared/lib/storage.ts";

test("blocked storage does not interrupt preferences or rendering", () => {
  const original = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    get() { throw new Error("SecurityError: storage blocked"); },
  });
  try {
    assert.equal(readStorage("locale"), null);
    assert.doesNotThrow(() => writeStorage("locale", "en"));
    assert.doesNotThrow(() => removeStorage("locale"));
  } finally {
    if (original) Object.defineProperty(globalThis, "localStorage", original);
    else Reflect.deleteProperty(globalThis, "localStorage");
  }
});
