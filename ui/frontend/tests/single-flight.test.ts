import assert from "node:assert/strict";
import test from "node:test";
import { SingleFlight } from "../src/shared/lib/singleFlight.ts";

test("warmup and send share one pending engine initialization", async () => {
  const starts = new SingleFlight<string>();
  let resolve!: (value: string) => void;
  let calls = 0;
  const initialize = () => {
    calls += 1;
    return new Promise<string>((done) => { resolve = done; });
  };
  const warmup = starts.run("pipeline", initialize);
  const send = starts.run("pipeline", initialize);
  await Promise.resolve();
  assert.equal(calls, 1);
  assert.equal(send, warmup);
  resolve("one-engine");
  assert.deepEqual(await Promise.all([warmup, send]), ["one-engine", "one-engine"]);
  assert.equal(await starts.run("pipeline", async () => "revalidated-engine"), "revalidated-engine");
});

test("different pipelines initialize independently", async () => {
  const starts = new SingleFlight<string>();
  const left = starts.run("left", async () => "left-engine");
  const right = starts.run("right", async () => "right-engine");
  assert.deepEqual(await Promise.all([left, right]), ["left-engine", "right-engine"]);
});

test("failed initialization is shared and can be retried", async () => {
  const starts = new SingleFlight<string>();
  const failure = new Error("unavailable");
  const first = starts.run("pipeline", () => { throw failure; });
  const second = starts.run("pipeline", async () => "must-not-start");
  const results = await Promise.allSettled([first, second]);
  for (const result of results) {
    assert.equal(result.status, "rejected");
    if (result.status === "rejected") assert.equal(result.reason, failure);
  }
  assert.equal(await starts.run("pipeline", async () => "recovered-engine"), "recovered-engine");
});
