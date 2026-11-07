import assert from "node:assert/strict";
import test from "node:test";
import { RequestOwner } from "../src/shared/lib/requestOwner.ts";

test("a replaced request cannot settle a newer view", () => {
  const owner = new RequestOwner();
  const first = owner.begin();
  const second = owner.begin();
  assert.equal(first.signal.aborted, true);
  assert.equal(owner.owns(first), false);
  assert.equal(owner.finish(first), false);
  assert.equal(owner.owns(second), true);
  assert.equal(owner.busy, true);
});

test("navigation cancels and detaches the old request", () => {
  const owner = new RequestOwner();
  const first = owner.begin();
  owner.cancel();
  assert.equal(first.signal.aborted, true);
  assert.equal(owner.owns(first), false);
  assert.equal(owner.busy, false);
});

test("explicit stop retains ownership until partial output settles", () => {
  const owner = new RequestOwner();
  const first = owner.begin();
  owner.abort();
  assert.equal(first.signal.aborted, true);
  assert.equal(owner.owns(first), true);
  assert.equal(owner.finish(first), true);
  assert.equal(owner.busy, false);
});

test("successful completion releases ownership without aborting", () => {
  const owner = new RequestOwner();
  const first = owner.begin();
  assert.equal(owner.finish(first), true);
  assert.equal(owner.finish(first), false);
  assert.equal(first.signal.aborted, false);
});
