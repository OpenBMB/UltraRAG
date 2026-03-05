import { dump as dumpYaml, load as parseYaml } from "js-yaml";

export type PipelinePreviewNode = {
  id: string;
  label: string;
  kind: "step" | "loop" | "branch" | "group";
  children: PipelinePreviewNode[];
};

type ParseResult = {
  nodes: PipelinePreviewNode[];
  error?: string;
};

function asObject(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function makeNodeId(): string {
  return `node_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function inferPipelineSteps(root: unknown): unknown[] {
  if (Array.isArray(root)) return root;
  const objectRoot = asObject(root);
  if (!objectRoot) return [];
  if (Array.isArray(objectRoot.pipeline)) return objectRoot.pipeline;
  if (Array.isArray(objectRoot.steps)) return objectRoot.steps;
  return [];
}

function node(id: string, label: string, kind: PipelinePreviewNode["kind"], children: PipelinePreviewNode[] = []) {
  return { id, label, kind, children };
}

function parseStep(step: unknown, path: string): PipelinePreviewNode {
  if (typeof step === "string") {
    return node(path, step, "step");
  }

  if (Array.isArray(step)) {
    return node(
      path,
      "Group",
      "group",
      step.map((item, index) => parseStep(item, `${path}.group.${index}`)),
    );
  }

  const objectStep = asObject(step);
  if (!objectStep) {
    return node(path, String(step), "step");
  }

  const entries = Object.entries(objectStep);
  if (!entries.length) {
    return node(path, "Empty Step", "step");
  }

  const [name, value] = entries[0];
  if (name === "loop") {
    const loopConfig = asObject(value);
    const timesValue = loopConfig?.times;
    const timesLabel = Number.isFinite(Number(timesValue)) ? ` x${Number(timesValue)}` : "";
    const nestedSteps = Array.isArray(loopConfig?.steps) ? loopConfig.steps : [];
    return node(
      path,
      `loop${timesLabel}`,
      "loop",
      nestedSteps.map((nested, index) => parseStep(nested, `${path}.loop.${index}`)),
    );
  }

  if (name.startsWith("branch")) {
    const branchConfig = asObject(value);
    const children: PipelinePreviewNode[] = [];
    const router = Array.isArray(branchConfig?.router) ? branchConfig.router : [];
    children.push(
      node(
        `${path}.router`,
        "router",
        "group",
        router.map((routerStep, index) => parseStep(routerStep, `${path}.router.${index}`)),
      ),
    );

    const branches = asObject(branchConfig?.branches);
    if (branches) {
      for (const [branchName, branchSteps] of Object.entries(branches)) {
        const list = Array.isArray(branchSteps) ? branchSteps : [];
        children.push(
          node(
            `${path}.branch.${branchName}`,
            `branch: ${branchName}`,
            "branch",
            list.map((branchStep, index) => parseStep(branchStep, `${path}.branch.${branchName}.${index}`)),
          ),
        );
      }
    }
    return node(path, name, "branch", children);
  }

  return node(path, name, "step");
}

export function parsePipelinePreview(yamlText: string): ParseResult {
  if (!yamlText.trim()) {
    return {
      nodes: [],
      error: "Pipeline YAML is empty.",
    };
  }

  try {
    const root = parseYaml(yamlText);
    const steps = inferPipelineSteps(root);
    return {
      nodes: steps.map((step, index) => parseStep(step, `step.${index}`)),
    };
  } catch (error) {
    return {
      nodes: [],
      error: error instanceof Error ? error.message : "Failed to parse YAML.",
    };
  }
}

export function flattenPreviewNodes(nodes: PipelinePreviewNode[]): PipelinePreviewNode[] {
  const output: PipelinePreviewNode[] = [];
  const walk = (current: PipelinePreviewNode[]) => {
    for (const item of current) {
      output.push(item);
      walk(item.children);
    }
  };
  walk(nodes);
  return output;
}

export function findPreviewNode(
  nodes: PipelinePreviewNode[],
  id: string,
): PipelinePreviewNode | undefined {
  for (const item of nodes) {
    if (item.id === id) return item;
    const nested = findPreviewNode(item.children, id);
    if (nested) return nested;
  }
  return undefined;
}

function cloneNodes(nodes: PipelinePreviewNode[]): PipelinePreviewNode[] {
  return nodes.map((item) => ({
    ...item,
    children: cloneNodes(item.children),
  }));
}

function collectDescendantIds(node: PipelinePreviewNode): Set<string> {
  const ids = new Set<string>();
  const walk = (current: PipelinePreviewNode) => {
    for (const child of current.children) {
      ids.add(child.id);
      walk(child);
    }
  };
  walk(node);
  return ids;
}

function updateNodeById(
  nodes: PipelinePreviewNode[],
  id: string,
  updater: (target: PipelinePreviewNode) => PipelinePreviewNode,
): PipelinePreviewNode[] {
  return nodes.map((item) => {
    if (item.id === id) {
      return updater(item);
    }
    return {
      ...item,
      children: updateNodeById(item.children, id, updater),
    };
  });
}

function detachNode(
  nodes: PipelinePreviewNode[],
  id: string,
): { nodes: PipelinePreviewNode[]; detached?: PipelinePreviewNode } {
  const next: PipelinePreviewNode[] = [];
  let detached: PipelinePreviewNode | undefined;
  for (const item of nodes) {
    if (item.id === id) {
      detached = item;
      continue;
    }
    const nested = detachNode(item.children, id);
    if (nested.detached) {
      detached = nested.detached;
    }
    next.push({
      ...item,
      children: nested.nodes,
    });
  }
  return { nodes: next, detached };
}

function insertAsChild(
  nodes: PipelinePreviewNode[],
  parentId: string,
  child: PipelinePreviewNode,
): PipelinePreviewNode[] {
  return nodes.map((item) => {
    if (item.id === parentId) {
      return { ...item, children: [...item.children, child] };
    }
    return {
      ...item,
      children: insertAsChild(item.children, parentId, child),
    };
  });
}

function insertAsChildAt(
  nodes: PipelinePreviewNode[],
  parentId: string,
  index: number,
  child: PipelinePreviewNode,
): PipelinePreviewNode[] {
  return nodes.map((item) => {
    if (item.id === parentId) {
      const insertIndex = Math.max(0, Math.min(index, item.children.length));
      const nextChildren = [...item.children];
      nextChildren.splice(insertIndex, 0, child);
      return { ...item, children: nextChildren };
    }
    return {
      ...item,
      children: insertAsChildAt(item.children, parentId, index, child),
    };
  });
}

export function updatePreviewNodeLabel(
  nodes: PipelinePreviewNode[],
  nodeId: string,
  nextLabel: string,
): PipelinePreviewNode[] {
  const trimmed = nextLabel.trim();
  if (!trimmed) return nodes;
  const cloned = cloneNodes(nodes);
  return updateNodeById(cloned, nodeId, (target) => ({ ...target, label: trimmed }));
}

export function addPreviewChildNode(
  nodes: PipelinePreviewNode[],
  parentId: string,
  kind: PipelinePreviewNode["kind"] = "step",
  label?: string,
  nodeId?: string,
): PipelinePreviewNode[] {
  const parent = findPreviewNode(nodes, parentId);
  if (!parent || parent.kind === "step") return nodes;
  const childLabel =
    label?.trim() ||
    (kind === "loop" ? "loop x1" : kind === "branch" ? "branch" : kind === "group" ? "Group" : "new.step");
  const newChild = node(nodeId ?? makeNodeId(), childLabel, kind, []);
  const cloned = cloneNodes(nodes);
  return insertAsChild(cloned, parentId, newChild);
}

export function insertPreviewChildNode(
  nodes: PipelinePreviewNode[],
  parentId: string,
  index: number,
  kind: PipelinePreviewNode["kind"] = "step",
  label?: string,
  nodeId?: string,
): PipelinePreviewNode[] {
  const parent = findPreviewNode(nodes, parentId);
  if (!parent || parent.kind === "step") return nodes;
  const childLabel =
    label?.trim() ||
    (kind === "loop" ? "loop x1" : kind === "branch" ? "branch" : kind === "group" ? "Group" : "new.step");
  const newChild = node(nodeId ?? makeNodeId(), childLabel, kind, []);
  const cloned = cloneNodes(nodes);
  return insertAsChildAt(cloned, parentId, index, newChild);
}

export function insertPreviewBranchNode(
  nodes: PipelinePreviewNode[],
  parentId: string,
  index: number,
  cases: string[],
  branchLabel = "branch",
  branchNodeId?: string,
  routerNodeId?: string,
): PipelinePreviewNode[] {
  const parent = findPreviewNode(nodes, parentId);
  if (!parent || parent.kind === "step") return nodes;
  const normalizedCases = cases
    .map((item) => item.trim())
    .filter(Boolean);
  const branchChildren: PipelinePreviewNode[] = [
    node(routerNodeId ?? makeNodeId(), "router", "group", []),
    ...(normalizedCases.length ? normalizedCases : ["case1", "case2"]).map((name) =>
      node(makeNodeId(), `branch: ${name}`, "branch", []),
    ),
  ];
  const branchNode = node(branchNodeId ?? makeNodeId(), branchLabel.trim() || "branch", "branch", branchChildren);
  const cloned = cloneNodes(nodes);
  return insertAsChildAt(cloned, parentId, index, branchNode);
}

export function deletePreviewNode(
  nodes: PipelinePreviewNode[],
  nodeId: string,
): PipelinePreviewNode[] {
  const cloned = cloneNodes(nodes);
  return detachNode(cloned, nodeId).nodes;
}

export function movePreviewNode(
  nodes: PipelinePreviewNode[],
  nodeId: string,
  nextParentId: string | null,
): PipelinePreviewNode[] {
  const target = findPreviewNode(nodes, nodeId);
  if (!target) return nodes;

  if (nextParentId) {
    const parent = findPreviewNode(nodes, nextParentId);
    if (!parent || parent.kind === "step") return nodes;
    const descendants = collectDescendantIds(target);
    if (descendants.has(nextParentId)) return nodes;
  }

  const cloned = cloneNodes(nodes);
  const detached = detachNode(cloned, nodeId);
  if (!detached.detached) return nodes;
  if (!nextParentId) {
    return [...detached.nodes, detached.detached];
  }
  return insertAsChild(detached.nodes, nextParentId, detached.detached);
}

function readLoopTimes(label: string): number {
  const match = label.match(/x\s*(\d+)/i);
  const parsed = match ? Number(match[1]) : NaN;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 1;
}

function branchKey(label: string): string {
  const trimmed = label.trim();
  if (trimmed.startsWith("branch")) return trimmed;
  return "branch";
}

function branchName(label: string): string {
  const trimmed = label.trim();
  if (trimmed.startsWith("branch:")) {
    const name = trimmed.slice("branch:".length).trim();
    return name || "default";
  }
  return trimmed || "default";
}

function toStep(nodeValue: PipelinePreviewNode): unknown {
  if (nodeValue.kind === "loop") {
    return {
      loop: {
        times: readLoopTimes(nodeValue.label),
        steps: nodeValue.children.map(toStep),
      },
    };
  }

  if (nodeValue.kind === "branch") {
    const routerNode = nodeValue.children.find(
      (child) => child.kind === "group" && child.label.toLowerCase() === "router",
    );
    const routerSteps = routerNode ? routerNode.children.map(toStep) : [];
    const branchNodes = nodeValue.children.filter((child) => child !== routerNode);
    const branches: Record<string, unknown[]> = {};
    for (const branchNode of branchNodes) {
      const name = branchName(branchNode.label);
      branches[name] = branchNode.children.map(toStep);
    }
    if (!Object.keys(branches).length) {
      branches.default = [];
    }
    return {
      [branchKey(nodeValue.label)]: {
        router: routerSteps,
        branches,
      },
    };
  }

  if (nodeValue.kind === "group") {
    return nodeValue.children.map(toStep);
  }

  const trimmed = nodeValue.label.trim();
  return trimmed || "step";
}

function normalizeSteps(steps: unknown[]): unknown[] {
  const normalized: unknown[] = [];
  for (const step of steps) {
    if (Array.isArray(step)) {
      normalized.push(...normalizeSteps(step));
    } else {
      normalized.push(step);
    }
  }
  return normalized;
}

export function serializePipelinePreviewToYaml(nodes: PipelinePreviewNode[]): string {
  const pipeline = normalizeSteps(nodes.map(toStep));
  return dumpYaml(
    { pipeline },
    {
      lineWidth: 120,
      noRefs: true,
      sortKeys: false,
    },
  ).trimEnd();
}
