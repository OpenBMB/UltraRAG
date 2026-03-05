import { Fragment, type ReactNode } from "react";
import { type PipelinePreviewNode } from "@/shared/lib/pipelinePreview";

type PipelineCanvasProps = {
  nodes: PipelinePreviewNode[];
  rootId?: string;
  activeContextId?: string;
  onInsertNode?: (parentId: string, index: number) => void;
  onAddBranchCase?: (branchNodeId: string) => void;
  onActivateContext?: (contextId: string) => void;
  onEditNode?: (nodeId: string) => void;
  onDeleteNode?: (nodeId: string) => void;
};

function inferServerLabel(node: PipelinePreviewNode): string {
  const normalized = node.label.trim();
  if (!normalized) return node.kind;
  const firstToken = normalized.split(/\s+/)[0];
  if (firstToken.includes(".")) return firstToken.split(".")[0];
  return firstToken;
}

function toolDisplayLabel(label: string): string {
  const normalized = label.trim();
  if (!normalized) return "";
  const firstToken = normalized.split(/\s+/)[0];
  if (!firstToken.includes(".")) return normalized;
  const [, ...rest] = firstToken.split(".");
  const display = rest.join(".").trim();
  return display || normalized;
}

function readLoopTimes(label: string): number {
  const match = label.match(/x\s*(\d+)/i);
  const parsed = match ? Number(match[1]) : NaN;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 1;
}

function caseDisplayName(label: string): string {
  const trimmed = label.trim();
  if (trimmed.toLowerCase().startsWith("branch:")) {
    return trimmed.slice("branch:".length).trim() || "default";
  }
  return trimmed || "default";
}

function renderInsertControl(
  parentId: string,
  index: number,
  onInsertNode?: (parentId: string, index: number) => void,
  options?: { prominent?: boolean },
): ReactNode {
  if (!onInsertNode) return null;
  const prominent = Boolean(options?.prominent);
  return (
    <div className={`flow-insert-control ${prominent ? "prominent" : ""}`}>
      <button
        type="button"
        className="flow-insert-button"
        title="Insert step"
        onClick={() => onInsertNode(parentId, index)}
      >
        <span>+</span>
        {prominent ? <span>添加节点</span> : null}
      </button>
    </div>
  );
}

function renderToolNode(
  node: PipelinePreviewNode,
  onEditNode?: (nodeId: string) => void,
  onDeleteNode?: (nodeId: string) => void,
): ReactNode {
  const serverLabel = inferServerLabel(node).toUpperCase();
  const moduleName = inferServerLabel(node).toLowerCase();
  const displayLabel = toolDisplayLabel(node.label);
  return (
    <div key={node.id} className="flow-node">
      <div className="flow-node-header">
        <div className="flow-node-server" data-module={moduleName}>
          {serverLabel}
        </div>
      </div>
      <div className="flow-node-body" title={node.label}>
        {displayLabel}
      </div>
      <div className="step-actions">
        {onEditNode ? (
          <button type="button" className="btn btn-outline-primary btn-sm" onClick={() => onEditNode(node.id)}>
            Edit
          </button>
        ) : null}
        {onDeleteNode ? (
          <button type="button" className="btn btn-outline-danger btn-sm" onClick={() => onDeleteNode(node.id)}>
            Delete
          </button>
        ) : null}
      </div>
    </div>
  );
}

function renderBranchCaseCard(
  caseNode: PipelinePreviewNode,
  onInsertNode?: (parentId: string, index: number) => void,
  onAddBranchCase?: (branchNodeId: string) => void,
  onActivateContext?: (contextId: string) => void,
  activeContextId?: string,
  onEditNode?: (nodeId: string) => void,
  onDeleteNode?: (nodeId: string) => void,
): ReactNode {
  return (
    <div key={caseNode.id} className={`branch-case ${activeContextId === caseNode.id ? "active" : ""}`}>
      <div className="d-flex justify-content-between mb-2">
        <span className="fw-bold text-xs">分支: {caseDisplayName(caseNode.label)}</span>
        <button
          type="button"
          className="btn btn-link btn-sm p-0 text-decoration-none"
          onClick={() => onActivateContext?.(caseNode.id)}
        >
          打开
        </button>
      </div>
      {renderStepList(
        caseNode.children,
        caseNode.id,
        onInsertNode,
        onAddBranchCase,
        onActivateContext,
        activeContextId,
        onEditNode,
        onDeleteNode,
      )}
    </div>
  );
}

function renderStepNode(
  node: PipelinePreviewNode,
  onInsertNode?: (parentId: string, index: number) => void,
  onAddBranchCase?: (branchNodeId: string) => void,
  onActivateContext?: (contextId: string) => void,
  activeContextId?: string,
  onEditNode?: (nodeId: string) => void,
  onDeleteNode?: (nodeId: string) => void,
): ReactNode {
  const normalizedLabel = node.label.trim().toLowerCase();
  const isBranchCaseNode =
    (node.kind === "group" || node.kind === "branch") &&
    normalizedLabel.startsWith("branch:");

  if (node.kind === "step" || isBranchCaseNode) {
    if (isBranchCaseNode) {
      return renderBranchCaseCard(
        node,
        onInsertNode,
        onAddBranchCase,
        onActivateContext,
        activeContextId,
        onEditNode,
        onDeleteNode,
      );
    }
    return renderToolNode(node, onEditNode, onDeleteNode);
  }

  if (node.kind === "loop") {
    const loopTimes = readLoopTimes(node.label);
    return (
      <div key={node.id} className={`loop-container ${activeContextId === node.id ? "active" : ""}`}>
        <div className="loop-header">
          <h6>循环 ({loopTimes}次)</h6>
          <button
            type="button"
            className="btn btn-sm btn-link text-decoration-none p-0"
            onClick={() => onActivateContext?.(node.id)}
          >
            打开上下文
          </button>
        </div>
        {renderStepList(
          node.children,
          node.id,
          onInsertNode,
          onAddBranchCase,
          onActivateContext,
          activeContextId,
          onEditNode,
          onDeleteNode,
        )}
        <div className="mt-2 d-flex justify-content-end gap-2">
          {onEditNode ? (
            <button
              type="button"
              className="btn btn-sm btn-outline-secondary border-0"
              onClick={() => onEditNode(node.id)}
            >
              Edit
            </button>
          ) : null}
          {onDeleteNode ? (
            <button
              type="button"
              className="btn btn-sm btn-outline-danger border-0"
              onClick={() => onDeleteNode(node.id)}
            >
              Delete
            </button>
          ) : null}
        </div>
      </div>
    );
  }

  if (node.kind === "branch") {
    const routerNode = node.children.find(
      (child) => child.kind === "group" && child.label.trim().toLowerCase() === "router",
    );
    const caseNodes = node.children.filter((child) => child !== routerNode);

    return (
      <div key={node.id} className="branch-container">
        <div className="branch-header">
          <h6>分支</h6>
          <button
            type="button"
            className="btn btn-sm btn-link text-decoration-none p-0"
            onClick={() => onActivateContext?.(routerNode?.id || node.id)}
          >
            打开 Router
          </button>
        </div>
        <div className={`branch-router ${activeContextId === routerNode?.id ? "active" : ""}`}>
          {routerNode
            ? renderStepList(
                routerNode.children,
                routerNode.id,
                onInsertNode,
                onAddBranchCase,
                onActivateContext,
                activeContextId,
                onEditNode,
                onDeleteNode,
              )
            : null}
        </div>
        <div className="branch-cases mt-3">
          {caseNodes.map((caseNode) =>
            renderBranchCaseCard(
              caseNode,
              onInsertNode,
              onAddBranchCase,
              onActivateContext,
              activeContextId,
              onEditNode,
              onDeleteNode,
            ),
          )}
        </div>
        <div className="mt-2 d-flex justify-content-end gap-2">
          {onAddBranchCase ? (
            <button type="button" className="btn btn-sm btn-light border" onClick={() => onAddBranchCase(node.id)}>
              + 分支
            </button>
          ) : null}
          {onDeleteNode ? (
            <button
              type="button"
              className="btn btn-sm btn-text text-danger"
              onClick={() => onDeleteNode(node.id)}
            >
              删除分支
            </button>
          ) : null}
        </div>
      </div>
    );
  }

  // Fallback for unknown kinds.
  return (
    <div key={node.id} className="flow-node">
      <div className="flow-node-header">
        <div className="flow-node-server" data-module="custom">
          CUSTOM
        </div>
      </div>
      <div className="flow-node-body" title={node.label}>
        {node.label}
      </div>
      <div className="step-actions">
        {onEditNode ? (
          <button type="button" className="btn btn-outline-primary btn-sm" onClick={() => onEditNode(node.id)}>
            Edit
          </button>
        ) : null}
        {onDeleteNode ? (
          <button type="button" className="btn btn-outline-danger btn-sm" onClick={() => onDeleteNode(node.id)}>
            Delete
          </button>
        ) : null}
      </div>
      {node.children.length
        ? renderStepList(
            node.children,
            node.id,
            onInsertNode,
            onAddBranchCase,
            onActivateContext,
            activeContextId,
            onEditNode,
            onDeleteNode,
          )
        : null}
    </div>
  );
}

function renderStepList(
  nodes: PipelinePreviewNode[],
  parentId: string,
  onInsertNode?: (parentId: string, index: number) => void,
  onAddBranchCase?: (branchNodeId: string) => void,
  onActivateContext?: (contextId: string) => void,
  activeContextId?: string,
  onEditNode?: (nodeId: string) => void,
  onDeleteNode?: (nodeId: string) => void,
): ReactNode {
  if (!nodes.length) {
    return (
      <div className="step-list">
        <div className="flow-placeholder">
          {renderInsertControl(parentId, 0, onInsertNode, {
            prominent: true,
          }) ?? <span>No pipeline steps found.</span>}
        </div>
      </div>
    );
  }

  return (
    <div className="step-list">
      {nodes.map((node, index) => (
        <Fragment key={node.id}>
          {renderInsertControl(parentId, index, onInsertNode)}
          {renderStepNode(
            node,
            onInsertNode,
            onAddBranchCase,
            onActivateContext,
            activeContextId,
            onEditNode,
            onDeleteNode,
          )}
        </Fragment>
      ))}
      {renderInsertControl(parentId, nodes.length, onInsertNode, { prominent: true })}
    </div>
  );
}

export function PipelineCanvas({
  nodes,
  rootId = "__pipeline_root__",
  activeContextId,
  onInsertNode,
  onAddBranchCase,
  onActivateContext,
  onEditNode,
  onDeleteNode,
}: PipelineCanvasProps) {
  if (!nodes.length) {
    return (
      <div className="step-list">
        <div className="flow-placeholder">
          {renderInsertControl(rootId, 0, onInsertNode, { prominent: true }) ?? <span>No pipeline steps found.</span>}
        </div>
      </div>
    );
  }
  return renderStepList(
    nodes,
    rootId,
    onInsertNode,
    onAddBranchCase,
    onActivateContext,
    activeContextId,
    onEditNode,
    onDeleteNode,
  );
}
