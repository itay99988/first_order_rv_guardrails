import { useCallback, useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

import { getPlaybookTrace } from "@/api/client";
import type { AsyncState, PlaybookTrace, PlaybookTraceEdge, PlaybookTraceNode } from "@/types";

interface Props {
  playbookId: string;
  sessionId: string;
}

const RADIUS = 26;
const SPINE_Y = 130;
const SPINE_GAP = 150;
const MARGIN_X = 90;
const TRAY_TOP = 250;
const TRAY_GAP = 130;
const TRAY_COLS = 5;
const TRAY_ROW_GAP = 90;

/**
 * Left-to-right visit order for the spine.
 *
 * The trace endpoint has no separate "visit order" field -- only aggregated
 * (from, to, count) edges, whose array order already reflects each
 * transition's first occurrence (the backend dict preserves insertion
 * order). Walking forward from nodes with no incoming edge reconstructs the
 * order a session actually moved through; leftover nodes (pure cycles,
 * disconnected visits) are appended the same way.
 */
function spineOrder(
  nodes: PlaybookTraceNode[],
  edges: PlaybookTraceEdge[],
): PlaybookTraceNode[] {
  const visited = nodes.filter((n) => n.visited);
  const byName = new Map(visited.map((n) => [n.name, n]));
  const outgoing = new Map<string, string[]>();
  const hasIncoming = new Set<string>();
  for (const e of edges) {
    if (!byName.has(e.from) || !byName.has(e.to)) continue;
    if (!outgoing.has(e.from)) outgoing.set(e.from, []);
    outgoing.get(e.from)?.push(e.to);
    hasIncoming.add(e.to);
  }

  const placed = new Set<string>();
  const order: PlaybookTraceNode[] = [];
  const walkFrom = (start: string) => {
    let current: string | undefined = start;
    while (current !== undefined && !placed.has(current) && byName.has(current)) {
      placed.add(current);
      order.push(byName.get(current)!);
      current = (outgoing.get(current) ?? []).find((t) => !placed.has(t));
    }
  };

  for (const n of visited) {
    if (!hasIncoming.has(n.name)) walkFrom(n.name);
  }
  for (const n of visited) {
    if (!placed.has(n.name)) walkFrom(n.name);
  }
  return order;
}

function shortLabel(name: string): string {
  return name.length <= 14 ? name : `${name.slice(0, 13)}…`;
}

function nodeClasses(node: PlaybookTraceNode, isCurrent: boolean): string {
  const parts = ["stroke-2"];
  if (node.flagged) {
    parts.push("fill-terminal-red/15 stroke-terminal-red");
  } else if (node.visited) {
    parts.push("fill-dark-elevated stroke-accent/60");
  } else {
    parts.push("fill-dark-surface stroke-border-strong");
  }
  if (!node.reachable) {
    parts.push("opacity-40 [stroke-dasharray:4_3]");
  }
  if (isCurrent) {
    parts.push("stroke-[3]");
  }
  return parts.join(" ");
}

export default function PlaybookGraph({ playbookId, sessionId }: Props) {
  const [state, setState] = useState<AsyncState<PlaybookTrace>>({ status: "idle" });

  const load = useCallback(async () => {
    setState({ status: "loading" });
    try {
      const data = await getPlaybookTrace(playbookId, sessionId);
      setState({ status: "success", data });
    } catch (err) {
      setState({
        status: "error",
        error: err instanceof Error ? err.message : "Failed to load the state graph",
      });
    }
  }, [playbookId, sessionId]);

  useEffect(() => {
    void load();
  }, [load]);

  if (state.status === "idle" || state.status === "loading") {
    return (
      <div
        className="flex items-center justify-center py-8"
        data-testid="playbook-graph-loading"
      >
        <Loader2 className="h-6 w-6 animate-spin text-accent" />
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <div
        className="flex items-center justify-between text-sm text-terminal-red"
        data-testid="playbook-graph-error"
      >
        <span>{state.error}</span>
        <button
          onClick={() => void load()}
          className="rounded-none border border-border px-3 py-1.5 text-xs font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
          data-testid="playbook-graph-retry"
        >
          Retry
        </button>
      </div>
    );
  }

  const { nodes, edges, current } = state.data;
  const spine = spineOrder(nodes, edges);
  const spineIndex = new Map(spine.map((n, i) => [n.name, i]));
  const unvisited = nodes.filter((n) => !n.visited);

  const positions = new Map<string, { x: number; y: number }>();
  spine.forEach((n, i) => positions.set(n.name, { x: MARGIN_X + i * SPINE_GAP, y: SPINE_Y }));
  unvisited.forEach((n, i) => {
    const row = Math.floor(i / TRAY_COLS);
    const col = i % TRAY_COLS;
    positions.set(n.name, {
      x: MARGIN_X + col * TRAY_GAP,
      y: TRAY_TOP + row * TRAY_ROW_GAP,
    });
  });

  const spineSpan = MARGIN_X * 2 + Math.max(spine.length - 1, 0) * SPINE_GAP;
  const traySpan =
    unvisited.length > 0
      ? MARGIN_X * 2 + (Math.min(unvisited.length, TRAY_COLS) - 1) * TRAY_GAP
      : 0;
  const width = Math.max(600, spineSpan, traySpan);
  const trayRows = Math.ceil(unvisited.length / TRAY_COLS);
  const height = TRAY_TOP + Math.max(trayRows, 1) * TRAY_ROW_GAP;

  return (
    <div className="space-y-2" data-testid="playbook-graph">
      <p className="text-xs text-terminal-dim">
        Nodes are behaviours; edges are transitions this session actually took, not every
        transition the playbook could take. Muted, dashed nodes are shaded by a{" "}
        <span className="text-terminal-amber">reachability heuristic</span> — an irrevocable
        (historically-quantified) member that has already gone False, syntactically detected.
        It is a heuristic, not a proof.
      </p>

      <div className="overflow-x-auto rounded-none border border-border bg-dark-primary">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="block"
          style={{ minWidth: `${width}px`, height: `${height}px` }}
        >
          {edges.map((edge) => {
            const from = positions.get(edge.from);
            const to = positions.get(edge.to);
            if (!from || !to) return null;
            const fromIdx = spineIndex.get(edge.from);
            const toIdx = spineIndex.get(edge.to);
            const isBack =
              fromIdx !== undefined && toIdx !== undefined ? toIdx <= fromIdx : false;
            const strokeWidth = Math.min(1 + edge.count, 8);
            const path = isBack
              ? `M ${from.x} ${from.y - RADIUS} Q ${(from.x + to.x) / 2} ${
                  SPINE_Y - 90
                } ${to.x} ${to.y - RADIUS}`
              : `M ${from.x + RADIUS} ${from.y} L ${to.x - RADIUS} ${to.y}`;
            return (
              <path
                key={`${edge.from}->${edge.to}`}
                data-testid={`edge-${edge.from}-${edge.to}`}
                d={path}
                fill="none"
                className="stroke-terminal-dim"
                strokeWidth={strokeWidth}
                markerEnd="url(#arrow)"
              />
            );
          })}

          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" className="fill-terminal-dim" />
            </marker>
          </defs>

          {unvisited.length > 0 && (
            <g data-testid="unvisited-tray">
              {unvisited.map((node) => {
                const pos = positions.get(node.name);
                if (!pos) return null;
                return (
                  <g
                    key={node.name}
                    data-testid={`node-${node.name}`}
                    data-visited="false"
                    data-current="false"
                    data-reachable={String(node.reachable)}
                    transform={`translate(${pos.x}, ${pos.y})`}
                  >
                    <circle r={RADIUS} className={nodeClasses(node, false)} />
                    <text
                      textAnchor="middle"
                      dominantBaseline="middle"
                      className="fill-terminal-dim text-[10px] font-mono"
                    >
                      {shortLabel(node.name)}
                    </text>
                  </g>
                );
              })}
            </g>
          )}

          {spine.map((node) => {
            const pos = positions.get(node.name);
            if (!pos) return null;
            const isCurrent = node.name === current;
            return (
              <g
                key={node.name}
                data-testid={`node-${node.name}`}
                data-visited="true"
                data-current={String(isCurrent)}
                data-reachable={String(node.reachable)}
                transform={`translate(${pos.x}, ${pos.y})`}
              >
                {isCurrent && (
                  <circle
                    r={RADIUS + 6}
                    fill="none"
                    className="stroke-accent"
                    strokeWidth={2}
                  />
                )}
                <circle r={RADIUS} className={nodeClasses(node, isCurrent)} />
                <text
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="fill-terminal-bright text-[10px] font-mono"
                >
                  {shortLabel(node.name)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
