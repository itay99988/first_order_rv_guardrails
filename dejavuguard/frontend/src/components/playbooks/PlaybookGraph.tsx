import { useCallback, useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

import { getPlaybookStates, getPlaybookTrace } from "@/api/client";
import type {
  AsyncState,
  PlaybookMember,
  PlaybookTrace,
  PlaybookTraceNode,
} from "@/types";

interface Props {
  playbookId: string;
  sessionId: string;
}

/**
 * The trace, plus the truth table behind it when it could be fetched.
 *
 * The trace says which behaviours exist and where the session went; only the
 * states endpoint knows the verdict combinations each behaviour covers, which
 * is what maps a node back to the policies that produced it. It is fetched
 * best-effort: losing the subtitle degrades the graph, losing the graph
 * because a subtitle failed would not be a trade worth making.
 */
interface Loaded {
  trace: PlaybookTrace;
  /** Behaviour name -> the verdict maps of every state it covers. */
  verdicts: Map<string, Record<string, boolean>[]>;
}

/**
 * Node box geometry. Nodes are boxes rather than circles because the label is
 * now a list -- one line per rule that applies -- and a circle has no room for
 * one. Every node is the same height so the spine stays level and the edge
 * geometry needs only the centre point.
 */
const NODE_W = 236;
const PAD_X = 12;
const RULE_LINE_H = 15;
const HEAD_H = 34;
const VERDICT_LINE_H = 13;
const FOOT_PAD = 32;
/** Rules past this many collapse into a "+N more" line. */
const MAX_RULE_LINES = 4;
/** Verdict tokens per subtitle line -- they wrap rather than truncate. */
const VERDICTS_PER_LINE = 4;
const NAME_CHARS = 28;

const SPINE_GAP = NODE_W + 74;
const MARGIN_X = NODE_W / 2 + 24;
const TRAY_COLS = 4;
const TRAY_COL_GAP = NODE_W + 26;
const BACK_EDGE_RISE = 74;

/**
 * Left-to-right visit order for the spine, taken from the server.
 *
 * `first_visit` is authoritative: the trace endpoint records it while walking
 * the session's messages in order, before collapsing them into aggregated
 * edges. Reconstructing it on the client from those edges cannot work once
 * the trace contains a cycle -- when a session returns to where it started,
 * every node has an incoming edge, so there is no unambiguous starting point
 * to walk forward from. That is the commonest shape, not a corner case: a
 * budget playbook goes clear, over budget, clear again.
 */
function spineOrder(nodes: PlaybookTraceNode[]): PlaybookTraceNode[] {
  return nodes
    .filter((n) => n.visited && n.first_visit !== null)
    .sort((a, b) => (a.first_visit ?? 0) - (b.first_visit ?? 0));
}

function clip(text: string, chars: number): string {
  return text.length <= chars ? text : `${text.slice(0, chars - 1)}…`;
}

/**
 * The rule names a node displays, one per line.
 *
 * Names rather than a joined-and-truncated caption: wave E measured the old
 * 14-character caption rendering "A-rule + B-rule" and "A-rule + B-rule +
 * C-rule" identically, so two different behaviours read as one. A line per
 * rule cannot collapse that way, and the count above them separates two nodes
 * before the reader has read a single name.
 */
function ruleLines(node: PlaybookTraceNode): string[] {
  const names = node.rule_names ?? node.rules;
  if (names.length === 0) return ["No guidance"];
  if (names.length <= MAX_RULE_LINES) return names;
  return [
    ...names.slice(0, MAX_RULE_LINES - 1),
    `+${names.length - MAX_RULE_LINES + 1} more`,
  ];
}

/**
 * One `M<n>=T|F|any` token per member, in member position order.
 *
 * A behaviour groups every state that behaves identically, so it usually
 * covers several verdict combinations: "any" is the honest rendering of a
 * member the behaviour does not care about, and listing all eight combinations
 * of a four-member playbook on a node would be unreadable. Members are
 * numbered rather than named because a member is identified by a uuid policy
 * id; the legend above the graph maps the numbers back.
 */
function verdictTokens(
  members: PlaybookMember[],
  rows: Record<string, boolean>[] | undefined,
): string[] {
  if (!rows || rows.length === 0) return [];
  return members.map((member, index) => {
    let sawTrue = false;
    let sawFalse = false;
    for (const row of rows) {
      if (row[member.policy_id] === true) sawTrue = true;
      else if (row[member.policy_id] === false) sawFalse = true;
    }
    const verdict = sawTrue && sawFalse ? "any" : sawTrue ? "T" : sawFalse ? "F" : "?";
    return `M${index + 1}=${verdict}`;
  });
}

function chunk(tokens: string[], size: number): string[][] {
  const out: string[][] = [];
  for (let i = 0; i < tokens.length; i += size) out.push(tokens.slice(i, i + size));
  return out;
}

interface Status {
  word: string;
  glyph: string;
  className: string;
}

/**
 * Current / visited / unvisited as a word and a glyph, not only a colour.
 *
 * The colour is the fastest cue for a sighted reader and carries none for
 * anyone else, so it is never the only one: the word is on the node, and the
 * same three states are on the group as data attributes.
 */
function statusOf(node: PlaybookTraceNode, isCurrent: boolean): Status {
  if (isCurrent) return { word: "Current", glyph: "▶", className: "fill-accent" };
  if (node.visited)
    return { word: "Visited", glyph: "✓", className: "fill-terminal-text" };
  return { word: "Not visited", glyph: "○", className: "fill-terminal-dim" };
}

function boxClasses(node: PlaybookTraceNode, isCurrent: boolean): string {
  const parts = [];
  if (node.flagged) {
    parts.push("fill-terminal-red/15 stroke-terminal-red stroke-2");
  } else if (node.visited) {
    parts.push("fill-dark-elevated stroke-accent/60 stroke-2");
  } else {
    parts.push("fill-dark-surface stroke-border-strong stroke-1");
  }
  if (!node.reachable) parts.push("opacity-50 [stroke-dasharray:5_3]");
  if (isCurrent) parts.push("stroke-[3]");
  return parts.join(" ");
}

function describe(
  node: PlaybookTraceNode,
  status: Status,
  tokens: string[],
): string {
  const names = node.rule_names ?? node.rules;
  const parts = [
    `${node.name}. ${status.word}`,
    node.flagged
      ? "Flagged: this state blocks the message"
      : "Not flagged: this state allows the message",
    names.length > 0 ? `Rules applied: ${names.join(", ")}` : "No guidance applies",
  ];
  if (tokens.length > 0) parts.push(`Verdicts: ${tokens.join(", ")}`);
  parts.push(`Covers ${node.state_count} state${node.state_count === 1 ? "" : "s"}`);
  if (!node.reachable) parts.push("Possibly unreachable");
  return `${parts.join(". ")}.`;
}

interface NodeProps {
  node: PlaybookTraceNode;
  isCurrent: boolean;
  x: number;
  y: number;
  height: number;
  /** Uniform across every node so the boxes line up. */
  ruleLineCount: number;
  verdictRows: string[][];
}

function GraphNode({
  node,
  isCurrent,
  x,
  y,
  height,
  ruleLineCount,
  verdictRows,
}: NodeProps) {
  const status = statusOf(node, isCurrent);
  const lines = ruleLines(node);
  const names = node.rule_names ?? node.rules;
  const top = -height / 2;
  const left = -NODE_W / 2 + PAD_X;
  const right = NODE_W / 2 - PAD_X;
  const rulesEnd = top + HEAD_H + ruleLineCount * RULE_LINE_H;

  return (
    <g
      data-testid={`node-${node.name}`}
      data-visited={String(node.visited)}
      data-current={String(isCurrent)}
      data-reachable={String(node.reachable)}
      data-flagged={String(node.flagged)}
      role="img"
      aria-label={describe(node, status, verdictRows.flat())}
      transform={`translate(${x}, ${y})`}
    >
      {/* "You are here", outside the box so it survives the flagged node's
          own red border -- a green ring around a red box reads as both at
          once, which is exactly the state it is. */}
      {isCurrent && (
        <rect
          x={-NODE_W / 2 - 6}
          y={top - 6}
          width={NODE_W + 12}
          height={height + 12}
          fill="none"
          className="stroke-accent"
          strokeWidth={2}
        />
      )}
      <rect
        x={-NODE_W / 2}
        y={top}
        width={NODE_W}
        height={height}
        className={boxClasses(node, isCurrent)}
      />

      {node.flagged && (
        <text
          x={left}
          y={top + 18}
          className="fill-terminal-red text-[10px] font-mono font-bold"
        >
          ⚑ BLOCKS
        </text>
      )}
      <text
        x={right}
        y={top + 18}
        textAnchor="end"
        className="fill-terminal-dim text-[10px] font-mono"
      >
        {names.length > 0
          ? `${names.length} rule${names.length === 1 ? "" : "s"}`
          : "no rules"}
      </text>
      <line
        x1={-NODE_W / 2}
        y1={top + 26}
        x2={NODE_W / 2}
        y2={top + 26}
        className="stroke-border"
        strokeWidth={1}
      />

      {lines.map((line, i) => (
        <text
          key={i}
          x={left}
          y={top + HEAD_H + 11 + i * RULE_LINE_H}
          className={
            names.length === 0
              ? "fill-terminal-dim text-[11px] font-mono italic"
              : "fill-terminal-bright text-[11px] font-mono"
          }
        >
          {names.length === 0 ? line : `· ${clip(line, NAME_CHARS)}`}
        </text>
      ))}

      {verdictRows.length > 0 && (
        <line
          x1={-NODE_W / 2}
          y1={rulesEnd + 5}
          x2={NODE_W / 2}
          y2={rulesEnd + 5}
          className="stroke-border"
          strokeWidth={1}
        />
      )}
      {verdictRows.map((row, i) => (
        <text
          key={i}
          x={left}
          y={rulesEnd + 19 + i * VERDICT_LINE_H}
          className="fill-terminal-amber text-[9px] font-mono"
        >
          {row.join(" · ")}
        </text>
      ))}

      <text
        x={left}
        y={top + height - 8}
        className={`${status.className} text-[10px] font-mono font-bold`}
      >
        {status.glyph} {status.word}
      </text>
      <text
        x={right}
        y={top + height - 8}
        textAnchor="end"
        className="fill-terminal-dim text-[9px] font-mono"
      >
        {node.reachable
          ? `${node.state_count} state${node.state_count === 1 ? "" : "s"}`
          : "⊘ unreachable?"}
      </text>
    </g>
  );
}

export default function PlaybookGraph({ playbookId, sessionId }: Props) {
  const [state, setState] = useState<AsyncState<Loaded>>({ status: "idle" });

  const load = useCallback(async () => {
    setState({ status: "loading" });
    try {
      const [trace, states] = await Promise.all([
        getPlaybookTrace(playbookId, sessionId),
        // Best-effort: see `Loaded`. Joined on behaviour name, which is what
        // identifies a behaviour everywhere else too -- the trace's visited
        // marker, the React key, the test id -- and the server hands both
        // endpoints the same grouping.
        (async () => {
          try {
            return await getPlaybookStates(playbookId);
          } catch {
            return null;
          }
        })(),
      ]);
      const verdicts = new Map(
        (states?.behaviours ?? []).map((b) => [
          b.name,
          b.states.map((s) => s.verdicts),
        ]),
      );
      setState({ status: "success", data: { trace, verdicts } });
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

  const { nodes, edges, current } = state.data.trace;
  const members = [...state.data.trace.members].sort(
    (a, b) => a.position - b.position,
  );
  const spine = spineOrder(nodes);
  const spineIndex = new Map(spine.map((n, i) => [n.name, i]));
  const unvisited = nodes.filter((n) => !n.visited);

  // Uniform across every node: boxes of one height keep the spine level, and
  // a node that shows fewer rules than its neighbour still reads as the same
  // kind of thing.
  const ruleLineCount = Math.max(1, ...nodes.map((n) => ruleLines(n).length));
  const verdictsFor = new Map(
    nodes.map((n) => [
      n.name,
      chunk(verdictTokens(members, state.data.verdicts.get(n.name)), VERDICTS_PER_LINE),
    ]),
  );
  const verdictRowCount = Math.max(
    0,
    ...[...verdictsFor.values()].map((rows) => rows.length),
  );
  const nodeH =
    HEAD_H + ruleLineCount * RULE_LINE_H + verdictRowCount * VERDICT_LINE_H + FOOT_PAD;

  const spineY = nodeH / 2 + BACK_EDGE_RISE + 26;
  const trayFirstRowY = spine.length
    ? spineY + nodeH + 62
    : nodeH / 2 + 16;

  const positions = new Map<string, { x: number; y: number }>();
  spine.forEach((n, i) => positions.set(n.name, { x: MARGIN_X + i * SPINE_GAP, y: spineY }));
  unvisited.forEach((n, i) => {
    positions.set(n.name, {
      x: MARGIN_X + (i % TRAY_COLS) * TRAY_COL_GAP,
      y: trayFirstRowY + Math.floor(i / TRAY_COLS) * (nodeH + 26),
    });
  });

  const spineSpan = spine.length
    ? MARGIN_X * 2 + (spine.length - 1) * SPINE_GAP
    : 0;
  const traySpan = unvisited.length
    ? MARGIN_X * 2 + (Math.min(unvisited.length, TRAY_COLS) - 1) * TRAY_COL_GAP
    : 0;
  const width = Math.max(600, spineSpan, traySpan);
  const trayRows = Math.ceil(unvisited.length / TRAY_COLS);
  const height = trayRows
    ? trayFirstRowY + (trayRows - 1) * (nodeH + 26) + nodeH / 2 + 20
    : spineY + nodeH / 2 + 20;

  return (
    <div className="space-y-2" data-testid="playbook-graph">
      <p className="text-xs text-terminal-dim">
        One node per <span className="text-terminal-text">behaviour</span> — the states
        that apply the same rules and block or allow alike are one node. The node lists
        the rules that apply in it; <span className="text-terminal-red">⚑ BLOCKS</span>{" "}
        marks a state that stops the message. Edges are transitions this session actually
        took, not every transition the playbook could take. Muted, dashed nodes are shaded
        by a <span className="text-terminal-amber">reachability heuristic</span> — an
        irrevocable (historically-quantified) member that has already gone False,
        syntactically detected. It is a heuristic, not a proof.
      </p>

      {members.length > 0 && (
        <p
          className="flex flex-wrap items-baseline gap-x-4 gap-y-1 text-[11px] text-terminal-dim"
          data-testid="graph-member-legend"
        >
          <span className="text-terminal-text">
            Members (T satisfied · F violated · any either):
          </span>
          {members.map((member, i) => (
            <span key={member.policy_id} className="font-mono">
              <span className="text-terminal-amber">M{i + 1}</span> {member.policy_id}
            </span>
          ))}
        </p>
      )}

      <div className="max-h-[65vh] overflow-auto rounded-none border border-border bg-dark-primary">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="block"
          style={{ minWidth: `${width}px`, height: `${height}px` }}
          role="group"
          aria-label="Playbook state machine"
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
              ? `M ${from.x} ${from.y - nodeH / 2} Q ${(from.x + to.x) / 2} ${
                  spineY - nodeH / 2 - BACK_EDGE_RISE
                } ${to.x} ${to.y - nodeH / 2}`
              : `M ${from.x + NODE_W / 2} ${from.y} L ${to.x - NODE_W / 2} ${to.y}`;
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
                  <GraphNode
                    key={node.name}
                    node={node}
                    isCurrent={false}
                    x={pos.x}
                    y={pos.y}
                    height={nodeH}
                    ruleLineCount={ruleLineCount}
                    verdictRows={verdictsFor.get(node.name) ?? []}
                  />
                );
              })}
            </g>
          )}

          {spine.map((node) => {
            const pos = positions.get(node.name);
            if (!pos) return null;
            return (
              <GraphNode
                key={node.name}
                node={node}
                isCurrent={node.name === current}
                x={pos.x}
                y={pos.y}
                height={nodeH}
                ruleLineCount={ruleLineCount}
                verdictRows={verdictsFor.get(node.name) ?? []}
              />
            );
          })}
        </svg>
      </div>
    </div>
  );
}
