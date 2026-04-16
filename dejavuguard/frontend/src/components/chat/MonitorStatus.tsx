interface MonitorStatusProps {
  monitorState: Record<string, boolean> | null;
}

export default function MonitorStatus({ monitorState }: MonitorStatusProps) {
  if (!monitorState || Object.keys(monitorState).length === 0) {
    return null;
  }

  const allPassing = Object.values(monitorState).every(Boolean);

  return (
    <div
      className="flex items-center gap-2 text-xs font-mono"
      data-testid="chat-monitor-status"
    >
      <span
        className={`text-sm ${allPassing ? "text-terminal-green" : "text-terminal-red"}`}
      >
        {"\u25A0"}
      </span>
      <span className={allPassing ? "text-terminal-green font-mono" : "text-terminal-red font-mono"}>
        {allPassing ? "All policies passing" : "Violation detected"}
      </span>
    </div>
  );
}
