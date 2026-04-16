import { MessageSquare, ScrollText, Settings } from "lucide-react";
import { NavLink } from "react-router-dom";

const navItems = [
  { to: "/chat", label: "Chat", icon: MessageSquare },
  { to: "/rules", label: "Rules", icon: ScrollText },
  { to: "/settings", label: "Settings", icon: Settings },
] as const;

interface SidebarProps {
  monitorStatus?: Record<string, boolean> | null;
}

export default function Sidebar({ monitorStatus }: SidebarProps) {
  const allPassing = monitorStatus
    ? Object.values(monitorStatus).every(Boolean)
    : true;

  return (
    <aside
      className="flex w-64 flex-col border-r border-border-subtle bg-dark-secondary"
      data-testid="sidebar"
    >
      <div className="px-0 pt-2 pb-1 overflow-hidden">
        <video
          src="/assets/intro_logo.mp4"
          autoPlay
          loop
          muted
          playsInline
          className="w-full h-auto mix-blend-screen"
          style={{
            maskImage: "radial-gradient(ellipse 60% 50% at center, black 20%, transparent 80%)",
            WebkitMaskImage: "radial-gradient(ellipse 60% 50% at center, black 20%, transparent 80%)",
          }}
        />
      </div>

      <nav className="flex-1 px-3 space-y-0.5" data-testid="sidebar-nav">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 text-sm font-medium font-mono transition-all duration-150 ${
                isActive
                  ? "border-l border-accent bg-accent-muted text-accent"
                  : "text-terminal-dim hover:text-terminal-green hover:bg-dark-hover"
              }`
            }
            data-testid={`nav-${label.toLowerCase()}`}
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>

      {monitorStatus && Object.keys(monitorStatus).length > 0 && (
        <div
          className="border-t border-border-subtle px-4 py-4"
          data-testid="monitor-status"
        >
          <p className="mb-2 text-xs font-medium uppercase tracking-widest text-terminal-dim font-mono">
            Monitor Status
          </p>
          <div className="flex items-center gap-2 text-sm font-mono">
            <span className={allPassing ? "text-terminal-green" : "text-terminal-red"}>
              ■
            </span>
            <span className={allPassing ? "text-terminal-green" : "text-terminal-red"}>
              {allPassing ? "All policies passing" : "Violation detected"}
            </span>
          </div>
          <div className="mt-2 space-y-1">
            {Object.entries(monitorStatus).map(([policyId, passing]) => (
              <div
                key={policyId}
                className="flex items-center gap-2 text-xs text-terminal-dim font-mono"
              >
                <span className={passing ? "text-terminal-green" : "text-terminal-red"}>
                  ■
                </span>
                {policyId}: {passing ? "Pass" : "Fail"}
              </div>
            ))}
          </div>
        </div>
      )}
    </aside>
  );
}
