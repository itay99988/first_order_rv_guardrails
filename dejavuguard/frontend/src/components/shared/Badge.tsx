interface BadgeProps {
  variant: "success" | "warning" | "error" | "info" | "neutral";
  children: React.ReactNode;
}

const variantClasses: Record<BadgeProps["variant"], string> = {
  success: "border border-terminal-green/40 text-terminal-green bg-terminal-green/5",
  warning: "border border-terminal-amber/40 text-terminal-amber bg-terminal-amber/5",
  error: "border border-terminal-red/40 text-terminal-red bg-terminal-red/5",
  info: "border border-terminal-cyan/40 text-terminal-cyan bg-terminal-cyan/5",
  neutral: "border border-terminal-dim/40 text-terminal-dim bg-terminal-dim/5",
};

export default function Badge({ variant, children }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center rounded-none px-2.5 py-0.5 text-xs font-medium font-mono ${variantClasses[variant]}`}
      data-testid="badge"
    >
      [ {children} ]
    </span>
  );
}
