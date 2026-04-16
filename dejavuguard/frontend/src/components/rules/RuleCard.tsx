import { Trash2 } from "lucide-react";

import Badge from "@/components/shared/Badge";
import type { Policy } from "@/types";

interface RuleCardProps {
  policy: Policy;
  onToggle: (policyId: string, enabled: boolean) => void;
  onDelete: (policyId: string) => void;
}

export default function RuleCard({
  policy,
  onToggle,
  onDelete,
}: RuleCardProps) {
  return (
    <div
      className="rounded-none border border-border bg-dark-surface p-4"
      data-testid={`policy-card-${policy.policy_id}`}
    >
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-sm text-terminal-bright font-mono font-bold">
            {policy.name}
          </span>
          <Badge variant={policy.enabled ? "success" : "neutral"}>
            {policy.enabled ? "Active" : "Disabled"}
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <label className="relative inline-flex cursor-pointer items-center">
            <input
              type="checkbox"
              checked={policy.enabled}
              onChange={(e) => onToggle(policy.policy_id, e.target.checked)}
              className="peer sr-only"
              aria-label={`Toggle policy ${policy.name}`}
              data-testid={`toggle-policy-${policy.policy_id}`}
            />
            <div className="h-5 w-9 rounded-full bg-dark-hover after:absolute after:left-[2px] after:top-[2px] after:h-4 after:w-4 after:rounded-full after:bg-terminal-dim after:transition-all after:content-[''] peer-checked:bg-accent peer-checked:after:translate-x-full peer-checked:after:bg-dark-primary peer-focus:ring-2 peer-focus:ring-accent/30" />
          </label>
          <button
            onClick={() => onDelete(policy.policy_id)}
            className="rounded-none p-1.5 text-terminal-dim hover:bg-terminal-red/10 hover:text-terminal-red"
            aria-label={`Delete policy ${policy.name}`}
            data-testid={`delete-policy-${policy.policy_id}`}
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>

      <div className="mb-2 bg-dark-primary border border-border rounded-none px-3 py-2">
        <code
          className="text-accent font-mono text-sm"
          data-testid="policy-formula"
        >
          {policy.formula_str}
        </code>
      </div>

      {policy.propositions.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {policy.propositions.map((propId) => (
            <span
              key={propId}
              className="rounded-none bg-accent-muted border border-accent/20 text-accent font-mono text-xs px-2 py-0.5"
            >
              {propId}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
