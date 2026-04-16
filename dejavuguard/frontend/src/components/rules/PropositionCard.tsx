import { FileText, Monitor, Pencil, Trash2, User } from "lucide-react";

import Badge from "@/components/shared/Badge";
import type { Proposition } from "@/types";

interface PropositionCardProps {
  proposition: Proposition;
  onEdit: (proposition: Proposition) => void;
  onDelete: (propId: string) => void;
  onViewPrompt: (proposition: Proposition) => void;
}

export default function PropositionCard({
  proposition,
  onEdit,
  onDelete,
  onViewPrompt,
}: PropositionCardProps) {
  return (
    <div
      className="border border-border bg-dark-surface p-4"
      data-testid={`proposition-card-${proposition.prop_id}`}
    >
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-accent font-mono font-bold text-sm">
            {proposition.prop_id}
          </span>
          <div className="flex items-center gap-1">
            {proposition.role === "user" ? (
              <User size={12} className="text-terminal-cyan" />
            ) : (
              <Monitor size={12} className="text-terminal-amber" />
            )}
            <Badge variant={proposition.role === "user" ? "info" : "warning"}>
              {proposition.role}
            </Badge>
          </div>
        </div>
        <div className="flex gap-1">
          <button
            onClick={() => onViewPrompt(proposition)}
            className="p-1.5 text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
            aria-label={`View grounding prompt for ${proposition.prop_id}`}
            data-testid={`view-prompt-${proposition.prop_id}`}
          >
            <FileText size={14} />
          </button>
          <button
            onClick={() => onEdit(proposition)}
            className="p-1.5 text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
            aria-label={`Edit predicate ${proposition.prop_id}`}
            data-testid={`edit-proposition-${proposition.prop_id}`}
          >
            <Pencil size={14} />
          </button>
          <button
            onClick={() => onDelete(proposition.prop_id)}
            className="p-1.5 text-terminal-dim hover:bg-terminal-red/10 hover:text-terminal-red"
            aria-label={`Delete predicate ${proposition.prop_id}`}
            data-testid={`delete-proposition-${proposition.prop_id}`}
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>
      <p className="text-sm text-terminal-text">{proposition.description}</p>
      {proposition.few_shot_generated_at && (
        <p className="mt-2 text-xs text-terminal-dim">
          Few-shots generated: {new Date(proposition.few_shot_generated_at).toLocaleString()}
        </p>
      )}
    </div>
  );
}
