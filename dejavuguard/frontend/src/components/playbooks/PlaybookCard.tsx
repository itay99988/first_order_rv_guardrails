import { AlertTriangle, Trash2 } from "lucide-react";

import type { Playbook } from "@/types";

interface Props {
  playbook: Playbook;
  onOpen: (playbookId: string) => void;
  onDelete: (playbookId: string) => void;
}

export default function PlaybookCard({ playbook, onOpen, onDelete }: Props) {
  return (
    <div
      className="rounded border border-gray-700 bg-dark-secondary p-4"
      data-testid={`playbook-card-${playbook.playbook_id}`}
    >
      <div className="flex items-start justify-between">
        <button
          className="text-left text-lg font-semibold text-terminal-green"
          onClick={() => onOpen(playbook.playbook_id)}
        >
          {playbook.name}
        </button>
        <button
          aria-label={`Delete ${playbook.name}`}
          onClick={() => onDelete(playbook.playbook_id)}
        >
          <Trash2 size={16} />
        </button>
      </div>

      <p className="mt-2 text-sm text-gray-400">
        {playbook.member_count} policies ·{" "}
        {`${playbook.state_count} states → ${playbook.behaviour_count} behaviours`}
      </p>

      {playbook.flagged_count === 0 && (
        <p
          className="mt-2 flex items-center gap-1 text-sm text-terminal-amber"
          data-testid="playbook-no-block-warning"
        >
          <AlertTriangle size={14} />
          No state is flagged — this playbook cannot block anything.
        </p>
      )}
    </div>
  );
}
