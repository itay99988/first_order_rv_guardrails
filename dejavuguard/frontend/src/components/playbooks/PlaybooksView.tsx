import { useState } from "react";
import { Loader2, Plus } from "lucide-react";

import { usePlaybooks } from "@/hooks/usePlaybooks";
import type { Playbook } from "@/types";
import PlaybookCard from "./PlaybookCard";
import PlaybookEditor from "./PlaybookEditor";

export default function PlaybooksView() {
  const { playbooks, fetchPlaybooks, createPlaybook, deletePlaybook } =
    usePlaybooks();

  const [openPlaybook, setOpenPlaybook] = useState<Playbook | null>(null);
  const [showNewForm, setShowNewForm] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  const loading = playbooks.status === "loading" || playbooks.status === "idle";
  const list = playbooks.status === "success" ? playbooks.data : [];

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newName.trim()) return;
    setCreating(true);
    setCreateError(null);
    try {
      await createPlaybook({
        name: newName.trim(),
        description: newDescription.trim() || undefined,
      });
      setNewName("");
      setNewDescription("");
      setShowNewForm(false);
    } catch (err) {
      setCreateError(
        err instanceof Error ? err.message : "Failed to create playbook",
      );
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (playbookId: string) => {
    await deletePlaybook(playbookId);
    if (openPlaybook?.playbook_id === playbookId) setOpenPlaybook(null);
  };

  const handleOpen = (playbookId: string) => {
    const found = list.find((p) => p.playbook_id === playbookId) ?? null;
    setOpenPlaybook(found);
  };

  if (openPlaybook) {
    return (
      <PlaybookEditor
        playbook={openPlaybook}
        onBack={() => {
          setOpenPlaybook(null);
          void fetchPlaybooks();
        }}
      />
    );
  }

  if (loading) {
    return (
      <div
        className="flex h-full items-center justify-center"
        data-testid="playbooks-loading"
      >
        <Loader2 className="h-8 w-8 animate-spin text-accent" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-8 p-6" data-testid="playbooks-view">
      <section>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-mono font-bold text-accent uppercase tracking-wider">
            Playbooks
          </h2>
          <button
            onClick={() => setShowNewForm((v) => !v)}
            className="btn-primary flex items-center gap-1.5 rounded-none px-3 py-2 text-sm font-medium"
            aria-label="Add playbook"
            data-testid="add-playbook"
          >
            <Plus size={16} />
            New playbook
          </button>
        </div>

        {showNewForm && (
          <form
            onSubmit={handleCreate}
            className="mb-4 space-y-3 rounded-none border border-border bg-dark-surface p-4"
            data-testid="new-playbook-form"
          >
            <div>
              <label
                className="mb-1 block text-terminal-text font-mono text-sm"
                htmlFor="new-playbook-name"
              >
                Name
              </label>
              <input
                id="new-playbook-name"
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Budget review"
                className="w-full rounded-none border border-border bg-dark-primary px-3 py-2 font-mono text-sm text-terminal-bright placeholder-terminal-dim focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20"
                data-testid="new-playbook-name-input"
              />
            </div>
            <div>
              <label
                className="mb-1 block text-terminal-text font-mono text-sm"
                htmlFor="new-playbook-description"
              >
                Description (optional)
              </label>
              <textarea
                id="new-playbook-description"
                value={newDescription}
                onChange={(e) => setNewDescription(e.target.value)}
                rows={2}
                className="w-full rounded-none border border-border bg-dark-primary px-3 py-2 text-sm text-terminal-bright placeholder-terminal-dim focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20"
                data-testid="new-playbook-description-input"
              />
            </div>
            {createError && (
              <p className="text-sm text-terminal-red" data-testid="new-playbook-error">
                {createError}
              </p>
            )}
            <div className="flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowNewForm(false)}
                className="rounded-none border border-border px-4 py-2 text-sm font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
                data-testid="new-playbook-cancel"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={!newName.trim() || creating}
                className="btn-primary rounded-none px-4 py-2 text-sm font-medium"
                data-testid="new-playbook-save"
              >
                {creating ? "Creating..." : "Create"}
              </button>
            </div>
          </form>
        )}

        {playbooks.status === "error" && (
          <p className="text-sm text-terminal-red" data-testid="playbooks-error">
            {playbooks.error}
          </p>
        )}

        {list.length === 0 && !loading && (
          <p className="text-sm text-terminal-dim" data-testid="no-playbooks">
            No playbooks defined yet. Click "New playbook" to create one.
          </p>
        )}

        <div className="space-y-3">
          {list.map((p) => (
            <PlaybookCard
              key={p.playbook_id}
              playbook={p}
              onOpen={handleOpen}
              onDelete={handleDelete}
            />
          ))}
        </div>
      </section>
    </div>
  );
}
