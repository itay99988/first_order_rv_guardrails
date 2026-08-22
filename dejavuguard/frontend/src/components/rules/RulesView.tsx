import { useState } from "react";
import { Loader2, Plus } from "lucide-react";

import { getPropositionGroundingPrompt } from "@/api/client";
import Modal from "@/components/shared/Modal";
import { usePolicies } from "@/hooks/usePolicies";
import type { GroundingPromptPreview, Proposition } from "@/types";
import FormulaBuilder from "./FormulaBuilder";
import PropositionCard from "./PropositionCard";
import PropositionEditor from "./PropositionEditor";
import RuleCard from "./RuleCard";

export default function RulesView() {
  const {
    propositions,
    policies,
    createProposition,
    updateProposition,
    deleteProposition,
    createPolicy,
    deletePolicy,
    togglePolicy,
    validateFormula,
  } = usePolicies();

  const [showPropEditor, setShowPropEditor] = useState(false);
  const [editingProp, setEditingProp] = useState<Proposition | null>(null);
  const [showFormulaBuilder, setShowFormulaBuilder] = useState(false);
  const [showPromptModal, setShowPromptModal] = useState(false);
  const [promptPreview, setPromptPreview] = useState<GroundingPromptPreview | null>(
    null,
  );
  const [promptLoading, setPromptLoading] = useState(false);
  const [promptError, setPromptError] = useState<string | null>(null);

  const [propWarning, setPropWarning] = useState<string | null>(null);

  const handleSaveProp = async (data: {
    prop_id: string;
    description: string;
    role: string;
    grounding_scope: "single_message" | "conversation_history";
    arity: number;
    arg_descriptions: string[];
  }) => {
    setPropWarning(null);
    if (editingProp) {
      await updateProposition(editingProp.prop_id, {
        description: data.description,
        role: data.role,
        grounding_scope: data.grounding_scope,
        arg_descriptions: data.arg_descriptions,
      });
    } else {
      const result = await createProposition(data);
      if (result.warning) {
        setPropWarning(result.warning);
      }
    }
    setShowPropEditor(false);
    setEditingProp(null);
  };

  const handleEditProp = (proposition: Proposition) => {
    setEditingProp(proposition);
    setShowPropEditor(true);
  };

  const handleDeleteProp = async (propId: string) => {
    await deleteProposition(propId);
  };

  const handleViewPrompt = async (proposition: Proposition) => {
    setShowPromptModal(true);
    setPromptLoading(true);
    setPromptError(null);
    setPromptPreview(null);
    try {
      const preview = await getPropositionGroundingPrompt(proposition.prop_id);
      setPromptPreview(preview);
    } catch (err) {
      setPromptError(
        err instanceof Error
          ? err.message
          : "Failed to load grounding prompt preview",
      );
    } finally {
      setPromptLoading(false);
    }
  };

  const handleSavePolicy = async (data: {
    name: string;
    formula_str: string;
  }) => {
    await createPolicy({ ...data, enabled: true });
    setShowFormulaBuilder(false);
  };

  const handleDeletePolicy = async (policyId: string) => {
    await deletePolicy(policyId);
  };

  const propsLoading =
    propositions.status === "loading" || propositions.status === "idle";
  const policiesLoading =
    policies.status === "loading" || policies.status === "idle";
  const propsList = propositions.status === "success" ? propositions.data : [];
  const policiesList = policies.status === "success" ? policies.data : [];

  if (propsLoading && policiesLoading) {
    return (
      <div
        className="flex h-full items-center justify-center"
        data-testid="rules-loading"
      >
        <Loader2 className="h-8 w-8 animate-spin text-accent" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-3xl space-y-8 p-6" data-testid="rules-view">
      {/* Zero-shot warning */}
      {propWarning && (
        <div className="rounded-none border border-terminal-amber/30 bg-terminal-amber/5 px-4 py-3 text-sm text-terminal-amber">
          <div className="flex items-start justify-between gap-2">
            <div>
              <p className="font-medium">Predicate saved with zero-shot mode</p>
              <p className="mt-1 text-xs text-terminal-amber/80">{propWarning}</p>
            </div>
            <button
              onClick={() => setPropWarning(null)}
              className="text-terminal-amber/60 hover:text-terminal-amber text-lg leading-none"
            >
              &times;
            </button>
          </div>
        </div>
      )}

      {/* Predicates Section */}
      <section>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-mono font-bold text-accent uppercase tracking-wider">Predicates</h2>
          <button
            onClick={() => {
              setEditingProp(null);
              setShowPropEditor(true);
            }}
            className="btn-primary flex items-center gap-1.5 rounded-none px-3 py-2 text-sm font-medium"
            aria-label="Add predicate"
            data-testid="add-proposition"
          >
            <Plus size={16} />
            Add
          </button>
        </div>

        {propositions.status === "error" && (
          <p className="text-sm text-terminal-red" data-testid="propositions-error">
            {propositions.error}
          </p>
        )}

        {propsList.length === 0 && !propsLoading && (
          <p className="text-sm text-terminal-dim" data-testid="no-propositions">
            No predicates defined yet. Click "Add" to create one.
          </p>
        )}

        <div className="space-y-3">
          {propsList.map((p) => (
            <PropositionCard
              key={p.prop_id}
              proposition={p}
              onEdit={handleEditProp}
              onDelete={handleDeleteProp}
              onViewPrompt={handleViewPrompt}
            />
          ))}
        </div>
      </section>

      {/* Policies Section */}
      <section>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-mono font-bold text-accent uppercase tracking-wider">Policies</h2>
          <button
            onClick={() => setShowFormulaBuilder(true)}
            className="btn-primary flex items-center gap-1.5 rounded-none px-3 py-2 text-sm font-medium"
            aria-label="Add policy"
            data-testid="add-policy"
          >
            <Plus size={16} />
            Add
          </button>
        </div>

        {policies.status === "error" && (
          <p className="text-sm text-terminal-red" data-testid="policies-error">
            {policies.error}
          </p>
        )}

        {policiesList.length === 0 && !policiesLoading && (
          <p className="text-sm text-terminal-dim" data-testid="no-policies">
            No policies defined yet.{" "}
            {propsList.length === 0
              ? 'You can still click "Add" and use the built-in predicate user_turn.'
              : 'Click "Add" to create one.'}
          </p>
        )}

        <div className="space-y-3">
          {policiesList.map((p) => (
            <RuleCard
              key={p.policy_id}
              policy={p}
              onToggle={togglePolicy}
              onDelete={handleDeletePolicy}
            />
          ))}
        </div>
      </section>

      {/* Predicate Editor Modal */}
      <Modal
        open={showPropEditor}
        onClose={() => {
          setShowPropEditor(false);
          setEditingProp(null);
        }}
        title={editingProp ? "Edit Predicate" : "New Predicate"}
      >
        <PropositionEditor
          initial={editingProp ?? undefined}
          onSave={handleSaveProp}
          onCancel={() => {
            setShowPropEditor(false);
            setEditingProp(null);
          }}
        />
      </Modal>

      {/* Formula Builder Modal */}
      <Modal
        open={showFormulaBuilder}
        onClose={() => setShowFormulaBuilder(false)}
        title="New Policy"
      >
        <FormulaBuilder
          propositions={propsList}
          onSave={handleSavePolicy}
          onCancel={() => setShowFormulaBuilder(false)}
          onValidate={validateFormula}
        />
      </Modal>

      <Modal
        open={showPromptModal}
        onClose={() => {
          setShowPromptModal(false);
          setPromptPreview(null);
          setPromptError(null);
          setPromptLoading(false);
        }}
        title="Grounding Prompt Preview"
      >
        <div className="space-y-4">
          {promptLoading && (
            <div className="flex items-center gap-2 text-sm text-terminal-dim">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading prompt...
            </div>
          )}
          {promptError && <p className="text-sm text-terminal-red">{promptError}</p>}
          {promptPreview && (
            <>
              <div>
                <p className="mb-1 text-xs font-semibold text-terminal-dim uppercase tracking-wider">
                  System Prompt
                </p>
                <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded-none border border-border bg-dark-primary p-3 text-xs text-terminal-text">
                  {promptPreview.system_prompt}
                </pre>
              </div>
              <div>
                <p className="mb-1 text-xs font-semibold text-terminal-dim uppercase tracking-wider">
                  User Prompt
                </p>
                <pre className="max-h-80 overflow-auto whitespace-pre-wrap rounded-none border border-border bg-dark-primary p-3 text-xs text-terminal-text">
                  {promptPreview.user_prompt}
                </pre>
              </div>
            </>
          )}
        </div>
      </Modal>
    </div>
  );
}
