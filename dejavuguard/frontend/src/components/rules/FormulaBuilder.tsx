import { useRef, useState } from "react";
import { CheckCircle, XCircle } from "lucide-react";

import type { FormulaValidation, Proposition } from "@/types";

interface FormulaBuilderProps {
  propositions: Proposition[];
  onSave: (data: { name: string; formula_str: string }) => void;
  onCancel: () => void;
  onValidate: (name: string, formulaStr: string) => Promise<FormulaValidation>;
}

const temporalOps = [
  { label: "H", insert: "H ", desc: "Historically (all past steps)" },
  { label: "P", insert: "P ", desc: "Previously (some past step)" },
  { label: "@", insert: "@ ", desc: "Previous step (Yesterday)" },
  { label: "S", insert: " S ", desc: "Since" },
  { label: "P[<=n]", insert: "P[<=] ", desc: "Previously within n steps" },
  { label: "H[>n]", insert: "H[>] ", desc: "Historically beyond n steps" },
] as const;

const quantifierOps = [
  { label: "forall", insert: "forall x . ", desc: "For all SEEN values of x (recommended default)" },
  { label: "exists", insert: "exists x . ", desc: "There exists a SEEN value of x (recommended default)" },
  { label: "Forall", insert: "Forall x . ", desc: "For ALL values of x (infinite domain)" },
  { label: "Exists", insert: "Exists x . ", desc: "There exists ANY value of x (infinite domain)" },
] as const;

const booleanOps = [
  { label: "!", insert: "!", desc: "Not" },
  { label: "&", insert: " & ", desc: "And" },
  { label: "|", insert: " | ", desc: "Or" },
  { label: "->", insert: " -> ", desc: "Implies" },
  { label: "(", insert: "(", desc: "Open paren" },
  { label: ")", insert: ")", desc: "Close paren" },
] as const;

// The `where` keyword introduces local rule definitions. Insert it on its
// own line for readability — rule bodies follow on indented lines.
const ruleOps = [
  { label: "where", insert: "\nwhere\n  ", desc: "Begin local rule definitions" },
  { label: ":=", insert: " := ", desc: "Define a rule body" },
] as const;



const builtInPropositions = [
  {
    prop_id: "user_turn",
    role: "builtin",
    description: "True when the current message is from the user, false otherwise.",
  },
] as const;

export default function FormulaBuilder({
  propositions,
  onSave,
  onCancel,
  onValidate,
}: FormulaBuilderProps) {
  const [name, setName] = useState("");
  const [formula, setFormula] = useState("");
  const [validation, setValidation] = useState<FormulaValidation | null>(null);
  const [validating, setValidating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [showReference, setShowReference] = useState(false);
  const formulaRef = useRef<HTMLTextAreaElement>(null);

  // Validation happens on save, not on every keystroke

  const insertAtCursor = (text: string) => {
    const input = formulaRef.current;
    if (!input) {
      setFormula((prev) => prev + text);
      return;
    }
    const start = input.selectionStart ?? formula.length;
    const end = input.selectionEnd ?? formula.length;
    const next = formula.slice(0, start) + text + formula.slice(end);
    setFormula(next);
    // Move cursor after inserted text
    requestAnimationFrame(() => {
      const pos = start + text.length;
      input.setSelectionRange(pos, pos);
      input.focus();
    });
  };

  // Detect when the user just finished typing the standalone keyword `where`
  // and auto-format it onto its own line followed by an indented blank line
  // ready for rule definitions. Triggers only on a freshly typed `where ` at
  // a word boundary.
  const WHERE_TRIGGER = /(^|[^A-Za-z_\n])where(\s)$/;
  const handleFormulaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const nextValue = e.target.value;
    const caret = e.target.selectionStart ?? nextValue.length;
    const match = WHERE_TRIGGER.exec(nextValue.slice(0, caret));
    if (match) {
      const matchStart = caret - match[0].length + match[1].length;
      const prefix = nextValue.slice(0, matchStart);
      const suffix = nextValue.slice(caret);
      const leadingNewline = prefix.endsWith("\n") ? "" : "\n";
      const replacement = `${leadingNewline}where\n  `;
      const formatted = `${prefix}${replacement}${suffix}`;
      setFormula(formatted);
      requestAnimationFrame(() => {
        const input = formulaRef.current;
        if (input) {
          const pos = prefix.length + replacement.length;
          input.setSelectionRange(pos, pos);
          input.focus();
        }
      });
      return;
    }
    setFormula(nextValue);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !formula.trim()) return;
    setValidating(true);
    setValidation(null);
    try {
      const result = await onValidate(name.trim(), formula.trim());
      setValidation(result);
      setValidating(false);
      if (!result.valid) return;
      setSaving(true);
      onSave({ name: name.trim(), formula_str: formula.trim() });
    } catch {
      setValidation({ valid: false, error: "Validation request failed", propositions: [] });
      setValidating(false);
    } finally {
      setSaving(false);
    }
  };

  const propositionChips = [
    ...propositions.map((p) => ({
      prop_id: p.prop_id,
      role: p.role,
      arity: p.arity ?? 0,
      description: p.description,
      // Build the display label: p_fraud or p_transfer(a1, a2, a3)
      label: p.arity > 0
        ? `${p.prop_id}(${Array.from({ length: p.arity }, (_, i) => `a${i + 1}`).join(",")})`
        : p.prop_id,
      // What to insert in the formula when clicked
      insert: p.arity > 0
        ? `${p.prop_id}()`
        : p.prop_id,
    })),
    ...builtInPropositions.map((p) => ({
      ...p,
      arity: 0,
      label: p.prop_id,
      insert: p.prop_id,
    })),
  ];

  return (
    <form onSubmit={handleSubmit} data-testid="formula-builder" className="flex flex-col max-h-[75vh]">
      {/* Scrollable content */}
      <div className="space-y-4 overflow-y-auto flex-1 pr-1">
        <div>
          <label className="mb-1 block text-terminal-text font-mono text-sm" htmlFor="policy-name">
            Policy Name
          </label>
          <input
            id="policy-name"
            name="policy_name"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Fraud Prevention Policy"
            className="w-full rounded-none border border-border bg-dark-primary px-3 py-2 text-sm text-terminal-bright placeholder-terminal-dim focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20"
            data-testid="policy-name-input"
          />
        </div>

        {propositionChips.length > 0 && (
          <div>
            <p className="mb-2 text-terminal-text font-mono text-sm">Available Predicates</p>
            <div className="flex flex-wrap gap-1.5" data-testid="proposition-chips">
              {propositionChips.map((p) => (
                <button
                  key={p.prop_id}
                  type="button"
                  onClick={() => insertAtCursor(p.insert)}
                  className="rounded-none bg-accent-muted border border-accent/20 text-accent font-mono text-xs px-2.5 py-1 hover:bg-accent/15 transition-colors"
                  title={`[${p.role}] ${p.arity > 0 ? `arity: ${p.arity} — ` : ""}${p.description}`}
                  data-testid={`chip-${p.prop_id}`}
                >
                  {p.label}
                </button>
              ))}
            </div>
            <p className="mt-1 text-xs text-terminal-dim">
              Built-in predicate: <code className="font-mono text-accent">user_turn</code> = true on user messages. Hover predicates for details.
            </p>
          </div>
        )}

        <div>
          <p className="mb-1 text-terminal-text font-mono text-sm">Operators</p>
          <div className="flex flex-wrap gap-1.5" data-testid="operator-buttons">
            {temporalOps.map(({ label, insert, desc }) => (
              <button key={label} type="button" onClick={() => insertAtCursor(insert)}
                className="rounded-none border border-border text-terminal-text font-mono px-2 py-0.5 text-xs hover:bg-dark-hover hover:text-accent transition-colors"
                title={desc} data-testid={`op-${label.replace(/[()[\]<>=]/g, "")}`}>{label}</button>
            ))}
            {quantifierOps.map(({ label, insert, desc }) => (
              <button key={label} type="button" onClick={() => insertAtCursor(insert)}
                className="rounded-none border border-accent/30 text-accent font-mono px-2 py-0.5 text-xs hover:bg-accent/10 transition-colors"
                title={desc} data-testid={`op-${label}`}>{label}</button>
            ))}
            {booleanOps.map(({ label, insert, desc }) => (
              <button key={label} type="button" onClick={() => insertAtCursor(insert)}
                className="rounded-none border border-border text-terminal-text font-mono px-2 py-0.5 text-xs hover:bg-dark-hover hover:text-accent transition-colors"
                title={desc} data-testid={`op-${label.replace(/[()]/g, "")}`}>{label}</button>
            ))}
            {ruleOps.map(({ label, insert, desc }) => (
              <button key={label} type="button" onClick={() => insertAtCursor(insert)}
                className="rounded-none border border-accent/30 text-accent font-mono px-2 py-0.5 text-xs hover:bg-accent/10 transition-colors"
                title={desc} data-testid={`op-${label.replace(/[:=]/g, "")}`}>{label}</button>
            ))}
          </div>
        </div>

        <div>
          <label className="mb-1 block text-terminal-text font-mono text-sm" htmlFor="formula">Formula</label>
          <textarea
            ref={formulaRef}
            id="formula"
            name="formula"
            rows={4}
            value={formula}
            onChange={handleFormulaChange}
            placeholder={"H (P p_fraud -> !q_comply)\n\nor with local rules:\nForall m . p(m) -> exists b . (@ r(m,b)) S q(m,b)\nwhere\n  r(m,b) := ...\n"}
            className="w-full rounded-none border border-border bg-dark-primary px-3 py-2 font-mono text-sm text-accent placeholder-terminal-dim focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20 resize-y"
            data-testid="formula-input"
          />
          {formula.trim() && !validating && validation && (
            <div className="mt-1 flex items-center gap-1.5 text-xs" data-testid="formula-validation">
              {validation.valid ? (
                <><CheckCircle size={14} className="text-terminal-green" /><span className="text-terminal-green">Formula is valid</span></>
              ) : (
                <><XCircle size={14} className="text-terminal-red" /><span className="text-terminal-red">{validation.error || "Invalid formula"}</span></>
              )}
            </div>
          )}
        </div>

        {/* Collapsible reference */}
        <button
          type="button"
          onClick={() => setShowReference(!showReference)}
          className="text-xs text-terminal-dim hover:text-accent transition-colors flex items-center gap-1"
        >
          <span>{showReference ? "▾" : "▸"}</span> DejaVu Operators Reference
        </button>
        {showReference && (
          <div className="rounded-none bg-dark-primary border border-border p-3 text-xs">
            <table className="w-full">
              <tbody className="text-terminal-dim">
                <tr><td className="pr-3 py-0.5 font-mono text-accent/70 whitespace-nowrap">H φ</td><td>φ held at every past step</td></tr>
                <tr><td className="pr-3 py-0.5 font-mono text-accent/70 whitespace-nowrap">P φ</td><td>φ held at some past step</td></tr>
                <tr><td className="pr-3 py-0.5 font-mono text-accent/70 whitespace-nowrap">@ φ</td><td>φ held at the previous step</td></tr>
                <tr><td className="pr-3 py-0.5 font-mono text-accent/70 whitespace-nowrap">φ S ψ</td><td>ψ occurred and φ held continuously since</td></tr>
                <tr><td className="pr-3 py-0.5 font-mono text-accent/70 whitespace-nowrap">P[&lt;=n] φ</td><td>φ held within the last n steps</td></tr>
                <tr><td className="pr-3 py-0.5 font-mono text-accent/70 whitespace-nowrap">H[&gt;n] φ</td><td>φ held at all steps beyond n ago</td></tr>
                <tr><td className="pr-3 py-0.5 font-mono text-accent/70 whitespace-nowrap">forall x . φ(x)</td><td>for all <strong>seen</strong> values of x, φ holds (default)</td></tr>
                <tr><td className="pr-3 py-0.5 font-mono text-accent/70 whitespace-nowrap">exists x . φ(x)</td><td>there exists a <strong>seen</strong> value of x where φ holds (default)</td></tr>
                <tr><td className="pr-3 py-0.5 font-mono text-accent/70 whitespace-nowrap">Forall x . φ(x)</td><td>for <strong>all</strong> values of x — infinite domain</td></tr>
                <tr><td className="pr-3 py-0.5 font-mono text-accent/70 whitespace-nowrap">Exists x . φ(x)</td><td>there exists <strong>any</strong> value of x — infinite domain</td></tr>
                <tr><td className="pr-3 py-0.5 font-mono text-accent/70 whitespace-nowrap">φ where r(x) := ψ</td><td>local rule definitions; <code>r(x)</code> in φ stands for ψ (rules called inside other rule bodies must be under <code>@</code>)</td></tr>
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Fixed footer with buttons */}
      <div className="flex justify-end gap-2 pt-4 mt-4 border-t border-border">
        <button
          type="button"
          onClick={onCancel}
          className="rounded-none border border-border px-4 py-2 text-sm font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
          data-testid="policy-cancel"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={!name.trim() || !formula.trim() || saving || validating}
          className="btn-primary rounded-none px-4 py-2 text-sm font-medium"
          data-testid="policy-save"
        >
          {saving || validating ? "Validating..." : "Save Policy"}
        </button>
      </div>
    </form>
  );
}
