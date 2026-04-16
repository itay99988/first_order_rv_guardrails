import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, Search } from "lucide-react";

import type { OpenRouterModel } from "@/types";

interface ModelComboboxProps {
  models: OpenRouterModel[];
  value: string;
  onChange: (modelId: string) => void;
  disabled?: boolean;
  placeholder?: string;
  "data-testid"?: string;
}

function formatContextLength(length: number | undefined): string | null {
  if (!length) return null;
  if (length >= 1_000_000) return `${(length / 1_000_000).toFixed(1)}M`;
  if (length >= 1_000) return `${Math.round(length / 1_000)}K`;
  return `${length}`;
}

function formatPrice(price: string | undefined): string | null {
  if (!price) return null;
  const num = parseFloat(price);
  if (isNaN(num)) return null;
  // Price is per token, convert to per 1M tokens
  const perMillion = num * 1_000_000;
  if (perMillion < 0.01) return "<$0.01";
  return `$${perMillion.toFixed(2)}`;
}

export default function ModelCombobox({
  models,
  value,
  onChange,
  disabled = false,
  placeholder = "Select a model",
  "data-testid": testId,
}: ModelComboboxProps) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");
  const [highlightIndex, setHighlightIndex] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const selectedModel = models.find((m) => m.id === value);
  const displayText = selectedModel?.name || value || placeholder;

  const filtered = search.trim()
    ? models.filter((m) => {
        const q = search.toLowerCase();
        return (
          m.id.toLowerCase().includes(q) || m.name.toLowerCase().includes(q)
        );
      })
    : models;

  // Reset highlight when filter changes
  useEffect(() => {
    setHighlightIndex(0);
  }, [search]);

  // Focus search input when dropdown opens
  useEffect(() => {
    if (open) {
      requestAnimationFrame(() => searchRef.current?.focus());
    } else {
      setSearch("");
    }
  }, [open]);

  // Click outside to close
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  // Scroll highlighted item into view
  useEffect(() => {
    if (!open || !listRef.current) return;
    const item = listRef.current.children[highlightIndex] as
      | HTMLElement
      | undefined;
    item?.scrollIntoView({ block: "nearest" });
  }, [highlightIndex, open]);

  const handleSelect = useCallback(
    (modelId: string) => {
      onChange(modelId);
      setOpen(false);
    },
    [onChange],
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      setOpen(false);
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlightIndex((prev) => Math.min(prev + 1, filtered.length - 1));
      return;
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightIndex((prev) => Math.max(prev - 1, 0));
      return;
    }
    if (e.key === "Enter" && filtered.length > 0) {
      e.preventDefault();
      handleSelect(filtered[highlightIndex].id);
    }
  };

  const safeTestId = (modelId: string) => modelId.replace(/\//g, "-");

  return (
    <div ref={containerRef} className="relative" data-testid={testId}>
      <button
        type="button"
        onClick={() => !disabled && setOpen(!open)}
        aria-disabled={disabled || undefined}
        className={`flex w-full items-center justify-between rounded-none border border-border bg-dark-primary px-3 py-2 text-left text-sm font-mono focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/30 ${
          disabled
            ? "cursor-not-allowed opacity-50"
            : "cursor-pointer hover:bg-dark-hover"
        }`}
        data-testid={testId ? `${testId}-trigger` : undefined}
      >
        <span className={value ? "text-terminal-text" : "text-terminal-dim"}>
          {displayText}
        </span>
        <ChevronDown size={16} className="text-terminal-dim" />
      </button>

      {open && !disabled && (
        <div
          className="absolute z-50 mt-1 w-full rounded-none border border-accent/20 bg-dark-primary"
          data-testid={testId ? `${testId}-dropdown` : undefined}
        >
          <div className="border-b border-border p-2">
            <div className="relative">
              <Search
                size={14}
                className="absolute left-2.5 top-1/2 -translate-y-1/2 text-terminal-dim"
              />
              <input
                ref={searchRef}
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Search models..."
                className="w-full rounded-none border-b border-border bg-dark-primary py-1.5 pl-8 pr-3 text-sm text-terminal-text placeholder-terminal-dim font-mono focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/30"
                data-testid={testId ? `${testId}-search` : undefined}
              />
            </div>
          </div>
          <div ref={listRef} className="max-h-64 overflow-y-auto">
            {filtered.length === 0 ? (
              <div className="px-3 py-6 text-center text-sm text-terminal-dim font-mono">
                No models found
              </div>
            ) : (
              filtered.map((m, i) => {
                const ctx = formatContextLength(m.context_length);
                const promptPrice = formatPrice(m.pricing?.prompt);
                const completionPrice = formatPrice(m.pricing?.completion);
                const priceStr =
                  promptPrice && completionPrice
                    ? `${promptPrice}/${completionPrice}`
                    : null;

                return (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => handleSelect(m.id)}
                    onMouseEnter={() => setHighlightIndex(i)}
                    className={`flex w-full items-center justify-between px-3 py-2 text-left text-sm font-mono ${
                      i === highlightIndex
                        ? "bg-accent-muted text-accent"
                        : "text-terminal-text hover:bg-dark-hover"
                    } ${m.id === value ? "font-medium" : ""}`}
                    data-testid={
                      testId
                        ? `${testId}-option-${safeTestId(m.id)}`
                        : undefined
                    }
                  >
                    <span className="min-w-0 truncate">{m.name}</span>
                    <span className="ml-2 flex shrink-0 items-center gap-2 text-xs text-terminal-dim">
                      {ctx && (
                        <span className="rounded-none bg-dark-secondary px-1.5 py-0.5 font-mono">
                          {ctx}
                        </span>
                      )}
                      {priceStr && <span className="bg-dark-secondary px-1.5 py-0.5 rounded-none">{priceStr}</span>}
                    </span>
                  </button>
                );
              })
            )}
          </div>
        </div>
      )}
    </div>
  );
}
