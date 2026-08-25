import { useEffect, useRef } from "react";
import { X } from "lucide-react";

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
}

export default function Modal({ open, onClose, title, children }: ModalProps) {
  const overlayRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
      data-testid="modal-overlay"
      onClick={(e) => {
        if (e.target === overlayRef.current) onClose();
      }}
    >
      <div
        // Capped at the viewport and scrolled internally: without this a tall
        // body -- the add-policy flow's rule list, say -- grows the panel past
        // the screen in both directions, and the confirm button ends up
        // somewhere the user cannot reach or scroll to. The header stays put
        // so the title and close button survive a long body.
        className="flex max-h-[90vh] w-full max-w-lg flex-col rounded-none border border-accent/30 bg-dark-primary"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        data-testid="modal"
      >
        <div className="shrink-0 px-6 pt-6">
        {/* Decorative terminal frame; the <h2> below carries the real title,
            so this must not be announced a second time. */}
        <div aria-hidden="true" className="mb-3 text-xs text-terminal-dim font-mono">
          ┌── {title} ──
        </div>
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-mono uppercase tracking-wider text-accent">{title}</h2>
          <button
            onClick={onClose}
            className="p-1 text-terminal-dim hover:text-terminal-red"
            aria-label="Close modal"
            data-testid="modal-close"
          >
            <X size={20} />
          </button>
        </div>
        </div>
        <div
          className="min-h-0 flex-1 overflow-y-auto px-6 pb-6 pt-4"
          data-testid="modal-body"
        >
          {children}
        </div>
      </div>
    </div>
  );
}
