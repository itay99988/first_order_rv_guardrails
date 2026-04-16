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
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80"
      data-testid="modal-overlay"
      onClick={(e) => {
        if (e.target === overlayRef.current) onClose();
      }}
    >
      <div
        className="w-full max-w-lg rounded-none border border-accent/30 bg-dark-primary p-6"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        data-testid="modal"
      >
        <div className="mb-3 text-xs text-terminal-dim font-mono">┌── {title} ──</div>
        <div className="mb-4 flex items-center justify-between">
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
        {children}
      </div>
    </div>
  );
}
