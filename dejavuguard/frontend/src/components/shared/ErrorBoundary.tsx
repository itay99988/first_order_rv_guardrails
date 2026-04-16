import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";
import { AlertTriangle } from "lucide-react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("ErrorBoundary caught:", error, info);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <div
          className="flex flex-col items-center justify-center gap-4 p-8"
          data-testid="error-boundary-fallback"
        >
          <AlertTriangle className="h-10 w-10 text-terminal-amber" />
          <h2 className="text-lg font-semibold text-terminal-bright font-mono">
            Something went wrong
          </h2>
          <p className="text-sm text-terminal-dim font-mono">{this.state.error?.message}</p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="btn-primary rounded-none px-4 py-2 text-sm font-medium font-mono"
          >
            Try again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
