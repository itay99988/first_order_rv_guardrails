import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { createSettings, createOpenRouterModelList } from "../../test/mocks";
import SettingsView from "./SettingsView";
import type { AppSettings, AsyncState } from "../../types";

// Mock the useSettings hook
const mockUseSettings = vi.fn();

vi.mock("@/hooks/useSettings", () => ({
  useSettings: () => mockUseSettings(),
}));

function createHookReturn(
  overrides: {
    settingsState?: AsyncState<AppSettings>;
    groundingModels?: string[];
    openRouterModels?: ReturnType<typeof createOpenRouterModelList>;
    groundingHealth?: { healthy: boolean; provider: string } | null;
  } = {},
) {
  return {
    settings: overrides.settingsState ?? {
      status: "success" as const,
      data: createSettings(),
    },
    groundingModels: overrides.groundingModels ?? ["mistral", "llama3"],
    openRouterModels: overrides.openRouterModels ?? createOpenRouterModelList(),
    groundingHealth: overrides.groundingHealth ?? null,
    fetchSettings: vi.fn(),
    updateSettings: vi.fn(),
    fetchGroundingModels: vi.fn(),
    fetchOpenRouterModels: vi.fn(),
    testGroundingConnection: vi
      .fn()
      .mockResolvedValue({ healthy: true, provider: "ollama" }),
  };
}

// jsdom does not implement scrollIntoView (needed by ModelCombobox rendered inside child components)
beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
});

describe("SettingsView", () => {
  // --- Loading state ---

  it('shows loading spinner when status is "loading"', () => {
    mockUseSettings.mockReturnValue(
      createHookReturn({ settingsState: { status: "loading" } }),
    );
    render(<SettingsView />);
    expect(screen.getByTestId("settings-loading")).toBeInTheDocument();
    expect(screen.queryByTestId("settings-view")).not.toBeInTheDocument();
  });

  it('shows loading spinner when status is "idle"', () => {
    mockUseSettings.mockReturnValue(
      createHookReturn({ settingsState: { status: "idle" } }),
    );
    render(<SettingsView />);
    expect(screen.getByTestId("settings-loading")).toBeInTheDocument();
  });

  // --- Error state ---

  it('shows error message when status is "error"', () => {
    mockUseSettings.mockReturnValue(
      createHookReturn({
        settingsState: { status: "error", error: "Network failure" },
      }),
    );
    render(<SettingsView />);
    expect(screen.getByTestId("settings-error")).toBeInTheDocument();
    expect(screen.getByText("Failed to load settings")).toBeInTheDocument();
    expect(screen.getByText("Network failure")).toBeInTheDocument();
  });

  // --- Success state ---

  it('renders full settings view when status is "success"', () => {
    mockUseSettings.mockReturnValue(createHookReturn());
    render(<SettingsView />);
    expect(screen.getByTestId("settings-view")).toBeInTheDocument();
    expect(screen.getByText("Settings")).toBeInTheDocument();
  });

  it("renders OpenRouterConfig section", () => {
    mockUseSettings.mockReturnValue(createHookReturn());
    render(<SettingsView />);
    expect(screen.getByText("Chat Model (OpenRouter)")).toBeInTheDocument();
  });

  it("renders GroundingConfig section", () => {
    mockUseSettings.mockReturnValue(createHookReturn());
    render(<SettingsView />);
    expect(screen.getByText("Grounding Model")).toBeInTheDocument();
  });

  it("renders GroundingPromptEditor section", () => {
    mockUseSettings.mockReturnValue(createHookReturn());
    render(<SettingsView />);
    expect(screen.getByText("Grounding Prompt")).toBeInTheDocument();
  });

  it('does not render loading or error states when status is "success"', () => {
    mockUseSettings.mockReturnValue(createHookReturn());
    render(<SettingsView />);
    expect(screen.queryByTestId("settings-loading")).not.toBeInTheDocument();
    expect(screen.queryByTestId("settings-error")).not.toBeInTheDocument();
  });
});
