import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor, act } from "@testing-library/react";

vi.mock("@/api/client", () => ({
  getSettings: vi.fn(),
  updateSettings: vi.fn(),
  getOpenRouterModels: vi.fn(),
  getGroundingModels: vi.fn(),
  getGroundingHealth: vi.fn(),
}));

import { useSettings } from "./useSettings";
import {
  getSettings,
  updateSettings as apiUpdateSettings,
  getOpenRouterModels,
  getGroundingModels,
  getGroundingHealth,
} from "@/api/client";
import type { AppSettings } from "@/types";
import { createSettings, createOpenRouterModelList } from "../test/mocks";

describe("useSettings", () => {
  const mockSettings = createSettings();
  const mockModels = createOpenRouterModelList(5);

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getSettings).mockResolvedValue(mockSettings);
    vi.mocked(getOpenRouterModels).mockResolvedValue(mockModels);
    vi.mocked(getGroundingModels).mockResolvedValue(["mistral", "llama3"]);
    vi.mocked(getGroundingHealth).mockResolvedValue({
      healthy: true,
      provider: "ollama",
    });
  });

  it("starts with idle status before mount effect runs", () => {
    vi.mocked(getSettings).mockReturnValue(new Promise(() => {})); // never resolves
    const { result } = renderHook(() => useSettings());
    // Status should transition away from idle very quickly, but initial render is idle
    expect(["idle", "loading"]).toContain(result.current.settings.status);
  });

  it("fetches settings on mount and transitions to loading then success", async () => {
    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("success");
    });

    expect(getSettings).toHaveBeenCalledOnce();
    expect(result.current.settings.status).toBe("success");
    if (result.current.settings.status === "success") {
      expect(result.current.settings.data).toEqual(mockSettings);
    }
  });

  it("contains fetched data in settings.data after success", async () => {
    const customSettings = createSettings({
      openrouter_model: "openai/gpt-4o",
    });
    vi.mocked(getSettings).mockResolvedValue(customSettings);

    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("success");
    });

    if (result.current.settings.status === "success") {
      expect(result.current.settings.data.openrouter_model).toBe(
        "openai/gpt-4o",
      );
      expect(result.current.settings.data.openrouter_api_key).toBe(
        "sk-or-v1-test-key-12345",
      );
    }
  });

  it("sets status to error when fetch fails", async () => {
    vi.mocked(getSettings).mockRejectedValue(new Error("Network error"));

    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("error");
    });

    if (result.current.settings.status === "error") {
      expect(result.current.settings.error).toBe("Network error");
    }
  });

  it("sets a generic error message when fetch fails with non-Error", async () => {
    vi.mocked(getSettings).mockRejectedValue("unknown failure");

    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("error");
    });

    if (result.current.settings.status === "error") {
      expect(result.current.settings.error).toBe("Failed to load settings");
    }
  });

  it("updateSettings calls apiUpdateSettings and updates state", async () => {
    const updatedSettings = createSettings({
      openrouter_model: "openai/gpt-4o-mini",
    });
    vi.mocked(apiUpdateSettings).mockResolvedValue(updatedSettings);

    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("success");
    });

    let returnValue: AppSettings | undefined;
    await act(async () => {
      returnValue = await result.current.updateSettings(updatedSettings);
    });

    expect(apiUpdateSettings).toHaveBeenCalledWith(updatedSettings);
    expect(returnValue).toEqual(updatedSettings);

    if (result.current.settings.status === "success") {
      expect(result.current.settings.data.openrouter_model).toBe(
        "openai/gpt-4o-mini",
      );
    }
  });

  it("updateSettings throws on API error", async () => {
    vi.mocked(apiUpdateSettings).mockRejectedValue(new Error("Save failed"));

    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("success");
    });

    await expect(
      act(async () => {
        await result.current.updateSettings(mockSettings);
      }),
    ).rejects.toThrow("Save failed");
  });

  it("auto-fetches grounding models after settings load", async () => {
    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("success");
    });

    await waitFor(() => {
      expect(result.current.groundingModels).toEqual(["mistral", "llama3"]);
    });

    expect(getGroundingModels).toHaveBeenCalled();
  });

  it("fetchGroundingModels calls getGroundingModels and sets groundingModels", async () => {
    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("success");
    });

    vi.mocked(getGroundingModels).mockClear();

    await act(async () => {
      await result.current.fetchGroundingModels();
    });

    expect(getGroundingModels).toHaveBeenCalledOnce();
    expect(result.current.groundingModels).toEqual(["mistral", "llama3"]);
  });

  it("fetchGroundingModels sets empty array on error", async () => {
    vi.mocked(getGroundingModels).mockRejectedValue(
      new Error("Server offline"),
    );

    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("success");
    });

    await act(async () => {
      await result.current.fetchGroundingModels();
    });

    expect(result.current.groundingModels).toEqual([]);
  });

  it("fetchGroundingModels sets groundingModelsError on error", async () => {
    vi.mocked(getGroundingModels).mockRejectedValue(
      new Error("Connection refused"),
    );

    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("success");
    });

    await act(async () => {
      await result.current.fetchGroundingModels();
    });

    expect(result.current.groundingModelsError).toBe("Connection refused");
  });

  it("fetchGroundingModels clears groundingModelsError on success", async () => {
    // First cause an error
    vi.mocked(getGroundingModels).mockRejectedValueOnce(new Error("Offline"));

    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("success");
    });

    // The auto-fetch on settings load may have already errored
    await waitFor(() => {
      expect(result.current.groundingModelsError).toBe("Offline");
    });

    // Now succeed
    vi.mocked(getGroundingModels).mockResolvedValue(["mistral", "llama3"]);

    await act(async () => {
      await result.current.fetchGroundingModels();
    });

    expect(result.current.groundingModelsError).toBeNull();
    expect(result.current.groundingModels).toEqual(["mistral", "llama3"]);
  });

  it("fetchGroundingModels sets generic message for non-Error throws", async () => {
    vi.mocked(getGroundingModels).mockRejectedValue("unknown");

    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("success");
    });

    await act(async () => {
      await result.current.fetchGroundingModels();
    });

    expect(result.current.groundingModelsError).toBe(
      "Failed to load grounding models",
    );
  });

  it("auto-fetches OpenRouter models after settings load", async () => {
    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("success");
    });

    await waitFor(() => {
      expect(result.current.openRouterModels).toEqual(mockModels);
    });

    expect(getOpenRouterModels).toHaveBeenCalled();
  });

  it("fetchOpenRouterModels calls getOpenRouterModels and sets openRouterModels", async () => {
    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("success");
    });

    vi.mocked(getOpenRouterModels).mockClear();

    await act(async () => {
      await result.current.fetchOpenRouterModels();
    });

    expect(getOpenRouterModels).toHaveBeenCalledOnce();
    expect(result.current.openRouterModels).toEqual(mockModels);
    expect(result.current.openRouterModels).toHaveLength(5);
  });

  it("fetchOpenRouterModels sets empty array on error", async () => {
    vi.mocked(getOpenRouterModels).mockRejectedValue(new Error("Unauthorized"));

    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("success");
    });

    await act(async () => {
      await result.current.fetchOpenRouterModels();
    });

    expect(result.current.openRouterModels).toEqual([]);
  });

  it("fetchOpenRouterModels sets openRouterModelsError on error", async () => {
    vi.mocked(getOpenRouterModels).mockRejectedValue(
      new Error("API key not configured"),
    );

    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("success");
    });

    await act(async () => {
      await result.current.fetchOpenRouterModels();
    });

    expect(result.current.openRouterModelsError).toBe("API key not configured");
  });

  it("fetchOpenRouterModels clears openRouterModelsError on success", async () => {
    // First cause an error
    vi.mocked(getOpenRouterModels).mockRejectedValueOnce(new Error("Bad key"));

    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("success");
    });

    await waitFor(() => {
      expect(result.current.openRouterModelsError).toBe("Bad key");
    });

    // Now succeed
    vi.mocked(getOpenRouterModels).mockResolvedValue(mockModels);

    await act(async () => {
      await result.current.fetchOpenRouterModels();
    });

    expect(result.current.openRouterModelsError).toBeNull();
    expect(result.current.openRouterModels).toEqual(mockModels);
  });

  it("groundingModelsError is initially null", () => {
    vi.mocked(getSettings).mockReturnValue(new Promise(() => {}));
    const { result } = renderHook(() => useSettings());
    expect(result.current.groundingModelsError).toBeNull();
  });

  it("openRouterModelsError is initially null", () => {
    vi.mocked(getSettings).mockReturnValue(new Promise(() => {}));
    const { result } = renderHook(() => useSettings());
    expect(result.current.openRouterModelsError).toBeNull();
  });

  it("testGroundingConnection calls getGroundingHealth and sets groundingHealth", async () => {
    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("success");
    });

    let healthResult: { healthy: boolean; provider: string } | undefined;
    await act(async () => {
      healthResult = await result.current.testGroundingConnection();
    });

    expect(getGroundingHealth).toHaveBeenCalledOnce();
    expect(result.current.groundingHealth).toEqual({
      healthy: true,
      provider: "ollama",
    });
    expect(healthResult).toEqual({ healthy: true, provider: "ollama" });
  });

  it("testGroundingConnection sets { healthy: false } on error", async () => {
    vi.mocked(getGroundingHealth).mockRejectedValue(
      new Error("Connection refused"),
    );

    const { result } = renderHook(() => useSettings());

    await waitFor(() => {
      expect(result.current.settings.status).toBe("success");
    });

    let healthResult: { healthy: boolean; provider: string } | undefined;
    await act(async () => {
      healthResult = await result.current.testGroundingConnection();
    });

    expect(result.current.groundingHealth).toEqual({
      healthy: false,
      provider: "",
    });
    expect(healthResult).toEqual({ healthy: false, provider: "" });
  });

  it("groundingHealth is initially null", () => {
    vi.mocked(getSettings).mockReturnValue(new Promise(() => {}));
    const { result } = renderHook(() => useSettings());
    expect(result.current.groundingHealth).toBeNull();
  });

  it("groundingModels is initially an empty array", () => {
    vi.mocked(getSettings).mockReturnValue(new Promise(() => {}));
    const { result } = renderHook(() => useSettings());
    expect(result.current.groundingModels).toEqual([]);
  });

  it("openRouterModels is initially an empty array", () => {
    vi.mocked(getSettings).mockReturnValue(new Promise(() => {}));
    const { result } = renderHook(() => useSettings());
    expect(result.current.openRouterModels).toEqual([]);
  });
});
