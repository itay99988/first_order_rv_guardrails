import { useCallback, useEffect, useState } from "react";

import {
  getGroundingHealth,
  getGroundingModels,
  getOpenRouterModels,
  getSettings,
  updateSettings as apiUpdateSettings,
} from "@/api/client";
import type { AppSettings, AsyncState, OpenRouterModel } from "@/types";

export function useSettings() {
  const [settings, setSettings] = useState<AsyncState<AppSettings>>({
    status: "idle",
  });
  const [groundingModels, setGroundingModels] = useState<string[]>([]);
  const [openRouterModels, setOpenRouterModels] = useState<OpenRouterModel[]>(
    [],
  );
  const [groundingModelsError, setGroundingModelsError] = useState<
    string | null
  >(null);
  const [openRouterModelsError, setOpenRouterModelsError] = useState<
    string | null
  >(null);
  const [groundingHealth, setGroundingHealth] = useState<{
    healthy: boolean;
    provider: string;
  } | null>(null);

  const fetchSettings = useCallback(async () => {
    setSettings({ status: "loading" });
    try {
      const data = await getSettings();
      setSettings({ status: "success", data });
    } catch (err) {
      setSettings({
        status: "error",
        error: err instanceof Error ? err.message : "Failed to load settings",
      });
    }
  }, []);

  const updateSettingsAction = useCallback(async (newSettings: AppSettings) => {
    try {
      const data = await apiUpdateSettings(newSettings);
      setSettings({ status: "success", data });
      return data;
    } catch (err) {
      throw err instanceof Error ? err : new Error("Failed to update settings");
    }
  }, []);

  const fetchGroundingModels = useCallback(
    async (provider?: string, baseUrl?: string) => {
      setGroundingModelsError(null);
      try {
        const models = await getGroundingModels(provider, baseUrl);
        setGroundingModels(models);
      } catch (err) {
        setGroundingModels([]);
        const msg =
          err instanceof Error
            ? err.message
            : "Failed to load grounding models";
        setGroundingModelsError(msg);
      }
    },
    [],
  );

  const fetchOpenRouterModels = useCallback(async () => {
    setOpenRouterModelsError(null);
    try {
      const models = await getOpenRouterModels();
      setOpenRouterModels(models);
    } catch (err) {
      setOpenRouterModels([]);
      const msg =
        err instanceof Error ? err.message : "Failed to load OpenRouter models";
      setOpenRouterModelsError(msg);
    }
  }, []);

  const testGroundingConnection = useCallback(async () => {
    try {
      const health = await getGroundingHealth();
      setGroundingHealth(health);
      return health;
    } catch {
      setGroundingHealth({ healthy: false, provider: "" });
      return { healthy: false, provider: "" };
    }
  }, []);

  useEffect(() => {
    fetchSettings();
  }, [fetchSettings]);

  // Auto-fetch model lists once settings are loaded
  useEffect(() => {
    if (settings.status === "success") {
      fetchOpenRouterModels();
      fetchGroundingModels();
    }
  }, [settings.status, fetchOpenRouterModels, fetchGroundingModels]);

  return {
    settings,
    groundingModels,
    openRouterModels,
    groundingModelsError,
    openRouterModelsError,
    groundingHealth,
    fetchSettings,
    updateSettings: updateSettingsAction,
    fetchGroundingModels,
    fetchOpenRouterModels,
    testGroundingConnection,
  };
}
