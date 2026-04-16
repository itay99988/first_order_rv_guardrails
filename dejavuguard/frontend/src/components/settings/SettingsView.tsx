import { Loader2 } from "lucide-react";

import { useSettings } from "@/hooks/useSettings";
import GroundingConfig from "./GroundingConfig";
import GroundingPromptEditor from "./GroundingPromptEditor";
import OpenRouterConfig from "./OpenRouterConfig";

export default function SettingsView() {
  const {
    settings,
    groundingModels,
    openRouterModels,
    groundingModelsError,
    openRouterModelsError,
    groundingHealth,
    updateSettings,
    fetchGroundingModels,
    fetchOpenRouterModels,
    testGroundingConnection,
  } = useSettings();

  if (settings.status === "loading" || settings.status === "idle") {
    return (
      <div
        className="flex h-full items-center justify-center"
        data-testid="settings-loading"
      >
        <Loader2 className="h-8 w-8 animate-spin text-accent" />
      </div>
    );
  }

  if (settings.status === "error") {
    return (
      <div
        className="flex h-full items-center justify-center"
        data-testid="settings-error"
      >
        <div className="text-center">
          <p className="text-terminal-bright font-mono">
            Failed to load settings
          </p>
          <p className="mt-1 text-sm text-terminal-dim">{settings.error}</p>
        </div>
      </div>
    );
  }

  const handleUpdate = async (newSettings: typeof settings.data) => {
    try {
      await updateSettings(newSettings);
    } catch {
      // Error is handled by the hook
    }
  };

  return (
    <div
      className="mx-auto max-w-3xl space-y-6 p-6"
      data-testid="settings-view"
    >
      <h2 className="text-lg font-mono font-bold text-accent uppercase tracking-wider">Settings</h2>

      <OpenRouterConfig
        settings={settings.data}
        openRouterModels={openRouterModels}
        modelsError={openRouterModelsError}
        onUpdate={handleUpdate}
        onFetchModels={fetchOpenRouterModels}
      />

      <GroundingConfig
        settings={settings.data}
        groundingModels={groundingModels}
        openRouterModels={openRouterModels}
        groundingModelsError={groundingModelsError}
        openRouterModelsError={openRouterModelsError}
        groundingHealth={groundingHealth}
        onUpdate={handleUpdate}
        onFetchModels={fetchGroundingModels}
        onFetchOpenRouterModels={fetchOpenRouterModels}
        onTestConnection={testGroundingConnection}
      />

      <GroundingPromptEditor settings={settings.data} onUpdate={handleUpdate} />
    </div>
  );
}
