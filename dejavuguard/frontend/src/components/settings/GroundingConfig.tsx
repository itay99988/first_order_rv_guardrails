import { useEffect, useState } from "react";
import { CheckCircle, Loader2, XCircle } from "lucide-react";

import ModelCombobox from "@/components/shared/ModelCombobox";
import type {
  AppSettings,
  GroundingProvider,
  GroundingSettings,
  OpenRouterModel,
} from "@/types";

interface GroundingConfigProps {
  settings: AppSettings;
  groundingModels: string[];
  openRouterModels: OpenRouterModel[];
  groundingModelsError: string | null;
  openRouterModelsError: string | null;
  groundingHealth: { healthy: boolean; provider: string } | null;
  onUpdate: (settings: AppSettings) => void;
  onFetchModels: (provider?: string, baseUrl?: string) => void;
  onFetchOpenRouterModels: () => void;
  onTestConnection: () => Promise<{ healthy: boolean; provider: string }>;
}

const providers: { value: GroundingProvider; label: string }[] = [
  { value: "ollama", label: "Ollama" },
  { value: "lmstudio", label: "LM Studio" },
  { value: "vllm", label: "vLLM" },
  { value: "custom", label: "Custom (OpenAI-compatible)" },
  { value: "openrouter", label: "OpenRouter" },
];

const defaultUrls: Record<GroundingProvider, string> = {
  ollama: "http://localhost:11434",
  lmstudio: "http://localhost:1234",
  vllm: "http://localhost:8000",
  custom: "http://localhost:8080",
  openrouter: "",
};

export default function GroundingConfig({
  settings,
  groundingModels,
  openRouterModels,
  groundingModelsError,
  openRouterModelsError,
  groundingHealth,
  onUpdate,
  onFetchModels,
  onFetchOpenRouterModels,
  onTestConnection,
}: GroundingConfigProps) {
  const [grounding, setGrounding] = useState<GroundingSettings>(
    settings.grounding,
  );
  const [testing, setTesting] = useState(false);

  useEffect(() => {
    setGrounding(settings.grounding);
  }, [settings]);

  const isOpenRouter = grounding.provider === "openrouter";

  const handleProviderChange = (provider: GroundingProvider) => {
    const newBaseUrl = defaultUrls[provider];
    const updated: GroundingSettings = {
      ...grounding,
      provider,
      base_url: newBaseUrl,
    };
    setGrounding(updated);
    if (provider === "openrouter") {
      onFetchOpenRouterModels();
    } else {
      // Fetch models for the new provider using query params
      // (avoids needing to save settings first)
      onFetchModels(provider, newBaseUrl);
    }
  };

  const handleTest = async () => {
    setTesting(true);
    // Save current grounding settings before testing
    onUpdate({ ...settings, grounding });
    await new Promise((r) => setTimeout(r, 300));
    await onTestConnection();
    onFetchModels();
    setTesting(false);
  };

  const handleSave = () => {
    onUpdate({ ...settings, grounding });
  };

  return (
    <div
      className="rounded-none border border-border bg-dark-surface p-6"
      data-testid="grounding-config"
    >
      <h3 className="mb-4 text-sm font-mono font-bold text-accent uppercase tracking-wider">
        Grounding Model
      </h3>

      <div className="space-y-4">
        <div>
          <label className="mb-2 block text-terminal-text font-mono text-sm">
            Provider
          </label>
          <div
            className="flex flex-wrap gap-2"
            data-testid="grounding-provider-select"
          >
            {providers.map(({ value, label }) => (
              <button
                key={value}
                onClick={() => handleProviderChange(value)}
                className={`rounded-none border px-3 py-1.5 text-sm font-medium transition-colors ${
                  grounding.provider === value
                    ? "border-accent/40 bg-accent-muted text-accent"
                    : "border-border text-terminal-dim hover:bg-dark-hover hover:text-terminal-text"
                }`}
                data-testid={`provider-${value}`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {isOpenRouter && (
          <div>
            <label className="mb-2 block text-terminal-text font-mono text-sm">
              API Key
            </label>
            <div className="space-y-2" data-testid="api-key-mode">
              <label className="flex items-center gap-2 text-sm text-terminal-text">
                <input
                  type="radio"
                  name="api-key-mode"
                  checked={!grounding.api_key}
                  onChange={() => setGrounding({ ...grounding, api_key: "" })}
                  data-testid="api-key-mode-same"
                />
                Same as Chat Model
              </label>
              <label className="flex items-center gap-2 text-sm text-terminal-text">
                <input
                  type="radio"
                  name="api-key-mode"
                  checked={!!grounding.api_key}
                  onChange={() =>
                    setGrounding({
                      ...grounding,
                      api_key: grounding.api_key || " ",
                    })
                  }
                  data-testid="api-key-mode-separate"
                />
                Use separate key
              </label>
              {!!grounding.api_key && (
                <input
                  type="password"
                  value={grounding.api_key.trim()}
                  onChange={(e) =>
                    setGrounding({ ...grounding, api_key: e.target.value })
                  }
                  placeholder="sk-or-v1-..."
                  className="w-full rounded-none border border-border bg-dark-primary px-3 py-2 text-sm text-terminal-bright placeholder-terminal-dim focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20"
                  data-testid="grounding-api-key"
                />
              )}
            </div>
          </div>
        )}

        {!isOpenRouter && (
          <div>
            <label
              className="mb-1 block text-terminal-text font-mono text-sm"
              htmlFor="base-url"
            >
              Base URL
            </label>
            <input
              id="base-url"
              type="text"
              value={grounding.base_url}
              onChange={(e) =>
                setGrounding({ ...grounding, base_url: e.target.value })
              }
              className="w-full rounded-none border border-border bg-dark-primary px-3 py-2 text-sm text-terminal-bright placeholder-terminal-dim focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20"
              data-testid="grounding-base-url"
            />
          </div>
        )}

        <div>
          <label
            className="mb-1 block text-terminal-text font-mono text-sm"
            htmlFor="grounding-model"
          >
            Model
          </label>
          <div className="flex gap-2">
            {isOpenRouter ? (
              <div className="flex-1">
                <ModelCombobox
                  models={openRouterModels}
                  value={grounding.model}
                  onChange={(m) => setGrounding({ ...grounding, model: m })}
                  placeholder="Select grounding model"
                  data-testid="grounding-model-select"
                />
              </div>
            ) : (
              <select
                id="grounding-model"
                value={grounding.model}
                onChange={(e) =>
                  setGrounding({ ...grounding, model: e.target.value })
                }
                className="flex-1 rounded-none border border-border bg-dark-primary px-3 py-2 text-sm text-terminal-bright focus:border-accent/50 focus:outline-none focus:ring-1 focus:ring-accent/20"
                data-testid="grounding-model-select"
              >
                {grounding.model &&
                  !groundingModels.includes(grounding.model) && (
                    <option value={grounding.model}>{grounding.model}</option>
                  )}
                {groundingModels.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
                {groundingModels.length === 0 && !grounding.model && (
                  <option value="">Click "Test" to load models</option>
                )}
              </select>
            )}
            <button
              onClick={handleTest}
              disabled={testing}
              className="rounded-none border border-border px-3 py-2 text-sm font-medium text-terminal-dim hover:bg-dark-hover hover:text-terminal-text disabled:opacity-50"
              data-testid="test-grounding"
            >
              {testing ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                "Test"
              )}
            </button>
          </div>
          {!isOpenRouter &&
            groundingModelsError &&
            groundingModels.length === 0 && (
              <p
                className="mt-1 text-xs text-terminal-amber"
                data-testid="grounding-models-error"
              >
                {groundingModelsError} — Click "Test" to retry.
              </p>
            )}
          {isOpenRouter &&
            openRouterModelsError &&
            openRouterModels.length === 0 && (
              <p
                className="mt-1 text-xs text-terminal-amber"
                data-testid="grounding-openrouter-models-error"
              >
                {openRouterModelsError}
              </p>
            )}
        </div>

        {groundingHealth && (
          <div
            className="flex items-center gap-1.5 text-xs"
            data-testid="grounding-health-status"
          >
            {groundingHealth.healthy ? (
              <>
                <CheckCircle size={14} className="text-terminal-green" />
                <span className="text-terminal-green">
                  Connected ({groundingHealth.provider || grounding.provider})
                </span>
              </>
            ) : (
              <>
                <XCircle size={14} className="text-terminal-red" />
                <span className="text-terminal-red">Connection failed</span>
              </>
            )}
          </div>
        )}

        <div className="flex justify-end">
          <button
            onClick={handleSave}
            className="btn-primary rounded-none px-4 py-2 text-sm font-medium"
            data-testid="save-grounding"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
