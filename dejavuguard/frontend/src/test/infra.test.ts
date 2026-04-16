import { describe, it, expect } from "vitest";
import {
  createOpenRouterModelList,
  createSettings,
  createProposition,
  createPolicy,
} from "./mocks";

describe("Test infrastructure", () => {
  it("creates mock models", () => {
    const models = createOpenRouterModelList(5);
    expect(models).toHaveLength(5);
    expect(models[0].id).toBe("anthropic/claude-3.5-sonnet");
  });

  it("creates mock settings", () => {
    const settings = createSettings();
    expect(settings.openrouter_api_key).toBe("sk-or-v1-test-key-12345");
    expect(settings.grounding.provider).toBe("ollama");
  });

  it("creates mock proposition", () => {
    const prop = createProposition();
    expect(prop.prop_id).toBe("p_fraud");
  });

  it("creates mock policy", () => {
    const policy = createPolicy();
    expect(policy.formula_str).toBe("H(p_fraud -> !q_comply)");
  });
});
