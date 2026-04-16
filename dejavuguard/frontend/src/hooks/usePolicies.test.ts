import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor, act } from "@testing-library/react";

vi.mock("@/api/client", () => ({
  getPropositions: vi.fn(),
  createProposition: vi.fn(),
  updateProposition: vi.fn(),
  deleteProposition: vi.fn(),
  getPolicies: vi.fn(),
  createPolicy: vi.fn(),
  updatePolicy: vi.fn(),
  deletePolicy: vi.fn(),
  validateFormula: vi.fn(),
}));

import { usePolicies } from "./usePolicies";
import {
  getPropositions,
  createProposition as apiCreateProposition,
  deleteProposition as apiDeleteProposition,
  getPolicies,
  createPolicy as apiCreatePolicy,
  updatePolicy as apiUpdatePolicy,
  deletePolicy as apiDeletePolicy,
} from "@/api/client";
import type { Proposition, Policy } from "@/types";
import {
  createProposition as mockProposition,
  createPolicy as mockPolicy,
} from "../test/mocks";

describe("usePolicies", () => {
  const propFraud = mockProposition();
  const propComply = mockProposition({
    prop_id: "q_comply",
    role: "assistant",
    description: "The assistant provides fraud instructions",
  });
  const policyFraud = mockPolicy();
  const policySensitive = mockPolicy({
    policy_id: "pol_sensitive",
    name: "Sensitive Data",
    formula_str: "H(@ p_sensitive -> q_warn)",
    propositions: ["p_sensitive", "q_warn"],
  });

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getPropositions).mockResolvedValue([propFraud, propComply]);
    vi.mocked(getPolicies).mockResolvedValue([policyFraud, policySensitive]);
  });

  it("fetches propositions on mount", async () => {
    const { result } = renderHook(() => usePolicies());

    await waitFor(() => {
      expect(result.current.propositions.status).toBe("success");
    });

    expect(getPropositions).toHaveBeenCalledOnce();
    if (result.current.propositions.status === "success") {
      expect(result.current.propositions.data).toEqual([
        propFraud,
        propComply,
      ]);
      expect(result.current.propositions.data).toHaveLength(2);
    }
  });

  it("fetches policies on mount", async () => {
    const { result } = renderHook(() => usePolicies());

    await waitFor(() => {
      expect(result.current.policies.status).toBe("success");
    });

    expect(getPolicies).toHaveBeenCalledOnce();
    if (result.current.policies.status === "success") {
      expect(result.current.policies.data).toEqual([
        policyFraud,
        policySensitive,
      ]);
      expect(result.current.policies.data).toHaveLength(2);
    }
  });

  it("createProposition calls API and adds to state", async () => {
    const newProp: Proposition = {
      prop_id: "p_frame",
      role: "user",
      description: "The user attempts to frame the conversation",
      created_at: "2025-01-02T00:00:00Z",
      updated_at: "2025-01-02T00:00:00Z",
    };
    vi.mocked(apiCreateProposition).mockResolvedValue(newProp);

    const { result } = renderHook(() => usePolicies());

    await waitFor(() => {
      expect(result.current.propositions.status).toBe("success");
    });

    await act(async () => {
      await result.current.createProposition({
        prop_id: "p_frame",
        description: "The user attempts to frame the conversation",
        role: "user",
      });
    });

    expect(apiCreateProposition).toHaveBeenCalledWith({
      prop_id: "p_frame",
      description: "The user attempts to frame the conversation",
      role: "user",
    });

    if (result.current.propositions.status === "success") {
      expect(result.current.propositions.data).toHaveLength(3);
      expect(result.current.propositions.data[2].prop_id).toBe("p_frame");
    }
  });

  it("deleteProposition calls API and removes from state", async () => {
    vi.mocked(apiDeleteProposition).mockResolvedValue(undefined);

    const { result } = renderHook(() => usePolicies());

    await waitFor(() => {
      expect(result.current.propositions.status).toBe("success");
    });

    await act(async () => {
      await result.current.deleteProposition("p_fraud");
    });

    expect(apiDeleteProposition).toHaveBeenCalledWith("p_fraud");

    if (result.current.propositions.status === "success") {
      expect(result.current.propositions.data).toHaveLength(1);
      expect(result.current.propositions.data[0].prop_id).toBe("q_comply");
    }
  });

  it("createPolicy calls API and adds to state", async () => {
    const newPolicy: Policy = {
      policy_id: "pol_jailbreak",
      name: "Anti-Jailbreak",
      formula_str: "H((p_escalate & P(p_frame)) -> !q_unsafe)",
      propositions: ["p_escalate", "p_frame", "q_unsafe"],
      enabled: true,
      created_at: "2025-01-02T00:00:00Z",
      updated_at: "2025-01-02T00:00:00Z",
    };
    vi.mocked(apiCreatePolicy).mockResolvedValue(newPolicy);

    const { result } = renderHook(() => usePolicies());

    await waitFor(() => {
      expect(result.current.policies.status).toBe("success");
    });

    await act(async () => {
      await result.current.createPolicy({
        name: "Anti-Jailbreak",
        formula_str: "H((p_escalate & P(p_frame)) -> !q_unsafe)",
      });
    });

    expect(apiCreatePolicy).toHaveBeenCalledWith({
      name: "Anti-Jailbreak",
      formula_str: "H((p_escalate & P(p_frame)) -> !q_unsafe)",
    });

    if (result.current.policies.status === "success") {
      expect(result.current.policies.data).toHaveLength(3);
      expect(result.current.policies.data[2].policy_id).toBe("pol_jailbreak");
    }
  });

  it("deletePolicy calls API and removes from state", async () => {
    vi.mocked(apiDeletePolicy).mockResolvedValue(undefined);

    const { result } = renderHook(() => usePolicies());

    await waitFor(() => {
      expect(result.current.policies.status).toBe("success");
    });

    await act(async () => {
      await result.current.deletePolicy("pol_fraud");
    });

    expect(apiDeletePolicy).toHaveBeenCalledWith("pol_fraud");

    if (result.current.policies.status === "success") {
      expect(result.current.policies.data).toHaveLength(1);
      expect(result.current.policies.data[0].policy_id).toBe("pol_sensitive");
    }
  });

  it("togglePolicy calls updatePolicy with toggled enabled flag", async () => {
    vi.mocked(apiUpdatePolicy).mockResolvedValue({
      ...policyFraud,
      enabled: false,
    });

    const { result } = renderHook(() => usePolicies());

    await waitFor(() => {
      expect(result.current.policies.status).toBe("success");
    });

    await act(async () => {
      await result.current.togglePolicy("pol_fraud", false);
    });

    expect(apiUpdatePolicy).toHaveBeenCalledWith("pol_fraud", {
      enabled: false,
    });
  });

  it("togglePolicy is optimistic and updates state immediately", async () => {
    // Make the API call hang so we can inspect the intermediate state
    let resolveUpdate: ((value: Policy) => void) | undefined;
    vi.mocked(apiUpdatePolicy).mockReturnValue(
      new Promise<Policy>((resolve) => {
        resolveUpdate = resolve;
      }),
    );

    const { result } = renderHook(() => usePolicies());

    await waitFor(() => {
      expect(result.current.policies.status).toBe("success");
    });

    // Start the toggle (don't await it)
    act(() => {
      void result.current.togglePolicy("pol_fraud", false);
    });

    // Check the optimistic update happened immediately
    if (result.current.policies.status === "success") {
      const toggled = result.current.policies.data.find(
        (p) => p.policy_id === "pol_fraud",
      );
      expect(toggled?.enabled).toBe(false);
    }

    // Resolve the API call
    await act(async () => {
      resolveUpdate?.({ ...policyFraud, enabled: false });
    });
  });

  it("togglePolicy reverts on API error", async () => {
    vi.mocked(apiUpdatePolicy).mockRejectedValue(new Error("Server error"));

    const { result } = renderHook(() => usePolicies());

    await waitFor(() => {
      expect(result.current.policies.status).toBe("success");
    });

    // Verify initially enabled
    if (result.current.policies.status === "success") {
      const before = result.current.policies.data.find(
        (p) => p.policy_id === "pol_fraud",
      );
      expect(before?.enabled).toBe(true);
    }

    await act(async () => {
      await result.current.togglePolicy("pol_fraud", false);
    });

    // Should revert back to enabled=true after error
    if (result.current.policies.status === "success") {
      const after = result.current.policies.data.find(
        (p) => p.policy_id === "pol_fraud",
      );
      expect(after?.enabled).toBe(true);
    }
  });

  it("sets propositions status to error when fetch fails", async () => {
    vi.mocked(getPropositions).mockRejectedValue(new Error("DB error"));

    const { result } = renderHook(() => usePolicies());

    await waitFor(() => {
      expect(result.current.propositions.status).toBe("error");
    });

    if (result.current.propositions.status === "error") {
      expect(result.current.propositions.error).toBe("DB error");
    }
  });

  it("sets policies status to error when fetch fails", async () => {
    vi.mocked(getPolicies).mockRejectedValue(new Error("DB error"));

    const { result } = renderHook(() => usePolicies());

    await waitFor(() => {
      expect(result.current.policies.status).toBe("error");
    });

    if (result.current.policies.status === "error") {
      expect(result.current.policies.error).toBe("DB error");
    }
  });
});
