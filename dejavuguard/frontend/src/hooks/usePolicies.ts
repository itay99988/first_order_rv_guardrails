import { useCallback, useEffect, useState } from "react";

import {
  createPolicy as apiCreatePolicy,
  createProposition as apiCreateProposition,
  deletePolicy as apiDeletePolicy,
  deleteProposition as apiDeleteProposition,
  getPolicies,
  getPropositions,
  updatePolicy as apiUpdatePolicy,
  updateProposition as apiUpdateProposition,
  validateFormula,
} from "@/api/client";
import type {
  AsyncState,
  FormulaValidation,
  Policy,
  Proposition,
} from "@/types";

export function usePolicies() {
  const [propositions, setPropositions] = useState<AsyncState<Proposition[]>>({
    status: "idle",
  });
  const [policies, setPolicies] = useState<AsyncState<Policy[]>>({
    status: "idle",
  });

  const fetchPropositions = useCallback(async () => {
    setPropositions({ status: "loading" });
    try {
      const data = await getPropositions();
      setPropositions({ status: "success", data });
    } catch (err) {
      setPropositions({
        status: "error",
        error:
          err instanceof Error ? err.message : "Failed to load predicates",
      });
    }
  }, []);

  const createProposition = useCallback(
    async (data: { prop_id: string; description: string; role: string; arity: number }) => {
      const result = await apiCreateProposition(data);
      setPropositions((prev) => {
        if (prev.status === "success") {
          return { status: "success", data: [...prev.data, result.proposition] };
        }
        return prev;
      });
      return result;
    },
    [],
  );

  const updateProposition = useCallback(
    async (propId: string, data: { description?: string; role?: string }) => {
      const updated = await apiUpdateProposition(propId, data);
      setPropositions((prev) => {
        if (prev.status === "success") {
          return {
            status: "success",
            data: prev.data.map((p) => (p.prop_id === propId ? updated : p)),
          };
        }
        return prev;
      });
      return updated;
    },
    [],
  );

  const deleteProposition = useCallback(async (propId: string) => {
    await apiDeleteProposition(propId);
    setPropositions((prev) => {
      if (prev.status === "success") {
        return {
          status: "success",
          data: prev.data.filter((p) => p.prop_id !== propId),
        };
      }
      return prev;
    });
  }, []);

  const fetchPolicies = useCallback(async () => {
    setPolicies({ status: "loading" });
    try {
      const data = await getPolicies();
      setPolicies({ status: "success", data });
    } catch (err) {
      setPolicies({
        status: "error",
        error: err instanceof Error ? err.message : "Failed to load policies",
      });
    }
  }, []);

  const createPolicy = useCallback(
    async (data: { name: string; formula_str: string; enabled?: boolean }) => {
      const created = await apiCreatePolicy(data);
      setPolicies((prev) => {
        if (prev.status === "success") {
          return { status: "success", data: [...prev.data, created] };
        }
        return prev;
      });
      return created;
    },
    [],
  );

  const updatePolicy = useCallback(
    async (
      policyId: string,
      data: { name?: string; formula_str?: string; enabled?: boolean },
    ) => {
      const updated = await apiUpdatePolicy(policyId, data);
      setPolicies((prev) => {
        if (prev.status === "success") {
          return {
            status: "success",
            data: prev.data.map((p) =>
              p.policy_id === policyId ? updated : p,
            ),
          };
        }
        return prev;
      });
      return updated;
    },
    [],
  );

  const deletePolicy = useCallback(async (policyId: string) => {
    await apiDeletePolicy(policyId);
    setPolicies((prev) => {
      if (prev.status === "success") {
        return {
          status: "success",
          data: prev.data.filter((p) => p.policy_id !== policyId),
        };
      }
      return prev;
    });
  }, []);

  const togglePolicy = useCallback(
    async (policyId: string, enabled: boolean) => {
      // Optimistic update
      setPolicies((prev) => {
        if (prev.status === "success") {
          return {
            status: "success",
            data: prev.data.map((p) =>
              p.policy_id === policyId ? { ...p, enabled } : p,
            ),
          };
        }
        return prev;
      });
      try {
        await apiUpdatePolicy(policyId, { enabled });
      } catch {
        // Revert on failure
        setPolicies((prev) => {
          if (prev.status === "success") {
            return {
              status: "success",
              data: prev.data.map((p) =>
                p.policy_id === policyId ? { ...p, enabled: !enabled } : p,
              ),
            };
          }
          return prev;
        });
      }
    },
    [],
  );

  const validateFormulaAction = useCallback(
    async (name: string, formulaStr: string): Promise<FormulaValidation> => {
      return validateFormula({ name, formula_str: formulaStr });
    },
    [],
  );

  useEffect(() => {
    fetchPropositions();
    fetchPolicies();
  }, [fetchPropositions, fetchPolicies]);

  return {
    propositions,
    policies,
    fetchPropositions,
    createProposition,
    updateProposition,
    deleteProposition,
    fetchPolicies,
    createPolicy,
    updatePolicy,
    deletePolicy,
    togglePolicy,
    validateFormula: validateFormulaAction,
  };
}
