import { useCallback, useEffect, useState } from "react";

import {
  createPlaybook as apiCreatePlaybook,
  deletePlaybook as apiDeletePlaybook,
  getPlaybooks,
} from "@/api/client";
import type { Playbook } from "@/types";

interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

export function usePlaybooks() {
  const [playbooks, setPlaybooks] = useState<AsyncState<Playbook[]>>({
    data: null,
    loading: true,
    error: null,
  });

  const fetchPlaybooks = useCallback(async () => {
    setPlaybooks((s) => ({ ...s, loading: true, error: null }));
    try {
      const data = await getPlaybooks();
      setPlaybooks({ data, loading: false, error: null });
    } catch (e) {
      setPlaybooks({
        data: null,
        loading: false,
        error: e instanceof Error ? e.message : "Failed to load playbooks",
      });
    }
  }, []);

  const createPlaybook = useCallback(
    async (data: { name: string; description?: string }) => {
      const created = await apiCreatePlaybook(data);
      await fetchPlaybooks();
      return created;
    },
    [fetchPlaybooks],
  );

  const deletePlaybook = useCallback(
    async (playbookId: string) => {
      await apiDeletePlaybook(playbookId);
      await fetchPlaybooks();
    },
    [fetchPlaybooks],
  );

  useEffect(() => {
    void fetchPlaybooks();
  }, [fetchPlaybooks]);

  return { playbooks, fetchPlaybooks, createPlaybook, deletePlaybook };
}
