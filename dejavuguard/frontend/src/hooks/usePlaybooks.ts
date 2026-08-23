import { useCallback, useEffect, useState } from "react";

import {
  createPlaybook as apiCreatePlaybook,
  deletePlaybook as apiDeletePlaybook,
  getPlaybooks,
} from "@/api/client";
import type { AsyncState, Playbook } from "@/types";

export function usePlaybooks() {
  const [playbooks, setPlaybooks] = useState<AsyncState<Playbook[]>>({
    status: "idle",
  });

  const fetchPlaybooks = useCallback(async () => {
    setPlaybooks({ status: "loading" });
    try {
      const data = await getPlaybooks();
      setPlaybooks({ status: "success", data });
    } catch (e) {
      setPlaybooks({
        status: "error",
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
