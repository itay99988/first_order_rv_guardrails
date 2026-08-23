import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { usePlaybooks } from "./usePlaybooks";

const mockGet = vi.fn();
const mockCreate = vi.fn();

vi.mock("@/api/client", () => ({
  getPlaybooks: (...args: unknown[]) => mockGet(...args),
  createPlaybook: (...args: unknown[]) => mockCreate(...args),
}));

describe("usePlaybooks", () => {
  beforeEach(() => {
    mockGet.mockReset();
    mockCreate.mockReset();
  });

  it("loads playbooks on mount", async () => {
    mockGet.mockResolvedValue([
      { playbook_id: "pb1", name: "Budget", member_count: 2,
        state_count: 4, behaviour_count: 2, flagged_count: 1 },
    ]);

    const { result } = renderHook(() => usePlaybooks());

    await waitFor(() => expect(result.current.playbooks.data).toHaveLength(1));
    expect(result.current.playbooks.data?.[0].name).toBe("Budget");
  });

  it("surfaces a load error instead of rendering an empty list", async () => {
    mockGet.mockRejectedValue(new Error("boom"));

    const { result } = renderHook(() => usePlaybooks());

    await waitFor(() => expect(result.current.playbooks.error).toBe("boom"));
  });

  it("refetches after creating a playbook", async () => {
    mockGet.mockResolvedValue([]);
    mockCreate.mockResolvedValue({ playbook_id: "pb1", name: "Budget" });

    const { result } = renderHook(() => usePlaybooks());
    await waitFor(() => expect(mockGet).toHaveBeenCalledTimes(1));

    await act(async () => {
      await result.current.createPlaybook({ name: "Budget" });
    });

    expect(mockGet).toHaveBeenCalledTimes(2);
  });
});
