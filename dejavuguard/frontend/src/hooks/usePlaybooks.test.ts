import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { AsyncState } from "@/types";

import { usePlaybooks } from "./usePlaybooks";

/**
 * Narrow an `AsyncState` before reading its payload.
 *
 * `.data` and `.error` live on one arm of the union each, so reaching for
 * them directly is a type error -- one that nothing reported, because
 * `npm run build` named two of the three tsconfig projects and the test
 * project was in neither. Failing here on the wrong status also says which
 * status it actually had, which a `?.` would swallow.
 */
function settled<T>(state: AsyncState<T>): Extract<
  AsyncState<T>,
  { status: "success" } | { status: "error" }
> {
  if (state.status !== "success" && state.status !== "error") {
    throw new Error(`expected a settled state, got "${state.status}"`);
  }
  return state;
}

function loaded<T>(state: AsyncState<T>): T {
  const done = settled(state);
  if (done.status !== "success") {
    throw new Error(`expected success, got error "${done.error}"`);
  }
  return done.data;
}

function failed<T>(state: AsyncState<T>): string {
  const done = settled(state);
  if (done.status !== "error") {
    throw new Error("expected an error, got success");
  }
  return done.error;
}

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

    await waitFor(() => expect(loaded(result.current.playbooks)).toHaveLength(1));
    expect(loaded(result.current.playbooks)[0].name).toBe("Budget");
  });

  it("surfaces a load error instead of rendering an empty list", async () => {
    mockGet.mockRejectedValue(new Error("boom"));

    const { result } = renderHook(() => usePlaybooks());

    await waitFor(() => expect(failed(result.current.playbooks)).toBe("boom"));
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
