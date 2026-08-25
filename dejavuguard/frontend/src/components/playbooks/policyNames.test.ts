import { describe, expect, it } from "vitest";

import type { PlaybookMember } from "@/types";
import { policyDisplayName, policyNamer } from "./policyNames";

function member(policy_id: string, name?: string | null): PlaybookMember {
  return { policy_id, name, position: 0, fires_on: false, guidance: "" };
}

describe("policyDisplayName", () => {
  it("calls a policy what its owner called it", () => {
    expect(
      policyDisplayName("8525fd4d-820c-4d23-b983-a054c7c3e211", "Budget cap"),
    ).toBe("Budget cap");
  });

  it("falls back to the id when nothing can name the policy", () => {
    // `null` is what the server sends for a member whose policy has been
    // deleted out from under the playbook. The row still has to say which
    // member it is, and the id is the only thing left that does.
    expect(policyDisplayName("p_gone", null)).toBe("p_gone");
    // `undefined` is a server too old to send the field at all.
    expect(policyDisplayName("p_old")).toBe("p_old");
  });

  it("treats a blank name as no name rather than drawing an empty label", () => {
    // The alternative is a member row whose bold identity line is empty --
    // the reader looks where the member's name goes and sees nothing, which
    // is worse than a uuid, not better.
    expect(policyDisplayName("p_blank", "   ")).toBe("p_blank");
  });

  it("trims a name rather than rendering the padding", () => {
    expect(policyDisplayName("p1", "  Budget cap  ")).toBe("Budget cap");
  });
});

describe("policyNamer", () => {
  it("names the members it was given", () => {
    const nameOf = policyNamer([member("p1", "Budget cap"), member("p2", "Tone")]);

    expect(nameOf("p1")).toBe("Budget cap");
    expect(nameOf("p2")).toBe("Tone");
  });

  it("hands back an id it holds no member for", () => {
    // A verdict map written before a membership change still names the
    // member that has since gone. Answering with the id keeps the badge
    // truthful; answering with "" or "unknown" would lose which bit it is.
    const nameOf = policyNamer([member("p1", "Budget cap")]);

    expect(nameOf("p_stale")).toBe("p_stale");
  });

  it("falls back per member, not for the whole list", () => {
    // One nameless member must not cost the others their names -- that is
    // the all-or-nothing shape this pane keeps producing.
    const nameOf = policyNamer([member("p1", "Budget cap"), member("p2", null)]);

    expect(nameOf("p1")).toBe("Budget cap");
    expect(nameOf("p2")).toBe("p2");
  });
});
