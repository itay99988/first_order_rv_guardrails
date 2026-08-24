import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import MessageBubble from "./MessageBubble";
import type {
  ViolationInfo,
  GroundingDetail,
  PlaybookStateInfo,
} from "../../types";

function createViolation(
  overrides: Partial<ViolationInfo> = {},
): ViolationInfo {
  return {
    policy_id: "pol_fraud",
    policy_name: "Fraud Prevention",
    formula_str: "H(p_fraud -> !q_comply)",
    violated_at_index: 2,
    labeling: { p_fraud: true, q_comply: true },
    grounding_details: [],
    ...overrides,
  };
}

function createGroundingDetail(
  overrides: Partial<GroundingDetail> = {},
): GroundingDetail {
  return {
    prop_id: "p_fraud",
    match: true,
    confidence: 0.95,
    reasoning: "User explicitly requested fraud methods",
    method: "llm",
    ...overrides,
  };
}

function createPlaybookState(
  overrides: Partial<PlaybookStateInfo> = {},
): PlaybookStateInfo {
  return {
    playbook_id: "pb1",
    playbook_name: "Budget",
    state_key: "s1",
    label: "Over budget",
    member_verdicts: { pol_budget: false },
    rules: ["Stay within budget.", "Offer a cheaper alternative."],
    flagged: true,
    ...overrides,
  };
}

describe("MessageBubble", () => {
  it("renders user message aligned to the right", () => {
    render(
      <MessageBubble
        role="user"
        content="Hello world"
        blocked={false}
        violationInfo={null}
        groundingDetails={null}
        monitorState={null}
      />,
    );
    const bubble = screen.getByTestId("message-user");
    expect(bubble).toBeInTheDocument();
    expect(bubble.className).toContain("justify-end");
  });

  it("renders assistant message aligned to the left", () => {
    render(
      <MessageBubble
        role="assistant"
        content="Hi there"
        blocked={false}
        violationInfo={null}
        groundingDetails={null}
        monitorState={null}
      />,
    );
    const bubble = screen.getByTestId("message-assistant");
    expect(bubble).toBeInTheDocument();
    expect(bubble.className).toContain("justify-start");
  });

  it("displays message content text", () => {
    render(
      <MessageBubble
        role="user"
        content="Test message content"
        blocked={false}
        violationInfo={null}
        groundingDetails={null}
        monitorState={null}
      />,
    );
    expect(screen.getByText("Test message content")).toBeInTheDocument();
  });

  it('shows "Passed" tag when message is not blocked', () => {
    render(
      <MessageBubble
        role="assistant"
        content="Safe response"
        blocked={false}
        violationInfo={null}
        groundingDetails={null}
        monitorState={null}
      />,
    );
    expect(screen.getByText("Passed")).toBeInTheDocument();
  });

  it('shows "BLOCKED" label and "Blocked" tag when message is blocked', () => {
    render(
      <MessageBubble
        role="assistant"
        content="Dangerous response"
        blocked={true}
        violationInfo={createViolation()}
        groundingDetails={null}
        monitorState={null}
      />,
    );
    expect(screen.getByText("BLOCKED")).toBeInTheDocument();
    expect(screen.getByText("Blocked")).toBeInTheDocument();
    expect(screen.getByTestId("message-blocked")).toBeInTheDocument();
  });

  it("applies line-through style to blocked message content", () => {
    render(
      <MessageBubble
        role="assistant"
        content="Blocked content"
        blocked={true}
        violationInfo={createViolation()}
        groundingDetails={null}
        monitorState={null}
      />,
    );
    const textEl = screen.getByText("Blocked content");
    expect(textEl.className).toContain("line-through");
  });

  it("shows details panel with violation info when toggle is clicked", async () => {
    const user = userEvent.setup();
    render(
      <MessageBubble
        role="assistant"
        content="Bad response"
        blocked={true}
        violationInfo={createViolation({ policy_name: "Fraud Prevention" })}
        groundingDetails={null}
        monitorState={null}
      />,
    );

    expect(screen.queryByTestId("message-details")).not.toBeInTheDocument();

    const toggle = screen.getByTestId("toggle-details");
    await user.click(toggle);

    expect(screen.getByTestId("message-details")).toBeInTheDocument();
    expect(
      screen.getByText("Violation: Fraud Prevention"),
    ).toBeInTheDocument();
    expect(screen.getByText("H(p_fraud -> !q_comply)")).toBeInTheDocument();
  });

  it("names the playbook and its state instead of an empty formula", async () => {
    const user = userEvent.setup();
    render(
      <MessageBubble
        role="assistant"
        content="Bad response"
        blocked={true}
        violationInfo={createViolation({
          policy_id: "pb1",
          policy_name: "Budget playbook",
          formula_str: "",
          playbook_id: "pb1",
          state_label: "Over budget",
        })}
        groundingDetails={null}
        monitorState={null}
      />,
    );

    await user.click(screen.getByTestId("toggle-details"));

    expect(
      screen.getByText("Blocked by playbook: Budget playbook"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("message-violation-state")).toHaveTextContent(
      "Over budget",
    );
  });

  it("shows grounding details in expanded panel", async () => {
    const user = userEvent.setup();
    const details = [
      createGroundingDetail({
        prop_id: "p_fraud",
        match: true,
        confidence: 0.95,
        reasoning: "Fraud request detected",
      }),
      createGroundingDetail({
        prop_id: "q_comply",
        match: false,
        confidence: 0.1,
        reasoning: "Refusal detected",
      }),
    ];

    render(
      <MessageBubble
        role="assistant"
        content="Refusal"
        blocked={false}
        violationInfo={null}
        groundingDetails={details}
        monitorState={null}
      />,
    );

    await user.click(screen.getByTestId("toggle-details"));

    expect(screen.getByText("Grounding:")).toBeInTheDocument();
    expect(screen.getByText("p_fraud")).toBeInTheDocument();
    expect(screen.getByText("Match")).toBeInTheDocument();
    expect(screen.getByText("(95%)")).toBeInTheDocument();
    expect(screen.getByText("Fraud request detected")).toBeInTheDocument();
    expect(screen.getByText("q_comply")).toBeInTheDocument();
    expect(screen.getByText("No match")).toBeInTheDocument();
  });

  it("shows grounding instances with mentions and canonical forms", async () => {
    const user = userEvent.setup();
    const details = [
      createGroundingDetail({
        prop_id: "p_car",
        match: true,
        reasoning: "Two car requests detected",
        instances: [
          {
            instance_id: "i1",
            object_mentions: [
              {
                object_id: "o1",
                mention: "Toyota",
                canonical_form: "Toyota",
              },
              {
                object_id: "o2",
                mention: "12000$",
                canonical_form: "12000 USD",
              },
            ],
          },
          {
            instance_id: "i2",
            object_mentions: [
              {
                object_id: "o1",
                mention: "Skoda",
                canonical_form: "Skoda",
              },
              {
                object_id: "o2",
                mention: "12500$",
                canonical_form: "12500 USD",
              },
            ],
          },
        ],
      }),
    ];

    render(
      <MessageBubble
        role="user"
        content="I'm considering Toyota under 12000$ and Skoda under 12500$."
        blocked={false}
        violationInfo={null}
        groundingDetails={details}
        monitorState={null}
      />,
    );

    await user.click(screen.getByTestId("toggle-details"));

    expect(screen.getByText("i1")).toBeInTheDocument();
    expect(screen.getByText("i2")).toBeInTheDocument();
    expect(screen.getAllByText("Toyota").length).toBeGreaterThan(0);
    expect(screen.getByText("12000 USD")).toBeInTheDocument();
    expect(screen.getAllByText("Skoda").length).toBeGreaterThan(0);
    expect(screen.getByText("12500 USD")).toBeInTheDocument();
  });

  it("shows monitor state in expanded panel", async () => {
    const user = userEvent.setup();
    render(
      <MessageBubble
        role="user"
        content="Hello"
        blocked={false}
        violationInfo={null}
        groundingDetails={null}
        monitorState={{ pol_fraud: true, pol_sensitive: false }}
      />,
    );

    await user.click(screen.getByTestId("toggle-details"));

    expect(screen.getByText("Monitor:")).toBeInTheDocument();
    expect(screen.getByText("pol_fraud: Pass")).toBeInTheDocument();
    expect(screen.getByText("pol_sensitive: Fail")).toBeInTheDocument();
  });

  it("hides details panel when toggle is clicked again", async () => {
    const user = userEvent.setup();
    render(
      <MessageBubble
        role="user"
        content="Hello"
        blocked={false}
        violationInfo={null}
        groundingDetails={null}
        monitorState={{ pol_fraud: true }}
      />,
    );

    const toggle = screen.getByTestId("toggle-details");
    await user.click(toggle);
    expect(screen.getByTestId("message-details")).toBeInTheDocument();

    await user.click(toggle);
    expect(screen.queryByTestId("message-details")).not.toBeInTheDocument();
  });

  // --- Playbook guidance stays out of the visible conversation ---

  it("never shows playbook guidance in the collapsed/default render", () => {
    render(
      <MessageBubble
        role="assistant"
        content="Here's a car under budget."
        blocked={false}
        violationInfo={null}
        groundingDetails={null}
        monitorState={null}
        playbookState={createPlaybookState()}
      />,
    );

    // The panel is collapsed by default -- none of the guidance text, or
    // the panel itself, may be present in the rendered conversation.
    expect(screen.queryByTestId("message-details")).not.toBeInTheDocument();
    expect(screen.queryByText("Stay within budget.")).not.toBeInTheDocument();
    expect(
      screen.queryByText("Offer a cheaper alternative."),
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/Over budget/)).not.toBeInTheDocument();
  });

  it("shows playbook state and its rules only once details are expanded", async () => {
    const user = userEvent.setup();
    render(
      <MessageBubble
        role="assistant"
        content="Here's a car under budget."
        blocked={false}
        violationInfo={null}
        groundingDetails={null}
        monitorState={null}
        playbookState={createPlaybookState()}
      />,
    );

    await user.click(screen.getByTestId("toggle-details"));

    expect(screen.getByTestId("message-details")).toBeInTheDocument();
    expect(
      screen.getByText("Playbook state after this turn (Over budget):"),
    ).toBeInTheDocument();
    expect(screen.getByText("Stay within budget.")).toBeInTheDocument();
    expect(
      screen.getByText("Offer a cheaper alternative."),
    ).toBeInTheDocument();
  });
});
