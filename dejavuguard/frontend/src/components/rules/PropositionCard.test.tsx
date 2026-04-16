import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import PropositionCard from "./PropositionCard";
import { createProposition } from "../../test/mocks";

describe("PropositionCard", () => {
  it("renders prop_id with monospace styling", () => {
    render(
      <PropositionCard
        proposition={createProposition()}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onViewPrompt={vi.fn()}
      />,
    );
    expect(screen.getByText("p_fraud")).toBeInTheDocument();
    expect(screen.getByText("p_fraud")).toHaveClass("font-mono");
  });

  it("renders role badge for user proposition", () => {
    render(
      <PropositionCard
        proposition={createProposition({ role: "user" })}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onViewPrompt={vi.fn()}
      />,
    );
    expect(screen.getByText("user")).toBeInTheDocument();
  });

  it("renders role badge for assistant proposition", () => {
    render(
      <PropositionCard
        proposition={createProposition({
          role: "assistant",
          prop_id: "q_comply",
        })}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onViewPrompt={vi.fn()}
      />,
    );
    expect(screen.getByText("assistant")).toBeInTheDocument();
  });

  it("renders description text", () => {
    render(
      <PropositionCard
        proposition={createProposition({
          description: "Requests fraud methods",
        })}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onViewPrompt={vi.fn()}
      />,
    );
    expect(
      screen.getByText("Requests fraud methods"),
    ).toBeInTheDocument();
  });

  it("has data-testid with prop_id", () => {
    render(
      <PropositionCard
        proposition={createProposition({ prop_id: "p_fraud" })}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onViewPrompt={vi.fn()}
      />,
    );
    expect(screen.getByTestId("proposition-card-p_fraud")).toBeInTheDocument();
  });

  it("clicking edit button calls onEdit with proposition", async () => {
    const user = userEvent.setup();
    const onEdit = vi.fn();
    const prop = createProposition();
    render(
      <PropositionCard
        proposition={prop}
        onEdit={onEdit}
        onDelete={vi.fn()}
        onViewPrompt={vi.fn()}
      />,
    );

    await user.click(screen.getByTestId("edit-proposition-p_fraud"));
    expect(onEdit).toHaveBeenCalledWith(prop);
  });

  it("clicking delete button calls onDelete with prop_id", async () => {
    const user = userEvent.setup();
    const onDelete = vi.fn();
    render(
      <PropositionCard
        proposition={createProposition()}
        onEdit={vi.fn()}
        onDelete={onDelete}
        onViewPrompt={vi.fn()}
      />,
    );

    await user.click(screen.getByTestId("delete-proposition-p_fraud"));
    expect(onDelete).toHaveBeenCalledWith("p_fraud");
  });

  it("edit button has accessible label", () => {
    render(
      <PropositionCard
        proposition={createProposition()}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onViewPrompt={vi.fn()}
      />,
    );
    expect(
      screen.getByLabelText("Edit proposition p_fraud"),
    ).toBeInTheDocument();
  });

  it("delete button has accessible label", () => {
    render(
      <PropositionCard
        proposition={createProposition()}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
        onViewPrompt={vi.fn()}
      />,
    );
    expect(
      screen.getByLabelText("Delete proposition p_fraud"),
    ).toBeInTheDocument();
  });
});
