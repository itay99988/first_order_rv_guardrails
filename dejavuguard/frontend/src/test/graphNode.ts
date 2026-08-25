/**
 * Reading a `PlaybookGraph` node in a test.
 *
 * A node's `textContent` swallows its `<title>`, and `tooltipOf` puts the
 * server's `_disambiguate`d behaviour name on the first line of it -- a string
 * that is unique per node by construction. So every assertion made through
 * `textContent` passes on the tooltip no matter what the drawn caption does,
 * and a caption that has collapsed two behaviours into one looks fine.
 *
 * That is not hypothetical: four assertions in `PlaybookGraph.test.tsx` --
 * including the one standing in for the spec's "legible at four members and
 * sixteen states" -- stayed green with `ruleLines` reverted to the very
 * 14-character join they exist to forbid. Two more had drifted into the same
 * shape in `ChatView.test.tsx` and `PlaybookEditor.test.tsx`, which is why
 * this lives here rather than in one test file: the trap is not local to the
 * graph's own tests.
 *
 * Ask a tooltip question with `tooltip`. Ask everything else with `drawn`.
 */

/** A node's drawn caption -- every `<text>` in it, tooltip excluded. */
export function drawn(node: HTMLElement): string {
  return Array.from(node.querySelectorAll("text"))
    .map((t) => t.textContent)
    .join("|");
}

/** A node's hover tooltip -- the `<title>` a pointer reveals. */
export function tooltip(node: HTMLElement): string {
  return node.querySelector("title")?.textContent ?? "";
}
