# Banking — bank-transfer-balance scenario set

100 hand-authored conversation scenarios (50 pass, 50 violate) for the
`bank-transfer-balance` policy. Every per-message `expected_verdict` is
**computed by DejaVu** from the intended event trace (see `_build.py`), never
hand-guessed — the numeric comparison can't be eyeballed reliably.

## Property

```
prop transfer :
  ( forall t . tx_g(t) -> exists b . ( ( !(exists b2 . bal_a(b2)) S bal_a(b) ) & !(b < t) ) )
  &
  ( (@ exists t . tx_g(t)) -> exists b . bal_a(b) )
```

1. **Affordability** — every transfer amount must be `<=` the **most recently
   reported** balance (and a balance must have been reported *before* any
   transfer). Validated to use DejaVu's genuine numeric `<`, comparing dollar
   values rather than enum/ID order.
2. **Disclosure** — **immediately after** each transfer, the assistant must
   report the updated balance.

Predicates: `bal_a(b)` (assistant reports the balance, USD) and `tx_g(t)` (the
agent/user initiates a transfer, USD). Roles: the autonomous agent maps to the
conversation **user**; the banking assistant maps to **assistant**.

## Composition

- **50 pass** — every transfer within the latest balance, a fresh balance
  reported after each. All per-message verdicts `true`.
- **50 violate** — a clean ledger that breaks exactly one rule on the **final
  message** (single `false` at the end, all earlier `true`):
  - **type A** — the last transfer exceeds the most recently reported balance.
  - **type B** — a transfer occurs but the next (last) message reports no balance.
  - plus *no-prior-balance* openings (a transfer before any balance is stated).
- **22 scenarios** intentionally **repeat identical message lines** (recurring
  allowances, sweeps, babysitter/storage payments) to exercise multi-cycle,
  multi-balance, multi-transfer ledgers.
- Conversation length: min 2 / **mean 8.6** / max 16 messages.

## Models (per scenario JSON)

- Grounding: `mistralai/ministral-8b-2512` (OpenRouter)
- Few-shot: `anthropic/claude-sonnet-4.6` (OpenRouter)

## Validation

- `_build.py` is the single source of truth; it boots its own DejaVu server on
  `127.0.0.1:8090`, replays each intended trace, and stamps the returned
  per-message verdicts. Re-run: `uv run --project ../../.. python3 _build.py`.
- Label invariant checked on every build: pass = all `true`; violate = last
  `false`, all earlier `true`. **0 bad labels** across all 100.
- Each scenario was additionally reviewed by an independent LLM judge agent
  (arithmetic of the running balance, natural dialogue, and the violation
  landing on the correct message): **100 OK / 0 PROBLEM**.

## Run

```
DEJAVU_URL=http://127.0.0.1:8090 \
  uv run python -m scenario_runner --dir scenarios/banking_scenario \
  --grounding "mistralai/ministral-8b-2512" --grounding-provider openrouter --overwrite
```
