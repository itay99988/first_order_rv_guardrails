# DejaVuGuard

Runtime verification for LLM conversations using first-order past-time temporal logic.

DejaVuGuard is a monitored chat application. A user converses with an assistant model through OpenRouter, while a separate grounding model converts natural-language messages into canonicalized first-order events. These events are checked incrementally by [DejaVu](https://github.com/havelund/dejavu). If an active policy is violated, DejaVuGuard blocks the user message before it reaches the assistant, or blocks the assistant response before it reaches the user.

This is the prototype implementation described in the paper *First-Order
Temporal Guardrails for LLMs*. The dataset-generation and standalone grounding
evaluation scripts used in the paper live outside this directory, under
`../extended_grounding_dataset/`, `../cloud_grounding_eval/`, and
`../gpu_grounding_eval/`.

## Current Capabilities

- **First-order temporal policies** using quantified DejaVu predicates and past-time operators.
- **Built-in turn predicate**: `user_turn` is true exactly at user-message positions and false at assistant-message positions.
- **Role-specific grounding** for user and assistant messages.
- **Structured grounding instances**: a single message may produce multiple instances of the same predicate.
- **Canonical object forms**: extracted mentions are normalized before being forwarded to DejaVu.
- **Conversation-aware canonicalization** using related-object context and prior canonical history.
- **Predicate-specific few-shot generation** when a predicate is created.
- **Composite events**: all matching predicate instances from one message are sent to DejaVu at one temporal position.
- **Editable grounding prompts** in Settings, with active optimized defaults automatically loaded for upgraded installations.
- **Persistent conversations and monitoring state** backed by SQLite and DejaVu session storage.

## Architecture

```text
User message
   |
   v
DejaVuGuard backend
   |-- grounds all user-role predicates with the grounding LLM
   |-- extracts zero, one, or multiple canonicalized instances
   |-- sends one composite event to DejaVu
   |-- blocks immediately if a policy is violated
   |
   v
Chat LLM via OpenRouter
   |
   v
Assistant response
   |
   v
DejaVuGuard backend
   |-- grounds all assistant-role predicates with the grounding LLM
   |-- sends canonicalized instances as one composite event to DejaVu
   |-- blocks the response if a policy is violated
   |
   v
User
```

The chat model and grounding model are independent. The grounding model may run locally through Ollama, LM Studio, vLLM, or another OpenAI-compatible endpoint, or remotely through OpenRouter.

## Quick Start

### Docker

```bash
docker compose up --build
```

This starts:

- DejaVuGuard frontend/backend on `http://localhost:8001`
- DejaVu runtime verification server on `http://localhost:8080`

Open `http://localhost:8001`, then configure:

1. The OpenRouter API key and chat model.
2. The grounding provider and grounding model.
3. Optionally, which model generates predicate few-shot examples.

The Settings page also loads the active optimized grounding prompt templates. Existing databases containing stale prompt templates are migrated when settings are loaded after an application upgrade.

### Development Setup

Prerequisites:

- Python 3.11+
- Node.js 18+
- Java 17+ for running the bundled DejaVu server outside Docker
- An OpenRouter API key for the chat model
- A configured grounding model

```bash
# Start DejaVu
java -jar backend/libs/dejavu.jar --server --port 8080 --storage ./sessions

# Backend
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn backend.main:app --reload

# Frontend, in another terminal
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Scenario Runner

The `scenario_runner/` package runs scripted conversations against the
DejaVuGuard backend and records pass/fail reports, per-scenario logs, grounding
details, and HTML/Markdown summaries. It is the easiest way to reproduce
end-to-end monitoring scenarios after the backend and DejaVu services are
available.

See `scenario_runner/README.md` for the exact local and Docker-based commands.

## Environment Configuration

The application can be configured in the UI. Environment variables are also supported:

```env
# Chat model
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=mistralai/mistral-7b-instruct
OPENROUTER_MODEL_CUSTOM=

# Grounding model
GROUNDING_PROVIDER=ollama          # ollama | lmstudio | vllm | custom | openrouter
GROUNDING_BASE_URL=http://localhost:11434
GROUNDING_MODEL=mistral
GROUNDING_API_KEY=                 # optional separate key for OpenRouter grounding

# Storage and server
DATABASE_PATH=./dejavuguard.db
HOST=0.0.0.0
PORT=8000
```

## Predicates

A predicate defines an observable natural-language fact:

- `prop_id`: unique predicate identifier used in formulas.
- `description`: declarative description of the fact to ground.
- `role`: whether the predicate can be matched in `user` or `assistant` messages.
- `arity`: number of object positions.
- `arg_descriptions`: description of each object position.

Example predicates:

| Predicate | Role | Objects | Description |
|---|---|---|---|
| `p_allergy(a)` | user | allergen | the user mentions a specific allergy they have |
| `q_recipe(d, i)` | assistant | dish, ingredient | the assistant provides a cooking recipe for a dish containing an ingredient |
| `p_budget(m, b)` | user | manufacturer, maximum price | the user requests a vehicle manufacturer under a maximum price |
| `q_offer(m, p)` | assistant | manufacturer, price | the assistant offers a vehicle from a manufacturer at a price |

### Predicate-Specific Few-Shot Examples

When a new predicate is created, DejaVuGuard attempts to generate structured few-shot examples for that predicate using the configured few-shot generation model. The generated demonstrations contain:

- Three positive examples.
- Three challenging negative near-misses.
- Positive examples in the same `found`/`instances` output format used at runtime.
- Multi-instance examples where semantically plausible.
- Related-object history canonicalization examples where appropriate.

The generated examples are validated before they are stored. Validation checks include exact mention spans, required object completeness, canonical source format, and valid history references.

If example generation fails or no usable generation model is configured, the predicate is still saved and the UI reports that it will be grounded without predicate-specific few-shot examples.

## Policies and Related Objects

Policies are DejaVu formulas over predicate events. For example:

```text
forall m . forall p .
  (q_offer(m,p) -> exists b . (P p_budget(m,b) & !(b < p)))
```

This policy requires an assistant offer to correspond to a previously requested manufacturer and to respect the stated maximum price.

DejaVuGuard derives **related object positions** from active policies:

- Positions using the same logical variable are related.
- Positions using different variables are related when those variables are compared by `=`, `!=`, `<`, `<=`, `>`, or `>=`.

In the policy above:

- `q_offer.o1` and `p_budget.o1` are related through `m`.
- `q_offer.o2` and `p_budget.o2` are related because `p` and `b` are compared by `b < p`.

The related-object database is updated when policies are added, updated, removed, or when referenced predicates are removed.

## Playbooks

A **playbook** groups several policies and reads their verdicts together. The
combination of verdicts selects a *state*, and each state carries guidance for
the assistant plus an optional violation flag. This turns policies from a
binary gate into a behaviour selector, without changing how policies are
written or evaluated.

### One mode per session

A conversation runs in exactly one of two modes, never both:

| Mode | Behaviour |
|---|---|
| Policies | Every enabled policy is monitored and blocks on its own `False` verdict. The default, and unchanged. |
| Playbook | Only the selected playbook's members are monitored, and only its flagged states block. |

The mode is chosen per session, in the chat view. Switching restarts that
session's monitoring, because the DejaVu specification itself changes: in
playbook mode the spec contains only the playbook's member policies, so the
state vector is exactly the truth table's axes.

Because at most one playbook is ever active, a policy may belong to any number
of playbooks with no ambiguity about which one governs it.

### Members and polarity

Each member declares whether its guidance applies when its policy is `True` or
`False`. The default is `False`, preserving the existing meaning that a false
verdict is the interesting one:

- a *safety property* — "stay within budget" — wants guidance when violated
- a *detector* — "the user disclosed an allergy" — wants guidance when satisfied

One global convention would have forced one of these to be written inside-out.

### States, defaults, and behaviours

With *n* members there are 2^*n* states. An unedited state derives its guidance
from its members: every member whose verdict matches its polarity contributes
its text, followed by any global rules marked *apply to all*. Only edits are
stored, so a five-member playbook with two customised states holds two rows,
not thirty-two.

States that produce **identical guidance and identical flagged status** are the
same *behaviour* and display as one group. Merging is derived, never stored, so
deliberately giving two states the same guidance visibly merges them. The
flagged bit is part of the identity: same words with different consequences are
not the same behaviour.

### Blocking

In playbook mode **only flagged states block**. A member returning `False` does
not block on its own — otherwise every state containing an `F` would be
unreachable and the truth table pointless.

The consequence is worth stating plainly: **a playbook with no flagged state
blocks nothing at all**, for the whole session. The editor warns when a member
can no longer cause a block, and the playbook list shows the flagged-state
count.

### Guidance delivery

Guidance is sent as a `system` message immediately before the current user
turn, and is never stored. Appending it to the user's text would make the model
treat guidance as something the user said, inviting it to reply to the
instructions or weigh them against the user's own wording.

It is never shown in the conversation. To see what was applied, expand a
message's details panel, beside the grounding details.

### Testing playbooks offline

`scenario_runner/scenarios/playbook_scenario/` contains scenarios that exercise
playbooks with no API key and no model, using the checked-in stub grounder:

```bash
java -jar backend/libs/dejavu.jar --server --port 8080 --storage /tmp/pb-sessions &
uv run python -m scenario_runner.support.stub_grounding --port 9099 \
    --rules scenario_runner/support/playbook_grounding.json &

DATABASE_PATH=/tmp/pb.db DEJAVU_URL=http://localhost:8080 \
  uv run python -m scenario_runner --dir scenario_runner/scenarios/playbook_scenario/ \
  --grounding-provider vllm --grounding stub-grounder \
  --grounding-base-url http://localhost:9099
```

`--grounding-base-url` is required here: the base URL is a property of the
machine rather than of the scenario, and without it the runner uses the stored
setting, which defaults to Ollama's port.

## Grounding Output

For every relevant predicate and message, the grounding LLM receives:

- The predicate description.
- The required objects and their descriptions.
- The message text and message role.
- Predicate-specific few-shot demonstrations.
- The related-object context, including related predicate descriptions and object descriptions.
- Related-object history from the current conversation, including earlier mentions and their canonical forms.

The grounding LLM returns JSON only.

### No Match

```json
{
  "found": false
}
```

### One or More Matches

```json
{
  "found": true,
  "instances": [
    {
      "instance_id": "i1",
      "object_mentions": [
        {
          "object_id": "o1",
          "mention": "Toyota",
          "canonical_form": "Toyota",
          "canonical_source": {"type": "new"}
        },
        {
          "object_id": "o2",
          "mention": "$12,000",
          "canonical_form": "12000 USD",
          "canonical_source": {"type": "new"}
        }
      ]
    },
    {
      "instance_id": "i2",
      "object_mentions": [
        {
          "object_id": "o1",
          "mention": "Skoda",
          "canonical_form": "Skoda",
          "canonical_source": {"type": "new"}
        },
        {
          "object_id": "o2",
          "mention": "$12,500",
          "canonical_form": "12500 USD",
          "canonical_source": {"type": "new"}
        }
      ]
    }
  ]
}
```

Each instance must contain one exact mention for every object required by the predicate. Mentions are copied verbatim from the message. The canonical form, not the mention, is forwarded to DejaVu.

## Canonicalization Across Predicates

Canonical forms allow policies to compare logically equivalent values even when their wording differs across messages or predicates.

For example, suppose the active policy relates:

- `p_allergy(a)`: an allergen named by the user.
- `q_recipe(d, i)`: an ingredient in a recipe returned by the assistant.

Conversation:

```text
User: I am allergic to sesame.
Assistant: Try a noodle bowl with tahini dressing.
```

When grounding the assistant message, the prompt includes:

- Related-object context indicating that `q_recipe`'s ingredient is related to `p_allergy`'s allergen.
- Related-object history containing a prior canonical value for `sesame`.

Although `tahini` would ordinarily be kept as its own ingredient name, the relationship context makes the policy-relevant canonicalization `sesame` appropriate:

```json
{
  "object_id": "o2",
  "mention": "tahini",
  "canonical_form": "sesame",
  "canonical_source": {
    "type": "history",
    "matched_history_index": 0
  }
}
```

The monitor therefore evaluates the assistant's ingredient against the same logical allergen value previously identified in the user message.

## Composite Events Sent to DejaVu

A single message may satisfy several predicates or satisfy one predicate multiple times. DejaVuGuard sends all grounded instances for that message in one composite event, preserving their shared conversation position.

For:

```text
User: I want a Toyota under $12,000 and a Skoda under $12,500.
```

the grounded instances may be sent to DejaVu as:

```json
[
  {"name": "p_budget", "args": ["Toyota", "12000 USD"]},
  {"name": "p_budget", "args": ["Skoda", "12500 USD"]}
]
```

Arguments use canonical forms. They are flat argument arrays for separate instances, not nested instance arrays inside one event.

## Monitoring Flow

For each message:

1. DejaVuGuard selects predicates whose role matches the message sender.
2. It loads each predicate's structured few-shot examples.
3. It computes related-object context from active policy relations.
4. It loads canonical history for related object positions from the current conversation.
5. The grounding LLM returns zero or more canonicalized predicate instances.
6. DejaVuGuard sends all matching instances from the message as a composite DejaVu event.
7. DejaVu evaluates the active policies incrementally.
8. If any policy is violated, the current message or response is blocked.
9. Grounding details displayed in the UI include matches, object mentions, and canonical forms.

The current grounding code also prints the related-object context and history blocks to standard output whenever a grounding prompt is submitted, to support inspection and debugging of canonicalization.

## Temporal Logic Operators

| Operator | Syntax | Meaning |
|---|---|---|
| Historically | `H phi` | `phi` held at every observed position up to now |
| Previously | `P phi` | `phi` held at some observed position up to now |
| Previous position | `@ phi` | `phi` held at the previous position |
| Since | `phi S psi` | `psi` occurred and `phi` held continuously since |
| Timed Previously | `P[<=n] phi` | `phi` held within the last `n` steps |
| Timed Historically | `H[>n] phi` | `phi` held at all positions earlier than `n` steps ago |
| For all seen | `forall x . phi(x)` | `phi` holds for all observed values of `x` |
| Exists seen | `exists x . phi(x)` | `phi` holds for some observed value of `x` |
| For all infinite | `Forall x . phi(x)` | `phi` holds over the unrestricted domain |
| Exists infinite | `Exists x . phi(x)` | `phi` holds for some unrestricted-domain value |
| Not | `!phi` | Negation |
| And | `phi & psi` | Conjunction |
| Or | `phi \| psi` | Disjunction |
| Implies | `phi -> psi` | Implication |

Cross-role formulas generally need temporal operators. At a user position, assistant-role predicates are not grounded at that same position, and vice versa.

## DejaVu HTTP API

DejaVuGuard communicates with the DejaVu server through:

```text
POST   /sessions              Create a monitor session
POST   /sessions/{id}/event   Send one event
POST   /sessions/{id}/events  Send a composite event
POST   /validate              Validate a specification
GET    /sessions/{id}         Get session status
GET    /sessions/{id}/history Get event history
DELETE /sessions/{id}         Delete a session
GET    /sessions              List sessions
GET    /health                Health check
```

Composite-event example:

```bash
curl -X POST localhost:8080/sessions/a1b2c3d4/events \
  -H "Content-Type: application/json" \
  -d '[{"name":"p_budget","args":["Toyota","12000 USD"]},{"name":"p_budget","args":["Skoda","12500 USD"]}]'
```

## Project Structure

```text
dejavuguard/
  backend/
    engine/
      dejavu_client.py          # DejaVu HTTP client
      grounding.py              # Prompt rendering and LLM grounding parsing
      monitor.py                # Conversation-to-composite-event orchestration
      spec_builder.py           # DejaVu specification construction
      trace.py                  # Conversation trace representation
    prompts/
      optimized_grounding.py    # Active system/user/assistant grounding prompts
    routers/
      chat.py                   # Monitored conversation endpoints
      policies.py               # Predicate/policy CRUD and few-shot generation
      settings.py               # Model settings and prompt migration
    services/                   # OpenRouter and grounding model clients
    store/db.py                 # SQLite persistence and schema migrations
    libs/                       # Bundled DejaVu server artifacts
  frontend/
    src/
      components/               # Chat, policy, and settings views
      api/                      # Typed API client
      hooks/                    # UI state hooks
  scenario_runner/              # Scripted end-to-end scenario execution
  scripts/                      # Helper scripts for local and batch runs
  tests/                        # Backend and end-to-end tests
  docker-compose.yml
```

## Testing

```bash
# Backend
python -m pytest tests/ --ignore=tests/e2e
python -m ruff check backend tests

# Frontend
cd frontend
npm test
npm run build
```

End-to-end tests require running DejaVuGuard and DejaVu.

## Troubleshooting

| Problem | Action |
|---|---|
| Updated prompts do not appear | Open Settings once after updating; prompt migration is applied when settings are loaded. |
| New predicate has no demonstrations | Check the selected few-shot generation model and its API/server connectivity; the predicate remains usable without demonstrations. |
| Grounding model unavailable | Configure and test the grounding provider in Settings. |
| OpenRouter chat fails | Check the chat-model OpenRouter API key and chosen model. |
| Related canonicalization is unexpected | Inspect the related-object context/history printed by the backend and shown through grounding details. |
| DejaVu is unreachable | Ensure the `dejavu` container or Java process is running and verify `DEJAVU_URL`. |
| Policy unexpectedly ignores another role | Use a temporal operator such as `P`, `@`, or `S` when linking events from different message positions. |
| Historical violation remains active | This is expected for irrevocable historical formulas such as `H (...)`; create a new conversation session to reset monitoring. |

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React, TypeScript, Tailwind CSS, Vite |
| Backend | FastAPI, Python, SQLite (`aiosqlite`) |
| Runtime verification | DejaVu, first-order past-time temporal logic |
| Chat model | OpenRouter |
| Grounding model | Ollama, LM Studio, vLLM, custom OpenAI-compatible server, or OpenRouter |

## License

MIT
