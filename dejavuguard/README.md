# DejaVuGuard

Runtime verification for LLM conversations using first-order past-time temporal logic.

DejaVuGuard is a web-based chat interface backed by a formal runtime verification engine. Users chat with any LLM model via OpenRouter while [DejaVu](https://github.com/havelund/dejavu) continuously monitors the conversation trace against user-defined safety policies. Atomic propositions are semantically grounded by a dedicated grounding model — either a local LLM (Ollama, LM Studio, vLLM, or any OpenAI-compatible server) or a cloud model via OpenRouter — keeping the monitor independent of the chat model. When a policy violation is detected, the offending message is blocked before reaching the user.

## Key Features

- **First-Order Temporal Logic** — Define safety policies using DejaVu's first-order past-time LTL with quantification (`Forall`, `Exists`), temporal operators (`H`, `P`, `@`, `S`), timed constraints (`P[<=n]`, `H[>n]`), and recursive rules
- **200+ Chat Models** — Connect to any model on OpenRouter (GPT-4, Claude, Llama, Mistral, etc.)
- **Flexible Grounding** — Semantic proposition evaluation runs on a local LLM for privacy, or via OpenRouter for convenience
- **Persistent Sessions** — Monitor sessions survive server restarts via DejaVu's event replay. Conversations can pause for days and resume with full state
- **Live Formula Validation** — Real-time formula syntax checking as you type
- **Searchable Model Selection** — ModelCombobox with search, context length badges, and pricing for 300+ models

## Architecture

```
  User                  DejaVuGuard             Grounding LLM       DejaVu         Chat LLM
   |                    (FastAPI backend)          (local/cloud)      (HTTP server)   (OpenRouter)
   |                         |                          |                 |               |
   |  user message           |                          |                 |               |
   |------------------------>|                          |                 |               |
   |                         |  ground user props       |                 |               |
   |                         |------------------------->|                 |               |
   |                         |  {p: T/F, ...}           |                 |               |
   |                         |<-------------------------|                 |               |
   |                         |                          |                 |               |
   |                         |  send true props as composite events       |               |
   |                         |--------------------------------------->|               |
   |                         |  verdicts: {prop: T/F}                 |               |
   |                         |<---------------------------------------|               |
   |                         |  VIOLATION? --> BLOCK                      |               |
   |                         |                          |                 |               |
   |                         |  forward message (PASS)                    |               |
   |                         |-------------------------------------------------------->|
   |                         |                          |                 |  LLM response |
   |                         |<--------------------------------------------------------|
   |                         |                          |                 |               |
   |                         |  [repeat grounding + DejaVu for assistant response]       |
   |                         |                          |                 |               |
   |  response or            |                          |                 |               |
   |  violation alert        |                          |                 |               |
   |<------------------------|                          |                 |               |
```

## Quick Start

### Docker (recommended — no setup required)

```bash
docker compose up --build
```

This starts:
- **DejaVuGuard** (frontend + backend) on http://localhost:8001
- **DejaVu** (runtime verification server) on port 8080

Open http://localhost:8001, go to **Settings**, enter your OpenRouter API key, and start chatting.

### Prerequisites (for development)

- Python 3.11+
- Node.js 18+
- A grounding model (any one of):
  - [Ollama](https://ollama.ai) (recommended) — `ollama pull mistral`
  - [LM Studio](https://lmstudio.ai)
  - [vLLM](https://github.com/vllm-project/vllm)
  - Any OpenAI-compatible server
  - OpenRouter (uses your API key — no local setup needed)
- An [OpenRouter](https://openrouter.ai) API key
- DejaVu server running (see below)

### Development Setup

```bash
# Start DejaVu server (requires Java 17+)
java -jar backend/libs/dejavu.jar --server --port 8080 --storage ./sessions &

# Backend
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Frontend
cd frontend
npm install
cd ..

# Seed example policies (optional)
python scripts/seed_examples.py

# Start backend (terminal 1)
uvicorn backend.main:app --reload

# Start frontend (terminal 2)
cd frontend && npm run dev
```

Open http://localhost:5173.

## Usage

### 1. Configure Settings

- **Chat Model**: Enter your OpenRouter API key and select a model from the searchable combobox
- **Grounding Model**: Select your provider (Ollama, LM Studio, vLLM, Custom, or OpenRouter), configure the base URL and model
- **Grounding Prompt**: Optionally customize the system/user prompts for the LLM judge

### 2. Define Predicates

Create predicates that describe observable behaviors. Each predicate has a name, role, arity (number of data arguments), and descriptions for each argument:

| ID | Role | Arity | Arg Descriptions | Description |
|----|------|-------|-----------------|-------------|
| `p_fraud` | user | 0 | — | The user requests methods for committing financial fraud |
| `p_transfer` | user | 2 | account, destination | The user requests a money transfer to a destination |
| `q_provide_instructions` | assistant | 1 | topic | The assistant provides step-by-step instructions on a topic |
| `p_report_bug` | user | 2 | software product, bug description | The user reports a software bug |

When arity > 0, the grounding LLM extracts verbatim mentions of each argument from the message. For example, if the user says "Transfer $50,000 from account 1234 to offshore", the grounding extracts `p_transfer("1234", "offshore")`.

### 3. Build Policies

Combine predicates into first-order temporal logic formulas using DejaVu syntax:

```
forall x . forall y . (p_transfer(x, y) -> @ P p_auth(x))
```

"For every transfer of account x to destination y, account x must have been authenticated at some previous step."

```
H (P p_fraud -> forall t . !q_provide_instructions(t))
```

"If the user ever requested fraud, the assistant must never provide instructions on any topic."

```
forall prod . forall bug . (p_report_bug(prod, bug) -> P q_acknowledge(prod))
```

"For every bug report about a product, the assistant must have previously acknowledged that product."

Formulas are validated by DejaVu's own parser when you click Save — syntax and wellformedness errors are shown immediately.

### 4. Chat

The monitor evaluates every message in real time. If a policy is violated, the message is blocked and a red alert explains which policy was breached. Sessions persist across server restarts — you can close the browser, restart the server, and continue the same conversation with the same monitor state.

## Temporal Logic Operators

DejaVu supports a rich set of first-order past-time temporal logic operators:

| Operator | Syntax | Meaning |
|----------|--------|---------|
| Historically | `H phi` | phi held at every step up to now |
| Previously | `P phi` | phi held at some past step or now |
| Yesterday/Previous | `@ phi` | phi held at the previous step |
| Since | `phi S psi` | psi occurred and phi held continuously since |
| Timed Previously | `P[<=n] phi` | phi held within the last n steps |
| Timed Historically | `H[>n] phi` | phi held at all steps beyond n ago |
| For all seen | `forall x . phi(x)` | phi holds for all **seen** values of x (recommended default) |
| Exists seen | `exists x . phi(x)` | phi holds for some **seen** value of x (recommended default) |
| For all (infinite) | `Forall x . phi(x)` | phi holds for **all** values of x (infinite domain) |
| Exists (infinite) | `Exists x . phi(x)` | phi holds for **some** value of x (infinite domain) |
| Not | `!phi` | negation |
| And | `phi & psi` | conjunction |
| Or | `phi \| psi` | disjunction |
| Implies | `phi -> psi` | implication |

### Example Policies

**Per-Account Transfer Authorization:**
```
forall acc . forall dest . (p_transfer(acc, dest) -> @ P p_auth(acc))
```
Predicates: `p_transfer(account, destination)` (user, arity 2), `p_auth(account)` (user, arity 1).
For every transfer request, the referenced account must have been authenticated in a previous step.

**Bug Report Tracking:**
```
forall prod . (p_report_bug(prod) -> P q_acknowledge(prod))
```
Predicates: `p_report_bug(software product)` (user, arity 1), `q_acknowledge(product)` (assistant, arity 1).
For every bug reported about a product, the assistant must have previously acknowledged that product.

**Enrollment Compliance:**
```
forall student . forall org . (p_enroll(student, org) -> P p_consent(student))
```
Predicates: `p_enroll(student, organization)` (user, arity 2), `p_consent(student)` (user, arity 1).
A student can only be enrolled in an organization if they previously gave consent.

**Medical Safety — No Diagnosis Without Caveat:**
```
H (@ p_symptom -> !q_diagnosis)
```
Predicates: `p_symptom` (user, arity 0), `q_diagnosis` (assistant, arity 0).
If the user described symptoms at the previous step, the assistant must not provide a diagnosis (without caveats).

**Fraud Prevention (Boolean):**
```
H (P p_fraud -> !q_comply)
```
Predicates: `p_fraud` (user, arity 0), `q_comply` (assistant, arity 0).
If the user ever requested fraud techniques, the assistant must never comply. Arity-0 predicates work as Boolean flags.

**Important:** Cross-role formulas must use temporal operators (`P`, `@`, `S`) to reference predicates from a different role. Each step only grounds predicates matching the message's role — other predicates default to `False`.

## How It Works

### Runtime Verification with DejaVu

DejaVuGuard uses [DejaVu](https://github.com/havelund/dejavu) as its runtime verification engine. DejaVu is a first-order past-time linear temporal logic monitor that uses Binary Decision Diagrams (BDDs) for efficient evaluation of quantified formulas over potentially infinite data domains.

For each conversation:
1. A **DejaVu session** is created with the policies compiled into a specification
2. Each message is **grounded**: the grounding LLM evaluates each predicate against the message text and extracts argument data
3. Matched predicates are sent to DejaVu as **composite events** with extracted arguments
4. DejaVu evaluates all temporal properties and returns **per-property verdicts**
5. If any property is violated, the message is **blocked**

### Semantic Grounding with Data Extraction

The grounding engine uses an LLM-as-judge approach to evaluate each predicate against the message text. For predicates with arguments (arity > 0), the LLM also extracts **verbatim mentions** of each argument from the message.

For example, given:
- Predicate: `p_transfer(account, destination)` — "the user requests a money transfer"
- Message: "Please transfer funds from account 1234 to the offshore branch"

The grounding LLM returns:
```json
{
  "found": true,
  "reasoning": "User requests a transfer; account=1234, destination=offshore branch",
  "object_mentions": [
    {"object_id": "o1", "mention": "1234"},
    {"object_id": "o2", "mention": "offshore branch"}
  ]
}
```

The extracted mentions are sent to DejaVu as event arguments: `p_transfer("1234", "offshore branch")`. This enables first-order quantification — DejaVu can track which specific accounts, products, or entities are involved across the conversation.

The grounding prompts include **built-in few-shot examples** (separate sets for user-role and assistant-role predicates) that teach the LLM to:
- Match predicates **literally** — subtle mismatches are rejected
- Distinguish requests from refusals, education, or adjacent topics
- Extract **exact verbatim substrings** from the message, not paraphrases
- Return empty `object_mentions` when `found=false`

### Session Persistence

DejaVu sessions persist to disk as JSON event logs. If the server restarts, sessions are automatically restored by replaying stored events through a fresh monitor. Since DejaVu monitors are deterministic (same spec + same events = same state), replay produces the exact same BDD state. This means conversations survive server restarts, container recreations, and even machine reboots — important for chatbot monitoring where messages can arrive days apart.

## DejaVu: The Runtime Verification Engine

DejaVu is the formal engine that powers DejaVuGuard. It implements first-order past-time linear temporal logic with recursive rules and time constraints. The implementation uses BDDs (Binary Decision Diagrams) for representing assignments to quantified variables, enabling efficient monitoring over large data domains.

### The Specification Logic

A DejaVu specification consists of predicate declarations, optional macros, and property definitions:

```
pred open(file, mode)
pred close(file)

pred isOpen(f) = !close(f) S open(f)

prop filePolicy : Forall f . (close(f) -> Exists m . @ [open(f,m), close(f)))
```

#### Grammar

```
<doc> ::= <def> ... <def>
<def> ::= <eventdef> | <macrodef> | <propertydef>

<eventdef>    ::= 'pred' <event>,...,<event>
<event>       ::= <id> [ '(' <id> ',' ... ',' <id> ')' ]

<macrodef>    ::= 'pred' <id> [ '(' <id> ',' ... ',' <id> ')' ] '=' <form>

<propertydef> ::= 'prop' <id> ':' <form> ['where' <ruledef> ',' ... ',' <ruledef>]
<ruledef>     ::= <id> ['(' <id> ',' ... ',' <id> ')'] ':=' <form>

<form> ::= 
     'true' | 'false' 
   | <id> [ '(' <param> ',' ... ',' <param> ')' ]
   | <form> <binop> <form> 
   | '[' <form> ',' <form> ')'
   | <unop> <form>
   | <id> <oper> (<id> | <const>)
   | '(' <form> ')'
   | <quantifier> <id> '.' <form>

<param>      ::= <id> | <const>
<const>      ::= <string> | <integer>
<binop>      ::= '->' | '|' | '&' | 'S' [<time>] | 'Z' <timeLE>
<unop>       ::= '!' | '@' | 'P' [<time>] | 'H' [<time>] 
<oper>       ::= '<' | '<=' | '=' | '>' | '>='    
<quantifier> ::= 'exists' | 'forall' | 'Exists' | 'Forall'
<time>       ::= <timeLE> | <timeGT>
<timeLE>     ::= '[<=' <number> ']'
<timeGT>     ::= '[>' <number> ']'    
```

#### Formula Semantics

```
true, false        Boolean truth and falsehood 
id(v1,...,vn)      event or call of predicate macro
p -> q             p implies q
p | q              p or q
p & q              p and q
p S q              p since q (q was true in the past, p held continuously since)
p S[<=d] q         p since q, where q occurred within d time units
p S[>d] q          p since q, where q occurred earlier than d time units
[p, q)             interval notation, equivalent to: !q S p
! p                not p
@ p                in previous state p is true
P p                in some previous state p is true
P[<=d] p           in some previous state within d time units p is true
P[>d] p            in some previous state earlier than d time units p is true
H p                in all previous states p is true
H[<=d] p           in all previous states within d time units p is true
H[>d] p            in all previous states earlier than d time units p is true
x op k             x is related to variable or constant k (e.g.: x < 10, x <= y)
exists x . p(x)    there exists an x such that seen(x) and p(x)
forall x . p(x)    for all x, if seen(x) then p(x)
Exists x . p(x)    there exists an x such that p(x) (infinite domain)
Forall x . p(x)    for all x, p(x) (infinite domain)
```

Note: `seen(x)` holds if the value `x` has been observed in any event in the past.

#### Rules (Recursive Definitions)

Rules allow expressing recursive temporal relationships such as transitive closure:

```
prop spawning :
  Forall x . Forall y . Forall d . report(y,x,d) -> spawned(x,y) 
  where 
    spawned(x,y) := 
        @ spawned(x,y) 
      | spawn(x,y) 
      | Exists z . (@spawned(x,z) & spawn(z,y))
```

This defines `spawned(x,y)` recursively: either it held in the previous state, or there is a direct `spawn(x,y)` now, or there is a transitive chain through some intermediate thread `z`.

#### Macros

Macros provide named shorthands for formulas:

```
pred isOpen(f) = !close(f) S open(f)

prop filePolicy : Forall f . (close(f) -> isOpen(f))
```

Macros can call other macros but cannot be recursive. Use rules for recursion.

### DejaVu Server API

DejaVuGuard communicates with DejaVu through its HTTP server API. The server manages persistent monitor sessions:

```
POST   /sessions              Create a new monitor session
POST   /sessions/{id}/event   Send one event, get verdict
POST   /sessions/{id}/events  Send composite events (simultaneous), get verdict
POST   /validate              Validate a spec without creating a session
GET    /sessions/{id}         Get session status
GET    /sessions/{id}/history Get full event log with per-event verdicts
DELETE /sessions/{id}         Close and delete a session
GET    /sessions              List all sessions
GET    /health                Health check
```

**Create a session:**
```bash
curl -X POST localhost:8080/sessions \
  -d '{"spec": "prop file : Forall f . (close(f) -> P open(f))", "bits": 20}'
# → {"session_id":"a1b2c3d4","properties":["file"],"status":"ready"}
```

**Send an event:**
```bash
curl -X POST localhost:8080/sessions/a1b2c3d4/event \
  -d '{"name": "open", "args": ["file1"]}'
# → {"event_number":1,"verdicts":{"file":true},"violations":[]}
```

**Send composite events (simultaneous):**
```bash
curl -X POST localhost:8080/sessions/a1b2c3d4/events \
  -d '[{"name":"open","args":["f1"]},{"name":"close","args":["f2"]}]'
# → {"event_number":2,"verdicts":{"file":false},"violations":["file"]}
```

**Validate a formula (no session created):**
```bash
curl -X POST localhost:8080/validate \
  -d '{"spec": "pred p_fraud\nprop test : H (P p_fraud -> !q_comply)"}'
# → {"valid":true,"properties":["test"]}
```

**Session persistence:** Sessions are saved as JSON event logs. On server restart, they are restored by replaying all stored events through a fresh monitor, producing the exact same state.

### How DejaVuGuard Maps to DejaVu

| DejaVuGuard Concept | DejaVu Concept |
|---------------------|---------------|
| Chat conversation | DejaVu session |
| Chat message (user/assistant) | Composite event (all true predicates sent simultaneously) |
| Predicate (e.g., `p_fraud`) | Event predicate in the spec |
| Policy formula | Property in the spec (`prop name : formula`) |
| Grounding result (true/false) | Event present/absent in the composite |
| Violation detected | Property evaluates to false |
| Session persistence | JSON event log + replay on restore |

## Project Structure

```
dejavuguard/
  backend/
    engine/
      dejavu_client.py   # HTTP client for DejaVu RV server
      spec_builder.py    # Converts policies to DejaVu spec format
      grounding.py       # LLM-as-judge semantic grounding
      monitor.py         # Orchestrator: grounding -> DejaVu -> verdict
      trace.py           # Conversation trace model
    libs/
      dejavu.jar         # Pre-built DejaVu runtime (self-contained)
      Dockerfile.dejavu  # Minimal Docker image for DejaVu server
    routers/             # FastAPI endpoints (chat, policies, settings)
    services/            # OpenRouter + local LLM + grounding clients
    store/db.py          # SQLite persistence (aiosqlite)
  frontend/
    src/
      components/        # React components (chat, rules, settings, shared)
      hooks/             # Custom hooks (useChat, usePolicies, useSettings)
      api/client.ts      # Typed API client
  tests/                 # pytest + Playwright E2E
  docker-compose.yml     # DejaVuGuard + DejaVu services
```

## Testing

```bash
# All backend tests
pytest tests/ --ignore=tests/e2e

# Frontend tests
cd frontend && npx vitest run

# E2E tests (requires running app)
cd tests/e2e && python -m pytest

# Lint
ruff check backend/ tests/
cd frontend && npx tsc --noEmit
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Grounding model not configured" banner | Go to Settings, select a grounding provider and model. For Ollama: ensure `ollama serve` is running and a model is pulled. |
| "No models found" in model selector | Enter your OpenRouter API key first, then click the model selector. |
| Chat input disabled | Enter your OpenRouter API key in Settings. |
| DejaVu server unreachable | Ensure DejaVu is running (`docker compose up` or `java -jar backend/libs/dejavu.jar --server`). Check `DEJAVU_URL` env var. |
| Policy always passes | Check that proposition roles match message roles. Use `P()` or `@` for cross-turn formulas. |
| `H()` violation won't clear | By design — `H()` is irrevocable. Start a new session to reset. |
| Docker shows "Not Found" | Access via http://localhost:8001. Ensure `docker compose up --build` completed. |

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, Tailwind CSS, Vite |
| Backend | FastAPI, Python 3.11+, SQLite (aiosqlite) |
| Runtime Verification | DejaVu (Scala 3, first-order past-time LTL, BDD-based) |
| Chat LLM | OpenRouter API (200+ models) |
| Grounding | Local LLM (Ollama / LM Studio / vLLM / custom) or OpenRouter |
| Testing | pytest, Vitest, Playwright |

## License

MIT
