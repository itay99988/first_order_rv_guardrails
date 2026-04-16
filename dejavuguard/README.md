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
   |                         |------------------------------------------->|               |
   |                         |  verdicts: {prop: T/F}                     |               |
   |                         |<-------------------------------------------|               |
   |                         |  VIOLATION? --> BLOCK                      |               |
   |                         |                          |                 |               |
   |                         |  forward message (PASS)                    |               |
   |                         |----------------------------------------------------------->|
   |                         |                          |                 |  LLM response |
   |                         |<-----------------------------------------------------------|
   |                         |                          |                 |               |
   |                         |   [repeat grounding + DejaVu for assistant response]       |
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

### 2. Define Propositions

Create atomic propositions that describe observable behaviors:

| ID | Role | Description |
|----|------|-------------|
| `p_fraud` | user | The user requests methods for committing financial fraud |
| `q_comply` | assistant | The assistant provides actionable fraud techniques |
| `p_allergy` | user | The user discloses a food allergy |
| `q_allergen` | assistant | The assistant suggests food containing the disclosed allergen |

### 3. Build Policies

Combine propositions into first-order temporal logic formulas using DejaVu syntax:

```
H (P p_fraud -> !q_comply)
```

"If the user ever requested fraud techniques, the assistant must never comply."

With quantification (future grounding with data extraction):
```
Forall acc . (p_transfer(acc, "offshore") & P p_fraud -> q_block(acc))
```

"For every account involved in an offshore transfer after a fraud request, block it."

Formulas are validated in real-time by DejaVu's own parser — syntax errors are caught as you type.

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

**Fraud Prevention:**
```
H (P p_fraud -> !q_comply)
```
If the user ever requested fraud techniques, the assistant must never comply. `P` remembers the request across turns. Irrevocable once violated.

**Sensitive Data Protection:**
```
H (@ p_sensitive -> q_warn) & H (P p_sensitive -> !q_echo)
```
After the user shares sensitive data, the assistant must warn on the next step (`@`) and must never echo that data back.

**Multi-Turn Jailbreak Prevention:**
```
H ((P p_escalate & P p_frame) -> !q_unsafe)
```
If the user has ever escalated and previously framed a harmful context, the assistant must not comply.

**Allergen Safety:**
```
H (@ p_allergy -> q_warn) & H (P p_allergy -> !q_allergen)
```
After a user discloses a food allergy, the assistant must warn on the next step and must never suggest food containing that allergen.

**First-Order: Track Specific Accounts (future, with data extraction):**
```
Forall acc . (p_transfer(acc, "offshore") & P p_fraud -> q_block(acc))
```
For every account involved in an offshore transfer after a prior fraud request, the assistant must block it. The `Forall` quantifier evaluates this for every account value seen in the conversation.

**Important:** Cross-role formulas must use temporal operators (`P`, `@`, `S`) to reference propositions from a different role. Each step only grounds propositions matching the message's role — other propositions default to `False`.

## How It Works

### Runtime Verification with DejaVu

DejaVuGuard uses [DejaVu](https://github.com/havelund/dejavu) as its runtime verification engine. DejaVu is a first-order past-time linear temporal logic monitor that uses Binary Decision Diagrams (BDDs) for efficient evaluation of quantified formulas over potentially infinite data domains.

For each conversation:
1. A **DejaVu session** is created with the policies compiled into a specification
2. Each message is **grounded**: the grounding LLM evaluates each predicate against the message text
3. True predicates are sent to DejaVu as **composite events** (simultaneous)
4. DejaVu evaluates all temporal properties and returns **per-property verdicts**
5. If any property is violated, the message is **blocked**

### Session Persistence

DejaVu sessions persist to disk as JSON event logs. If the server restarts, sessions are automatically restored by replaying stored events through a fresh monitor. Since DejaVu monitors are deterministic (same spec + same events = same state), replay produces the exact same BDD state. This means conversations survive server restarts, container recreations, and even machine reboots — important for chatbot monitoring where messages can arrive days apart.

### Future: Grounding with Extracted Data

Currently, predicates are Boolean (true/false). In the future, the grounding layer will extract entity data from messages, enabling first-order quantification:

```
# Current: Boolean grounding
p_fraud → True/False

# Future: Data-bearing predicates
p_transfer("1234", "offshore", "50000") → extracted from message

# Enables first-order specs:
Forall acc . (p_transfer(acc, "offshore") & P p_fraud) -> q_block(acc)
```

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
