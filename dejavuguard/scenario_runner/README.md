# DejaVuGuard Scenario Runner

Replay pre-recorded user/assistant conversations through the grounding and
DejaVu monitoring pipeline without contacting a chat LLM. Used for:

- regression testing policies and predicates as the grounding model changes
- demoing/reproducing how a specific conversation triggers a policy
- bootstrapping predicates and policies into DejaVuGuard from a single JSON
- sweeping the same scenario set across multiple grounding models

## Requirements

- The DejaVuGuard backend's SQLite DB is reachable via the running app's
  config (`backend.config.get_config()` → defaults to `./data/dejavuguard.db`
  with the bind-mounted compose setup, or `./dejavuguard.db` otherwise).
- The DejaVu RV server is reachable at `DEJAVU_URL` (started by
  `docker compose up dejavu` or `java -jar dejavu.jar --server`).
- The grounding LLM specified in each scenario (or via `--grounding`) is
  reachable — Ollama, LM Studio, vLLM, OpenAI-compatible, or OpenRouter.
- If any scenario predicates lack inline `few_shot_examples`, the chat
  model configured in the DejaVuGuard Settings page (or
  `model.few_shot_model` in the scenario) must be reachable too.

## CLI

```bash
# Single scenario
uv run python -m scenario_runner scenario_runner/scenarios/allergan_scenario/allergy-pass-001.json

# Whole folder (alphabetical order)
uv run python -m scenario_runner --dir scenario_runner/scenarios/allergan_scenario/

# Allow updating existing predicates/policies whose shape differs
uv run python -m scenario_runner --dir scenario_runner/scenarios/ --overwrite

# Custom output directory, skip HTML report
uv run python -m scenario_runner --dir scenario_runner/scenarios/ --log-dir /tmp/logs --no-html

# Preserve the DejaVu session after the run (useful for debugging)
uv run python -m scenario_runner foo.json --keep-session

# Override the grounding model from every scenario without editing JSON files
uv run python -m scenario_runner --dir scenario_runner/scenarios/allergan_scenario \
  --grounding anthropic/claude-haiku-4.5

# Override both provider and model — useful for switching providers entirely
uv run python -m scenario_runner --dir scenario_runner/scenarios/allergan_scenario \
  --grounding-provider ollama --grounding llama3:8b

# Combine overrides with --overwrite to force a full refresh sweep
uv run python -m scenario_runner --dir scenario_runner/scenarios/allergan_scenario \
  --grounding mistralai/ministral-8b-2512 --overwrite
```

### Flag reference

| Flag | Purpose |
|---|---|
| `--dir <path>` | Run every `.json` file in `<path>` in alphabetical order. |
| `--log-dir <path>` | Root directory for batch output. Default: `scenario_runner/logs/`. |
| `--overwrite` | Update existing predicates/policies/related-objects when the scenario disagrees. Also triggers fresh few-shot regeneration when shape matches. |
| `--keep-session` | Don't delete the DejaVu session at the end of each scenario. Useful for inspecting state via the DejaVu API. |
| `--no-html` | Skip the HTML batch report. Markdown is always written. |
| `--grounding <model>` | Override every scenario's `model.grounding_model`. Useful for sweeping a scenario set across multiple models without editing JSON. |
| `--grounding-provider <provider>` | Override every scenario's `model.grounding_provider`. One of: `ollama`, `lm_studio`, `vllm`, `openai_compatible`, `openrouter`. |

### Wrapper script for macOS Docker users

`scripts/run_scenarios.sh` stops the `dejavuguard` UI container before the
run and restarts it afterward. This is required on macOS Docker Desktop
because its bind-mount fsync is asynchronous — UI-driven SQLite writes
don't become visible to the host until the container stops. The DejaVu
RV container stays up across runs.

```bash
# Defaults to the whole scenarios/ folder if no target given
./scripts/run_scenarios.sh --overwrite

# Pass any scenario_runner flag through
./scripts/run_scenarios.sh --dir scenario_runner/scenarios/allergan_scenario \
  --grounding anthropic/claude-haiku-4.5 --overwrite

# Single scenario
./scripts/run_scenarios.sh scenario_runner/scenarios/allergan_scenario/allergy-violate-001.json \
  --overwrite
```

### Exit codes

| Code | Meaning |
|---|---|
| 0 | All scenarios passed (or none had `expected_verdict`). |
| 1 | One or more scenarios had verdict mismatches. |
| 2 | Setup conflict — existing predicate/policy disagrees with scenario (without `--overwrite`). |
| 3 | Malformed scenario JSON / schema error. |
| 4 | Runtime error during the pipeline (DejaVu or grounding LLM unreachable). |

## Scenario JSON format

```json
{
  "scenario_id": "ride-sharing-1",
  "description": "Booking must follow request.",
  "model": {
    "grounding_provider": "openrouter",
    "grounding_model": "anthropic/claude-haiku-4.5",
    "few_shot_provider": "openrouter",
    "few_shot_model": "anthropic/claude-haiku-4.5"
  },
  "predicates": [
    {
      "prop_id": "user_car",
      "description": "the user requests a ride between two locations",
      "role": "user",
      "objects": [
        {"object_id": "o1", "description": "pickup location", "entity_type": "Location"},
        {"object_id": "o2", "description": "dropoff location", "entity_type": "Location"}
      ]
    }
  ],
  "policies": [
    {
      "policy_id": "car-booking-order",
      "name": "Assistant car offer must follow a matching user request",
      "formula_str": "Forall m . Forall p . assistant_car(m,p) -> exists b . ((@ noOther(m, b)) S user_car(m, b) & (b = p)) where noOther(m, b) := ! exists m2 . exists b2 . (user_car(m2, b2) & (!(m2 = m) | !(b2 = b)))"
    }
  ],
  "related_objects": [
    {
      "policy_id": "car-booking-order",
      "pairs": [
        ["user_car.o1", "assistant_car.o1"],
        ["user_car.o2", "assistant_car.o2"]
      ]
    }
  ],
  "messages": [
    {
      "role": "user",
      "text": "I need a ride from Berlin to Munich tomorrow.",
      "expected_verdict": {"car-booking-order": true}
    },
    {
      "role": "assistant",
      "text": "I can also offer a car from Hamburg to Bremen.",
      "expected_verdict": {"car-booking-order": false}
    }
  ]
}
```

### Field reference

| Field | Required | Notes |
|---|---|---|
| `scenario_id` | yes | Used in log filenames; should be filesystem-safe. |
| `description` | no | Free-form. Surfaced in logs and report. |
| `model.grounding_provider` | yes | `ollama` / `lm_studio` / `vllm` / `openai_compatible` / `openrouter`. Overridable via `--grounding-provider`. |
| `model.grounding_model` | yes | Model identifier expected by the provider. Overridable via `--grounding`. |
| `model.few_shot_model` | no | Only used when a predicate has no `few_shot_examples`. Falls back to the chat model configured in the DejaVuGuard Settings page. |
| `model.few_shot_provider` | no | Informational only — auto-generation uses OpenRouter via the configured chat-model API key. |
| `predicates[].prop_id` | yes | Used as the DejaVu predicate name and DB key. |
| `predicates[].description` | yes | Surfaced to the grounding LLM as the predicate definition. |
| `predicates[].role` | yes | `user` or `assistant` — restricts grounding to messages of that role. |
| `predicates[].arity` | no | Inferred from `objects` if omitted. Must agree if both supplied. |
| `predicates[].arg_descriptions` | no | Inferred from `objects[].description` if omitted. |
| `predicates[].objects` | no | List of `{object_id, description, entity_type}`. Cleaner than passing `arity` + `arg_descriptions` separately. |
| `predicates[].few_shot_examples` | no | **Recommended: omit.** Few-shots are auto-generated by the configured chat model when the predicate is created, mirroring the live app's behavior. Supply inline examples only if you need bit-exact reproducibility. With `--overwrite`, stored few-shots are refreshed (regenerated when the scenario omits them; replaced when it supplies them). |
| `policies[].policy_id` | yes | Used as the DB key and DejaVu property name. |
| `policies[].name` | yes | Human-readable name. |
| `policies[].formula_str` | yes | DejaVu first-order past-time formula. `where` clauses for local rule definitions are supported. |
| `policies[].enabled` | no | Defaults to `true`. |
| `related_objects` | no | Per-policy list of bidirectional object-slot links the grounding engine uses to surface canonical-form history across predicates. See below. |
| `messages[].role` | yes | `user` or `assistant`. |
| `messages[].text` | yes | Message body to ground. |
| `messages[].expected_verdict` | no | Map of `policy_id -> bool` checked after this message is processed. Mismatches counted as failures. |

### related_objects

Tell the grounding engine that two object slots refer to the same conceptual
entity, scoped to a policy. Each entry expands to **two** rows in the DB
(both directions) so the canonical_form history flows either way.

```json
"related_objects": [
  {
    "policy_id": "car-booking-order",
    "pairs": [
      ["user_car.o1", "assistant_car.o1"],
      ["user_car.o2", "assistant_car.o2"]
    ]
  }
]
```

This is the JSON equivalent of clicking "Related objects" in the policy UI
and linking `user_car.o1 ↔ assistant_car.o1`. Existing rows for the same
`policy_id` are reused as-is when they match exactly, or replaced when
`--overwrite` is passed (a diff is shown otherwise).

## Output layout

Each run writes one batch directory:

```
{log-dir}/
└── batch__YYYYMMDD-HHMMSS/
    ├── report.html                                          # color-coded summary table
    ├── report.md                                            # same in Markdown
    ├── {scenario}__{provider}_{model}__YYYYMMDD-HHMMSS.log  # human-readable per-msg log
    ├── {scenario}__{provider}_{model}__YYYYMMDD-HHMMSS.json # structured machine-parseable log
    └── failures/
        └── {scenario}__...log                               # only mismatched messages
```

The per-scenario `.log` is human-readable (one block per message:
grounding details with mentions/canonical_forms, composite event sent to
DejaVu, per-policy verdicts, expected vs actual marker). The `.json`
sibling has the same content in a CI-friendly structured form. The
`failures/` subdirectory is created only when at least one message
diverged from its expected verdict.

## Idempotency rules

- **Predicate** already exists with identical `role` / `arity` /
  `arg_descriptions` / `description` → reused. Different shape → abort
  with a clear diff, unless `--overwrite` is set (in which case the
  stored definition is updated).
- **Policy** already exists with identical `formula_str` + `name` →
  reused. Different → abort with a diff or update on `--overwrite`. The
  replacement formula is re-validated against the DejaVu server before
  insert.
- **Related objects** already exist for the same `policy_id` with the
  same pair set → reused. Different → abort with `+`/`-` diff or replace
  on `--overwrite`.
- **Few-shots** are not part of shape comparison. On `--overwrite`, they
  are refreshed: replaced if the scenario supplies them, regenerated via
  the configured chat model otherwise.

## Adding new scenarios

Drop a JSON file under `scenario_runner/scenarios/` (or a sub-folder)
and run:

```bash
./scripts/run_scenarios.sh --dir scenario_runner/scenarios/<your-folder>/ --overwrite
```

The first scenario that creates each predicate / policy / related_objects
entry pays the setup cost; subsequent scenarios reuse them.
