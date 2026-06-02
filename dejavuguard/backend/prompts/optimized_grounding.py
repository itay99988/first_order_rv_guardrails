"""Optimized prompt templates for instance-aware grounding."""

SYSTEM_PROMPT = """You are a strict JSON-only extraction model for extended first-order grounding.

Step 1 - decide found=true or found=false. Return found=false unless the message ACTIVELY AND EXPLICITLY performs the predicate right now. Specifically, return found=false when:
- The message uses information-request framing to ask whether the predicate holds ("Can you tell me whether X", "Please confirm if X", "I need to confirm whether X", "Can you confirm if X")
- The message queries availability or existence ("Are there flights from X to Y?", "Is X available?")
- The predicate action is purely historical, conditional, or hypothetical - applies to BOTH questions AND statements: "Last year X was with Y", "previously X held Y", "if X were to..."
- The relevant entities appear only as background context, not as the direct subject of the predicate ("I need info about the case involving X and Y" - X and Y are context only)
- The message looks for or wants to find something, rather than actually requesting or providing it ("I want the title that X held")
- Not all required objects (o1, o2, ...) are explicitly present as distinct named mentions in the message (pronouns and vague references like "my wife", "him", "her" are NOT sufficient)

Note: the grammatical form does not determine found. Direct questions, tag questions, declarative statements, and checking expressions can all be found=true as long as the predicate relationship is directly expressed between explicitly named entities. In particular, for predicates that describe "the user asks about/for X", a direct question that queries that specific relationship ("Was X on Y?", "Is X at Y?") IS the predicate (found=true), as long as it is not phrased with information-request framing.

Return found=true only when the message itself directly performs or states the predicate as a current, active fact.

Step 2 - if found=true, extract instances. Each instance is one complete predicate occurrence:
- Scan the FULL message for EVERY occurrence of the predicate. Conjunctions like "and", "and also", "as well as" often introduce additional instances - extract each one separately.
- One instance per satisfying tuple (binary) or entity (unary)
- Every required object_id must appear exactly once per instance
- Never merge two separate occurrences into one instance
- If the same entity appears in the message under different names or aliases, each distinct mention creates its own separate instance
- mention = exact substring copied from the MESSAGE TEXT - never use a span from related_object_history
- CRITICAL: Every instance must contain a non-null, non-empty mention for EVERY required object_id. If you cannot find an explicit mention for any required object, do NOT output found=true - return {"found": false} instead.
- canonical_form = the value used to compare this mention with objects in related predicates. Determine it using BOTH related_object_context and related_object_history.
- The related-object context explains why objects are comparable. Use that relationship when normalization depends on the policy meaning rather than ordinary naming.
- Reuse a canonical_form from related_object_history when the current mention denotes or implies the same value in the related-object context, even if the surface words differ.
- canonical_source = {"type": "history", "matched_history_index": N} for history matches, {"type": "new"} otherwise

Output valid JSON only. No markdown."""


INSTANCE_RULES = """Instance rules:
One instance per entity/pair satisfying the predicate. "A and B" -> two instances, never one.
If the same entity appears under different names in the message, each distinct mention is its own instance.
mention = exact span from the MESSAGE TEXT.
-canonical_form = the value used to compare this mention with objects in related predicates. Determine it using BOTH related_object_context and related_object_history.
The related-object context explains why objects are comparable. Use that relationship when normalization depends on the policy meaning rather than ordinary naming.
The default is to reuse (copy) a canonical_form from related_object_history when the current mention denotes or implies the same value in the related-object context, even if the surface words differ. A new value for a  canonical_form should be used only when no canonical_form from history matches. 
canonical_source = {"type": "history", "matched_history_index": N} for history matches, {"type": "new"} otherwise.
**important: if you decided that a canonical_source is of "type": "history", then the current canonical_form cannot contain a new value!! Make sure this applies to all the generated canonical forms. if you decided that canonical_source is of {"type": "history"}, then the corresponding canonical_form must be copied from related object history, even if the few shot examples tell something else.  **"""


USER_MESSAGE_PROMPT = """You are grounding a USER message.

Predicate information:
{predicate_block}

Related object context:
{related_object_context}

Related object history:
{related_object_history}

Few-shot examples for this predicate:
{few_shot_block}

{instance_rules}

Message role: {role}
Reminder: return {"found": false} unless this message actively and exactly expresses: "{predicate_description}". Closely related or similar actions do not qualify.
Message text:
{text}

Return JSON only.
If not found: {"found": false}
If found: {"found": true, "instances": [...]}"""


ASSISTANT_MESSAGE_PROMPT = """You are grounding an ASSISTANT message.

Predicate information:
{predicate_block}

Related object context:
{related_object_context}

Related object history:
{related_object_history}

Few-shot examples for this predicate:
{few_shot_block}

{instance_rules}

Message role: {role}
Reminder: return {"found": false} unless this message actively and exactly expresses: "{predicate_description}". Closely related or similar actions do not qualify.
Message text:
{text}

Return JSON only.
If not found: {"found": false}
If found: {"found": true, "instances": [...]}"""
