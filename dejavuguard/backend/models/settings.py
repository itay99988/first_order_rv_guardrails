"""
Pydantic models for application settings and LLM provider configuration.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class GroundingProvider(StrEnum):
    """Supported grounding LLM providers."""

    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    VLLM = "vllm"
    CUSTOM = "custom"  # Any OpenAI-compatible server
    OPENROUTER = "openrouter"


# Default grounding prompts — duplicated here to avoid circular imports
# with backend.engine.grounding.
#
# The new prompt strategy uses a SINGLE user message (no system prompt) with
# built-in few-shot examples from the grounding_dataset evaluation work.
# Separate templates exist for user-role and assistant-role predicates.

DEFAULT_GROUNDING_SYSTEM_PROMPT = 'You are a text annotation assistant for first-order grounding.\n\nYour task:\n- Decide whether a message expresses a given predicate description.\n- If it does, extract each complete predicate instance expressed in the message.\n- For each instance, extract exact verbatim mentions for each required object_id.\n- For each extracted mention, also provide a canonical_form.\n\nStrict rules:\n- Read the predicate literally and precisely.\n- Only return found=true if the message explicitly satisfies the exact predicate (not adjacent or related meaning).\n- If found=false, instances must be [].\n- If found=true, instances must include one item for each complete predicate occurrence in the message.\n- Each instance must include all required object_ids with exact substrings from the message and canonical forms.\n- Do not merge objects across different predicate occurrences.\n- Preserve object pairings/groups exactly as expressed in the message.\n- Do not paraphrase mentions.\n- The mention must be copied exactly from the message.\n- The canonical_form is the normalized identity/value for that mention, chosen using the related-object context and history when provided.\n- Output JSON only.\n\nOutput schema:\n{\n  "reasoning": "brief rationale",\n  "found": true,\n  "instances": [\n    {\n      "instance_id": "i1",\n      "object_mentions": [\n        {\n          "object_id": "o1",\n          "mention": "exact span",\n          "canonical_form": "canonical identity or value"\n        }\n      ]\n    }\n  ]\n}'

# ---------------------------------------------------------------------------
# Built-in few-shot examples for USER predicates
# ---------------------------------------------------------------------------

_USER_EXAMPLES_TEXT = r"""Message: "Could you walk me through enrollment deadlines and registration steps at Georgia Tech?"
Predicate: the user requests enrollment information for an organization
Objects:
  - o1: educational institution
Output: {"reasoning": "Enrollment deadlines and registration steps = enrollment information; Georgia Tech is the organization. MATCH.", "found": true, "instances": [{"instance_id": "i1", "object_mentions": [{"object_id": "o1", "mention": "Georgia Tech", "canonical_form": "Georgia Tech"}]}]}

Message: "Is there an outage affecting T-Mobile customers in Phoenix today?"
Predicate: the user asks about mobile network coverage in location
Objects:
  - o1: service area
Output: {"reasoning": "Asking about an outage ≠ asking about coverage quality. NO MATCH.", "found": false, "instances": []}

Message: "Could you get Priya Nair admitted to Coursera Academy's machine learning track?"
Predicate: the user asks about enrolling a person in an organization
Objects:
  - o1: prospective student
  - o2: educational institution
Output: {"reasoning": "Getting Priya Nair admitted = enrolling a person; Coursera Academy is the organization. MATCH.", "found": true, "instances": [{"instance_id": "i1", "object_mentions": [{"object_id": "o1", "mention": "Priya Nair", "canonical_form": "Priya Nair"}, {"object_id": "o2", "mention": "Coursera Academy", "canonical_form": "Coursera Academy"}]}]}

Message: "Could you list the airlines that operate out of Haneda Airport on Friday?"
Predicate: the user requests flights arriving at an airport
Objects:
  - o1: arrival airport
Output: {"reasoning": "Airlines operating OUT OF (departing) ≠ flights arriving. NO MATCH.", "found": false, "instances": []}

Message: "I'm planning to grow tomatoes in Andalusia, Spain."
Predicate: the user requests planting guidance for crop cultivation in a country region
Objects:
  - o1: country region
Output: {"reasoning": "Planning to grow tomatoes = requesting planting guidance; Andalusia, Spain is the country region. MATCH.", "found": true, "instances": [{"instance_id": "i1", "object_mentions": [{"object_id": "o1", "mention": "Andalusia, Spain", "canonical_form": "Andalusia, Spain"}]}]}

Message: "Can you tell me the acceptance rate for Stanford University?"
Predicate: the user requests enrollment information for an organization
Objects:
  - o1: educational institution
Output: {"reasoning": "Acceptance rate = admissions statistics, not enrollment process information. NO MATCH.", "found": false, "instances": []}

Message: "Could you list a few recipes from the Campbell's website that are popular right now?"
Predicate: the user requests a recipe that uses a named product as an ingredient
Objects:
  - o1: named product
Output: {"reasoning": "Requesting recipes FROM a website ≠ requesting a recipe that uses the product as ingredient. NO MATCH.", "found": false, "instances": []}

Message: "I need the scholarship deadline for MIT, not the regular application date."
Predicate: the user asks for the application deadline of an academic organization
Objects:
  - o1: academic organization
Output: {"reasoning": "Scholarship deadline ≠ application deadline; user explicitly says 'not the regular application date'. NO MATCH.", "found": false, "instances": []}

Message: "Please don't enroll Sarah in the leadership workshop yet \u2014 she hasn't completed the prerequisite."
Predicate: the user requests training enrollment for an employee in a skill
Objects:
  - o1: employee
  - o2: training program
Output: {"reasoning": "'Don't enroll' is a request NOT to do X, which is the opposite of requesting X. NO MATCH.", "found": false, "instances": []}

Message: "I need the university for Jason Murphy, but I can't remember whether it was USC or UC Irvine."
Predicate: the user provides a student and the university they attend
Objects:
  - o1: student name
  - o2: university
Output: {"reasoning": "User says 'can't remember' — requesting information, not providing it. NO MATCH.", "found": false, "instances": []}

Message: "There's a bug in Slack: when I try to share a screen in a call, the app freezes and disconnects all other participants."
Predicate: the user reports that a software product has a bug
Objects:
  - o1: software product
  - o2: bug description
Output: {"reasoning": "Reports a bug in Slack = match; bug description = the full clause describing the issue. MATCH.", "found": true, "instances": [{"instance_id": "i1", "object_mentions": [{"object_id": "o1", "mention": "Slack", "canonical_form": "Slack"}, {"object_id": "o2", "mention": "when I try to share a screen in a call, the app freezes and disconnects all other participants", "canonical_form": "when I try to share a screen in a call, the app freezes and disconnects all other participants"}]}]}"""

# ---------------------------------------------------------------------------
# Built-in few-shot examples for ASSISTANT predicates
# ---------------------------------------------------------------------------

_ASSISTANT_EXAMPLES_TEXT = r"""Message: "The film Spirited Away was directed by Hayao Miyazaki."
Predicate: the assistant provides the director of a film
Objects:
  - o1: film title
  - o2: director name
Output: {"reasoning": "Spirited Away = film, Hayao Miyazaki = director; predicate requires both. MATCH.", "found": true, "instances": [{"instance_id": "i1", "object_mentions": [{"object_id": "o1", "mention": "Spirited Away", "canonical_form": "Spirited Away"}, {"object_id": "o2", "mention": "Hayao Miyazaki", "canonical_form": "Hayao Miyazaki"}]}]}

Message: "Cristiano Ronaldo transferred from Manchester United before that season started."
Predicate: the assistant states that a person plays for a sports organization
Objects:
  - o1: athlete
  - o2: sports team
Output: {"reasoning": "'Transferred from' = past move away, not current membership; 'plays for' ≠ 'transferred from'. NO MATCH.", "found": false, "instances": []}

Message: "The ruling in Brown v. Board of Education was issued by the Supreme Court of the United States."
Predicate: the assistant provides the court that issued a legal ruling
Objects:
  - o1: issuing court
Output: {"reasoning": "Ruling 'was issued by the Supreme Court of the United States' = naming the issuing court. MATCH.", "found": true, "instances": [{"instance_id": "i1", "object_mentions": [{"object_id": "o1", "mention": "Supreme Court of the United States", "canonical_form": "Supreme Court of the United States"}]}]}

Message: "PyTorch and CUDA need to be compatible, so check the official compatibility matrix before installing."
Predicate: the assistant provides a package version for a computing product
Objects:
  - o1: software package
  - o2: version number
Output: {"reasoning": "Packages mentioned but no specific version number given; predicate requires a version. NO MATCH.", "found": false, "instances": []}

Message: "The address 10.0.4.15 is just the internal workstation that appears infected, not the malware's command server."
Predicate: the assistant provides the IP address of a command-and-control server associated with malware activity
Objects:
  - o1: C2 server IP address
Output: {"reasoning": "10.0.4.15 is explicitly described as the workstation, NOT the command server; predicate requires the C2 IP. NO MATCH.", "found": false, "instances": []}

Message: "The boiling point of water at sea level is 100 degrees Celsius."
Predicate: the assistant provides the boiling point of a substance
Objects:
  - o1: substance
  - o2: temperature value
Output: {"reasoning": "Boiling point of water = 100 degrees Celsius; both substance and temperature value present. MATCH.", "found": true, "instances": [{"instance_id": "i1", "object_mentions": [{"object_id": "o1", "mention": "water", "canonical_form": "water"}, {"object_id": "o2", "mention": "100 degrees Celsius", "canonical_form": "100 degrees Celsius"}]}]}

Message: "The quickest way to get from Lisbon Airport into downtown is usually by metro or taxi."
Predicate: the assistant provides the airport serving a city
Objects:
  - o1: airport name
  - o2: city name
Output: {"reasoning": "Sentence is about transit options from the airport, not stating which airport serves the city. NO MATCH.", "found": false, "instances": []}

Message: "The usual adult dose of lisinopril is often 10 mg once daily, though your prescriber may adjust it."
Predicate: the assistant provides the dosage duration for a medication
Objects:
  - o1: medication
  - o2: dosage duration
Output: {"reasoning": "'once daily' is dosage frequency (how often to take it), not duration (how long to take it). NO MATCH.", "found": false, "instances": []}

Message: "Johns Hopkins Medicine offers sleep studies, which could help figure out why you're snoring and tired."
Predicate: the assistant gives a diagnosis associated with a medical organization
Objects:
  - o1: medical organization
  - o2: diagnosis
Output: {"reasoning": "Offering a diagnostic test (sleep study) ≠ giving a diagnosis. NO MATCH.", "found": false, "instances": []}"""

# ---------------------------------------------------------------------------
# Prompt templates with objects (arity > 0)
# ---------------------------------------------------------------------------

# Escape curly braces in examples so they survive str.format()
_USER_EXAMPLES_ESCAPED = _USER_EXAMPLES_TEXT.replace("{", "{{").replace("}", "}}")
_ASSISTANT_EXAMPLES_ESCAPED = _ASSISTANT_EXAMPLES_TEXT.replace("{", "{{").replace("}", "}}")

DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_USER = 'You are a text annotation assistant. Your task is to determine if a user message matches a predicate description, and if so, extract each complete predicate instance from the message.\n\nRules:\n- Read the predicate description LITERALLY and PRECISELY. Only mark found=true if the message explicitly and specifically satisfies the exact predicate - not something adjacent, related, or similar.\n- Subtle mismatches count as NOT found. E.g. "asking about an outage" != "asking about coverage"; "departing from airport" != "arriving at airport"; "requests acceptance rate" != "requests enrollment information".\n- Mentions must be exact substrings copied verbatim from the message - do not paraphrase or generalize.\n- If the predicate is expressed multiple times in the same message, return one instances item per complete predicate occurrence.\n- Each instance must include all required object_ids for that occurrence.\n- Do not merge objects across different occurrences.\n- Preserve the object pairings/groups exactly as expressed in the message.\n- For every extracted object mention, include a canonical_form.\n- To choose canonical_form for each object, use the related object context and related object history below.\n- For each current object, the related object context gives:\n  - the related predicate\n  - the related object from that predicate\n- The related object history gives prior mention strings and their canonical forms for that related object, when such history exists.\n- For each current mention, do one of two things:\n  - pick one canonical_form from the related object history if the current mention refers to the same entity/value/concept\n  - define a new canonical_form if no prior canonical_form fits\n- canonical_form should be concise, stable, and not tied to the wording of this one message unless the mention itself is already the best canonical form.\n- If found is false, instances must be [].\n- Output a JSON object with fields: "reasoning" (brief check of whether the predicate matches), "found" (bool), "instances" (list). No other text.\n- Each instances item must have fields: "instance_id" and "object_mentions".\n- Each object_mentions item must have fields: "object_id", "mention", "canonical_form".\n\nExamples:\n{{USER_EXAMPLES_BLOCK}}\n\nAdditional multi-instance example:\nMessage: "I\'m considering Toyota under 12000$ and Skoda under 12500$."\nPredicate: the user requests a car brand under a maximum price\nObjects:\n  - o1: car brand\n  - o2: maximum price\nOutput: {"reasoning": "The user gives two complete car-brand/maximum-price requests. Toyota pairs with 12000$, and Skoda pairs with 12500$.", "found": true, "instances": [{"instance_id": "i1", "object_mentions": [{"object_id": "o1", "mention": "Toyota", "canonical_form": "Toyota"}, {"object_id": "o2", "mention": "12000$", "canonical_form": "12000 USD"}]}, {"instance_id": "i2", "object_mentions": [{"object_id": "o1", "mention": "Skoda", "canonical_form": "Skoda"}, {"object_id": "o2", "mention": "12500$", "canonical_form": "12500 USD"}]}]}\n\nNow annotate the following:\n\nMessage: "{{TEXT}}"\nPredicate: {{PREDICATE_DESCRIPTION}}\nObjects:\n{{OBJECTS_BLOCK}}\nRelated object context:\n{{RELATED_OBJECT_CONTEXT_BLOCK}}\nRelated object mention and canonical history:\n{{RELATED_OBJECT_HISTORY_BLOCK}}\nOutput:'

DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_ASSISTANT = 'You are a text annotation assistant.\n\nExamples:\n{{ASSISTANT_EXAMPLES_BLOCK}}\n\nTask: determine if an assistant message matches a predicate description and extract each complete predicate instance from the message.\nRules:\n- Read the predicate LITERALLY. Only mark found=true if the message explicitly satisfies the exact predicate - not something adjacent or similar.\n- Subtle mismatches = NOT found: "transferred from team" != "plays for team"; "no version given" != "provides version"; "workstation IP" != "C2 server IP"; "nominated for award" != "won award".\n- If the message explicitly says it CANNOT confirm or it does NOT satisfy the predicate fact -> NOT found.\n- A shopping list, food pairing suggestion, or ingredient substitution != a recipe using a product as an ingredient.\n- Mentions must be exact verbatim substrings - do not paraphrase.\n- If the predicate is expressed multiple times in the same message, return one instances item per complete predicate occurrence.\n- Each instance must include all required object_ids for that occurrence.\n- Do not merge objects across different occurrences.\n- Preserve the object pairings/groups exactly as expressed in the message.\n- For every extracted object mention, include a canonical_form.\n- To choose canonical_form for each object, use the related object context and related object history below.\n- For each current object, the related object context gives:\n  - the related predicate\n  - the related object from that predicate\n- The related object history gives prior mention strings and their canonical forms for that related object, when such history exists.\n- For each current mention, do one of two things:\n  - pick one canonical_form from the related object history if the current mention refers to the same entity/value/concept\n  - define a new canonical_form if no prior canonical_form fits\n- canonical_form should be concise, stable, and not tied to the wording of this one message unless the mention itself is already the best canonical form.\n- If found is false, instances must be [].\n- Output a JSON object with fields: "reasoning" (brief check), "found" (bool), "instances" (list). No other text.\n- Each instances item must have fields: "instance_id" and "object_mentions".\n- Each object_mentions item must have fields: "object_id", "mention", "canonical_form".\n\nAdditional multi-instance example:\nMessage: "The Toyota Corolla is available for 11500$, and the Skoda Octavia is listed at 12400$."\nPredicate: the assistant provides a car model and price\nObjects:\n  - o1: car model\n  - o2: price\nOutput: {"reasoning": "The assistant provides two complete car-model/price facts. Toyota Corolla pairs with 11500$, and Skoda Octavia pairs with 12400$.", "found": true, "instances": [{"instance_id": "i1", "object_mentions": [{"object_id": "o1", "mention": "Toyota Corolla", "canonical_form": "Toyota Corolla"}, {"object_id": "o2", "mention": "11500$", "canonical_form": "11500 USD"}]}, {"instance_id": "i2", "object_mentions": [{"object_id": "o1", "mention": "Skoda Octavia", "canonical_form": "Skoda Octavia"}, {"object_id": "o2", "mention": "12400$", "canonical_form": "12400 USD"}]}]}\n\nAnnotate:\n\nMessage: "{{TEXT}}"\nPredicate: {{PREDICATE_DESCRIPTION}}\nObjects:\n{{OBJECTS_BLOCK}}\nRelated object context:\n{{RELATED_OBJECT_CONTEXT_BLOCK}}\nRelated object mention and canonical history:\n{{RELATED_OBJECT_HISTORY_BLOCK}}\nOutput:'

# Backward compatibility aliases.
DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE = DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_USER


class GroundingSettings(BaseModel):
    """Configuration for the grounding LLM.

    Attributes:
        provider: Grounding provider type.
        base_url: Server base URL (not used for OpenRouter).
        model: Model name on the grounding server.
        system_prompt: Shared system prompt for all predicates.
        user_prompt_template_user: User-prompt template for user-message predicates.
        user_prompt_template_assistant: User-prompt template for assistant-message predicates.
        api_key: API key for OpenRouter grounding (falls back to openrouter_api_key).
    """

    provider: str = GroundingProvider.OLLAMA
    base_url: str = "http://localhost:11434"
    model: str = "mistral"
    system_prompt: str = DEFAULT_GROUNDING_SYSTEM_PROMPT
    user_prompt_template_user: str = DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_USER
    user_prompt_template_assistant: str = DEFAULT_GROUNDING_USER_PROMPT_TEMPLATE_ASSISTANT
    api_key: str = ""


class AppSettings(BaseModel):
    """Full application settings.

    Attributes:
        openrouter_api_key: API key for OpenRouter.
        openrouter_model: Model identifier for the chat LLM (from dropdown).
        openrouter_model_custom: Custom model ID override (overrides dropdown when non-empty).
        grounding: Grounding LLM configuration.
    """

    openrouter_api_key: str = ""
    openrouter_model: str = ""
    openrouter_model_custom: str = ""
    few_shot_model: str = "chat"  # "chat" or "grounding" — which model generates few-shot examples
    grounding: GroundingSettings = GroundingSettings()
