"""
Grounding prompt strategy. This is the only file you should edit.

Experiment with different prompting strategies to maximize accuracy on the grounding task.
evaluate.py imports this file and calls predict() on each of the 200 records.

Task:
  Given a text message, a predicate description, and a list of objects,
  determine whether the message matches the predicate, and if so, extract
  the exact verbatim mention of each object from the message.

predict() input:
  record: dict with keys:
    - text (str): the message to analyze
    - role (str): "user" or "assistant" — the speaker role in the conversation
    - predicate_description (str): NL description of what we're looking for
    - predicate_role (str): "user" or "assistant" — the role the predicate applies to
    - objects (list): [{"object_id": "o1", "description": "country region"}, ...]

predict() output:
  {"found": bool, "object_mentions": [{"object_id": "o1", "mention": "exact text"}, ...]}
  - found: True only if the message genuinely matches the predicate description
  - object_mentions: verbatim substrings from the text (only when found=True, else [])

Accuracy is strict: found must be exactly right AND all mention strings must be verbatim.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODEL = "google/gemma-3-4b-it"

# ---------------------------------------------------------------------------
# Few-shot examples for USER predicates
# ---------------------------------------------------------------------------

USER_EXAMPLES = [
    {
        "text": "Could you walk me through enrollment deadlines and registration steps at Georgia Tech?",
        "predicate_description": "the user requests enrollment information for an organization",
        "objects": [{"object_id": "o1", "description": "educational institution"}],
        "output": {"reasoning": "Enrollment deadlines and registration steps = enrollment information; Georgia Tech is the organization. MATCH.", "found": True, "object_mentions": [{"object_id": "o1", "mention": "Georgia Tech"}]},
    },
    {
        "text": "Is there an outage affecting T-Mobile customers in Phoenix today?",
        "predicate_description": "the user asks about mobile network coverage in location",
        "objects": [{"object_id": "o1", "description": "service area"}],
        "output": {"reasoning": "Asking about an outage ≠ asking about coverage quality. NO MATCH.", "found": False, "object_mentions": []},
    },
    {
        "text": "Could you get Priya Nair admitted to Coursera Academy's machine learning track?",
        "predicate_description": "the user asks about enrolling a person in an organization",
        "objects": [
            {"object_id": "o1", "description": "prospective student"},
            {"object_id": "o2", "description": "educational institution"},
        ],
        "output": {
            "reasoning": "Getting Priya Nair admitted = enrolling a person; Coursera Academy is the organization. MATCH.",
            "found": True,
            "object_mentions": [
                {"object_id": "o1", "mention": "Priya Nair"},
                {"object_id": "o2", "mention": "Coursera Academy"},
            ],
        },
    },
    {
        "text": "Could you list the airlines that operate out of Haneda Airport on Friday?",
        "predicate_description": "the user requests flights arriving at an airport",
        "objects": [{"object_id": "o1", "description": "arrival airport"}],
        "output": {"reasoning": "Airlines operating OUT OF (departing) ≠ flights arriving. NO MATCH.", "found": False, "object_mentions": []},
    },
    {
        "text": "I'm planning to grow tomatoes in Andalusia, Spain.",
        "predicate_description": "the user requests planting guidance for crop cultivation in a country region",
        "objects": [{"object_id": "o1", "description": "country region"}],
        "output": {"reasoning": "Planning to grow tomatoes = requesting planting guidance; Andalusia, Spain is the country region. MATCH.", "found": True, "object_mentions": [{"object_id": "o1", "mention": "Andalusia, Spain"}]},
    },
    {
        "text": "Can you tell me the acceptance rate for Stanford University?",
        "predicate_description": "the user requests enrollment information for an organization",
        "objects": [{"object_id": "o1", "description": "educational institution"}],
        "output": {"reasoning": "Acceptance rate = admissions statistics, not enrollment process information. NO MATCH.", "found": False, "object_mentions": []},
    },
    {
        "text": "Could you list a few recipes from the Campbell's website that are popular right now?",
        "predicate_description": "the user requests a recipe that uses a named product as an ingredient",
        "objects": [{"object_id": "o1", "description": "named product"}],
        "output": {"reasoning": "Requesting recipes FROM a website ≠ requesting a recipe that uses the product as ingredient. NO MATCH.", "found": False, "object_mentions": []},
    },
    {
        "text": "I need the scholarship deadline for MIT, not the regular application date.",
        "predicate_description": "the user asks for the application deadline of an academic organization",
        "objects": [{"object_id": "o1", "description": "academic organization"}],
        "output": {"reasoning": "Scholarship deadline ≠ application deadline; user explicitly says 'not the regular application date'. NO MATCH.", "found": False, "object_mentions": []},
    },
    {
        "text": "Please don't enroll Sarah in the leadership workshop yet — she hasn't completed the prerequisite.",
        "predicate_description": "the user requests training enrollment for an employee in a skill",
        "objects": [{"object_id": "o1", "description": "employee"}, {"object_id": "o2", "description": "training program"}],
        "output": {"reasoning": "'Don't enroll' is a request NOT to do X, which is the opposite of requesting X. NO MATCH.", "found": False, "object_mentions": []},
    },
    {
        "text": "I need the university for Jason Murphy, but I can't remember whether it was USC or UC Irvine.",
        "predicate_description": "the user provides a student and the university they attend",
        "objects": [{"object_id": "o1", "description": "student name"}, {"object_id": "o2", "description": "university"}],
        "output": {"reasoning": "User says 'can't remember' — requesting information, not providing it. NO MATCH.", "found": False, "object_mentions": []},
    },
    {
        "text": "There's a bug in Slack: when I try to share a screen in a call, the app freezes and disconnects all other participants.",
        "predicate_description": "the user reports that a software product has a bug",
        "objects": [
            {"object_id": "o1", "description": "software product"},
            {"object_id": "o2", "description": "bug description"},
        ],
        "output": {
            "reasoning": "Reports a bug in Slack = match; bug description = the full clause describing the issue. MATCH.",
            "found": True,
            "object_mentions": [
                {"object_id": "o1", "mention": "Slack"},
                {"object_id": "o2", "mention": "when I try to share a screen in a call, the app freezes and disconnects all other participants"},
            ],
        },
    },
]

# ---------------------------------------------------------------------------
# Few-shot examples for ASSISTANT predicates
# ---------------------------------------------------------------------------

ASSISTANT_EXAMPLES = [
    {
        "text": "The film Spirited Away was directed by Hayao Miyazaki.",
        "predicate_description": "the assistant provides the director of a film",
        "objects": [
            {"object_id": "o1", "description": "film title"},
            {"object_id": "o2", "description": "director name"},
        ],
        "output": {
            "reasoning": "Spirited Away = film, Hayao Miyazaki = director; predicate requires both. MATCH.",
            "found": True,
            "object_mentions": [
                {"object_id": "o1", "mention": "Spirited Away"},
                {"object_id": "o2", "mention": "Hayao Miyazaki"},
            ],
        },
    },
    {
        "text": "Cristiano Ronaldo transferred from Manchester United before that season started.",
        "predicate_description": "the assistant states that a person plays for a sports organization",
        "objects": [
            {"object_id": "o1", "description": "athlete"},
            {"object_id": "o2", "description": "sports team"},
        ],
        "output": {"reasoning": "'Transferred from' = past move away, not current membership; 'plays for' ≠ 'transferred from'. NO MATCH.", "found": False, "object_mentions": []},
    },
    {
        "text": "The ruling in Brown v. Board of Education was issued by the Supreme Court of the United States.",
        "predicate_description": "the assistant provides the court that issued a legal ruling",
        "objects": [{"object_id": "o1", "description": "issuing court"}],
        "output": {
            "reasoning": "Ruling 'was issued by the Supreme Court of the United States' = naming the issuing court. MATCH.",
            "found": True,
            "object_mentions": [{"object_id": "o1", "mention": "Supreme Court of the United States"}],
        },
    },
    {
        "text": "PyTorch and CUDA need to be compatible, so check the official compatibility matrix before installing.",
        "predicate_description": "the assistant provides a package version for a computing product",
        "objects": [
            {"object_id": "o1", "description": "software package"},
            {"object_id": "o2", "description": "version number"},
        ],
        "output": {"reasoning": "Packages mentioned but no specific version number given; predicate requires a version. NO MATCH.", "found": False, "object_mentions": []},
    },
    {
        "text": "The address 10.0.4.15 is just the internal workstation that appears infected, not the malware's command server.",
        "predicate_description": "the assistant provides the IP address of a command-and-control server associated with malware activity",
        "objects": [{"object_id": "o1", "description": "C2 server IP address"}],
        "output": {"reasoning": "10.0.4.15 is explicitly described as the workstation, NOT the command server; predicate requires the C2 IP. NO MATCH.", "found": False, "object_mentions": []},
    },
    {
        "text": "The boiling point of water at sea level is 100 degrees Celsius.",
        "predicate_description": "the assistant provides the boiling point of a substance",
        "objects": [{"object_id": "o1", "description": "substance"}, {"object_id": "o2", "description": "temperature value"}],
        "output": {
            "reasoning": "Boiling point of water = 100 degrees Celsius; both substance and temperature value present. MATCH.",
            "found": True,
            "object_mentions": [
                {"object_id": "o1", "mention": "water"},
                {"object_id": "o2", "mention": "100 degrees Celsius"},
            ],
        },
    },
    {
        "text": "The quickest way to get from Lisbon Airport into downtown is usually by metro or taxi.",
        "predicate_description": "the assistant provides the airport serving a city",
        "objects": [{"object_id": "o1", "description": "airport name"}, {"object_id": "o2", "description": "city name"}],
        "output": {"reasoning": "Sentence is about transit options from the airport, not stating which airport serves the city. NO MATCH.", "found": False, "object_mentions": []},
    },
    {
        "text": "The usual adult dose of lisinopril is often 10 mg once daily, though your prescriber may adjust it.",
        "predicate_description": "the assistant provides the dosage duration for a medication",
        "objects": [{"object_id": "o1", "description": "medication"}, {"object_id": "o2", "description": "dosage duration"}],
        "output": {"reasoning": "'once daily' is dosage frequency (how often to take it), not duration (how long to take it). NO MATCH.", "found": False, "object_mentions": []},
    },
    {
        "text": "Johns Hopkins Medicine offers sleep studies, which could help figure out why you're snoring and tired.",
        "predicate_description": "the assistant gives a diagnosis associated with a medical organization",
        "objects": [{"object_id": "o1", "description": "medical organization"}, {"object_id": "o2", "description": "diagnosis"}],
        "output": {"reasoning": "Offering a diagnostic test (sleep study) ≠ giving a diagnosis. NO MATCH.", "found": False, "object_mentions": []},
    },
]


def _format_example(ex):
    objects_str = "\n".join(f'  - {o["object_id"]}: {o["description"]}' for o in ex["objects"])
    return (
        f'Message: "{ex["text"]}"\n'
        f"Predicate: {ex['predicate_description']}\n"
        f"Objects:\n{objects_str}\n"
        f"Output: {json.dumps(ex['output'])}"
    )


def _build_user_prompt(record):
    objects_str = "\n".join(
        f'  - {o["object_id"]}: {o["description"]}' for o in record["objects"]
    )
    examples_str = "\n\n".join(_format_example(ex) for ex in USER_EXAMPLES)

    return f"""You are a text annotation assistant. Your task is to determine if a user message matches a predicate description, and if so, extract the exact verbatim mention of each object from the message.

Rules:
- Read the predicate description LITERALLY and PRECISELY. Only mark found=true if the message explicitly and specifically satisfies the exact predicate — not something adjacent, related, or similar.
- Subtle mismatches count as NOT found. E.g. "asking about an outage" ≠ "asking about coverage"; "departing from airport" ≠ "arriving at airport"; "requests acceptance rate" ≠ "requests enrollment information".
- Mentions must be exact substrings copied verbatim from the message — do not paraphrase or generalize.
- If found is false, object_mentions must be [].
- Output a JSON object with fields: "reasoning" (brief check of whether the predicate matches), "found" (bool), "object_mentions" (list). No other text.

Examples:

{examples_str}

Now annotate the following:

Message: "{record["text"]}"
Predicate: {record["predicate_description"]}
Objects:
{objects_str}
Output:"""


def _build_assistant_prompt(record):
    objects_str = "\n".join(
        f'  - {o["object_id"]}: {o["description"]}' for o in record["objects"]
    )
    examples_str = "\n\n".join(_format_example(ex) for ex in ASSISTANT_EXAMPLES)

    return f"""You are a text annotation assistant.

Examples:

{examples_str}

Task: determine if an assistant message matches a predicate description and extract verbatim object mentions.
Rules:
- Read the predicate LITERALLY. Only mark found=true if the message explicitly satisfies the exact predicate — not something adjacent or similar.
- Subtle mismatches = NOT found: "transferred from team" ≠ "plays for team"; "no version given" ≠ "provides version"; "workstation IP" ≠ "C2 server IP"; "nominated for award" ≠ "won award".
- If the message explicitly says it CANNOT confirm or it does NOT satisfy the predicate fact → NOT found.
- A shopping list, food pairing suggestion, or ingredient substitution ≠ a recipe using a product as an ingredient.
- Mentions must be exact verbatim substrings — do not paraphrase.
- If found is false, object_mentions must be [].
- Output a JSON object with fields: "reasoning" (brief check), "found" (bool), "object_mentions" (list). No other text.

Annotate:

Message: "{record["text"]}"
Predicate: {record["predicate_description"]}
Objects:
{objects_str}
Output:"""


def _sanitize(text):
    """Replace non-printable control characters (except tab/newline) with a space."""
    return "".join(c if c >= " " or c in "\t\n" else " " for c in text)


def predict(record):
    # Sanitize text to avoid control characters breaking JSON in model output
    record = {**record, "text": _sanitize(record["text"])}
    if record["predicate_role"] == "user":
        prompt_text = _build_user_prompt(record)
    else:
        prompt_text = _build_assistant_prompt(record)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0,
        max_tokens=256,
    )

    content = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract a JSON object from the response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(content[start:end])
        else:
            raise

    found = bool(result.get("found", False))
    object_mentions = result.get("object_mentions", []) if found else []
    return {"found": found, "object_mentions": object_mentions}
