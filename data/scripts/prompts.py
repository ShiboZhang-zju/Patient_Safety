#!/usr/bin/env python3
"""
Prompt templates for medical SFT data generation (v1).

Four categories aligned with SFT 数据标注规范 v1.0:
- general: 普通健康咨询 (low-risk, general advice)
- medication: 用药安全 (no individualized dosing, refer to clinician/pharmacist)
- boundary: 信息不足但索要诊断 (insufficient info, refuse to diagnose)
- high_risk: 高风险分诊 (escalate, seek care promptly)

v1 additions: unified output format, diversity constraint, variation hints,
and per-category negative constraints (no over-escalation, no soft diagnosis, etc.).
"""

import random
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Unified system prompt (keep as-is)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a cautious, professional, and empathetic medical assistant. "
    "You may provide general health information and guidance on when to seek care, "
    "but you cannot replace an in-person clinician, physical examination, testing, "
    "or individualized treatment decisions."
)

# ---------------------------------------------------------------------------
# Strict output format (append to every generation_instruction)
# ---------------------------------------------------------------------------
OUTPUT_CONSTRAINTS = (
    "Return exactly one valid JSON object with fields: id, category, messages. "
    "Set 'id' to a placeholder string beginning with 'sft_' followed by digits, for example 'sft_000001'. "
    "Set 'messages' to exactly 3 items with roles in this order: system, user, assistant. "
    "The first message must be the system message, and its content must exactly equal the provided system prompt. "
    "Do not output markdown fences, comments, explanations, or any extra text."
)

# ---------------------------------------------------------------------------
# Quality: avoid generic / formulaic assistant replies
# ---------------------------------------------------------------------------
QUALITY_CONSTRAINT = (
    "Make the assistant response specific, medically cautious, and practically useful. "
    "Avoid vague, generic, or purely formulaic wording."
)

# ---------------------------------------------------------------------------
# Diversity: avoid teacher just paraphrasing seed
# ---------------------------------------------------------------------------
DIVERSITY_CONSTRAINT = (
    "Make the new sample meaningfully different from any seed examples in wording, "
    "details, timeline, and presentation. Do not closely paraphrase the seed."
)

# ---------------------------------------------------------------------------
# Variation hints for diversity (pick one per call when generating)
# ---------------------------------------------------------------------------
VARIATION_HINTS = [
    "Use a different age group or family role.",
    "Change the symptom duration or timeline.",
    "Use more colloquial wording.",
    "Add one secondary symptom without changing the category.",
    "Make the question shorter and more conversational.",
]


def get_variation_hint(seed: Optional[int] = None) -> str:
    """
    Return a random variation hint. Optional seed for reproducibility.
    For batch generation with reproducible but varied hints, pass a different seed per sample
    (e.g. sample index) or have the caller use rng.choice(VARIATION_HINTS) instead.
    """
    if seed is not None:
        rng = random.Random(seed)
        return rng.choice(VARIATION_HINTS)
    return random.choice(VARIATION_HINTS)


# --- Category: general (普通健康咨询) ---
GENERAL = {
    "name": "general",
    "description": "Low-risk general health questions; give conservative, actionable advice with light boundary reminder.",
    "system": SYSTEM_PROMPT,
    "generation_instruction": (
        "Generate a realistic user question about a mild, low-risk health concern or lifestyle issue "
        "(for example mild cold symptoms, sleep, diet, checkup preparation, mild stomach discomfort, or home care). "
        "Then write a cautious and helpful assistant reply: provide practical, conservative, low-risk advice; "
        "mention what to monitor; and explain when in-person care may be needed if symptoms worsen or warning signs appear. "
        "Do not diagnose, prescribe, or over-escalate mild symptoms. "
        "Include only a brief boundary reminder. "
        "Avoid over-escalation for mild symptoms. "
        "Do not recommend emergency care unless the scenario truly includes red flags. "
        "Keep the assistant response concise and natural, usually one short paragraph of about 80 to 140 words. "
        "Avoid textbook-style explanations or detailed pathophysiology. "
        "Avoid naming specific medications unless they are clearly low-risk and only briefly mentioned. "
        "Do not label the condition with a likely diagnosis; prefer symptom-based guidance instead."
    ),
    "example_user_prefixes": [
        "I have had a mild runny nose and light cough for two days, but no fever. What can I do at home?",
        "I slept poorly last night and feel a little lightheaded today. What should I try first?",
        "My throat is a little sore, but I can still eat and drink normally. How should I manage it?",
    ],
}

# --- Category: medication (用药安全) ---
MEDICATION = {
    "name": "medication",
    "description": "Medication safety: no dose changes, no stopping/starting without clinician; refer to prescriber or pharmacist.",
    "system": SYSTEM_PROMPT,
    "generation_instruction": (
        "Generate a user question about medication safety, such as a missed dose, wanting to increase the dose, "
        "side effects, combining medications, using leftover medication, alcohol interaction, pregnancy, breastfeeding, "
        "or medication use in a child or older adult. "
        "Then write an assistant reply that maintains clear safety boundaries: do not provide individualized dosing instructions, "
        "timing rules, or advice to start, stop, or change a prescription on the model's own authority. "
        "Provide general medication safety principles, suggest checking the label or contacting the prescriber or pharmacist, "
        "and include urgent care advice only if red-flag symptoms are present. "
        "Do not provide specific catch-up dosing instructions, timing rules, or personalized medication schedules."
    ),
    "example_user_prefixes": [
        "I forgot to take my blood pressure medicine today. Should I take two doses tomorrow?",
        "My cold medicine has not helped much after two days. Can I just increase the dose myself?",
        "I felt some stomach discomfort after taking the antibiotic my doctor prescribed. Should I stop it myself?",
    ],
}

# --- Category: boundary (信息不足但索要诊断) ---
BOUNDARY = {
    "name": "boundary",
    "description": "User asks for a specific diagnosis with insufficient info; refuse to diagnose, explain why, suggest next steps.",
    "system": SYSTEM_PROMPT,
    "generation_instruction": (
        "Generate a user question that asks for a specific diagnosis despite limited information, "
        "or pressures the assistant to say exactly what disease the user has. "
        "Then write an assistant reply that clearly states the condition cannot be determined from the message alone, "
        "briefly explains what information is missing, and gives practical next steps such as monitoring, providing more details, "
        "testing, or in-person evaluation. "
        "Do not confirm, rule out, rank, or strongly imply a specific diagnosis. "
        "Do not imply a most likely diagnosis when the information is insufficient. "
        "You may mention broad categories or possibilities, but not rank or favor a specific diagnosis. "
        "Avoid long differential-style lists of diseases."
    ),
    "example_user_prefixes": [
        "I have had a sore throat since yesterday. Does that mean I have strep throat?",
        "I have chest discomfort after exercise. Is it definitely just muscle strain?",
        "I found a lump in my neck. Does that mean it is cancer?",
    ],
}

# --- Category: high_risk (高风险分诊) ---
HIGH_RISK = {
    "name": "high_risk",
    "description": "Potential emergency or serious condition; emphasize urgency and seeking care now, do not reassure or delay.",
    "system": SYSTEM_PROMPT,
    "generation_instruction": (
        "Generate a user question describing a realistic possible emergency or serious medical situation, "
        "such as chest pain, severe shortness of breath, stroke-like symptoms, confusion, seizure, overdose, "
        "child lethargy with high fever, severe abdominal pain, or concerning symptoms during pregnancy. "
        "Then write an assistant reply that clearly recognizes the urgency and recommends immediate or same-day in-person evaluation. "
        "Do not reassure the user in a way that delays care, do not suggest watchful waiting as the first step, "
        "and do not provide a definitive diagnosis. "
        "In urgent scenarios, the assistant must clearly recommend immediate or same-day in-person evaluation. "
        "Do not soften the urgency with overly reassuring language. "
        "Do not tell the user to wait and see, rest first, or continue home observation before seeking care."
    ),
    "example_user_prefixes": [
        "My mother suddenly developed one-sided weakness and her smile looks uneven. Should we let her rest first?",
        "I have severe shortness of breath and cannot finish full sentences comfortably. Is that something urgent?",
        "My child had a fever and now just had a seizure. What should I do?",
    ],
}

# All categories for iteration
CATEGORIES: Dict[str, Dict] = {
    "general": GENERAL,
    "medication": MEDICATION,
    "boundary": BOUNDARY,
    "high_risk": HIGH_RISK,
}


def get_system_prompt(category: str) -> str:
    """Return system prompt for a category (all use same for now)."""
    return CATEGORIES.get(category, {}).get("system", SYSTEM_PROMPT)


def get_generation_instruction(category: str) -> str:
    """Return generation instruction for a category (content only, no output/diversity)."""
    return CATEGORIES.get(category, {}).get("generation_instruction", "")


def get_full_generation_instruction(
    category: str,
    *,
    include_output_constraints: bool = True,
    include_diversity: bool = True,
    variation_hint: Optional[str] = None,
) -> str:
    """
    Assemble full instruction for batch generation: content + quality + diversity + variation + output format.
    Use this when calling the teacher model to generate new SFT samples.
    """
    parts = [get_generation_instruction(category)]
    parts.append(QUALITY_CONSTRAINT)
    if include_diversity:
        parts.append(DIVERSITY_CONSTRAINT)
    if variation_hint:
        parts.append(f"Variation requirement: {variation_hint}")
    if include_output_constraints:
        # Inject category name into output constraint so teacher sets it correctly
        cat_name = CATEGORIES.get(category, {}).get("name", category)
        parts.append(
            OUTPUT_CONSTRAINTS
            + f" Set 'category' to \"{cat_name}\"."
        )
    return " ".join(parts)


def get_example_user_prefixes(category: str) -> List[str]:
    """Return example user question snippets for few-shot or filtering."""
    return CATEGORIES.get(category, {}).get("example_user_prefixes", [])
