import os
import json
import re
import copy
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Literal, TYPE_CHECKING
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ---- OpenAI import: runtime only ----
try:
    from openai import OpenAI as OpenAIClient  # runtime name
except Exception:
    OpenAIClient = None  # type: ignore

if TYPE_CHECKING:
    from openai import OpenAI  # noqa: F401


# -------------------
# CONFIG
# -------------------
LIB_PATH = "Question_Rates_Merged_backbone_ready.xlsx"
OUT_PATH = "draft_survey_arrival_v10.json"

SUPPORTED_OPS = {"equals", "not_equals", "in", "not_in", "contains", "exists"}

DROP_LIBRARY_CONDITIONS_FOR_CONSTRUCTS = {
    "audience_identity",
    "journey_stage",
    "purpose_intent",
    "prior_knowledge",
    "trigger",
}

# -------------------
# L2 POLICY
# -------------------
L2_POLICY = os.getenv("L2_POLICY", "user").strip().lower()
# allowed: "off", "user", "auto"
# - off: do not plan or add L2
# - user: plan suggestions only; user adds via interactive command
# - auto: auto-add only when plan is strong (high confidence)
L2_AUTO_CONFIDENCE = float(os.getenv("L2_AUTO_CONFIDENCE", "0.87"))
L2_MAX_DISTINCT = int(os.getenv("L2_MAX_DISTINCT", "2"))


def _get_setting(name: str, default: str) -> str:
    try:
        import streamlit as st
        v = st.secrets.get(name)
        return str(v) if v is not None else os.getenv(name, default)
    except Exception:
        return os.getenv(name, default)


MODEL = _get_setting("OPENAI_MODEL", "gpt-4o-mini")
TEMP = float(_get_setting("OPENAI_TEMPERATURE", "0.2"))

AUTO_GENERATE_L2 = str(os.getenv("AUTO_GENERATE_L2", "0")).strip().lower() in {"1", "true", "yes", "y"}

LLM_SELECT_PER_CONSTRUCT = str(os.getenv("LLM_SELECT_PER_CONSTRUCT", "1")).strip().lower() in {"1", "true", "yes", "y"}
LLM_SELECT_ALWAYS = str(os.getenv("LLM_SELECT_ALWAYS", "0")).strip().lower() in {"1", "true", "yes", "y"}

LLM_REWRITE_SELECTED = str(os.getenv("LLM_REWRITE_SELECTED", "1")).strip().lower() in {"1", "true", "yes", "y"}
LLM_FALLBACK_ON_MISSING = str(os.getenv("LLM_FALLBACK_ON_MISSING", "1")).strip().lower() in {"1", "true", "yes", "y"}

MAX_QUESTIONS = int(str(os.getenv("MAX_QUESTIONS", "5")).strip() or "5")
CANDIDATE_K = int(str(os.getenv("CANDIDATE_K", "12")).strip() or "12")
SELECT_MARGIN = int(str(os.getenv("SELECT_MARGIN", "4")).strip() or "4")
STRICT_DEPHARMA = str(os.getenv("STRICT_DEPHARMA", "1")).strip().lower() in {"1", "true", "yes", "y"}

ARRIVAL_FLOW = [
    "audience_identity",
    "journey_stage",
    "hcp_profile",
    "hcp_profile_l2",
    "purpose_intent",
    "prior_knowledge",
    "trigger",
]

# Construct -> (slot, level)
SLOT_LEVEL_BY_CONSTRUCT = {
    "audience_identity": ("1_Audience", "L1"),
    "journey_stage": ("2_Context", "L1"),
    "hcp_profile": ("1b_HCP_Profile", "L1"),
    "purpose_intent": ("3_Goal", "L1"),
    "prior_knowledge": ("4_Past", "L1"),
    "trigger": ("4_Trigger", "L1"),
}

# Heuristic keywords (deterministic) to decide constructs
GOAL_SIGNALS = {
    "journey_stage": ["journey", "stage", "lifecycle", "where are", "funnel", "readiness", "consideration", "decision"],
    "prior_knowledge": ["familiar", "awareness", "knowledge", "understand", "clarity", "baseline"],
    "trigger": ["why now", "trigger", "campaign", "ad", "email", "why today", "time-sensitive"],
}

# Core exemplars for fallback generation when library has no candidates (cross-domain)
CORE_EXEMPLARS: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "audience_identity": {
        "generic": [
            {"q": "Which best describes you?", "type": "SingleSelection",
             "opts": ["Homeowner", "Contractor/Builder", "Architect/Designer", "DIY shopper", "Other"]},
            {"q": "What role best describes you today?", "type": "SingleSelection",
             "opts": ["Customer", "Professional", "Just researching", "Other"]},
        ],
    },
    "journey_stage": {
        "generic": [
            {"q": "Where are you in your decision process?", "type": "SingleSelection",
             "opts": ["Just browsing", "Comparing options", "Ready to get pricing", "Ready to talk to someone", "Other"]},
        ],
    },
    "purpose_intent": {
        "generic": [
            {"q": "What are you here to do today?", "type": "SingleSelection",
             "opts": ["Get pricing", "Explore options", "Check requirements/specs", "Plan installation", "Contact sales/support", "Other"]},
        ],
    },
    "prior_knowledge": {
        "generic": [
            {"q": "How familiar are you with this topic?", "type": "SingleSelection",
             "opts": ["Not at all", "A little", "Somewhat", "Very familiar", "Prefer not to say"]},
        ],
    },
    "trigger": {
        "generic": [
            {"q": "What prompted your visit today?", "type": "SingleSelection",
             "opts": ["A new project", "Replacing something existing", "Recommendation", "Search or ad", "Just exploring", "Other"]},
        ],
    },
}

MIN_QUESTIONS = int(os.getenv("MIN_QUESTIONS", "4"))


# -------------------
# DATA STRUCTURES
# -------------------
@dataclass
class AnswerOption:
    key: str
    label: str


@dataclass
class SurveyItem:
    id: str
    module_key: str
    construct: str
    slot: str
    phase: str
    level: str
    question_id: str
    question_type: str
    question_text: str
    answer_options: List[AnswerOption]
    display_condition_json: Optional[dict] = None
    display_condition: str = ""
    ai_actions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuilderContext:
    site_purpose: str
    survey_goal: str
    site_category: str


# -------------------
# OPENAI (Structured Output)
# -------------------
QuestionType = Literal[
    "SingleSelection",
    "MultiSelection",
    "SingleSelectionWithOther",
    "MultiSelectionWithOther",
    "OpenText",
]



class OptionSuggestions(BaseModel):
    candidates: List[str] = Field(min_length=5)
    notes: str = ""


class L2Draft(BaseModel):
    question_text: str = Field(min_length=5)
    question_type: QuestionType
    answer_options: List[str] = Field(default_factory=list)


class RewriteQuestion(BaseModel):
    question_text: str = Field(min_length=5)


class RewriteOptions(BaseModel):
    answer_options: List[str] = Field(min_length=2)


class L1Draft(BaseModel):
    question_text: str = Field(min_length=5)
    question_type: QuestionType
    answer_options: List[str] = Field(default_factory=list)


class SelectionDecision(BaseModel):
    chosen_question_id: str = Field(min_length=1)
    rewrite_needed: bool = Field(default=True)
    why: str = Field(min_length=3)
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)


class IntentDraft(BaseModel):
    question_text: str = Field(min_length=5)
    question_type: QuestionType
    answer_options: List[str] = Field(default_factory=list)


class AudienceFollowupPlan(BaseModel):
    enabled: bool = Field(default=False)
    trigger_audience_keys: List[str] = Field(default_factory=list)
    question_text: str = Field(default="")
    question_type: QuestionType = Field(default="SingleSelection")
    answer_options: List[str] = Field(default_factory=list)
    why: str = Field(default="")
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)


class L2QuestionDraft(BaseModel):
    trigger_answer_keys: List[str] = Field(min_length=1)
    question_text: str = Field(min_length=5)
    question_type: QuestionType
    answer_options: List[str] = Field(default_factory=list)
    why: str = Field(min_length=3)


class L2Plan(BaseModel):
    enabled: bool = False
    overall_why: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    drafts: List[L2QuestionDraft] = Field(default_factory=list)


def plan_audience_followup_openai(
    client: Any,
    *,
    ctx: BuilderContext,
    audience_question_text: str,
    audience_options: List[Dict[str, str]],  # [{"key":..., "label":...}]
) -> AudienceFollowupPlan:
    constraints = (
        "- This is a SECOND question shown right after the Audience question, only for some selections.\n"
        "- The follow-up MUST refine the visitor's ROLE/RELATIONSHIP (who they are / how they relate to the project),\n"
        "  NOT their intent/purpose (we already ask intent separately).\n"
        "- Enable only if it meaningfully helps the survey goal.\n"
        "- Keep it answerable in <=10 seconds.\n"
        "- Prefer closed-ended with 2–8 options.\n"
        "- Do NOT ask analytics/trackable topics.\n"
        "- Choose trigger_audience_keys ONLY from the provided keys.\n"
    )

    prompt = (
        "We are building an Arrival survey.\n"
        "We already asked an Audience (who are you) question.\n"
        "Decide if we should ask ONE conditional follow-up question that appears only for certain audience selections.\n"
        "The follow-up must refine ROLE/RELATIONSHIP (e.g., project involvement, professional capacity), not intent.\n\n"
        f"Site category: {ctx.site_category}\n"
        f"Site purpose: {ctx.site_purpose}\n"
        f"Survey goal: {ctx.survey_goal}\n\n"
        f"Audience question: {audience_question_text}\n"
        f"Audience options (keys must be used):\n{json.dumps(audience_options, ensure_ascii=False, indent=2)}\n\n"
        f"Constraints:\n{constraints}\n\n"
        "Return:\n"
        "- enabled (bool)\n"
        "- trigger_audience_keys (list of keys)\n"
        "- question_text, question_type, answer_options (if enabled)\n"
        "- why, confidence\n"
    )

    resp = client.responses.parse(
        model=MODEL,
        temperature=0.0,
        input=[
            {"role": "system", "content": "You plan a conditional role-refinement follow-up survey question based on audience selection."},
            {"role": "user", "content": prompt},
        ],
        text_format=AudienceFollowupPlan,
    )

    out = resp.output_parsed

    # Enforce triggers are subset of provided keys
    allowed = {o["key"] for o in (audience_options or []) if o.get("key")}
    out.trigger_audience_keys = [k for k in (out.trigger_audience_keys or []) if k in allowed]  # type: ignore

    if not out.enabled or not out.trigger_audience_keys:
        return AudienceFollowupPlan(
            enabled=False,
            trigger_audience_keys=[],
            why=getattr(out, "why", ""),
            confidence=float(getattr(out, "confidence", 0.7)),
        )

    qt, ao = _ensure_closed_ended(out.question_type, out.answer_options)
    out.question_type = qt  # type: ignore
    out.answer_options = dedupe_options(ao)  # type: ignore
    return out


def generate_l1_by_intent_openai(
    client: Any,
    *,
    ctx: BuilderContext,
    intent: str,
    slot: str,
    level: str,
    force_type: Optional[str] = None,
) -> IntentDraft:
    constraints = (
        "- Must be concise and answerable in <=10 seconds.\n"
        "- Avoid analytics/trackable topics.\n"
        "- If closed-ended: 2–8 options.\n"
        "- Include 'Other' only if helpful.\n"
    )
    type_line = f"Use question_type={force_type}." if force_type and force_type != "auto" else \
                "Choose the best question_type among: SingleSelection, MultiSelection, OpenText."
    prompt = (
        "Draft ONE on-site survey question from intent.\n"
        f"{type_line}\n"
        f"Slot: {slot}  Level: {level}\n\n"
        f"Site category: {ctx.site_category}\n"
        f"Site purpose: {ctx.site_purpose}\n"
        f"Survey goal: {ctx.survey_goal}\n\n"
        f"Intent: {intent}\n\n"
        f"Constraints:\n{constraints}\n"
        "Return only question_text, question_type, answer_options."
    )
    resp = client.responses.parse(
        model=MODEL,
        temperature=0.25,
        input=[
            {"role": "system", "content": "You draft concise on-site survey questions from intent."},
            {"role": "user", "content": prompt},
        ],
        text_format=IntentDraft,
    )
    out = resp.output_parsed
    qt, ao = _ensure_closed_ended(out.question_type, out.answer_options)
    out.question_type = qt  # type: ignore
    out.answer_options = dedupe_options(ao)  # type: ignore
    return out

def normalize_qtype(qtype: str) -> str:
    qt = (qtype or "").strip()
    mapping = {
        "single": "SingleSelection",
        "singleselection": "SingleSelection",
        "single_selection": "SingleSelection",
        "multi": "MultiSelection",
        "multiselection": "MultiSelection",
        "multi_selection": "MultiSelection",
        "opentext": "OpenText",
        "open_text": "OpenText",
        "open": "OpenText",
        "text": "OpenText",
    }
    key = re.sub(r"[^a-z_]", "", qt.lower())
    return mapping.get(key, qt or "SingleSelection")

def make_client() -> Any:
    """
    Works in:
    - Local dev: reads OPENAI_API_KEY from env or .env (optional)
    - Streamlit Cloud: reads OPENAI_API_KEY from st.secrets
    """
    # Try Streamlit secrets first (Streamlit Cloud)
    key = None
    try:
        import streamlit as st
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass

    # Fallback to env / .env (local)
    if not key:
        try:
            load_dotenv()
        except Exception:
            pass
        key = os.getenv("OPENAI_API_KEY")

    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it to Streamlit secrets or your environment."
        )
    if OpenAIClient is None:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    return OpenAIClient(api_key=key)


def _ensure_closed_ended(qtype: str, opts: List[str]) -> Tuple[str, List[str]]:
    qt = (qtype or "").strip()
    if qt == "OpenText":
        return "OpenText", []
    cleaned = [o.strip() for o in (opts or []) if str(o).strip()]
    cleaned = cleaned[:8]
    if len(cleaned) < 2:
        cleaned = ["Option 1", "Option 2", "Other"]
    return qt or "SingleSelection", cleaned


def generate_l2_followup_openai(
    client: Any,
    *,
    parent_question_text: str,
    parent_answer_label: str,
    parent_signal: str,
    site_purpose: str,
    survey_goal: str,
    site_category: str,
) -> L2Draft:
    constraints = (
        "- Must be concise and answerable in <=10 seconds.\n"
        "- Must be closed-ended unless OpenText is clearly better.\n"
        "- If closed-ended: 2–8 options.\n"
        "- Avoid analytics/trackable topics (device/referrer/browser/time on site/pages/clicks).\n"
    )
    prompt = (
        "Create ONE follow-up (L2) on-site survey question.\n"
        "This follow-up is shown ONLY if the user selected the specified parent answer.\n"
        "Return only the L2 question and its answer options.\n\n"
        f"Signal: {parent_signal}\n"
        f"Site category: {site_category}\n"
        f"Site purpose: {site_purpose}\n"
        f"Survey goal: {survey_goal}\n\n"
        f"Parent question: {parent_question_text}\n"
        f"Selected parent answer: {parent_answer_label}\n\n"
        f"Constraints:\n{constraints}"
    )

    resp = client.responses.parse(
        model=MODEL,
        temperature=TEMP,
        input=[
            {"role": "system", "content": "You write concise on-site survey follow-up questions."},
            {"role": "user", "content": prompt},
        ],
        text_format=L2Draft,
    )
    out = resp.output_parsed
    qt, ao = _ensure_closed_ended(out.question_type, out.answer_options)
    out.question_type = qt  # type: ignore
    out.answer_options = ao  # type: ignore
    return out


def plan_l2_followups_openai(
    client: Any,
    *,
    ctx: BuilderContext,
    parent_construct: str,
    parent_question_text: str,
    parent_question_type: str,
    parent_answer_options: List[Dict[str, str]],  # [{"key","label"}]
    max_l2_questions: int = 2,
) -> L2Plan:
    """
    Plan (not add) L2 followups:
    - decides enabled vs disabled
    - returns 0..max_l2_questions drafts
    - each draft is gated on a SUBSET of answer keys
    """
    constraints = (
        "- FIRST decide if an L2 follow-up is warranted. If not, set enabled=false.\n"
        "- L2 must have a clear purpose tied to the survey goal (disambiguation, routing, qualification).\n"
        "- L2 should be shown only for a SUBSET of answers (trigger_answer_keys).\n"
        "- Do NOT add L2 for every answer by default.\n"
        "- Keep L2 answerable in <=10 seconds.\n"
        "- Avoid analytics/trackable topics.\n"
        "- If parent is OpenText: generally avoid L2 (enabled=false) unless there's a strong reason.\n"
        f"- Maximum distinct L2 questions: {max_l2_questions}.\n"
        "- trigger_answer_keys MUST be chosen from provided keys.\n"
        "- Prefer closed-ended with 2–6 options; include 'Other' only if it truly helps.\n"
        "- The L2 must still align with the parent construct (do not drift into a different construct).\n"
    )

    prompt = (
        "You are planning optional L2 follow-ups for an on-site Arrival survey.\n"
        "An L2 follow-up is a conditional question shown ONLY after specific parent answers.\n\n"
        f"Site category: {ctx.site_category}\n"
        f"Site purpose: {ctx.site_purpose}\n"
        f"Survey goal: {ctx.survey_goal}\n\n"
        f"Parent construct: {parent_construct}\n"
        f"Parent question: {parent_question_text}\n"
        f"Parent question type: {parent_question_type}\n"
        f"Parent answer options (keys must be used):\n{json.dumps(parent_answer_options, ensure_ascii=False, indent=2)}\n\n"
        f"Constraints:\n{constraints}\n\n"
        "Return an L2Plan:\n"
        "- enabled\n"
        "- overall_why\n"
        "- confidence\n"
        "- drafts: list (0..max) of L2QuestionDraft with trigger_answer_keys, question_text, question_type, answer_options, why\n"
    )

    resp = client.responses.parse(
        model=MODEL,
        temperature=0.0,
        input=[
            {"role": "system", "content": "You plan reasoned, subset-triggered L2 survey follow-ups."},
            {"role": "user", "content": prompt},
        ],
        text_format=L2Plan,
    )

    plan = resp.output_parsed

    # Enforce triggers are subset of provided keys
    allowed = {o["key"] for o in (parent_answer_options or []) if o.get("key")}
    cleaned: List[L2QuestionDraft] = []
    for d in (plan.drafts or []):
        keys = [k for k in (d.trigger_answer_keys or []) if k in allowed]
        if not keys:
            continue
        qt, ao = _ensure_closed_ended(d.question_type, d.answer_options)
        ao = dedupe_options(ao)[:8]
        cleaned.append(
            L2QuestionDraft(
                trigger_answer_keys=keys,
                question_text=(d.question_text or "").strip(),
                question_type=qt,  # type: ignore
                answer_options=ao,  # type: ignore
                why=(d.why or "").strip(),
            )
        )

    plan.drafts = cleaned  # type: ignore
    if not plan.drafts:
        plan.enabled = False  # type: ignore
    return plan


def is_multi_select(qtype: str) -> bool:
    qt = (qtype or "").strip()
    return qt in {"MultiSelection", "MultiSelectionWithOther"}


def build_l2_condition(parent_var: str, parent_qtype: str, trigger_keys: List[str]) -> dict:
    """
    For SingleSelection: use op=in
    For MultiSelection: use any/contains (safer because stored value is typically list)
    """
    trigger_keys = [k for k in trigger_keys if k]
    if not trigger_keys:
        return {"all": []}

    if is_multi_select(parent_qtype):
        # Show L2 if parent selections contain ANY of the trigger keys
        return {"any": [{"var": parent_var, "op": "contains", "value": k} for k in trigger_keys]}

    # SingleSelection / SingleSelectionWithOther
    if len(trigger_keys) == 1:
        return {"all": [{"var": parent_var, "op": "equals", "value": trigger_keys[0]}]}
    return {"all": [{"var": parent_var, "op": "in", "value": trigger_keys}]}


def suggest_answer_options_openai(
    client: Any,
    *,
    site_purpose: str,
    survey_goal: str,
    site_category: str,
    question_text: str,
    question_type: str,
    n_candidates: int = 12,
) -> List[str]:
    prompt = (
        "Generate a candidate list of answer options for this on-site survey question.\n"
        f"- Return {n_candidates} to {n_candidates+4} options.\n"
        "- Options should be mutually exclusive where possible.\n"
        "- Avoid analytics/trackable topics.\n"
        "- Keep wording short.\n"
        "- Include 'Other' only if helpful.\n\n"
        f"Site category: {site_category}\n"
        f"Site purpose: {site_purpose}\n"
        f"Survey goal: {survey_goal}\n\n"
        f"Question: {question_text}\n"
        f"Question type: {question_type}\n"
        "Return only: candidates, notes."
    )
    resp = client.responses.parse(
        model=MODEL,
        temperature=0.3,
        input=[
            {"role": "system", "content": "You propose survey answer options."},
            {"role": "user", "content": prompt},
        ],
        text_format=OptionSuggestions,
    )
    cands = [c.strip() for c in resp.output_parsed.candidates if c and c.strip()]
    return dedupe_options(cands)[:16]


def generate_l1_from_exemplars_openai(
    client: Any,
    *,
    construct: str,
    site_purpose: str,
    survey_goal: str,
    site_category: str,
    exemplars: List[Dict[str, Any]],
) -> L1Draft:
    constraints = (
        "- Must be concise and answerable in <=10 seconds.\n"
        "- Must be closed-ended unless OpenText is clearly better.\n"
        "- If closed-ended: 2–8 options.\n"
        "- Avoid analytics/trackable topics (device/referrer/browser/time on site/pages/clicks).\n"
        "- Keep the same construct intent as the examples (do not invent a new construct).\n"
    )
    prompt = (
        f"Adapt ONE on-site survey question for the construct: {construct}\n"
        "Use the examples ONLY as patterns. Do NOT copy domain-specific wording if it doesn't fit.\n\n"
        f"Site category: {site_category}\n"
        f"Site purpose: {site_purpose}\n"
        f"Survey goal: {survey_goal}\n\n"
        f"Examples:\n{json.dumps(exemplars, ensure_ascii=False, indent=2)}\n\n"
        f"Constraints:\n{constraints}\n"
        "Return only: question_text, question_type, answer_options."
    )
    resp = client.responses.parse(
        model=MODEL,
        temperature=0.25,
        input=[
            {"role": "system", "content": "You adapt survey questions to new site contexts using examples."},
            {"role": "user", "content": prompt},
        ],
        text_format=L1Draft,
    )
    out = resp.output_parsed
    qt, ao = _ensure_closed_ended(out.question_type, out.answer_options)
    out.question_type = qt  # type: ignore
    out.answer_options = ao  # type: ignore
    return out


def rewrite_question_text_openai(
    client: Any,
    *,
    site_purpose: str,
    survey_goal: str,
    site_category: str,
    original_question_text: str,
    instruction: str,
) -> str:
    prompt = (
        "Rewrite the survey question text according to the instruction.\n"
        "Keep it concise and neutral. Do not add analytics tracking topics.\n\n"
        f"Site category: {site_category}\n"
        f"Site purpose: {site_purpose}\n"
        f"Survey goal: {survey_goal}\n\n"
        f"Original question: {original_question_text}\n"
        f"Instruction: {instruction}\n"
        "Return only the rewritten question_text."
    )
    resp = client.responses.parse(
        model=MODEL,
        temperature=0.3,
        input=[
            {"role": "system", "content": "You rewrite survey questions."},
            {"role": "user", "content": prompt},
        ],
        text_format=RewriteQuestion,
    )
    return resp.output_parsed.question_text.strip()


def rewrite_answer_options_openai(
    client: Any,
    *,
    site_purpose: str,
    survey_goal: str,
    site_category: str,
    question_text: str,
    original_options: List[str],
    instruction: str,
    keep_other_if_present: bool = True,
) -> List[str]:
    other_present = any(o.strip().lower() == "other" for o in original_options)

    other_rule_line = (
        'Keep an "Other" option if one exists.'
        if (keep_other_if_present and other_present)
        else "Include Other only if needed."
    )

    prompt = (
        "Rewrite the answer options according to the instruction.\n"
        "Rules:\n"
        "- Return 2 to 8 options.\n"
        "- Make options mutually exclusive and clear.\n"
        "- Avoid analytics/trackable topics.\n"
        f"- {other_rule_line}\n\n"
        f"Site category: {site_category}\n"
        f"Site purpose: {site_purpose}\n"
        f"Survey goal: {survey_goal}\n\n"
        f"Question: {question_text}\n"
        f"Original options: {original_options}\n"
        f"Instruction: {instruction}\n"
        "Return only the revised answer_options list."
    )

    resp = client.responses.parse(
        model=MODEL,
        temperature=0.3,
        input=[
            {"role": "system", "content": "You rewrite survey answer options."},
            {"role": "user", "content": prompt},
        ],
        text_format=RewriteOptions,
    )
    opts = [str(x).strip() for x in (resp.output_parsed.answer_options or []) if str(x).strip()]
    opts = opts[:8]

    if keep_other_if_present and other_present and not any(o.lower() == "other" for o in opts):
        if len(opts) >= 8:
            opts[-1] = "Other"
        else:
            opts.append("Other")

    if len(opts) < 2:
        opts = ["Option 1", "Option 2", "Other"]

    return opts


def choose_best_candidate_openai(
    client: Any,
    *,
    ctx: BuilderContext,
    construct: str,
    candidates: List[Dict[str, Any]],
) -> SelectionDecision:
    """
    Bounded selection: MUST choose one of the provided question_ids.
    """
    rubric = [
        "Best matches the construct intent",
        "Fits site_category + site_purpose",
        "Fits survey_goal",
        "Neutral, concise, <=10 seconds",
        "Avoids analytics/trackable topics",
        "Options are usable and mutually exclusive (when applicable)",
    ]
    prompt = (
        "You are selecting ONE survey question from a provided shortlist.\n"
        "Hard rules:\n"
        "- You MUST choose exactly one `question_id` from the candidates.\n"
        "- Do NOT invent new questions or rewrite content here.\n"
        "- Prefer candidates that can be easily adapted with small wording changes if needed.\n\n"
        f"Construct: {construct}\n"
        f"Site category: {ctx.site_category}\n"
        f"Site purpose: {ctx.site_purpose}\n"
        f"Survey goal: {ctx.survey_goal}\n\n"
        f"Rubric:\n- " + "\n- ".join(rubric) + "\n\n"
        f"Candidates:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
        "Return only: chosen_question_id, rewrite_needed, why, confidence."
    )

    resp = client.responses.parse(
        model=MODEL,
        temperature=0.0,  # selection should be stable
        input=[
            {"role": "system", "content": "You select the best candidate from a shortlist using a rubric."},
            {"role": "user", "content": prompt},
        ],
        text_format=SelectionDecision,
    )
    dec = resp.output_parsed

    # Safety: ensure chosen id is in candidates
    allowed = {str(c.get("question_id", "")).strip() for c in candidates}
    if dec.chosen_question_id.strip() not in allowed:
        # fallback to first candidate
        return SelectionDecision(
            chosen_question_id=next(iter(allowed)) if allowed else candidates[0].get("question_id", "unknown"),
            rewrite_needed=True,
            why="Model returned an invalid id; fell back to first candidate.",
            confidence=0.2,
        )
    return dec


# -------------------
# UTILITIES
# -------------------
def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "opt"


def _norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def dedupe_options(options: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for o in (options or []):
        t = (o or "").strip()
        if not t:
            continue
        n = _norm_label(t)
        if n in seen:
            continue
        seen.add(n)
        out.append(t)
    return out


def build_option_dicts(slot: str, labels: List[str]) -> List[Dict[str, str]]:
    labels = dedupe_options(labels)
    used = set()
    out: List[Dict[str, str]] = []
    for lab in labels:
        base = canonical_option_key(slot, lab)
        k = base
        i = 2
        while k in used:
            k = f"{base}_{i}"
            i += 1
        used.add(k)
        out.append({"key": k, "label": lab})
    return out


def _try_json_load(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def canonical_option_key(slot: str, label: str) -> str:
    lab = (label or "").strip().lower()
    if slot == "1_Audience":
        if "health care professional" in lab or "healthcare professional" in lab or lab == "hcp":
            return "hcp"
        if lab == "patient":
            return "patient"
        if "caregiver" in lab:
            return "caregiver"
        if "relative" in lab or "friend" in lab:
            return "friend_family"
        if "office" in lab and ("staff" in lab or "manager" in lab):
            return "office_staff"
        if lab == "other":
            return "other"
    return slugify(label)


def parse_answer_options(cell: Any, slot_for_keying: Optional[str] = None) -> List[AnswerOption]:
    if cell is None:
        return []
    if isinstance(cell, list):
        arr = cell
    else:
        s = str(cell).strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            parsed = _try_json_load(s)
            if isinstance(parsed, list):
                arr = parsed
            else:
                arr = [p.strip() for p in re.split(r";|\||\n|,", s) if p.strip()]
        else:
            arr = [p.strip() for p in re.split(r";|\||\n|,", s) if p.strip()]

    out: List[AnswerOption] = []
    used = set()
    for item in arr[:30]:
        if isinstance(item, dict):
            label = str(item.get("label") or item.get("value") or item.get("text") or "").strip()
            key = str(item.get("key") or "").strip()
            if not key:
                key = canonical_option_key(slot_for_keying or "", label) if slot_for_keying else slugify(label)
        else:
            label = str(item).strip()
            key = canonical_option_key(slot_for_keying or "", label) if slot_for_keying else slugify(label)

        if not label:
            continue

        base = key
        i = 2
        while key in used:
            key = f"{base}_{i}"
            i += 1
        used.add(key)
        out.append(AnswerOption(key=key, label=label))
    return out


def parse_condition_json(cell: Any) -> Optional[dict]:
    if cell is None:
        return None
    if isinstance(cell, dict):
        return cell
    s = str(cell).strip()
    if not s:
        return None
    parsed = _try_json_load(s)
    return parsed if isinstance(parsed, dict) else None


def parse_condition_string_fallback(cond: str) -> Optional[dict]:
    if not cond:
        return None
    s = cond.strip()
    s = re.sub(r"^\s*if\s+", "", s, flags=re.I).strip()

    m = re.match(r'^(?P<var>\w+)\s+contains\s+"?(?P<val>[^"]+)"?$', s, flags=re.I)
    if m:
        return {"all": [{"var": m.group("var"), "op": "contains", "value": m.group("val").strip()}]}

    m = re.match(r'^(?P<var>\w+)\s*(=|==|equals)\s*"?([^"]+)"?$', s, flags=re.I)
    if m:
        var = m.group("var")
        val = s.split(m.group(2), 1)[1].strip().strip('"').strip("'")
        return {"all": [{"var": var, "op": "equals", "value": val}]}

    m = re.match(r'^(?P<var>\w+)\s*(!=|not_equals)\s*"?([^"]+)"?$', s, flags=re.I)
    if m:
        var = m.group("var")
        val = s.split(m.group(2), 1)[1].strip().strip('"').strip("'")
        return {"all": [{"var": var, "op": "not_equals", "value": val}]}

    m = re.match(r'^(?P<var>\w+)\s+in\s+(?P<list>\[.*\])$', s, flags=re.I)
    if m:
        var = m.group("var")
        arr = _try_json_load(m.group("list"))
        if isinstance(arr, list):
            return {"all": [{"var": var, "op": "in", "value": arr}]}
    return None


def var_name_for_slot(slot: str, level: str) -> str:
    lvl = (level or "").strip().upper()
    if slot == "1_Audience":
        base = "Audience"
    elif slot in {"1b_HCP_Profile", "2b_HCP_Profile"}:
        base = "HCP_Profile"
    elif slot == "2_Context":
        base = "Context"
    elif slot == "3_Goal":
        base = "Purpose"
    elif slot == "4_Past":
        base = "Past_Knowledge"
    elif slot == "4_Trigger":
        base = "Trigger"
    else:
        base = slugify(slot).title()
    return f"{base}_{lvl}"


def normalize_condition_vars(cond: Optional[dict]) -> Optional[dict]:
    if not cond or not isinstance(cond, dict):
        return cond

    mapping = {
        "1b_HCP_Profile": var_name_for_slot("1b_HCP_Profile", "L1"),
        "2b_HCP_Profile": var_name_for_slot("1b_HCP_Profile", "L1"),
        "1_Audience": var_name_for_slot("1_Audience", "L1"),
        "3_Goal": var_name_for_slot("3_Goal", "L1"),
        "4_Past": var_name_for_slot("4_Past", "L1"),
        "4_Trigger": var_name_for_slot("4_Trigger", "L1"),
        "2_Context": var_name_for_slot("2_Context", "L1"),
    }

    def _rewrite_pred(p: dict) -> dict:
        p2 = dict(p)
        v = str(p2.get("var") or "").strip()
        if v in mapping:
            p2["var"] = mapping[v]
        return p2

    out = dict(cond)
    if "all" in out and isinstance(out["all"], list):
        out["all"] = [_rewrite_pred(p) if isinstance(p, dict) else p for p in out["all"]]
    if "any" in out and isinstance(out["any"], list):
        out["any"] = [_rewrite_pred(p) if isinstance(p, dict) else p for p in out["any"]]
    if "var" in out and isinstance(out.get("var"), str):
        out = _rewrite_pred(out)
    return out


def extract_simple_gate(cond: Optional[dict]) -> Optional[Tuple[str, str, str]]:
    if not cond or not isinstance(cond, dict):
        return None

    preds = None
    if "all" in cond and isinstance(cond["all"], list) and len(cond["all"]) == 1:
        preds = cond["all"]
    elif "any" in cond and isinstance(cond["any"], list) and len(cond["any"]) == 1:
        preds = cond["any"]
    elif "var" in cond and "op" in cond:
        preds = [cond]

    if not preds:
        return None

    p = preds[0]
    if not isinstance(p, dict):
        return None
    var = str(p.get("var") or "").strip()
    op = str(p.get("op") or "").strip()
    val = p.get("value", "")
    if not var or op not in SUPPORTED_OPS:
        return None
    if isinstance(val, (list, dict)):
        return None
    return (var, op, slugify(str(val).strip()))


def _split_tags(s: str) -> List[str]:
    if not s:
        return []
    parts = [p.strip().lower() for p in re.split(r"[;,|]", str(s)) if p.strip()]
    return [p for p in parts if p]


# -------------------
# LIBRARY LOAD
# -------------------
def load_library(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=str).fillna("")

    for c in [
        "question_id", "question_text", "answer_options", "question_type",
        "purpose", "survey_phase", "slot", "level", "is_backbone",
        "display_condition", "display_condition_json", "category",
        # optional columns
        "construct", "category_tags", "time_to_answer", "allow_rewrite",
    ]:
        if c not in df.columns:
            df[c] = ""

    df["survey_phase"] = df["survey_phase"].astype(str).str.strip().str.title()
    df["slot"] = df["slot"].astype(str).str.strip()
    df["level"] = df["level"].astype(str).str.strip().str.upper()
    df["is_backbone"] = df["is_backbone"].astype(str).str.strip().str.upper()
    df["purpose"] = df["purpose"].astype(str).str.strip().str.lower()
    df["category"] = df["category"].astype(str).str.strip().str.title()
    df["construct"] = df["construct"].astype(str).str.strip()

    df["_answers"] = df.apply(lambda r: parse_answer_options(r.get("answer_options", ""), r.get("slot", "")), axis=1)
    df["_cond_json"] = df["display_condition_json"].apply(parse_condition_json)

    def _fallback(row):
        cj = row["_cond_json"]
        if cj:
            return cj
        return parse_condition_string_fallback(row.get("display_condition", ""))

    df["_cond_json_final_raw"] = df.apply(_fallback, axis=1)
    df["_cond_json_final"] = df["_cond_json_final_raw"].apply(normalize_condition_vars)
    return df


# -------------------
# HEURISTIC SCORING + CANDIDATE SHORTLIST
# -------------------
def score_row(row: pd.Series, ctx: BuilderContext, construct: str) -> int:
    ans_len = len(row.get("_answers") or [])
    score = 0

    # backbone prior (SOFT, not a filter)
    score += 3 if str(row.get("is_backbone", "")).upper() == "TRUE" else 0

    # completeness
    score += 1 if str(row.get("question_text", "")).strip() else 0
    score += 2 if ans_len >= 4 else (1 if ans_len >= 2 else 0)

    # category match
    cat = str(row.get("category", "")).strip().lower()
    ctx_cat = (ctx.site_category or "").strip().lower()
    if cat and ctx_cat:
        score += 3 if cat == ctx_cat else 0

    # category tags match
    tags = _split_tags(row.get("category_tags", ""))
    if tags and ctx_cat:
        score += 2 if ctx_cat.lower() in tags else 0

    # construct match if column is used
    r_construct = str(row.get("construct", "")).strip().lower()
    if r_construct:
        score += 3 if r_construct == construct.lower() else -2

    # goal signal boosts
    goal = (ctx.survey_goal or "").lower()
    if construct == "journey_stage" and any(x in goal for x in GOAL_SIGNALS["journey_stage"]):
        score += 2
    if construct == "prior_knowledge" and any(x in goal for x in GOAL_SIGNALS["prior_knowledge"]):
        score += 2
    if construct == "trigger" and any(x in goal for x in GOAL_SIGNALS["trigger"]):
        score += 2

    # allow_rewrite boost
    allow_rewrite = str(row.get("allow_rewrite", "")).strip().upper()
    if allow_rewrite == "TRUE":
        score += 1

    return score


def get_candidates_for_construct(
    df: pd.DataFrame,
    ctx: BuilderContext,
    *,
    construct: str,
    phase: str,
    level: str,
    k: int,
) -> pd.DataFrame:
    slot, _lvl = SLOT_LEVEL_BY_CONSTRUCT.get(construct, ("", ""))
    if not slot:
        return pd.DataFrame()

    cand = df[
        (df["slot"] == slot)
        & (df["level"] == level.upper())
        & (df["survey_phase"].isin([phase, "Both", ""]))
    ].copy()

    # If construct column exists and has content, filter to that construct
    if "construct" in cand.columns and cand["construct"].astype(str).str.strip().ne("").any():
        cand2 = cand[cand["construct"].astype(str).str.strip().str.lower() == construct.lower()].copy()
        if not cand2.empty:
            cand = cand2

    if cand.empty:
        return cand

    cand["_score"] = cand.apply(lambda r: score_row(r, ctx, construct), axis=1)
    cand = cand.sort_values("_score", ascending=False)
    return cand.head(max(1, k)).copy()


def should_call_llm_selection(
    cand: pd.DataFrame,
    client: Optional[Any],
) -> bool:
    if client is None:
        return False
    if not LLM_SELECT_PER_CONSTRUCT:
        return False
    if LLM_SELECT_ALWAYS:
        return True
    if cand is None or cand.empty:
        return False
    if len(cand) < 2:
        return False
    try:
        best = int(cand.iloc[0]["_score"])
        second = int(cand.iloc[1]["_score"])
        return (best - second) < SELECT_MARGIN
    except Exception:
        return True


# -------------------
# CONSTRUCT POLICY (DETERMINISTIC FLOW)
# -------------------
LLM_PLAN_CONSTRUCTS = str(os.getenv("LLM_PLAN_CONSTRUCTS", "1")).strip().lower() in {"1", "true", "yes", "y"}

NEGATION_PATTERNS = {
    "journey_stage": [
        "don't ask journey", "do not ask journey", "no journey stage", "skip journey", "avoid journey stage",
        "don't include journey", "exclude journey", "no need to ask journey", "measure satisfaction only"
    ],
    "prior_knowledge": [
        "don't ask familiarity", "skip prior knowledge", "avoid familiarity",
        "don't ask prior knowledge", "exclude prior knowledge"
    ],
    "trigger": [
        "don't ask trigger", "skip trigger", "avoid trigger",
        "don't ask what prompted", "exclude trigger"
    ],
}


def _has_any(text: str, *phrases: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in phrases)


def is_negated(goal: str, construct: str) -> bool:
    g = (goal or "").lower()
    return any(p in g for p in NEGATION_PATTERNS.get(construct, []))


class ConstructPlanDecision(BaseModel):
    include_journey_stage: bool = False
    include_prior_knowledge: bool = False
    include_trigger: bool = False
    why: str = ""


def plan_constructs_openai(client: Any, ctx: BuilderContext) -> ConstructPlanDecision:
    prompt = (
        "Decide which optional constructs to include in an Arrival survey plan.\n"
        "Optional constructs: journey_stage, prior_knowledge, trigger.\n"
        "Honor explicit negation in the survey goal (e.g., 'don't ask journey stage').\n"
        "Return only booleans + a short why.\n\n"
        f"Site category: {ctx.site_category}\n"
        f"Site purpose: {ctx.site_purpose}\n"
        f"Survey goal: {ctx.survey_goal}\n"
    )
    resp = client.responses.parse(
        model=MODEL,
        temperature=0.0,
        input=[
            {"role": "system", "content": "You choose which optional constructs to include."},
            {"role": "user", "content": prompt},
        ],
        text_format=ConstructPlanDecision,
    )
    return resp.output_parsed


def plan_constructs(ctx: BuilderContext, client: Optional[Any] = None) -> Dict[str, Any]:
    goal = (ctx.survey_goal or "").lower()
    cat = (ctx.site_category or "").lower()

    plan: List[Dict[str, Any]] = []
    reasons: Dict[str, str] = {}

    # Required
    for c in ["audience_identity", "purpose_intent"]:
        plan.append({"construct": c, "required": True})
        reasons[c] = "required_construct"

    # ---- OPTIONALS: start with heuristics ----
    want_journey = _has_any(goal, *GOAL_SIGNALS["journey_stage"]) or cat in {"education", "ecommerce", "saas", "content", "university"}
    want_knowledge = _has_any(goal, *GOAL_SIGNALS["prior_knowledge"])
    want_trigger = _has_any(goal, *GOAL_SIGNALS["trigger"])

    # ---- HARD OVERRIDES: explicit negation wins ----
    if is_negated(goal, "journey_stage"):
        want_journey = False
        reasons["journey_stage"] = "explicitly_skipped_by_user"
    if is_negated(goal, "prior_knowledge"):
        want_knowledge = False
        reasons["prior_knowledge"] = "explicitly_skipped_by_user"
    if is_negated(goal, "trigger"):
        want_trigger = False
        reasons["trigger"] = "explicitly_skipped_by_user"

    # ---- OPTIONAL: let LLM decide *only for optionals* (never removes required) ----
    if client is not None and LLM_PLAN_CONSTRUCTS:
        dec = plan_constructs_openai(client, ctx)

        # keep negation as a hard rule
        if not is_negated(goal, "journey_stage"):
            want_journey = bool(dec.include_journey_stage)
            reasons["journey_stage"] = reasons.get("journey_stage") or f"llm_plan: {dec.why}".strip()
        if not is_negated(goal, "prior_knowledge"):
            want_knowledge = bool(dec.include_prior_knowledge)
            reasons["prior_knowledge"] = reasons.get("prior_knowledge") or f"llm_plan: {dec.why}".strip()
        if not is_negated(goal, "trigger"):
            want_trigger = bool(dec.include_trigger)
            reasons["trigger"] = reasons.get("trigger") or f"llm_plan: {dec.why}".strip()

    # Apply optionals
    if want_journey:
        plan.append({"construct": "journey_stage", "required": False})
        reasons.setdefault("journey_stage", "goal_or_category_signal")
    if want_knowledge:
        plan.append({"construct": "prior_knowledge", "required": False})
        reasons.setdefault("prior_knowledge", "goal_signal")
    if want_trigger:
        plan.append({"construct": "trigger", "required": False})
        reasons.setdefault("trigger", "goal_signal")

    # Order + cap
    order = ["audience_identity", "journey_stage", "purpose_intent", "prior_knowledge", "trigger"]
    ordered = [x for x in order if any(p["construct"] == x for p in plan)]
    new_plan = [{"construct": c, "required": (c in {"audience_identity", "purpose_intent"})} for c in ordered]

    # Backfill to reach MIN_QUESTIONS (unless explicitly negated)
    if len(new_plan) < MIN_QUESTIONS:
        for c in ["journey_stage", "trigger", "prior_knowledge"]:
            if len(new_plan) >= MIN_QUESTIONS:
                break
            if any(p["construct"] == c for p in new_plan):
                continue
            if is_negated(goal, c):
                continue
            new_plan.append({"construct": c, "required": False})
            reasons.setdefault(c, "backfilled_to_meet_min_questions")

    while len(new_plan) > MAX_QUESTIONS:
        idx = next((i for i in range(len(new_plan) - 1, -1, -1) if not new_plan[i]["required"]), None)
        if idx is None:
            break
        removed = new_plan.pop(idx)
        reasons[removed["construct"]] = f"dropped_by_max_questions_{MAX_QUESTIONS}"

    return {"constructs": new_plan, "reasons": reasons, "max_questions": MAX_QUESTIONS}


# -------------------
# L2 RESOLUTION
# -------------------
def l2_candidates_by_answer(l2_rows: pd.DataFrame, parent_var: str) -> Dict[str, List[pd.Series]]:
    bucket: Dict[str, List[pd.Series]] = {}
    for _, r in l2_rows.iterrows():
        gate = extract_simple_gate(r.get("_cond_json_final"))
        if not gate:
            continue
        var, op, val = gate
        if var != parent_var:
            continue
        if op not in {"equals", "contains"}:
            continue
        bucket.setdefault(val, []).append(r)
    return bucket


def resolve_l2_mode(*, bucket: Dict[str, List[pd.Series]], answer_key: str) -> Dict[str, Any]:
    ak = slugify(answer_key or "")

    if ak in bucket:
        rows = bucket[ak]
        ids = [str(x.get("question_id", "")).strip() for x in rows if str(x.get("question_id", "")).strip()]
        if len(rows) == 1:
            return {"mode": "library", "reason": "single_matching_library_template", "chosen_question_id": ids[0] if ids else "", "candidate_question_ids": ids}
        return {"mode": "llm_needed", "reason": "multiple_matching_library_templates", "chosen_question_id": "", "candidate_question_ids": ids}

    for k, rows in bucket.items():
        if k and (k in ak or ak in k):
            ids = [str(x.get("question_id", "")).strip() for x in rows if str(x.get("question_id", "")).strip()]
            if len(rows) == 1:
                return {"mode": "library", "reason": "single_matching_library_template_loose", "chosen_question_id": ids[0] if ids else "", "candidate_question_ids": ids}
            return {"mode": "llm_needed", "reason": "multiple_matching_library_templates_loose", "chosen_question_id": "", "candidate_question_ids": ids}

    return {"mode": "llm_needed", "reason": "no_matching_library_template", "chosen_question_id": "", "candidate_question_ids": []}


# -------------------
# ITEM BUILDING
# -------------------
def row_to_item(
    row: Optional[pd.Series],
    qnum: int,
    *,
    construct: str,
    slot: str,
    level: str,
) -> SurveyItem:
    if row is None:
        return SurveyItem(
            id=str(qnum),
            module_key=construct,
            construct=construct,
            slot=slot,
            phase="Arrival",
            level=level,
            question_id=f"missing::{slot}::{level}",
            question_type="SingleSelection",
            question_text=f"[MISSING TEMPLATE] {construct} ({slot} {level})",
            answer_options=[],
            display_condition_json=None,
            display_condition="",
            ai_actions={"missing_template": True},
        )

    return SurveyItem(
        id=str(qnum),
        module_key=construct,
        construct=construct,
        slot=slot,
        phase="Arrival",
        level=level,
        question_id=str(row.get("question_id", "")).strip(),
        question_type=str(row.get("question_type", "") or "SingleSelection").strip() or "SingleSelection",
        question_text=str(row.get("question_text", "")).strip(),
        answer_options=row.get("_answers", []) or [],
        display_condition_json=(
            None if construct in DROP_LIBRARY_CONDITIONS_FOR_CONSTRUCTS else row.get("_cond_json_final", None)
        ),
        display_condition=(
            "" if construct in DROP_LIBRARY_CONDITIONS_FOR_CONSTRUCTS else str(row.get("display_condition", "")).strip()
        ),
        ai_actions={},
    )


def _maybe_rewrite_selected_item(it: SurveyItem, ctx: BuilderContext, client: Optional[Any]) -> None:
    """
    Rewrite policy (safer):
    - Never rewrite audience_identity question text (prevents drift into "purpose").
    - For purpose_intent, allow rewrite but guard against turning it into "role"/audience.
    - For any rewrite: if guardrail fails, revert to original text/options.
    - Always preserve option keys via canonical_option_key(slot, label).
    """
    if not LLM_REWRITE_SELECTED or client is None:
        return

    # Snapshot originals so we can revert on drift
    orig_q = (it.question_text or "").strip()
    orig_opt_labels = [o.label for o in (it.answer_options or [])]

    def _contains_any(text: str, phrases: List[str]) -> bool:
        t = (text or "").lower()
        return any(p.lower() in t for p in phrases)

    # --- Construct-specific guardrails ---
    PURPOSE_BAD = [
        "which best describes you", "your role", "who are you", "i am a", "i'm a",
    ]
    PURPOSE_GOOD = [
        "looking for", "here to", "trying to", "do today", "goal", "purpose",
        "information", "pricing", "quote", "contact",
    ]

    depharma_line = "Remove any healthcare/pharma/clinical wording if present. " if STRICT_DEPHARMA else ""

    # =================
    # 1) QUESTION TEXT
    # =================
    if it.construct != "audience_identity":
        instruction_q = (
            f"Rewrite for a {ctx.site_category} site. {depharma_line}"
            "Keep the same construct intent. Keep concise and neutral. "
            "Do not change the construct (do not turn role into purpose or purpose into role)."
        )

        rewritten_q = rewrite_question_text_openai(
            client,
            site_purpose=ctx.site_purpose,
            survey_goal=ctx.survey_goal,
            site_category=ctx.site_category,
            original_question_text=orig_q,
            instruction=instruction_q,
        ).strip()

        # Guardrails (only needed for constructs we rewrite)
        if it.construct == "purpose_intent":
            if _contains_any(rewritten_q, PURPOSE_BAD) and not _contains_any(rewritten_q, PURPOSE_GOOD):
                rewritten_q = orig_q

        # Apply
        it.question_text = rewritten_q or orig_q

    # =================
    # 2) ANSWER OPTIONS
    # =================
    # Default: do NOT rewrite audience_identity options (keys used downstream)
    if it.construct != "audience_identity" and it.answer_options:
        instruction_o = (
            f"Rewrite options for a {ctx.site_category} site. {depharma_line}"
            "Keep options mutually exclusive and aligned to the question. "
            "Keep quick to answer. Do not introduce analytics/trackable topics."
        )

        new_opts = rewrite_answer_options_openai(
            client,
            site_purpose=ctx.site_purpose,
            survey_goal=ctx.survey_goal,
            site_category=ctx.site_category,
            question_text=it.question_text,
            original_options=orig_opt_labels,
            instruction=instruction_o,
            keep_other_if_present=True,
        )
        new_opts = dedupe_options(new_opts)

        # Ensure enough options for closed-ended
        if it.question_type != "OpenText" and len(new_opts) < 2:
            new_opts = orig_opt_labels[:] if len(orig_opt_labels) >= 2 else ["Option 1", "Option 2", "Other"]

        # Guardrail: prevent purpose_intent options drifting into roles
        if it.construct == "purpose_intent":
            roleish = ["patient", "health care", "healthcare", "hcp", "doctor", "nurse", "provider", "caregiver", "parent"]
            if any(any(r in (o or "").lower() for r in roleish) for o in new_opts):
                new_opts = orig_opt_labels[:]

        # Rebuild AnswerOption list with stable unique keys
        used = set()
        out_opts: List[AnswerOption] = []
        for lab in new_opts[:8]:
            base = canonical_option_key(it.slot, lab)
            k = base
            i = 2
            while k in used:
                k = f"{base}_{i}"
                i += 1
            used.add(k)
            out_opts.append(AnswerOption(key=k, label=lab))

        it.answer_options = out_opts

    # ==========================
    # 3) FINAL SAFETY REVERSION
    # ==========================
    if not it.question_text or len(it.question_text.strip()) < 5:
        it.question_text = orig_q

    # If we somehow ended with no options on a closed-ended Q, revert
    if it.question_type != "OpenText":
        cur_labels = [o.label for o in (it.answer_options or [])]
        if len(cur_labels) < 2 and orig_opt_labels:
            used = set()
            out_opts: List[AnswerOption] = []
            for lab in orig_opt_labels[:8]:
                base = canonical_option_key(it.slot, lab)
                k = base
                i = 2
                while k in used:
                    k = f"{base}_{i}"
                    i += 1
                used.add(k)
                out_opts.append(AnswerOption(key=k, label=lab))
            it.answer_options = out_opts


def _llm_fallback_item(
    ctx: BuilderContext,
    client: Any,
    *,
    construct: str,
    qnum: int,
    slot: str,
    level: str,
) -> SurveyItem:
    ex = CORE_EXEMPLARS.get(construct, {}).get("generic", [])
    draft = generate_l1_from_exemplars_openai(
        client,
        construct=construct,
        site_purpose=ctx.site_purpose,
        survey_goal=ctx.survey_goal,
        site_category=ctx.site_category,
        exemplars=ex,
    )
    qt, ao = _ensure_closed_ended(draft.question_type, draft.answer_options)
    ao = dedupe_options(ao)

    # ensure unique keys
    used = set()
    opts: List[AnswerOption] = []
    for lab in ao[:8]:
        base = canonical_option_key(slot, lab)
        k = base
        i = 2
        while k in used:
            k = f"{base}_{i}"
            i += 1
        used.add(k)
        opts.append(AnswerOption(key=k, label=lab))

    return SurveyItem(
        id=str(qnum),
        module_key=construct,
        construct=construct,
        slot=slot,
        phase="Arrival",
        level=level,
        question_id=f"llm_fallback::{slot}::{level}",
        question_type=qt,
        question_text=draft.question_text,
        answer_options=opts,
        display_condition_json=None,
        display_condition="",
        ai_actions={"draft": True, "fallback_generated": True, "generated_by": {"provider": "openai", "model": MODEL}},
    )


def _candidate_summary_rows(cand: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Make candidates JSON-safe and small (don’t send giant fields to the model).
    """
    out: List[Dict[str, Any]] = []
    for _, r in cand.iterrows():
        opts = r.get("_answers", []) or []
        out.append(
            {
                "question_id": str(r.get("question_id", "")).strip(),
                "score": int(r.get("_score", 0)),
                "category": str(r.get("category", "")).strip(),
                "is_backbone": str(r.get("is_backbone", "")).strip(),
                "question_type": str(r.get("question_type", "")).strip(),
                "question_text": str(r.get("question_text", "")).strip(),
                "answer_options": [o.label for o in opts[:10]] if isinstance(opts, list) else [],
            }
        )
    # keep order as-is (already sorted by score)
    return out


def select_row_for_construct(
    df: pd.DataFrame,
    ctx: BuilderContext,
    client: Optional[Any],
    *,
    construct: str,
    phase: str,
    level: str,
) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
    """
    Returns (chosen_row, selection_trace).
    selection_trace includes candidates + heuristic vs LLM decision.
    """
    slot, _lvl = SLOT_LEVEL_BY_CONSTRUCT.get(construct, ("", ""))
    cand = get_candidates_for_construct(df, ctx, construct=construct, phase=phase, level=level, k=CANDIDATE_K)

    trace: Dict[str, Any] = {
        "construct": construct,
        "slot": slot,
        "phase": phase,
        "level": level,
        "candidate_k": CANDIDATE_K,
        "used_llm_selection": False,
        "chosen_by": "heuristic",
        "candidates": _candidate_summary_rows(cand) if not cand.empty else [],
    }

    if cand.empty:
        trace["chosen_by"] = "none"
        return None, trace

    # If heuristic confident (or no client), just pick top
    if not should_call_llm_selection(cand, client):
        top = cand.iloc[0]
        trace["chosen_question_id"] = str(top.get("question_id", "")).strip()
        trace["chosen_score"] = int(top.get("_score", 0))
        return top, trace

    # LLM selection from shortlist
    dec = choose_best_candidate_openai(
        client,
        ctx=ctx,
        construct=construct,
        candidates=trace["candidates"],
    )
    trace["used_llm_selection"] = True
    trace["chosen_by"] = "llm_selection"
    trace["chosen_question_id"] = dec.chosen_question_id
    trace["rewrite_needed"] = dec.rewrite_needed
    trace["why"] = dec.why
    trace["confidence"] = dec.confidence

    chosen = cand[cand["question_id"].astype(str).str.strip() == dec.chosen_question_id.strip()]
    if not chosen.empty:
        return chosen.iloc[0], trace

    # fallback to top if somehow not found
    top = cand.iloc[0]
    trace["chosen_by"] = "heuristic_fallback_after_llm"
    trace["chosen_question_id"] = str(top.get("question_id", "")).strip()
    trace["chosen_score"] = int(top.get("_score", 0))
    return top, trace


# -------------------
# BUILD ARRIVAL SURVEY (Dynamic + LLM Selection)
# -------------------
def build_arrival(df: pd.DataFrame, ctx: BuilderContext, client: Optional[Any] = None) -> Dict[str, Any]:
    """
    Updated build_arrival():
    - For NON-PHARMA: LLM-draft audience_identity (question + options) because library is pharma-only.
    - Adds an optional, LLM-planned Audience follow-up (role/relationship refinement) inserted
      immediately after audience_identity (never last).
    - Keeps legacy Pharma-only HCP profile behavior but prevents it from appearing for non-Pharma.
    - Ensures anything stored in payload/meta/ai_actions is JSON-serializable (no Pydantic objects).
    """

    # -----------------------
    # helpers
    # -----------------------
    def _to_plain(obj: Any) -> Any:
        # Pydantic v2
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump()
            except Exception:
                pass
        # Pydantic v1
        if hasattr(obj, "dict"):
            try:
                return obj.dict()
            except Exception:
                pass
        return obj

    def _build_followup_item_from_plan(
        qnum: int,
        *,
        followup_plan: Any,  # AudienceFollowupPlan
        audience_var: str,
    ) -> SurveyItem:
        qt, ao = _ensure_closed_ended(followup_plan.question_type, followup_plan.answer_options)
        ao = dedupe_options(ao)

        cond = {"all": [{"var": audience_var, "op": "in", "value": list(followup_plan.trigger_audience_keys)}]}

        slot_name = "1b_Audience_Profile"

        # unique keys
        used = set()
        opts: List[AnswerOption] = []
        for lab in ao[:8]:
            base = canonical_option_key(slot_name, lab)
            k = base
            i = 2
            while k in used:
                k = f"{base}_{i}"
                i += 1
            used.add(k)
            opts.append(AnswerOption(key=k, label=lab))

        return SurveyItem(
            id=str(qnum),
            module_key="audience_followup",
            construct="audience_followup",
            slot=slot_name,
            phase="Arrival",
            level="L1",
            question_id="llm_followup::audience",
            question_type=qt,
            question_text=str(getattr(followup_plan, "question_text", "") or "").strip(),
            answer_options=opts,
            display_condition_json=cond,
            display_condition="",
            ai_actions={
                "draft": True,
                "generated_by": {"provider": "openai", "model": MODEL},
                "followup_plan": _to_plain(followup_plan),
            },
        )

    def _draft_audience_identity_non_pharma(
        qnum: int,
        *,
        slot: str,
        level: str,
    ) -> SurveyItem:
        """
        LLM-generate a sensible Audience (role) question for non-Pharma sites,
        since the library is pharma-only.
        """
        if client is None:
            # Fall back to missing template if no client
            return SurveyItem(
                id=str(qnum),
                module_key="audience_identity",
                construct="audience_identity",
                slot=slot,
                phase="Arrival",
                level=level,
                question_id=f"missing::{slot}::{level}",
                question_type="SingleSelection",
                question_text="[MISSING TEMPLATE] audience_identity",
                answer_options=[],
                display_condition_json=None,
                display_condition="",
                ai_actions={"missing_template": True, "reason": "non_pharma_needs_llm_but_no_client"},
            )

        # Use the intent-based generator so it adapts to site_purpose + survey_goal
        draft = generate_l1_by_intent_openai(
            client,
            ctx=ctx,
            intent="Identify the visitor's role/relationship to the product or project (who they are), not their purpose/intent.",
            slot=slot,
            level=level,
            force_type="SingleSelection",
        )
        qt, ao = _ensure_closed_ended(draft.question_type, draft.answer_options)
        ao = dedupe_options(ao)

        used = set()
        opts: List[AnswerOption] = []
        for lab in ao[:8]:
            base = canonical_option_key(slot, lab)
            k = base
            i = 2
            while k in used:
                k = f"{base}_{i}"
                i += 1
            used.add(k)
            opts.append(AnswerOption(key=k, label=lab))

        return SurveyItem(
            id=str(qnum),
            module_key="audience_identity",
            construct="audience_identity",
            slot=slot,
            phase="Arrival",
            level=level,
            question_id=f"llm_audience::{slot}::{level}",
            question_type=qt,
            question_text=(draft.question_text or "").strip(),
            answer_options=opts,
            display_condition_json=None,
            display_condition="",
            ai_actions={
                "draft": True,
                "fallback_generated": True,
                "generated_by": {"provider": "openai", "model": MODEL},
                "reason": "library_pharma_only_bypass_for_non_pharma",
            },
        )

    # -----------------------
    # main
    # -----------------------
    phase = "Arrival"
    construct_plan = plan_constructs(ctx, client=client)
    constructs = [p["construct"] for p in construct_plan["constructs"]]

    items: List[SurveyItem] = []
    qnum = 1

    is_pharma = (ctx.site_category or "").strip().lower() == "pharma"

    # 1) Build planned constructs
    for construct in constructs:
        if construct not in SLOT_LEVEL_BY_CONSTRUCT:
            continue

        slot, lvl = SLOT_LEVEL_BY_CONSTRUCT[construct]

        # ✅ audience_identity for non-Pharma is LLM-generated (bypass pharma-only library)
        if construct == "audience_identity" and not is_pharma:
            it = _draft_audience_identity_non_pharma(qnum, slot=slot, level=lvl)
            items.append(it)
            qnum += 1
            continue

        chosen_row, sel_trace = select_row_for_construct(
            df,
            ctx,
            client,
            construct=construct,
            phase=phase,
            level=lvl,
        )

        it = row_to_item(chosen_row, qnum, construct=construct, slot=slot, level=lvl)
        it.ai_actions["selection_trace"] = _to_plain(sel_trace)
        it.ai_actions.setdefault("suggest_wording", {"enabled": True, "mode": "llm_optional"})

        # If missing and fallback enabled
        if it.ai_actions.get("missing_template") and LLM_FALLBACK_ON_MISSING and client is not None:
            it = _llm_fallback_item(ctx, client, construct=construct, qnum=qnum, slot=slot, level=lvl)

        # Optional rewrite (bounded, with guardrails)
        _maybe_rewrite_selected_item(it, ctx, client)

        # -----------------------
        # L2 SUGGESTIONS (plan-only by default)
        # -----------------------
        if client is not None and L2_POLICY in {"user", "auto"}:
            try:
                if it.level == "L1" and it.question_type != "OpenText" and it.answer_options:
                    parent_opts = [{"key": o.key, "label": o.label} for o in (it.answer_options or [])]
                    l2_plan = plan_l2_followups_openai(
                        client,
                        ctx=ctx,
                        parent_construct=it.construct,
                        parent_question_text=it.question_text,
                        parent_question_type=it.question_type,
                        parent_answer_options=parent_opts,
                        max_l2_questions=L2_MAX_DISTINCT,
                    )
                    it.ai_actions["l2_suggestions"] = l2_plan.model_dump() if hasattr(l2_plan, "model_dump") else l2_plan.dict()

                    # Auto-add only when explicitly configured and strong
                    if (
                        L2_POLICY == "auto"
                        and getattr(l2_plan, "enabled", False)
                        and float(getattr(l2_plan, "confidence", 0.0)) >= L2_AUTO_CONFIDENCE
                        and (l2_plan.drafts or [])
                    ):
                        # Add the first draft automatically (conservative)
                        d0 = (l2_plan.drafts or [])[0]
                        parent_var = var_name_for_slot(it.slot, it.level)
                        cond = build_l2_condition(parent_var, it.question_type, list(d0.trigger_answer_keys))
                        qt, ao = _ensure_closed_ended(d0.question_type, d0.answer_options)
                        ao = dedupe_options(ao)

                        items.append(it)  # ensure parent is appended before L2
                        qnum += 1

                        # unique keys
                        used = set()
                        opts: List[AnswerOption] = []
                        if qt != "OpenText":
                            for lab in ao[:8]:
                                base = canonical_option_key(it.slot, lab)
                                k = base
                                i = 2
                                while k in used:
                                    k = f"{base}_{i}"
                                    i += 1
                                used.add(k)
                                opts.append(AnswerOption(key=k, label=lab))

                        items.append(
                            SurveyItem(
                                id=str(qnum),
                                module_key="l2_followup",
                                construct="l2_followup",
                                slot=it.slot,
                                phase="Arrival",
                                level="L2",
                                question_id=f"llm_l2::{it.slot}::{slugify(','.join(d0.trigger_answer_keys))}",
                                question_type=qt,
                                question_text=d0.question_text,
                                answer_options=opts,
                                display_condition_json=cond,
                                display_condition="",
                                ai_actions={
                                    "draft": True,
                                    "generated_by": {"provider": "openai", "model": MODEL},
                                    "why": d0.why,
                                    "trigger_answer_keys": list(d0.trigger_answer_keys),
                                    "auto_added": True,
                                },
                            )
                        )
                        qnum += 1
                        continue  # we already appended parent + l2

            except Exception as e:
                it.ai_actions.setdefault("warnings", [])
                it.ai_actions["warnings"].append({"type": "l2_planning_failed", "error": str(e)})

        items.append(it)
        qnum += 1

    # 2) Optional: LLM-planned Audience follow-up inserted right after audience_identity
    #    This is a dynamic "secondary audience" question (role refinement), not purpose duplication.
    audience_item = next((x for x in items if x.construct == "audience_identity"), None)
    if client is not None and audience_item and audience_item.answer_options:
        try:
            aud_opts = [{"key": o.key, "label": o.label} for o in audience_item.answer_options]
            followup_plan = plan_audience_followup_openai(
                client,
                ctx=ctx,
                audience_question_text=audience_item.question_text,
                audience_options=aud_opts,
            )

            if bool(getattr(followup_plan, "enabled", False)) and list(
                getattr(followup_plan, "trigger_audience_keys", [])
            ):
                audience_var = var_name_for_slot("1_Audience", "L1")
                followup_item = _build_followup_item_from_plan(
                    qnum,
                    followup_plan=followup_plan,
                    audience_var=audience_var,
                )

                # Insert immediately after audience_identity (never last)
                aud_idx = next(i for i, it2 in enumerate(items) if it2.construct == "audience_identity")
                items.insert(aud_idx + 1, followup_item)
                qnum += 1

        except Exception as e:
            if audience_item:
                audience_item.ai_actions.setdefault("warnings", [])
                audience_item.ai_actions["warnings"].append(
                    {"type": "audience_followup_planning_failed", "error": str(e)}
                )

    # 3) Legacy: Pharma-only HCP profile block (fixed)
    if is_pharma:
        audience_item = next((x for x in items if x.construct == "audience_identity"), None)

        # Add HCP profile only if the audience question actually has an "hcp" key
        need_hcp_profile = bool(
            audience_item
            and audience_item.answer_options
            and any((o.key or "").strip() == "hcp" for o in (audience_item.answer_options or []))
        )

        if need_hcp_profile:
            slot, lvl = SLOT_LEVEL_BY_CONSTRUCT["hcp_profile"]

            chosen_row, sel_trace = select_row_for_construct(
                df,
                ctx,
                client,
                construct="hcp_profile",
                phase=phase,
                level=lvl,
            )

            hcp_l1 = row_to_item(chosen_row, qnum, construct="hcp_profile", slot=slot, level=lvl)
            hcp_l1.ai_actions["selection_trace"] = _to_plain(sel_trace)
            hcp_l1.ai_actions.setdefault("suggest_wording", {"enabled": True, "mode": "llm_optional"})

            if hcp_l1.ai_actions.get("missing_template") and LLM_FALLBACK_ON_MISSING and client is not None:
                hcp_l1 = _llm_fallback_item(ctx, client, construct="hcp_profile", qnum=qnum, slot=slot, level=lvl)

            _maybe_rewrite_selected_item(hcp_l1, ctx, client)

            # Gate on Audience_L1 == hcp
            hcp_l1.display_condition_json = {
                "all": [{"var": var_name_for_slot("1_Audience", "L1"), "op": "equals", "value": "hcp"}]
            }

            items.append(hcp_l1)
            qnum += 1

            # HCP L2: library matching + trace only (no auto L2 generation unless AUTO_GENERATE_L2 is enabled)
            parent_var = var_name_for_slot("1b_HCP_Profile", "L1")

            l2_1b = df[
                (df["slot"].isin(["1b_HCP_Profile", "2b_HCP_Profile"]))
                & (df["level"] == "L2")
                & (df["survey_phase"].isin([phase, "Both", ""]))
            ].copy()

            # Prefer backbone but don't require it
            if not l2_1b.empty:
                l2_1b["_score"] = l2_1b.apply(lambda r: score_row(r, ctx, "hcp_profile"), axis=1)
                l2_all = l2_1b.sort_values("_score", ascending=False).copy()
            else:
                l2_all = pd.DataFrame()

            bucket = l2_candidates_by_answer(l2_all, parent_var) if not l2_all.empty else {}

            by_answer: Dict[str, Any] = {}
            parent_answers = hcp_l1.answer_options or []

            for opt in parent_answers:
                res = resolve_l2_mode(bucket=bucket, answer_key=opt.key)
                by_answer[opt.key] = _to_plain(res)

                if res["mode"] == "library" and res.get("chosen_question_id"):
                    chosen = l2_all[l2_all["question_id"].astype(str).str.strip() == res["chosen_question_id"]]
                    if not chosen.empty:
                        row = chosen.iloc[0]
                        l2_item = row_to_item(
                            row,
                            qnum,
                            construct="hcp_profile_l2",
                            slot=str(row.get("slot", "")).strip() or "2b_HCP_Profile",
                            level="L2",
                        )
                        items.append(l2_item)
                        qnum += 1

                if res["mode"] == "llm_needed" and AUTO_GENERATE_L2:
                    # Intentionally not auto-generating here (legacy path stays deterministic)
                    continue

            hcp_l1.ai_actions["l2_followups"] = {
                "enabled": True,
                "parent_var": parent_var,
                "by_answer_key": by_answer,
                "llm_policy": {
                    "generate_when": ["multiple_matching_library_templates", "no_matching_library_template"],
                    "auto_generate_enabled": AUTO_GENERATE_L2,
                    "model": MODEL,
                },
            }

    # 4) Renumber to ensure sequential ids after inserts
    for i, it in enumerate(items, start=1):
        it.id = str(i)

    # 5) Payload (ensure meta values are plain dicts)
    payload = {
        "meta": {
            "builder": "arrival_backbone_v10_llm_selection_openai_interactive",
            "site_purpose": ctx.site_purpose,
            "survey_goal": ctx.survey_goal,
            "site_category": ctx.site_category,
            "library_path": LIB_PATH,
            "arrival_flow": ARRIVAL_FLOW,
            "construct_plan": _to_plain(construct_plan),
            "llm": {
                "selection_enabled": LLM_SELECT_PER_CONSTRUCT,
                "selection_always": LLM_SELECT_ALWAYS,
                "rewrite_selected": LLM_REWRITE_SELECTED,
                "fallback_on_missing": LLM_FALLBACK_ON_MISSING,
                "auto_generate_l2": AUTO_GENERATE_L2,
                "provider": "openai"
                if (LLM_SELECT_PER_CONSTRUCT or LLM_REWRITE_SELECTED or LLM_FALLBACK_ON_MISSING or AUTO_GENERATE_L2)
                else "none",
                "model": MODEL
                if (LLM_SELECT_PER_CONSTRUCT or LLM_REWRITE_SELECTED or LLM_FALLBACK_ON_MISSING or AUTO_GENERATE_L2)
                else "",
                "temperature": TEMP
                if (LLM_SELECT_PER_CONSTRUCT or LLM_REWRITE_SELECTED or LLM_FALLBACK_ON_MISSING or AUTO_GENERATE_L2)
                else None,
                "candidate_k": CANDIDATE_K,
                "select_margin": SELECT_MARGIN,
            },
        },
        "phase": "Arrival",
        "items": [
            {
                **{k: v for k, v in asdict(it).items() if k != "answer_options"},
                "answer_options": [asdict(o) for o in it.answer_options],
                "ai_actions": _to_plain(asdict(it).get("ai_actions", {})),
            }
            for it in items
        ],
    }

    return payload


# -------------------
# PRETTY PRINT
# -------------------
def neat_preview(payload: Dict[str, Any], show_keys: bool = False) -> str:
    lines: List[str] = []
    lines.append("\n=== NEAT PREVIEW (Arrival) ===\n")

    plan = payload.get("meta", {}).get("construct_plan", {})
    if hasattr(plan, "model_dump"):
        plan = plan.model_dump()
    elif hasattr(plan, "dict"):
        plan = plan.dict()

    if plan:
        lines.append("Construct plan:")
        for p in plan.get("constructs", []):
            c = p.get("construct")
            reason = (plan.get("reasons", {}) or {}).get(c, "")
            req = "required" if p.get("required") else "optional"
            lines.append(f" - {c} ({req})  reason={reason}")
        lines.append("")

    for it in payload.get("items", []):
        qid = it.get("id")
        slot = it.get("slot")
        lvl = it.get("level")
        c = it.get("construct")
        qtext = it.get("question_text")
        lines.append(f"Q{qid} [{c}] [{slot}/{lvl}] {qtext}")
        if it.get("display_condition_json"):
            lines.append(f"   cond={json.dumps(it['display_condition_json'], ensure_ascii=False)}")

        opts = it.get("answer_options", []) or []
        for o in opts:
            if isinstance(o, dict):
                lab = o.get("label", "")
                if show_keys and o.get("key"):
                    lines.append(f"   - {lab}  (key={o['key']})")
                else:
                    lines.append(f"   - {lab}")
            else:
                lines.append(f"   - {str(o)}")

        sel = (it.get("ai_actions") or {}).get("selection_trace")
        if sel and isinstance(sel, dict) and sel.get("used_llm_selection"):
            lines.append(f"   selected_by=llm  id={sel.get('chosen_question_id')}  conf={sel.get('confidence')}")

        l2p = (it.get("ai_actions") or {}).get("l2_followups")
        if l2p and isinstance(l2p, dict):
            by = l2p.get("by_answer_key", {}) or {}
            needed = [k for k, v in by.items() if isinstance(v, dict) and v.get("mode") == "llm_needed"]
            if needed:
                lines.append(f"   L2 needed for: {', '.join(needed)}")

    return "\n".join(lines)


# -------------------
# INTERACTIVE LOOP
# -------------------

COMMAND_HELP = """
VIEW
  show
  showkeys
  keys <qid>

EDIT (AI)
  tuneq <qid> | <instruction>
  tuneopts <qid> | <instruction>

EDIT OPTIONS (no AI)
  moveopt <qid> <from_idx> <to_idx>        (1-based)
  reorderopts <qid> <idx1,idx2,...>        (1-based)

CONDITIONAL (AI)
  l2plan <parent_qid>                      (generate/print L2 suggestions for parent)
  l2add <parent_qid> <draft_index>         (add an L2 from the printed plan; draft_index is 1-based)
  l2 <parent_qid> <key1,key2,...>          (interactive: choose keys -> enter intent -> AI drafts -> auto-add)
  l2 <parent_qid> <key1,key2,...> | <question_type> | <intent>
                                          (non-interactive: SingleSelection/MultiSelection/OpenText)

ADD QUESTION (no AI)
  addq <after_qid|end> <slot> <level> <question_type> | <question_text> | <opt1;opt2;...>

ADD QUESTION (AI)
  addqai <after_qid|end> <slot> <level> <question_type|auto> | <intent>

SYSTEM
  save
  quit
""".strip()


def _split_pipe(cmd: str, expected_parts: int) -> Optional[List[str]]:
    parts = [p.strip() for p in cmd.split("|")]
    if len(parts) < expected_parts:
        return None
    # re-join any extra pipes into the last part
    head = parts[0]
    tail = parts[1:]
    while len(tail) > expected_parts - 1:
        tail[-2] = tail[-2] + " | " + tail[-1]
        tail.pop()
    return [head] + tail


def _find_item(payload: Dict[str, Any], qid: str) -> Optional[Dict[str, Any]]:
    for it in payload.get("items", []):
        if str(it.get("id")) == str(qid):
            return it
    return None


def _clone_item(it: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(it)


def _dedupe_labels_preserve_order(labels: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in labels:
        t = (x or "").strip()
        if not t:
            continue
        n = re.sub(r"\s+", " ", t.lower())
        if n in seen:
            continue
        seen.add(n)
        out.append(t)
    return out


def _set_options_from_labels(it: Dict[str, Any], labels: List[str]) -> None:
    slot = it.get("slot", "")
    labels = _dedupe_labels_preserve_order(labels)

    used = set()
    out = []
    for lab in labels:
        base = canonical_option_key(slot, lab)
        k = base
        i = 2
        while k in used:
            k = f"{base}_{i}"
            i += 1
        used.add(k)
        out.append({"key": k, "label": lab})

    it["answer_options"] = out


def _print_update(before_it: Dict[str, Any], after_it: Dict[str, Any]) -> None:
    print("\n--- UPDATE APPLIED ---")
    print(f"Q{after_it.get('id')} [{after_it.get('construct')}] [{after_it.get('slot')}/{after_it.get('level')}]")
    if before_it.get("question_text") != after_it.get("question_text"):
        print("Question text:")
        print("  BEFORE:", before_it.get("question_text"))
        print("  AFTER: ", after_it.get("question_text"))

    bopts = [o.get("label") for o in (before_it.get("answer_options") or []) if isinstance(o, dict)]
    aopts = [o.get("label") for o in (after_it.get("answer_options") or []) if isinstance(o, dict)]
    if bopts != aopts:
        print("Answer options:")
        print("  BEFORE:", bopts)
        print("  AFTER: ", aopts)
    print("----------------------\n")


def ai_generate_l2_draft(
    client,
    *,
    ctx,
    parent_it: Dict[str, Any],
    trigger_labels: List[str],
    user_intent: str,
    desired_qtype: str,
):
    """
    Use existing rewrite_* helpers to generate an L2 question + options draft.
    Returns: (question_text: str, options: List[str])
    """
    parent_q = (parent_it.get("question_text") or "").strip()
    trig = ", ".join(trigger_labels) if trigger_labels else ""

    # 1) Generate L2 question text
    draft_l2 = (
        f"Follow-up question when user selected: {trig}. "
        f"Parent question: {parent_q}. "
        f"User wants: {user_intent}."
    )

    q_instruction = (
        "Write ONE concise follow-up survey question that is directly relevant to the selected answer(s). "
        "Avoid pharma/healthcare wording unless site category is Pharma. "
        "Keep it short and clear. "
        f"Desired question type: {desired_qtype}. "
        "Do not mention 'selected answer' explicitly. "
        "Do not reference internal keys."
    )

    l2_text = rewrite_question_text_openai(
        client,
        site_purpose=ctx.site_purpose,
        survey_goal=ctx.survey_goal,
        site_category=ctx.site_category,
        original_question_text=draft_l2,
        instruction=q_instruction,
    )

    # 2) Generate options if closed-ended
    if desired_qtype == "OpenText":
        return l2_text.strip(), []

    # seed options to guide the model
    seed = ["Option A", "Option B", "Other"]

    opt_instruction = (
        "Generate answer options for the follow-up question. "
        "Return short option labels only. "
        "Make them mutually exclusive where possible. "
        "Include 'Other' only if useful. "
        f"Follow-up question: {l2_text}"
    )

    l2_opts = rewrite_answer_options_openai(
        client,
        site_purpose=ctx.site_purpose,
        survey_goal=ctx.survey_goal,
        site_category=ctx.site_category,
        question_text=l2_text,
        original_options=seed,
        instruction=opt_instruction,
        keep_other_if_present=True,
    )

    # dedupe/cap
    l2_opts = _dedupe_labels_preserve_order([str(x) for x in (l2_opts or [])])[:8]
    return l2_text.strip(), l2_opts


def build_l2_item_from_draft(
    *,
    parent_item: Dict[str, Any],
    draft: Dict[str, Any],
    new_id: str,
) -> Dict[str, Any]:
    pslot = str(parent_item.get("slot") or "")
    plevel = str(parent_item.get("level") or "L1")
    parent_var = var_name_for_slot(pslot, plevel)

    parent_qtype = normalize_qtype(str(parent_item.get("question_type") or "SingleSelection"))
    trigger_keys = [str(x).strip() for x in (draft.get("trigger_answer_keys") or []) if str(x).strip()]

    cond = build_l2_condition(parent_var, parent_qtype, trigger_keys)

    qtype = normalize_qtype(str(draft.get("question_type") or "SingleSelection"))
    qtext = str(draft.get("question_text") or "").strip()
    opts = list(draft.get("answer_options") or [])

    qtype, opts = _ensure_closed_ended(qtype, opts)

    item = {
        "id": str(new_id),
        "module_key": "l2_followup",
        "construct": "l2_followup",
        "slot": pslot,
        "phase": "Arrival",
        "level": "L2",
        "question_id": f"llm_l2::{pslot}::{slugify(','.join(trigger_keys))}",
        "question_type": qtype,
        "question_text": qtext,
        "answer_options": [],
        "display_condition_json": cond,
        "display_condition": "",
        "ai_actions": {
            "draft": True,
            "source": "l2_user_requirements_ui",
            "trigger_answer_keys": trigger_keys,
            "why": str(draft.get("why") or ""),
        },
    }

    if qtype != "OpenText":
        opts = _dedupe_labels_preserve_order([str(x) for x in opts])
        _set_options_from_labels(item, opts[:8])

    return item


def interactive_loop(payload: Dict[str, Any], ctx: BuilderContext, client: Any) -> Dict[str, Any]:
    print("\nInteractive refinement started. Type 'help' to see commands.")

    while True:
        cmd = input("\n> ").strip()
        if not cmd:
            continue
        low = cmd.lower()

        if low in {"help", "h", "?"}:
            print("\n" + COMMAND_HELP + "\n")
            continue

        # Hidden alias for backwards compatibility (not shown in help)
        if low.startswith("genl2 "):
            cmd = "l2 " + cmd.split(" ", 1)[1]
            low = cmd.lower()

        if low in {"quit", "exit", "q"}:
            return payload

        if low == "show":
            print(neat_preview(payload, show_keys=False))
            continue

        if low == "showkeys":
            print(neat_preview(payload, show_keys=True))
            continue

        if low == "save":
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"✅ Saved: {OUT_PATH}")
            continue

        if low.startswith("keys "):
            parts = cmd.split()
            if len(parts) != 2:
                print("Usage: keys <qid>")
                continue
            qid = parts[1].strip()
            it = _find_item(payload, qid)
            if not it:
                print(f"No question found with id={qid}")
                continue
            opts = it.get("answer_options", []) or []
            if not opts:
                print("No answer options.")
                continue
            print("Answer keys:")
            for o in opts:
                if isinstance(o, dict):
                    print(f" - {o.get('key')}: {o.get('label')}")
            continue

        if low.startswith("moveopt "):
            parts = cmd.split()
            if len(parts) != 4:
                print("Usage: moveopt <qid> <from_idx> <to_idx>")
                continue
            it = _find_item(payload, parts[1])
            if not it:
                print("No question found.")
                continue
            labels = [o.get("label") for o in (it.get("answer_options") or []) if isinstance(o, dict)]
            try:
                a = int(parts[2]) - 1
                b = int(parts[3]) - 1
            except Exception:
                print("Indices must be integers.")
                continue
            if not (0 <= a < len(labels) and 0 <= b < len(labels)):
                print(f"Index out of range (1..{len(labels)})")
                continue
            before = _clone_item(it)
            x = labels.pop(a)
            labels.insert(b, x)
            _set_options_from_labels(it, labels)
            _print_update(before, it)
            continue

        if low.startswith("reorderopts "):
            parts = cmd.split(" ", 2)
            if len(parts) != 3:
                print("Usage: reorderopts <qid> <idx1,idx2,...>")
                continue
            it = _find_item(payload, parts[1])
            if not it:
                print("No question found.")
                continue
            labels = [o.get("label") for o in (it.get("answer_options") or []) if isinstance(o, dict)]
            try:
                order = [int(x.strip()) - 1 for x in parts[2].split(",")]
            except Exception:
                print("Bad index list.")
                continue
            if sorted(order) != list(range(len(labels))):
                print(f"Must include each index exactly once (1..{len(labels)})")
                continue
            before = _clone_item(it)
            new_labels = [labels[i] for i in order]
            _set_options_from_labels(it, new_labels)
            _print_update(before, it)
            continue

        if low.startswith("tuneq "):
            parts = _split_pipe(cmd, 2)
            if not parts:
                print("Usage: tuneq <qid> | <instruction>")
                continue
            head, instruction = parts[0], parts[1]
            hp = head.split()
            if len(hp) < 2:
                print("Usage: tuneq <qid> | <instruction>")
                continue
            qid = hp[1].strip()
            it = _find_item(payload, qid)
            if not it:
                print(f"No question found with id={qid}")
                continue
            before = _clone_item(it)
            it["question_text"] = rewrite_question_text_openai(
                client,
                site_purpose=ctx.site_purpose,
                survey_goal=ctx.survey_goal,
                site_category=ctx.site_category,
                original_question_text=it.get("question_text", ""),
                instruction=instruction,
            )
            _print_update(before, it)
            print("✅ Updated question text.")
            continue

        if low.startswith("tuneopts "):
            parts = _split_pipe(cmd, 2)
            if not parts:
                print("Usage: tuneopts <qid> | <instruction>")
                continue
            head, instruction = parts[0], parts[1]
            hp = head.split()
            if len(hp) < 2:
                print("Usage: tuneopts <qid> | <instruction>")
                continue
            qid = hp[1].strip()
            it = _find_item(payload, qid)
            if not it:
                print(f"No question found with id={qid}")
                continue
            before = _clone_item(it)
            orig_opts = [o.get("label") for o in (it.get("answer_options") or []) if isinstance(o, dict)]
            new_opts = rewrite_answer_options_openai(
                client,
                site_purpose=ctx.site_purpose,
                survey_goal=ctx.survey_goal,
                site_category=ctx.site_category,
                question_text=it.get("question_text", ""),
                original_options=orig_opts,
                instruction=instruction,
                keep_other_if_present=True,
            )
            _set_options_from_labels(it, new_opts)
            _print_update(before, it)
            print("✅ Updated answer options.")
            continue

        if low.startswith("addq "):
            try:
                head, qtext, optblob = [x.strip() for x in cmd.split("|", 2)]
            except Exception:
                print("Usage: addq <after_qid|end> <slot> <level> <question_type> | <question_text> | <opt1;opt2;...>")
                continue

            hp = head.split()
            if len(hp) < 5:
                print("Usage: addq <after_qid|end> <slot> <level> <question_type> | <question_text> | <opt1;opt2;...>")
                continue

            after = hp[1].strip()
            slot = hp[2].strip()
            level = hp[3].strip().upper()
            qtype = " ".join(hp[4:]).strip()

            labels = [x.strip() for x in optblob.split(";") if x.strip()]
            if qtype != "OpenText":
                labels = _dedupe_labels_preserve_order(labels)
                if len(labels) < 2:
                    labels = ["Option 1", "Option 2", "Other"]

            new_item = {
                "id": "TBD",
                "module_key": "manual_add",
                "construct": "manual_add",
                "slot": slot,
                "phase": "Arrival",
                "level": level,
                "question_id": f"manual::{slot}::{level}::{slugify(qtype)}",
                "question_type": qtype,
                "question_text": qtext,
                "display_condition_json": None,
                "display_condition": "",
                "ai_actions": {"manual_added": True},
                "answer_options": [] if qtype == "OpenText" else [],
            }
            if qtype != "OpenText":
                _set_options_from_labels(new_item, labels)

            items = payload.get("items", [])
            if after.lower() == "end":
                items.append(new_item)
            else:
                idx = next((i for i, it2 in enumerate(items) if str(it2.get("id")) == after), None)
                if idx is None:
                    print("after_qid not found (use end).")
                    continue
                items.insert(idx + 1, new_item)

            for i, it2 in enumerate(items, start=1):
                it2["id"] = str(i)
            payload["items"] = items
            print("✅ Added question.")
            continue

        if low.startswith("addqai "):
            if client is None:
                print("OpenAI client not available.")
                continue
            try:
                head, intent = [x.strip() for x in cmd.split("|", 1)]
            except Exception:
                print("Usage: addqai <after_qid|end> <slot> <level> <question_type|auto> | <intent>")
                continue

            hp = head.split()
            if len(hp) < 5:
                print("Usage: addqai <after_qid|end> <slot> <level> <question_type|auto> | <intent>")
                continue

            after = hp[1].strip()
            slot = hp[2].strip()
            level = hp[3].strip().upper()
            qtype = " ".join(hp[4:]).strip()

            draft = generate_l1_by_intent_openai(
                client,
                ctx=ctx,
                intent=intent,
                slot=slot,
                level=level,
                force_type=qtype,
            )
            qt, ao = _ensure_closed_ended(draft.question_type, draft.answer_options)
            ao = _dedupe_labels_preserve_order(ao)

            new_item = {
                "id": "TBD",
                "module_key": "llm_add",
                "construct": "llm_add",
                "slot": slot,
                "phase": "Arrival",
                "level": level,
                "question_id": f"llm_add::{slot}::{level}",
                "question_type": qt,
                "question_text": draft.question_text,
                "display_condition_json": None,
                "display_condition": "",
                "ai_actions": {"draft": True, "manual_added": True, "generated_by": {"provider": "openai", "model": MODEL}},
                "answer_options": [] if qt == "OpenText" else [],
            }
            if qt != "OpenText":
                _set_options_from_labels(new_item, ao)

            items = payload.get("items", [])
            if after.lower() == "end":
                items.append(new_item)
            else:
                idx = next((i for i, it2 in enumerate(items) if str(it2.get("id")) == after), None)
                if idx is None:
                    print("after_qid not found (use end).")
                    continue
                items.insert(idx + 1, new_item)

            for i, it2 in enumerate(items, start=1):
                it2["id"] = str(i)
            payload["items"] = items
            print("✅ Added LLM draft question.")
            continue

        # ==========================
        # UPDATED L2 COMMAND (NEW)
        # ==========================
        if low.startswith("l2 "):
            # Supports:
            #   l2 <parent_qid> <key1,key2,...>
            #   l2 <parent_qid> <key1,key2,...> | <question_type> | <intent>
            #
            # If pipe form not used, we prompt interactively for intent + type.
            pipe_parts = _split_pipe(cmd, 3)
            if pipe_parts:
                head, qtype_part, intent_part = pipe_parts[0], pipe_parts[1], pipe_parts[2]
                hp = head.split()
                if len(hp) < 3:
                    print("Usage: l2 <parent_qid> <key1,key2,...> | <question_type> | <intent>")
                    continue
                parent_qid = hp[1].strip()
                keys_blob = hp[2].strip()
                desired_qtype = (qtype_part or "").strip() or "SingleSelection"
                user_intent = (intent_part or "").strip()
            else:
                parts = cmd.split()
                if len(parts) < 3:
                    print("Usage: l2 <parent_qid> <key1,key2,...>")
                    continue
                parent_qid = parts[1].strip()
                keys_blob = parts[2].strip()
                desired_qtype = ""
                user_intent = ""

            parent = _find_item(payload, parent_qid)
            if not parent:
                print(f"No question found with id={parent_qid}")
                continue

            parent_opts = parent.get("answer_options", []) or []
            key_to_label = {o.get("key"): o.get("label") for o in parent_opts if isinstance(o, dict)}
            trigger_keys = [k.strip() for k in keys_blob.split(",") if k.strip()]
            trigger_keys = [k for k in trigger_keys if k in key_to_label]

            if not trigger_keys:
                print("No valid answer keys provided.")
                print("Available keys:", list(key_to_label.keys()))
                continue

            # Step 1: collect intent + type if not provided
            if not user_intent:
                print("\nWhat do you want this L2 follow-up to ask? (free text)")
                user_intent = input("> ").strip()
                if not user_intent:
                    print("Canceled: intent is required.")
                    continue

            if not desired_qtype:
                print("\nChoose L2 question type: SingleSelection / MultiSelection / OpenText  [SingleSelection]")
                desired_qtype = (input("> ").strip() or "SingleSelection").strip()

            # Normalize type
            desired_qtype_norm = desired_qtype.strip()
            if desired_qtype_norm.lower() in {"single", "singleselect", "single_selection"}:
                desired_qtype_norm = "SingleSelection"
            if desired_qtype_norm.lower() in {"multi", "multiselect", "multi_selection"}:
                desired_qtype_norm = "MultiSelection"
            if desired_qtype_norm.lower() in {"open", "text", "opentext"}:
                desired_qtype_norm = "OpenText"
            if desired_qtype_norm not in {"SingleSelection", "MultiSelection", "OpenText"}:
                print("Invalid question_type. Use: SingleSelection / MultiSelection / OpenText")
                continue

            # Step 2: call AI to draft question + options
            selected_labels = [key_to_label[k] for k in trigger_keys]
            l2_text, l2_opts = ai_generate_l2_draft(
                client,
                ctx=ctx,
                parent_it=parent,
                trigger_labels=selected_labels,
                user_intent=user_intent,
                desired_qtype=desired_qtype_norm,
            )

            # Build a draft dict compatible with build_l2_item_from_draft
            draft = {
                "trigger_answer_keys": trigger_keys,
                "question_text": l2_text,
                "question_type": desired_qtype_norm,
                "answer_options": l2_opts,
                "why": f"user_intent: {user_intent}",
            }

            # Step 3: build the L2 item and insert right after parent
            items = payload.get("items", [])

            new_item = build_l2_item_from_draft(
                parent_item=parent,
                draft=draft,
                new_id=str(len(items) + 1),
            )

            # Tag it clearly as the simplified interactive flow
            new_item.setdefault("ai_actions", {})
            new_item["ai_actions"]["mode"] = "user_intent_then_ai"
            new_item["ai_actions"]["user_intent"] = user_intent
            new_item["ai_actions"]["desired_qtype"] = desired_qtype_norm

            parent_idx = next(
                (i for i, it2 in enumerate(items) if str(it2.get("id")) == str(parent_qid)),
                None,
            )

            if parent_idx is None:
                items.append(new_item)
            else:
                items.insert(parent_idx + 1, new_item)

            for i, it2 in enumerate(items, start=1):
                it2["id"] = str(i)
            payload["items"] = items

            print("✅ Added L2 draft question.")
            continue

        if low.startswith("l2plan "):
            parts = cmd.split()
            if len(parts) != 2:
                print("Usage: l2plan <parent_qid>")
                continue
            parent_qid = parts[1].strip()
            parent = _find_item(payload, parent_qid)
            if not parent:
                print(f"No question found with id={parent_qid}")
                continue

            # If no stored suggestions, generate now
            sug = (parent.get("ai_actions") or {}).get("l2_suggestions")
            if not sug and client is not None:
                try:
                    parent_opts = [{"key": o.get("key"), "label": o.get("label")} for o in (parent.get("answer_options") or []) if isinstance(o, dict)]
                    if not parent_opts:
                        print("Parent has no answer options; cannot plan L2.")
                        continue
                    plan = plan_l2_followups_openai(
                        client,
                        ctx=ctx,
                        parent_construct=str(parent.get("construct") or ""),
                        parent_question_text=str(parent.get("question_text") or ""),
                        parent_question_type=str(parent.get("question_type") or "SingleSelection"),
                        parent_answer_options=parent_opts,
                        max_l2_questions=L2_MAX_DISTINCT,
                    )
                    sug = plan.model_dump() if hasattr(plan, "model_dump") else plan.dict()
                    parent.setdefault("ai_actions", {})
                    parent["ai_actions"]["l2_suggestions"] = sug
                except Exception as e:
                    print("Failed to generate L2 plan:", str(e))
                    continue

            if not sug:
                print("No L2 suggestions available.")
                continue

            enabled = bool(sug.get("enabled", False))
            print("\n--- L2 SUGGESTIONS ---")
            print(f"enabled={enabled}  confidence={sug.get('confidence')}  why={sug.get('overall_why','')}")
            drafts = sug.get("drafts", []) or []
            if not drafts:
                print("(no drafts)")
                continue
            for i, d in enumerate(drafts, start=1):
                print(f"\n[{i}] triggers={d.get('trigger_answer_keys')}")
                print("    why:", d.get("why", ""))
                print("    Q:", d.get("question_text", ""))
                print("    type:", d.get("question_type", ""))
                opts = d.get("answer_options", []) or []
                for o in opts:
                    print("     -", o)
            print("----------------------\n")
            continue

        if low.startswith("l2add "):
            parts = cmd.split()
            if len(parts) != 3:
                print("Usage: l2add <parent_qid> <draft_index>")
                continue
            parent_qid = parts[1].strip()
            try:
                draft_idx = int(parts[2]) - 1
            except Exception:
                print("draft_index must be an integer (1-based).")
                continue

            parent = _find_item(payload, parent_qid)
            if not parent:
                print(f"No question found with id={parent_qid}")
                continue

            sug = (parent.get("ai_actions") or {}).get("l2_suggestions") or {}
            drafts = sug.get("drafts", []) or []
            if not drafts:
                print("No L2 drafts found. Run: l2plan <parent_qid>")
                continue
            if not (0 <= draft_idx < len(drafts)):
                print(f"draft_index out of range (1..{len(drafts)})")
                continue

            draft = drafts[draft_idx]
            items = payload.get("items", [])

            new_item = build_l2_item_from_draft(
                parent_item=parent,
                draft=draft,
                new_id=str(len(items) + 1),
            )

            # Insert right after parent
            parent_idx = next(
                (i for i, it2 in enumerate(items) if str(it2.get("id")) == str(parent_qid)),
                None,
            )

            if parent_idx is None:
                items.append(new_item)
            else:
                items.insert(parent_idx + 1, new_item)

            for i, it2 in enumerate(items, start=1):
                it2["id"] = str(i)
            payload["items"] = items

            print("✅ Added L2 from plan.")
            continue

        print("Unknown command. Type 'help' to see commands.")


# -------------------
# SETUP FORM
# -------------------
def prompt_builder_context() -> BuilderContext:
    print("\n=== Arrival Survey Builder (v10 LLM Selection + Optional Rewrite/Fallback) ===")

    site_purpose = input("In one sentence, what is this site for? (required): ").strip()
    while not site_purpose:
        site_purpose = input("Site purpose is required. Please enter 1–2 sentences: ").strip()

    survey_goal = input("Survey goal (required): ").strip()
    while not survey_goal:
        survey_goal = input("Survey goal is required (e.g., improve clarity, measure intent): ").strip()

    site_category = (input("Site category (Pharma/Ecommerce/Education/SaaS/Content/University) [Pharma]: ").strip() or "Pharma").strip()
    if site_category.lower() == "saas":
        site_category = "SaaS"
    else:
        site_category = site_category.title()

    return BuilderContext(site_purpose=site_purpose, survey_goal=survey_goal, site_category=site_category)


# -------------------
# MAIN
# -------------------
def main():
    df = load_library(LIB_PATH)
    ctx = prompt_builder_context()

    client = None
    want_client = (LLM_SELECT_PER_CONSTRUCT or LLM_REWRITE_SELECTED or LLM_FALLBACK_ON_MISSING or AUTO_GENERATE_L2)
    try:
        if want_client:
            client = make_client()
    except Exception as e:
        if want_client:
            raise
        print("⚠️ OpenAI client not available:", str(e))
        print("   You can still build a draft, but LLM selection/rewrites won't run.")

    payload = build_arrival(df, ctx, client=client)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Exported: {OUT_PATH}")
    print(neat_preview(payload, show_keys=False))

    if client is not None:
        payload = interactive_loop(payload, ctx, client)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Final saved: {OUT_PATH}")
    else:
        print("\n(No OpenAI client available, so interactive refinement was skipped.)")


if __name__ == "__main__":
    main()