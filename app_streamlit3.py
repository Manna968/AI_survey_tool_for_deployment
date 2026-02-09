import os
import json
import streamlit as st
import re
import copy
from typing import List
from typing import List, Optional, Any, Dict

# Import from your existing script file:
from survey_template_0209 import (
    load_library,
    build_arrival,
    neat_preview,
    BuilderContext,
    make_client,
    rewrite_question_text_openai,
    rewrite_answer_options_openai,
    var_name_for_slot,
    canonical_option_key,
)

LIB_PATH = "Question_Rates_Merged_backbone_ready.xlsx"

@st.cache_data
def get_df():
    return load_library(LIB_PATH)

df = get_df()

# ------------------------------------------------------------
# Streamlit config (ONLY ONCE, MUST BE FIRST st.* call)
# ------------------------------------------------------------
st.set_page_config(page_title="Arrival Survey Builder", layout="wide")
st.title("Arrival Survey Builder (Internal)")

# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "payload" not in st.session_state:
    st.session_state.payload = None
if "client" not in st.session_state:
    st.session_state.client = None
if "ctx" not in st.session_state:
    st.session_state.ctx = None
if "logs" not in st.session_state:
    st.session_state.logs = []
if "history" not in st.session_state:
    st.session_state.history = []
if "option_suggestions" not in st.session_state:
    # { qid(str): {"candidates": [labels...], "selected": [labels...], "instruction": str} }
    st.session_state.option_suggestions = {}

if "last_suggest_qid" not in st.session_state:
    st.session_state.last_suggest_qid = None

# Used to detect category changes and reset dependent widgets
if "last_site_category" not in st.session_state:
    st.session_state.last_site_category = None

# -----------------------
# Small helpers (UI-side only)
# -----------------------

def get_ui_options(site_category: str) -> dict:
    cat = (site_category or "").strip().lower()

    if cat in {"education", "university"}:
        return {
            "primary_audience": [
                "Prospective student",
                "Current student",
                "Parent/guardian",
                "Educator / faculty",
                "Administrator / staff",
                "Alumni",
                "Other",
            ],
            "primary_actions": [
                "Explore programs/courses",
                "Check admissions requirements",
                "Compare schools/programs",
                "Request information",
                "Apply / enroll",
                "Find tuition/financial aid info",
                "Download brochures/resources",
                "Contact admissions/support",
                "Other",
            ],
            "domain_placeholder": "e.g., programs, majors, learning differences, tutoring, certifications…",
            "value_placeholder": "e.g., help students compare programs and apply; find resources; request info…",
            "site_name_placeholder": "e.g., Riverside University",
            "extra_context_placeholder": "Tone/constraints (e.g., avoid jargon; emphasize admissions)…",
        }

    if cat == "saas":
        return {
            "primary_audience": [
                "Individual user",
                "Business buyer",
                "Admin / IT",
                "Developer",
                "Partner",
                "Other",
            ],
            "primary_actions": [
                "Explore features",
                "Compare plans",
                "View pricing",
                "Start a trial / sign up",
                "Book a demo / contact sales",
                "Read docs / tutorials",
                "Contact support",
                "Other",
            ],
            "domain_placeholder": "e.g., CRM, analytics, workflow automation, HRIS…",
            "value_placeholder": "e.g., help teams evaluate plans and start a trial; book a demo…",
            "site_name_placeholder": "e.g., Acme CRM",
            "extra_context_placeholder": "Constraints (e.g., enterprise buyers; emphasize security)…",
        }

    if cat == "content":
        return {
            "primary_audience": [
                "General reader",
                "Subscriber/member",
                "Professional researcher",
                "Student",
                "Other",
            ],
            "primary_actions": [
                "Read articles",
                "Find answers/how-tos",
                "Browse topics",
                "Subscribe / join",
                "Download resources",
                "Other",
            ],
            "domain_placeholder": "e.g., home renovation tips, health education, finance guides…",
            "value_placeholder": "e.g., help readers find answers quickly and subscribe…",
            "site_name_placeholder": "e.g., The Renovation Guide",
            "extra_context_placeholder": "Any constraints (e.g., no sales language; neutral tone)…",
        }

    if cat == "pharma":
        return {
            "primary_audience": [
                "Patient",
                "Health care professional (HCP)",
                "Caregiver",
                "Office staff / practice admin",
                "Other",
            ],
            "primary_actions": [
                "Learn about a condition or treatment",
                "Review product information",
                "Find patient support resources",
                "Find dosing/safety info",
                "Talk to a representative",
                "Other",
            ],
            "domain_placeholder": "e.g., diabetes treatment, migraine prevention, oncology support…",
            "value_placeholder": "e.g., help patients understand treatment options and find support…",
            "site_name_placeholder": "e.g., Brandname Support",
            "extra_context_placeholder": "Constraints (e.g., compliance tone; avoid claims)…",
        }

    # Default = Ecommerce (or “Other”)
    return {
        "primary_audience": [
            "Individual consumer",
            "Professional / business buyer",
            "DIY shopper",
            "Contractor / installer",
            "Architect / designer",
            "Mixed / not sure",
            "Other",
        ],
        "primary_actions": [
            "Browse products",
            "Compare options",
            "Use a configurator / planner",
            "Request a quote / contact sales",
            "Find pricing",
            "Download resources",
            "Other",
        ],
        "domain_placeholder": "e.g., custom stairs, flooring, furniture, appliances…",
        "value_placeholder": "e.g., compare options and request a quote; find pricing; plan a project…",
        "site_name_placeholder": "e.g., Paragon Stairs",
        "extra_context_placeholder": "Constraints, tone, things to avoid…",
    }


def reset_multiselect_if_invalid(state_key: str, valid_options: list):
    """If session_state has selections no longer in valid_options, drop invalid selections."""
    cur = st.session_state.get(state_key, [])
    if not cur:
        return
    cur2 = [x for x in cur if x in valid_options]
    if cur2 != cur:
        st.session_state[state_key] = cur2


def hard_reset_on_category_change(new_category: str):
    """
    When site_category changes, clear dependent selections so UI feels "dynamic"
    and users don't carry Ecommerce picks into Education.
    """
    prev = st.session_state.get("last_site_category")
    if prev is None:
        st.session_state.last_site_category = new_category
        return

    if prev != new_category:
        # Clear dependent widget state
        st.session_state.primary_audience = []
        st.session_state.primary_actions = []
        # Optional: clear text inputs if you want a full reset
        # st.session_state.domain_topic = ""
        # st.session_state.core_value = ""
        st.session_state.last_site_category = new_category


def _find_item(payload, qid: str):
    for it in payload.get("items", []):
        if str(it.get("id")) == str(qid):
            return it
    return None


def _dedupe_labels_preserve_order(labels):
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


def _get_candidates_block(it: dict) -> dict:
    """
    Backward-compatible reader for ai_actions["option_candidates"].

    Supports BOTH:
      A) list[str]                           (legacy)
      B) {"generated": [...], "selected": [...], "max_select": 8} (new)

    Returns:
      {"generated": List[str], "selected": List[str], "max_select": int}
    """
    ai = it.get("ai_actions") or {}
    raw = ai.get("option_candidates")

    # Default
    generated: List[str] = []
    selected: List[str] = []
    max_select = 8

    # Case A: legacy list[str]
    if isinstance(raw, list):
        generated = [str(x).strip() for x in raw if x and str(x).strip()]
        # default selection = first N
        # do NOT auto-select by default
        return {"generated": generated, "selected": [], "max_select": max_select}


    # Case B: new dict
    if isinstance(raw, dict):
        generated = raw.get("generated") or []
        selected = raw.get("selected") or []
        max_select = int(raw.get("max_select") or 8)

        # normalize
        generated = [str(x).strip() for x in generated if x and str(x).strip()]
        selected = [str(x).strip() for x in selected if x and str(x).strip()]


        return {"generated": generated, "selected": selected, "max_select": max_select}

    # None/unknown
    return {"generated": [], "selected": [], "max_select": max_select}



def _set_candidates_block(it: dict, *, generated: list, selected: list = None, max_select: int = 8) -> None:
    it.setdefault("ai_actions", {})
    it["ai_actions"]["option_candidates"] = {
        "generated": [x for x in (generated or []) if x and str(x).strip()],
        "selected": [x for x in (selected or []) if x and str(x).strip()] if selected is not None else [],
        "max_select": int(max_select),
    }

def _as_label_list(x) -> list:
    """
    Normalize candidate/applied selections into List[str] labels.
    Accepts:
      - list[str]
      - list[dict] with 'label'
      - dict with 'selected' or 'generated'
      - single str
    """
    if x is None:
        return []
    if isinstance(x, str):
        return [x.strip()] if x.strip() else []
    if isinstance(x, dict):
        # prefer selected if present
        if isinstance(x.get("selected"), list):
            return [str(i).strip() for i in x["selected"] if i and str(i).strip()]
        if isinstance(x.get("generated"), list):
            return [str(i).strip() for i in x["generated"] if i and str(i).strip()]
        return []
    if isinstance(x, list):
        out = []
        for i in x:
            if isinstance(i, str):
                t = i.strip()
                if t:
                    out.append(t)
            elif isinstance(i, dict) and i.get("label"):
                t = str(i["label"]).strip()
                if t:
                    out.append(t)
        return out
    return []



def _set_options_from_labels(it, labels):
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


def suggest_options_for_item(
    client: Any,
    *,
    ctx: Any,              # BuilderContext
    it: Dict[str, Any],    # item dict from payload
    instruction: str = "",
    n: int = 12,
) -> List[str]:
    """
    Generate a candidate pool of answer option labels using the existing rewrite_answer_options_openai
    but in a 'suggest' mode: we don't apply automatically; we just return candidates for the user to pick.

    Returns: list[str] labels (deduped).
    """
    qtext = (it.get("question_text") or "").strip()
    if not qtext:
        return []

    orig_opts = [o.get("label") for o in (it.get("answer_options") or []) if isinstance(o, dict)]
    # If there's no baseline options, give the model a lightweight seed so it doesn't return junk.
    seed = orig_opts if orig_opts else ["Option A", "Option B", "Other"]

    base_instruction = (
        "Propose a candidate pool of answer options (do NOT rewrite the question). "
        "Return short option labels only. "
        "Make them mutually exclusive where possible. "
        "Avoid pharma/healthcare wording unless the site category is Pharma. "
        "Include 'Other' only if useful."
    )
    if instruction.strip():
        base_instruction += " " + instruction.strip()

    # Use your existing function; it returns a list[str]
    cands = rewrite_answer_options_openai(
        client,
        site_purpose=ctx.site_purpose,
        survey_goal=ctx.survey_goal,
        site_category=ctx.site_category,
        question_text=qtext,
        original_options=seed,
        instruction=base_instruction,
        keep_other_if_present=True,
    )

    # Dedupe + cap
    cands = _dedupe_labels_preserve_order(cands)[: max(8, min(16, n))]
    return cands



def _split_pipe(cmd: str, expected_parts: int):
    parts = [p.strip() for p in cmd.split("|")]
    if len(parts) < expected_parts:
        return None
    head = parts[0]
    tail = parts[1:]
    while len(tail) > expected_parts - 1:
        tail[-2] = tail[-2] + " | " + tail[-1]
        tail.pop()
    return [head] + tail


def _bullet_join(xs: List[str]) -> str:
    xs = [x.strip() for x in (xs or []) if x and x.strip()]
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    if len(xs) == 2:
        return f"{xs[0]} and {xs[1]}"
    return ", ".join(xs[:-1]) + f", and {xs[-1]}"


def compose_site_purpose(
    *,
    site_name: str,
    site_type: str,
    primary_audience: List[str],
    domain_topic: str,
    core_value: str,
    primary_actions: List[str],
    extra_context: str,
) -> str:
    aud = _bullet_join(primary_audience) or "visitors"
    domain = domain_topic.strip()
    value = core_value.strip()
    actions = _bullet_join(primary_actions)

    parts = []
    if site_name.strip():
        parts.append(f"{site_name.strip()} is a {site_type.strip().lower()} site")
    else:
        parts.append(f"A {site_type.strip().lower()} site")

    parts.append(f"for {aud.lower()}")

    if domain:
        parts.append(f"focused on {domain}")

    if value:
        parts.append(f"that helps users {value}")
    elif actions:
        parts.append(f"that helps users {actions.lower()}")

    s = " ".join(parts).strip()
    if not s.endswith("."):
        s += "."
    if extra_context.strip():
        ec = extra_context.strip()
        if not ec.endswith("."):
            ec += "."
        s += f" {ec}"
    return s


def compose_survey_goal(
    *,
    goal_type: str,
    goal_details: str,
    identify_roles: bool,
    capture_intent: bool,
    measure_satisfaction: bool,
    find_blockers: bool,
    gauge_readiness: bool,
    desired_next_steps: List[str],
    dont_ask: List[str],
    max_questions_hint: str,
) -> str:
    objectives = []

    primary = goal_type.strip()
    if primary:
        objectives.append(primary)

    if identify_roles:
        objectives.append("understand who’s visiting")
    if capture_intent:
        objectives.append("understand what they’re trying to do / find")
    if measure_satisfaction:
        objectives.append("measure satisfaction")
    if find_blockers:
        objectives.append("identify blockers or confusion")
    if gauge_readiness:
        objectives.append("understand readiness / next step")

    steps = _bullet_join(desired_next_steps)
    if steps:
        objectives.append(f"and how close they are to {steps.lower()}")

    detail = goal_details.strip()
    if detail:
        objectives.append(f"Context: {detail}")

    neg_map = {
        "journey_stage": "Don't ask journey stage.",
        "prior_knowledge": "Don't ask familiarity/prior knowledge.",
        "trigger": "Don't ask trigger.",
    }
    neg_lines = [neg_map[k] for k in dont_ask if k in neg_map]

    mq = max_questions_hint.strip()
    if mq:
        objectives.append(mq)

    base = "Survey goal: " + "; ".join([o for o in objectives if o])
    if not base.endswith("."):
        base += "."
    if neg_lines:
        base += " " + " ".join(neg_lines)
    return base


# -----------------------
# Command console
# -----------------------
def apply_command(payload, ctx, client, cmd: str):
    cmd = (cmd or "").strip()
    if not cmd:
        return payload, "Empty command."

    low = cmd.lower().strip()

    # -----------------------
    # helpers
    # -----------------------
    def _renumber_items(pl):
        for i, it in enumerate(pl.get("items", []), start=1):
            it["id"] = str(i)
        return pl

    def _parse_semicolon_list(s: str) -> list:
        if not s:
            return []
        return [x.strip() for x in s.split(";") if x.strip()]

    def _new_manual_item(new_id: str, question_text: str, option_labels: list) -> dict:
        item = {
            "id": str(new_id),
            "module_key": "custom_manual",
            "construct": "custom_manual",
            "slot": "X_Custom",
            "phase": "Arrival",
            "level": "L1",
            "question_id": f"manual::{new_id}",
            "question_type": "SingleSelect" if option_labels else "OpenText",
            "question_text": (question_text or "").strip(),
            "answer_options": [],
            "display_condition_json": None,
            "display_condition": "",
            "ai_actions": {"source": "manual", "draft": True},
        }
        if option_labels:
            _set_options_from_labels(item, option_labels[:10])
        return item

    def _ensure_has_options(it: dict) -> bool:
        opts = it.get("answer_options") or []
        return any(isinstance(o, dict) and (o.get("label") or "").strip() for o in opts)

    # -----------------------
    # legacy alias
    # -----------------------
    if low.startswith("genl2 "):
        cmd = "l2 " + cmd.split(" ", 1)[1]
        low = cmd.lower()
    

    # =========================================================
    # RULE-BASED: add/delete/edit questions/options
    # =========================================================
    if low.startswith("addq "):
        parts = _split_pipe(cmd, 3)
        if not parts:
            return payload, (
                "Usage:\n"
                "  addq after <qid> | <question_text> | <opt1; opt2; ...>\n"
                "  addq end | <question_text> | <opt1; opt2; ...>"
            )
        head = parts[0].strip()  # "addq after 2" or "addq end"
        qtext = parts[1].strip()
        opt_labels = _parse_semicolon_list(parts[2].strip())

        new_payload = copy.deepcopy(payload)
        items = new_payload.get("items", [])

        head_low = head.lower()
        insert_idx = len(items)
        if head_low.startswith("addq after "):
            toks = head.split()
            if len(toks) != 3:
                return payload, "Usage: addq after <qid> | <question_text> | <opt1; opt2; ...>"
            after_qid = toks[2].strip()
            idx = next((i for i, it in enumerate(items) if str(it.get("id")) == str(after_qid)), None)
            if idx is None:
                return payload, f"No question found with id={after_qid}"
            insert_idx = idx + 1
        elif head_low.strip() == "addq end":
            insert_idx = len(items)
        else:
            return payload, (
                "Usage:\n"
                "  addq after <qid> | <question_text> | <opt1; opt2; ...>\n"
                "  addq end | <question_text> | <opt1; opt2; ...>"
            )

        new_id = str(len(items) + 1)
        items.insert(insert_idx, _new_manual_item(new_id, qtext, opt_labels))
        new_payload["items"] = items
        _renumber_items(new_payload)
        return new_payload, f"✅ Added manual question at position {insert_idx + 1}."

    if low.startswith("delq "):
        parts = cmd.split()
        if len(parts) != 2:
            return payload, "Usage: delq <qid>"
        qid = parts[1].strip()

        new_payload = copy.deepcopy(payload)
        before = len(new_payload.get("items", []))
        new_payload["items"] = [it for it in new_payload.get("items", []) if str(it.get("id")) != str(qid)]
        after = len(new_payload.get("items", []))
        if before == after:
            return payload, f"No question found with id={qid}"

        _renumber_items(new_payload)
        return new_payload, f"🗑️ Deleted Q{qid}."

    if low.startswith("editq "):
        parts = _split_pipe(cmd, 2)
        if not parts:
            return payload, "Usage: editq <qid> | <new question text>"
        head, new_text = parts[0], parts[1]
        hp = head.split()
        if len(hp) != 2:
            return payload, "Usage: editq <qid> | <new question text>"
        qid = hp[1].strip()

        it = _find_item(payload, qid)
        if not it:
            return payload, f"No question found with id={qid}"

        new_payload = copy.deepcopy(payload)
        it2 = _find_item(new_payload, qid)
        it2["question_text"] = new_text.strip()
        it2.setdefault("ai_actions", {})
        it2["ai_actions"]["edited_by_user"] = True
        return new_payload, f"✅ Updated question text for Q{qid}."

    # if low.startswith("addopt "):
    #     parts = _split_pipe(cmd, 2)
    #     if not parts:
    #         return payload, "Usage: addopt <qid> | <label>"
    #     head, label = parts[0], parts[1].strip()
    #     hp = head.split()
    #     if len(hp) != 2:
    #         return payload, "Usage: addopt <qid> | <label>"
    #     qid = hp[1].strip()

    #     it = _find_item(payload, qid)
    #     if not it:
    #         return payload, f"No question found with id={qid}"
    #     if not _ensure_has_options(it):
    #         return payload, "This question has no answer options (maybe OpenText)."

    #     new_payload = copy.deepcopy(payload)
    #     it2 = _find_item(new_payload, qid)
    #     labels = [o.get("label") for o in (it2.get("answer_options") or []) if isinstance(o, dict)]
    #     labels.append(label)
    #     _set_options_from_labels(it2, labels)
    #     return new_payload, f"✅ Added option to Q{qid}."

    # if low.startswith("delopt "):
    #     parts = cmd.split()
    #     if len(parts) != 3:
    #         return payload, "Usage: delopt <qid> <idx> (1-based)"
    #     qid = parts[1].strip()
    #     try:
    #         idx = int(parts[2]) - 1
    #     except Exception:
    #         return payload, "idx must be an integer."

    #     it = _find_item(payload, qid)
    #     if not it:
    #         return payload, f"No question found with id={qid}"
    #     labels = [o.get("label") for o in (it.get("answer_options") or []) if isinstance(o, dict)]
    #     if not labels:
    #         return payload, "No answer options."
    #     if not (0 <= idx < len(labels)):
    #         return payload, f"Index out of range (1..{len(labels)})"
    #     if len(labels) <= 2:
    #         return payload, "Refusing to delete: would leave fewer than 2 options."

    #     new_payload = copy.deepcopy(payload)
    #     it2 = _find_item(new_payload, qid)
    #     labels2 = [o.get("label") for o in (it2.get("answer_options") or []) if isinstance(o, dict)]
    #     removed = labels2.pop(idx)
    #     _set_options_from_labels(it2, labels2)
    #     return new_payload, f"🗑️ Deleted option '{removed}' from Q{qid}."

    # if low.startswith("editopt "):
    #     parts = _split_pipe(cmd, 2)
    #     if not parts:
    #         return payload, "Usage: editopt <qid> <idx> | <new label>"
    #     head, new_label = parts[0], parts[1].strip()
    #     hp = head.split()
    #     if len(hp) != 3:
    #         return payload, "Usage: editopt <qid> <idx> | <new label>"
    #     qid = hp[1].strip()
    #     try:
    #         idx = int(hp[2]) - 1
    #     except Exception:
    #         return payload, "idx must be an integer."

    #     it = _find_item(payload, qid)
    #     if not it:
    #         return payload, f"No question found with id={qid}"
    #     labels = [o.get("label") for o in (it.get("answer_options") or []) if isinstance(o, dict)]
    #     if not (0 <= idx < len(labels)):
    #         return payload, f"Index out of range (1..{len(labels)})"

    #     new_payload = copy.deepcopy(payload)
    #     it2 = _find_item(new_payload, qid)
    #     labels2 = [o.get("label") for o in (it2.get("answer_options") or []) if isinstance(o, dict)]
    #     labels2[idx] = new_label
    #     _set_options_from_labels(it2, labels2)
    #     return new_payload, f"✅ Updated option {idx+1} for Q{qid}."

    # =========================================================
    # AI: suggest candidates (store), show, apply selected, clear
    # =========================================================
    # if low.startswith("suggestopts "):
    #     parts = _split_pipe(cmd, 2)
    #     if not parts:
    #         return payload, "Usage: suggestopts <qid> | <instruction>"
    #     head, instruction = parts[0], parts[1]
    #     hp = head.split()
    #     if len(hp) < 2:
    #         return payload, "Usage: suggestopts <qid> | <instruction>"
    #     qid = hp[1].strip()

    #     it = _find_item(payload, qid)
    #     if not it:
    #         return payload, f"No question found with id={qid}"
    #     if client is None:
    #         return payload, "OpenAI client unavailable (enable key + LLM)."

    #     orig_opts = [o.get("label") for o in (it.get("answer_options") or []) if isinstance(o, dict)]
    #     if not orig_opts:
    #         return payload, "No answer options to extend for this question."

    #     instruction2 = (
    #         "Generate additional answer options that fit the question. "
    #         "Do NOT remove or rewrite the existing options. "
    #         "Return a list that includes the original options plus new ones. "
    #         "Keep them short, mutually exclusive, and practical. "
    #         f"User instruction: {instruction}"
    #     )

    #     expanded = rewrite_answer_options_openai(
    #         client,
    #         site_purpose=ctx.site_purpose,
    #         survey_goal=ctx.survey_goal,
    #         site_category=ctx.site_category,
    #         question_text=it.get("question_text", ""),
    #         original_options=orig_opts,
    #         instruction=instruction2,
    #         keep_other_if_present=True,
    #     )
    #     expanded = _dedupe_labels_preserve_order(expanded)

    #     # Only keep *new* candidates
    #     existing_norm = {x.strip().lower() for x in orig_opts}
    #     cands = []
    #     for x in expanded:
    #         t = (x or "").strip()
    #         if not t:
    #             continue
    #         if t.lower() in existing_norm:
    #             continue
    #         if t.lower() in {c.lower() for c in cands}:
    #             continue
    #         cands.append(t)

    #     if not cands:
    #         return payload, "No new candidates generated (try a different instruction)."

    #     new_payload = copy.deepcopy(payload)
    #     it2 = _find_item(new_payload, qid)
    #     it2.setdefault("ai_actions", {})
    #     _set_candidates_block(it2, generated=cands[:20], selected=[], max_select=8)
    #     it2["ai_actions"]["option_candidates_instruction"] = instruction.strip()

    #     return new_payload, f"✅ Saved {min(len(cands),20)} candidate options for Q{qid}. Use: showcands {qid} then applycands {qid} 1,2,..."

    # if low.startswith("showcands "):
    #     parts = cmd.split()
    #     if len(parts) != 2:
    #         return payload, "Usage: showcands <qid>"
    #     qid = parts[1].strip()

    #     it = _find_item(payload, qid)
    #     if not it:
    #         return payload, f"No question found with id={qid}"

    #     c = _get_candidates_block(it)
    #     cands = c["generated"] or []
    #     if not cands:
    #         return payload, f"No candidates stored for Q{qid}. Run: suggestopts {qid} | <instruction>"

    #     out = [f"Candidates for Q{qid} (max_select={c['max_select']}):"]
    #     for i, lab in enumerate(cands, start=1):
    #         out.append(f" {i}. {lab}")
    #     return payload, "\n".join(out)


    # if low.startswith("applycands "):
    #     parts = cmd.split(" ", 2)
    #     if len(parts) != 3:
    #         return payload, "Usage: applycands <qid> <idx1,idx2,...> (1-based from candidates list)"
    #     qid = parts[1].strip()
    #     idxs_raw = parts[2].strip()

    #     it = _find_item(payload, qid)
    #     if not it:
    #         return payload, f"No question found with id={qid}"

    #     # option_candidates may be:
    #     #   - list[str]  (your current format)
    #     #   - dict {"generated":[...], "selected":[...], "max_select":8} (older/newer UI format)
    #     oc = (it.get("ai_actions") or {}).get("option_candidates") or []
    #     if isinstance(oc, dict):
    #         cands = oc.get("generated") or []
    #     else:
    #         cands = oc

    #     if not isinstance(cands, list) or not cands:
    #         return payload, f"No candidates found for Q{qid}. Run: suggestopts {qid} | <instruction>"

    #     try:
    #         idxs = [int(x.strip()) - 1 for x in idxs_raw.split(",") if x.strip()]
    #     except Exception:
    #         return payload, "Bad index list."

    #     if not idxs:
    #         return payload, "No indices provided."
    #     if any(i < 0 or i >= len(cands) for i in idxs):
    #         return payload, f"Index out of range (1..{len(cands)})"

    #     # Selected candidate labels
    #     chosen = [str(cands[i]).strip() for i in idxs if str(cands[i]).strip()]
    #     chosen = _dedupe_labels_preserve_order(chosen)
    #     if len(chosen) < 1:
    #         return payload, "No valid candidates selected."

    #     new_payload = copy.deepcopy(payload)
    #     it2 = _find_item(new_payload, qid)

    #     # Merge into existing options (do NOT wipe the originals)
    #     existing = [o.get("label") for o in (it2.get("answer_options") or []) if isinstance(o, dict)]
    #     merged = _dedupe_labels_preserve_order(existing + chosen)

    #     # Keep a reasonable cap (optional)
    #     merged = merged[:10]  # adjust cap if you want

    #     _set_options_from_labels(it2, merged)

    #     it2.setdefault("ai_actions", {})
    #     it2["ai_actions"]["applied_from_candidates"] = {
    #         "added": chosen,
    #         "result": merged,
    #     }

    #     # If option_candidates is a dict-format block, also persist "selected" for UI
    #     if isinstance(oc, dict):
    #         it2["ai_actions"]["option_candidates"]["selected"] = chosen

    #     return new_payload, f"✅ Added {len(chosen)} candidate option(s) to Q{qid} (now {len(merged)} total)."


    # if low.startswith("clearcands "):
    #     parts = cmd.split()
    #     if len(parts) != 2:
    #         return payload, "Usage: clearcands <qid>"
    #     qid = parts[1].strip()

    #     it = _find_item(payload, qid)
    #     if not it:
    #         return payload, f"No question found with id={qid}"

    #     new_payload = copy.deepcopy(payload)
    #     it2 = _find_item(new_payload, qid)
    #     it2.setdefault("ai_actions", {})
    #     it2["ai_actions"].pop("option_candidates", None)
    #     it2["ai_actions"].pop("option_candidates_instruction", None)
    #     it2["ai_actions"].pop("applied_from_candidates", None)
    #     return new_payload, f"✅ Cleared candidate options for Q{qid}."

    # -----------------------
    # reorder/move (existing)
    # -----------------------
    if low.startswith("moveopt "):
        parts = cmd.split()
        if len(parts) != 4:
            return payload, "Usage: moveopt <qid> <from_idx> <to_idx> (1-based)"
        it = _find_item(payload, parts[1])
        if not it:
            return payload, "No question found."
        labels = [o.get("label") for o in (it.get("answer_options") or []) if isinstance(o, dict)]
        try:
            a = int(parts[2]) - 1
            b = int(parts[3]) - 1
        except Exception:
            return payload, "Indices must be integers."
        if not (0 <= a < len(labels) and 0 <= b < len(labels)):
            return payload, f"Index out of range (1..{len(labels)})"
        new_payload = copy.deepcopy(payload)
        it2 = _find_item(new_payload, parts[1])
        labels2 = [o.get("label") for o in (it2.get("answer_options") or []) if isinstance(o, dict)]
        x = labels2.pop(a)
        labels2.insert(b, x)
        _set_options_from_labels(it2, labels2)
        return new_payload, f"✅ Moved option {parts[2]} → {parts[3]} for Q{parts[1]}."

    if low.startswith("reorderopts "):
        parts = cmd.split(" ", 2)
        if len(parts) != 3:
            return payload, "Usage: reorderopts <qid> <idx1,idx2,...> (1-based)"
        qid = parts[1].strip()
        it = _find_item(payload, qid)
        if not it:
            return payload, "No question found."
        labels = [o.get("label") for o in (it.get("answer_options") or []) if isinstance(o, dict)]
        try:
            order = [int(x.strip()) - 1 for x in parts[2].split(",")]
        except Exception:
            return payload, "Bad index list."
        if sorted(order) != list(range(len(labels))):
            return payload, f"Must include each index exactly once (1..{len(labels)})"
        new_payload = copy.deepcopy(payload)
        it2 = _find_item(new_payload, qid)
        labels2 = [o.get("label") for o in (it2.get("answer_options") or []) if isinstance(o, dict)]
        new_labels = [labels2[i] for i in order]
        _set_options_from_labels(it2, new_labels)
        return new_payload, f"✅ Reordered options for Q{qid}."

    # -----------------------
    # LLM rewrite (existing)
    # -----------------------
    if low.startswith("tuneq "):
        parts = _split_pipe(cmd, 2)
        if not parts:
            return payload, "Usage: tuneq <qid> | <instruction>"
        head, instruction = parts[0], parts[1]
        hp = head.split()
        if len(hp) < 2:
            return payload, "Usage: tuneq <qid> | <instruction>"
        qid = hp[1].strip()
        it = _find_item(payload, qid)
        if not it:
            return payload, f"No question found with id={qid}"
        if client is None:
            return payload, "OpenAI client unavailable (enable key + LLM)."
        new_payload = copy.deepcopy(payload)
        it2 = _find_item(new_payload, qid)
        it2["question_text"] = rewrite_question_text_openai(
            client,
            site_purpose=ctx.site_purpose,
            survey_goal=ctx.survey_goal,
            site_category=ctx.site_category,
            original_question_text=it2.get("question_text", ""),
            instruction=instruction,
        )
        return new_payload, f"✅ Updated question text for Q{qid}."

    if low.startswith("tuneopts "):
        parts = _split_pipe(cmd, 2)
        if not parts:
            return payload, "Usage: tuneopts <qid> | <instruction>"
        head, instruction = parts[0], parts[1]
        hp = head.split()
        if len(hp) < 2:
            return payload, "Usage: tuneopts <qid> | <instruction>"
        qid = hp[1].strip()
        it = _find_item(payload, qid)
        if not it:
            return payload, f"No question found with id={qid}"
        if client is None:
            return payload, "OpenAI client unavailable (enable key + LLM)."
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
        new_payload = copy.deepcopy(payload)
        it2 = _find_item(new_payload, qid)
        _set_options_from_labels(it2, new_opts)
        return new_payload, f"✅ Updated answer options for Q{qid}."

    return payload, "Unknown command."



# -----------------------
# Sidebar: Config
# -----------------------
st.sidebar.header("Config")
st.sidebar.markdown(
    "- Fill the structured **Build** form\n"
    "- Click **Build survey**\n"
    "- View output in **Survey Preview**\n"
    "- Refine in **Command Console**\n"
)

lib_path = st.sidebar.text_input(
    "Library path (xlsx)",
    value=os.getenv("LIB_PATH", "Question_Rates_Merged_backbone_ready.xlsx"),
)
out_name = st.sidebar.text_input("Output filename", value="draft_survey_arrival_v10.json")

st.sidebar.subheader("LLM toggles (env)")
def env_toggle(label, env_key, default="1"):
    cur = os.getenv(env_key, default).strip().lower() in {"1", "true", "yes", "y"}
    val = st.sidebar.checkbox(label, value=cur)
    os.environ[env_key] = "1" if val else "0"
    return val

LLM_SELECT_PER_CONSTRUCT = env_toggle("LLM select per construct", "LLM_SELECT_PER_CONSTRUCT", "1")
LLM_REWRITE_SELECTED = env_toggle("Rewrite selected question/options", "LLM_REWRITE_SELECTED", "1")
LLM_FALLBACK_ON_MISSING = env_toggle("Fallback generate if missing", "LLM_FALLBACK_ON_MISSING", "1")
AUTO_GENERATE_L2 = env_toggle("Auto-generate L2 (if needed)", "AUTO_GENERATE_L2", "0")
LLM_PLAN_CONSTRUCTS = env_toggle("LLM plan optionals (honor negation)", "LLM_PLAN_CONSTRUCTS", "1")
STRICT_DEPHARMA = env_toggle("Strict de-pharma wording", "STRICT_DEPHARMA", "1")

st.sidebar.subheader("OpenAI")
model = st.sidebar.text_input("OPENAI_MODEL", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
temp = st.sidebar.slider(
    "OPENAI_TEMPERATURE",
    min_value=0.0,
    max_value=1.0,
    value=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
    step=0.05,
)
os.environ["OPENAI_MODEL"] = model
os.environ["OPENAI_TEMPERATURE"] = str(temp)

key_mode = st.sidebar.radio("API key source", ["Use env var", "Paste in UI"], index=0)
if key_mode == "Paste in UI":
    ui_key = st.sidebar.text_input("OPENAI_API_KEY", type="password", value="")
    if ui_key:
        os.environ["OPENAI_API_KEY"] = ui_key

@st.cache_data(show_spinner=False)
def cached_load_library(path: str):
    return load_library(path)

# Tabs
tab_build, tab_preview, tab_console, tab_advanced = st.tabs(
    ["Build", "Survey Preview", "Command Console", "Advanced"]
)

# -----------------------
# BUILD TAB
# -----------------------
with tab_build:
    st.subheader("Website basics")

    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        # placeholder depends on site_category, so set it later after category selection
        site_name = st.text_input("Site name (optional)", placeholder="e.g., Paragon Stairs", key="site_name")

    with c2:
        site_category = st.selectbox(
            "Site category",
            ["Pharma", "Ecommerce", "Education", "SaaS", "Content", "University"],
            index=1,
            key="site_category",
        )

    # Hard reset dependent widgets when category changes (makes UI feel truly dynamic)
    hard_reset_on_category_change(site_category)

    ui = get_ui_options(site_category)

    with c3:
        site_type = st.selectbox(
            "Site type (experience on arrival)",
            [
                "Marketing / brand site",
                "Product browsing & pricing",
                "Configurator / planning tool",
                "Support / help center",
                "Account / logged-in experience",
                "Other",
            ],
            index=1,
            key="site_type",
        )

    # Ensure selections stay valid when category changes
    reset_multiselect_if_invalid("primary_audience", ui["primary_audience"])
    reset_multiselect_if_invalid("primary_actions", ui["primary_actions"])

    primary_audience = st.multiselect(
        "Primary audience (who is the main visitor?)",
        ui["primary_audience"],
        default=st.session_state.get("primary_audience", []),
        key="primary_audience",
    )

    domain_topic = st.text_input(
        "Topic / domain (what is it about?)",
        placeholder=ui["domain_placeholder"],
        key="domain_topic",
    )

    core_value = st.text_input(
        "Core value (what does it help people do?)",
        placeholder=ui["value_placeholder"],
        key="core_value",
    )

    primary_actions = st.multiselect(
        "Primary actions (what visitors do here?)",
        ui["primary_actions"],
        default=st.session_state.get("primary_actions", []),
        key="primary_actions",
    )

    extra_context = st.text_area(
        "Extra context (optional)",
        height=80,
        placeholder=ui.get("extra_context_placeholder", "Constraints, tone, things to avoid…"),
        key="extra_context",
    )

    st.divider()
    st.subheader("Survey goal builder")

    goal_type = st.selectbox(
        "Primary goal (pick one)",
        [
            "Understand what visitors are trying to do",
            "Understand who’s visiting",
            "Identify blockers / confusion",
            "Gauge readiness / next step",
            "Measure satisfaction",
            "Mixed / general understanding",
        ],
        index=0,
        key="goal_type",
    )

    st.markdown("**What do you want to measure?** (optional)")
    identify_roles = st.checkbox("Who’s visiting (role)", value=True, key="identify_roles")
    capture_intent = st.checkbox("What they want (intent)", value=True, key="capture_intent")
    measure_satisfaction = st.checkbox("Satisfaction", value=False, key="measure_satisfaction")
    find_blockers = st.checkbox("Blockers/confusion", value=False, key="find_blockers")
    gauge_readiness = st.checkbox("Readiness/next step", value=True, key="gauge_readiness")

    desired_next_steps = st.multiselect(
        "Desired next step (optional)",
        ["request a quote", "contact sales", "browse options", "purchase", "sign up", "other"],
        default=[],
        key="desired_next_steps",
    )

    dont_ask = st.multiselect(
        "Do NOT ask about (optional)",
        ["journey_stage", "prior_knowledge", "trigger"],
        default=[],
        key="dont_ask",
    )

    length_pref = st.selectbox(
        "Survey length preference",
        ["Keep it short (3)", "Standard (4–5)", "Longer (5+)"],
        index=1,
        key="length_pref",
    )
    if length_pref.startswith("Keep"):
        mq_text = "Prefer 3 questions if possible."
    elif length_pref.startswith("Standard"):
        mq_text = "Prefer 4–5 questions if possible."
    else:
        mq_text = "Okay to include more than 5 questions if needed."

    goal_details = st.text_area("Goal details (optional)", height=80, key="goal_details")

    composed_purpose = compose_site_purpose(
        site_name=site_name,
        site_type=site_type,
        primary_audience=primary_audience,
        domain_topic=domain_topic,
        core_value=core_value,
        primary_actions=primary_actions,
        extra_context=extra_context,
    )
    composed_goal = compose_survey_goal(
        goal_type=goal_type,
        goal_details=goal_details,
        identify_roles=identify_roles,
        capture_intent=capture_intent,
        measure_satisfaction=measure_satisfaction,
        find_blockers=find_blockers,
        gauge_readiness=gauge_readiness,
        desired_next_steps=desired_next_steps,
        dont_ask=dont_ask,
        max_questions_hint=mq_text,
    )

    st.divider()
    st.subheader("What will be sent to the builder")
    pcol, gcol = st.columns(2)
    with pcol:
        st.caption("Composed Site Purpose")
        st.code(composed_purpose, language="text")
    with gcol:
        st.caption("Composed Survey Goal")
        st.code(composed_goal, language="text")

    st.divider()
    build_btn = st.button("Build survey", type="primary")

    if build_btn:
        try:
            df = cached_load_library(lib_path)
        except Exception as e:
            st.error(f"Failed to load library: {e}")
            st.stop()

        llm_needed = any([
            LLM_SELECT_PER_CONSTRUCT,
            LLM_REWRITE_SELECTED,
            LLM_FALLBACK_ON_MISSING,
            AUTO_GENERATE_L2,
            LLM_PLAN_CONSTRUCTS,
        ])

        client = None
        if llm_needed:
            try:
                client = make_client()
            except Exception as e:
                st.error(f"OpenAI client unavailable: {e}")
                st.stop()

        ctx = BuilderContext(
            site_purpose=composed_purpose.strip(),
            survey_goal=composed_goal.strip(),
            site_category=site_category,
        )

        with st.spinner("Building…"):
            payload = build_arrival(df, ctx, client=client)

        st.session_state.payload = payload
        st.session_state.client = client
        st.session_state.ctx = ctx

        st.success("Built! Go to **Survey Preview** or **Command Console**.")

# -----------------------
# PREVIEW TAB
# -----------------------
with tab_preview:
    st.subheader("Neat preview")
    if st.session_state.payload is None:
        st.info("Build a survey in the **Build** tab first.")
    else:
        # 1) Existing neat preview
        st.code(neat_preview(st.session_state.payload, show_keys=False), language="text")

        blob = json.dumps(st.session_state.payload, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            label=f"Download {out_name}",
            data=blob,
            file_name=out_name,
            mime="application/json",
        )

        # 2) NEW: Option Review (Suggest → Select → Apply)
        st.divider()
        st.subheader("Option Review (Suggest → Select → Apply)")

        items = st.session_state.payload.get("items", []) or []
        any_found = False

        for it in items:
            c = _get_candidates_block(it)
            if not c["generated"]:
                continue

            any_found = True
            qid = it.get("id")
            qtext = (it.get("question_text") or "").strip()
            slot = it.get("slot", "")
            construct = it.get("construct", "")

            with st.expander(f"Q{qid} • {construct} • {slot}", expanded=False):
                if qtext:
                    st.markdown(qtext)
                else:
                    st.markdown("_(no question text)_")

                selected = st.multiselect(
                    "Select the answer options to keep",
                    options=c["generated"],
                    default=c["selected"],
                    key=f"cand_select_{qid}",
                )

                # soft max cap
                if len(selected) > c["max_select"]:
                    st.warning(f"Please select at most {c['max_select']} options.")
                    selected = selected[: c["max_select"]]

                colA, colB = st.columns([1, 1])
                with colA:
                    if st.button("Apply selected to survey output", key=f"apply_cands_btn_{qid}"):
                        new_payload = copy.deepcopy(st.session_state.payload)
                        it2 = _find_item(new_payload, qid)

                        _set_candidates_block(
                            it2,
                            generated=c["generated"],
                            selected=selected,
                            max_select=c["max_select"],
                        )
                        _set_options_from_labels(it2, selected)

                        st.session_state.payload = new_payload

                        st.success("Applied! The preview above reflects your updated options.")
                        st.rerun()


                with colB:
                    if st.button("Clear candidates", key=f"clear_cands_btn_{qid}"):
                        new_payload = copy.deepcopy(st.session_state.payload)
                        it2 = _find_item(new_payload, qid)
                        it2.setdefault("ai_actions", {})
                        it2["ai_actions"].pop("option_candidates", None)
                        it2["ai_actions"].pop("option_candidates_instruction", None)
                        it2["ai_actions"].pop("applied_from_candidates", None)

                        st.session_state.payload = new_payload

                        preview_key = f"cand_select_{qid}"
                        if preview_key in st.session_state:
                            del st.session_state[preview_key]

                        st.rerun()




        if not any_found:
            st.info(
                "No candidate options stored yet. "
                "Use **Command Console → AI option recommender → Suggest options** first."
            )


# -----------------------
# COMMAND CONSOLE TAB
# -----------------------
with tab_console:
    st.subheader("Command Console")

    # --- Command catalog (UI only) ---
    RULE_BASED = [
        ("moveopt <qid> <from_idx> <to_idx>", "Move an option (1-based)"),
        ("reorderopts <qid> <idx1,idx2,...>", "Reorder options by index (1-based)"),
        ("editq <qid> | <new question text>", "Edit question text"),
        ("addq after <qid> | <question_text> | <opt1; opt2; ...>", "Insert a new manual question after Q<qid>"),
        ("addq end | <question_text> | <opt1; opt2; ...>", "Append a new manual question"),
        ("delq <qid>", "Delete a question"),
    ]

    AI_ASSISTED = [
        ("tuneq <qid> | <instruction>", "Rewrite question text (LLM; applies immediately)"),
        ("tuneopts <qid> | <instruction>", "Rewrite all options (LLM; applies immediately)"),
    ]

    if st.session_state.payload is None:
        st.info("Build a survey in the **Build** tab first.")
    else:
        payload = st.session_state.payload

        # ==========================================================
        # 1) Command input (TOP) + reference right above it
        # ==========================================================
        st.markdown("## Command input (power-user)")
        st.caption("Run typed commands. Reference is right here so you don’t have to scroll.")

        with st.expander("Command reference", expanded=True):
            st.markdown("### Rule-based")
            for cmd_ref, desc in RULE_BASED:
                st.markdown(f"- `{cmd_ref}` — {desc}")

            st.markdown("### AI-assisted")
            for cmd_ref, desc in AI_ASSISTED:
                st.markdown(f"- `{cmd_ref}` — {desc}")

        cmd_col, hist_col = st.columns([2, 1], gap="large")

        with cmd_col:
            cmd = st.text_input(
                "Command",
                placeholder=(
                    "Examples:\n"
                    "  addq after 2 | What matters most? | Price; Style; Install time; Other\n"
                    "  delq 4\n"
                    "  tuneq 1 | Make it shorter"
                ),
                key="cmd_console_power",
            )

            run = st.button("Run command", key="run_cmd_console_power")

            if run:
                if not cmd.strip():
                    st.warning("Please enter a command.")
                else:
                    new_payload, msg = apply_command(
                        st.session_state.payload,
                        st.session_state.ctx,
                        st.session_state.client,
                        cmd,
                    )
                    st.session_state.payload = new_payload
                    st.session_state.history.append(cmd)
                    st.session_state.logs.append(msg)

                    # ✅ Don’t dump full survey output here; just status
                    if msg.lower().startswith("unknown command"):
                        st.error(msg)
                    else:
                        st.success(msg)

        with hist_col:
            st.markdown("**History**")
            for h in reversed(st.session_state.history[-25:]):
                st.code(h, language="text")

        if st.session_state.logs:
            st.markdown("**Recent logs**")
            st.code("\n\n".join(st.session_state.logs[-10:]), language="text")

        st.divider()

        # ==========================================================
        # 2) AI option recommender (interactive) (BELOW)
        # ==========================================================
        st.markdown("## AI option recommender (interactive)")
        st.caption("Suggest candidate answer options with AI, then append selected ones. You can also delete current options inline.")

        items = payload.get("items", []) or []

        # Build question choices for dropdown
        q_choices = []
        for it in items:
            qid = str(it.get("id", "")).strip()
            qtext = (it.get("question_text") or "").strip()
            if qid:
                label = f"Q{qid}: {qtext[:90]}{'…' if len(qtext) > 90 else ''}"
                q_choices.append((qid, label))

        if not q_choices:
            st.warning("No questions found in payload.")
        else:
            qid_to_label = {qid: label for qid, label in q_choices}

            selected_qid = st.selectbox(
                "Select a question",
                options=[qid for qid, _ in q_choices],
                format_func=lambda qid: qid_to_label.get(qid, qid),
                key="ai_selected_qid_console",
            )

            sel_item = _find_item(payload, selected_qid)
            if not sel_item:
                st.error("Could not find the selected question.")
            else:
                st.markdown("### Current answer options")

                cur_opts = [
                    o.get("label")
                    for o in (sel_item.get("answer_options") or [])
                    if isinstance(o, dict) and (o.get("label") or "").strip()
                ]
                cur_opts = _dedupe_labels_preserve_order(cur_opts)

                if not cur_opts:
                    st.info("This question currently has no answer options (might be OpenText).")
                else:
                    for i, lab in enumerate(cur_opts):
                        row_key = f"delopt_ui_{selected_qid}_{i}"
                        c_text, c_edit, c_save, c_del = st.columns([5, 3, 1, 1])
                        with c_text:
                            st.write(lab)
                        
                        with c_edit:
                            new_lab = st.text_input(
                                "Rename",
                                value=lab,
                                key=f"rename_opt_{selected_qid}_{i}",
                                label_visibility="collapsed",
                            )

                        with c_save:
                            if st.button("💾", key=f"save_rename_{selected_qid}_{i}", help="Save rename"):
                                new_payload = copy.deepcopy(payload)
                                it2 = _find_item(new_payload, selected_qid)

                                labels2 = [o.get("label") for o in (it2.get("answer_options") or []) if isinstance(o, dict)]
                                labels2 = _dedupe_labels_preserve_order(labels2)

                                # replace only the i-th option label (by position in cur_opts)
                                labels2[i] = new_lab.strip() if new_lab.strip() else labels2[i]
                                _set_options_from_labels(it2, labels2)

                                st.session_state.payload = new_payload
                                st.session_state.logs.append(f"✅ Renamed option {i+1} in Q{selected_qid}.")
                                st.rerun()
                        with c_del:  # ✅ was c_btn
                            if st.button("🗑️", key=row_key, help="Delete this option"):
                                new_payload = copy.deepcopy(payload)
                                it2 = _find_item(new_payload, selected_qid)

                                labels2 = [
                                    o.get("label")
                                    for o in (it2.get("answer_options") or [])
                                    if isinstance(o, dict)
                                ]
                                labels2 = _dedupe_labels_preserve_order(labels2)
                                labels2 = [x for x in labels2 if x != lab]

                                if len(labels2) < 2:
                                    st.warning("Refusing to delete: would leave fewer than 2 options.")
                                else:
                                    _set_options_from_labels(it2, labels2)
                                    it2.setdefault("ai_actions", {})
                                    it2["ai_actions"]["deleted_option_labels"] = (
                                        (it2["ai_actions"].get("deleted_option_labels") or []) + [lab]
                                    )

                                    st.session_state.payload = new_payload
                                    st.session_state.history.append(f"[UI] del option '{lab}' from Q{selected_qid}")
                                    st.session_state.logs.append(f"🗑️ Deleted option '{lab}' from Q{selected_qid}.")
                                    payload = new_payload
                                    st.rerun()
                    

                st.divider()
                st.markdown("### Add options")

                add_col1, add_col2 = st.columns([3, 1])
                with add_col1:
                    manual_new_label = st.text_input(
                        "Add a new option (manual)",
                        placeholder="e.g., Supplier / Distributor",
                        key=f"manual_add_opt_{selected_qid}",
                    )
                with add_col2:
                    add_manual_btn = st.button("Add", key=f"btn_add_manual_{selected_qid}")

                if add_manual_btn:
                    if not manual_new_label.strip():
                        st.error("Please enter an option label.")
                    else:
                        new_payload = copy.deepcopy(payload)
                        it2 = _find_item(new_payload, selected_qid)

                        labels2 = [
                            o.get("label")
                            for o in (it2.get("answer_options") or [])
                            if isinstance(o, dict)
                        ]
                        merged = _dedupe_labels_preserve_order(labels2 + [manual_new_label.strip()])
                        _set_options_from_labels(it2, merged)

                        st.session_state.payload = new_payload
                        st.session_state.history.append(f"[UI] add option '{manual_new_label.strip()}' to Q{selected_qid}")
                        st.session_state.logs.append(f"✅ Added option '{manual_new_label.strip()}' to Q{selected_qid}.")
                        payload = new_payload
                        st.rerun()

                st.divider()
                st.markdown("### AI suggestions (append)")

                c1, c2 = st.columns([3, 1])
                with c1:
                    suggest_instruction = st.text_input(
                        "Instruction (optional)",
                        value=(sel_item.get("ai_actions") or {}).get("option_candidates_instruction", ""),
                        placeholder="e.g., Add more role options; keep short; avoid overlap",
                        key="ai_suggest_instruction_console",
                    )
                with c2:
                    run_suggest = st.button("Suggest options", key="btn_suggest_opts_console")

                if run_suggest:
                    if st.session_state.client is None:
                        st.error("OpenAI client unavailable (enable key + LLM).")
                    else:
                        new_payload = copy.deepcopy(payload)
                        it2 = _find_item(new_payload, selected_qid)

                        cands = suggest_options_for_item(
                            st.session_state.client,
                            ctx=st.session_state.ctx,
                            it=it2,
                            instruction=suggest_instruction,
                            n=12,
                        )

                        if not cands:
                            st.warning("No candidates generated. Try a different instruction.")
                        else:
                            _set_candidates_block(it2, generated=cands[:20], selected=[], max_select=8)
                            it2.setdefault("ai_actions", {})
                            it2["ai_actions"]["option_candidates_instruction"] = suggest_instruction.strip()
                            st.session_state.payload = new_payload

                            cand_key = f"ai_cand_multiselect_{selected_qid}"
                            if cand_key in st.session_state:
                                del st.session_state[cand_key]

                        st.rerun()

                sel_item = _find_item(payload, selected_qid)
                cand_block = _get_candidates_block(sel_item)
                cands = cand_block["generated"]

                if cands:
                    st.markdown("**Candidate options (select what to append)**")
                    cand_key = f"ai_cand_multiselect_{selected_qid}"
                    chosen = st.multiselect(
                        "Select candidates to append",
                        options=cands,
                        default=[],
                        key=cand_key,
                    )

                    b1, b2, b3 = st.columns([1, 1, 2])
                    with b1:
                        append_selected = st.button("Append selected", key="btn_append_cands_console")
                    with b2:
                        clear_candidates = st.button("Clear candidates", key="btn_clear_cands_console")
                    with b3:
                        st.caption("Append will merge your selections into existing options (deduped).")

                    if append_selected:
                        if not chosen:
                            st.error("Please select at least 1 candidate to append.")
                        else:
                            new_payload = copy.deepcopy(payload)
                            it2 = _find_item(new_payload, selected_qid)

                            existing = [
                                o.get("label")
                                for o in (it2.get("answer_options") or [])
                                if isinstance(o, dict)
                            ]
                            merged = _dedupe_labels_preserve_order(existing + chosen)[:12]
                            _set_options_from_labels(it2, merged)

                            it2.setdefault("ai_actions", {})
                            it2["ai_actions"]["applied_from_candidates"] = {"added": chosen, "result": merged}
                            _set_candidates_block(
                                it2,
                                generated=cands,
                                selected=chosen,
                                max_select=cand_block["max_select"],
                            )

                            st.session_state.payload = new_payload
                            st.session_state.history.append(f"[UI] append {len(chosen)} candidates to Q{selected_qid}")
                            st.session_state.logs.append(f"✅ Appended {len(chosen)} candidate option(s) to Q{selected_qid}.")
                            payload = new_payload
                            st.rerun()

                    if clear_candidates:
                        new_payload = copy.deepcopy(payload)
                        it2 = _find_item(new_payload, selected_qid)
                        it2.setdefault("ai_actions", {})
                        it2["ai_actions"].pop("option_candidates", None)
                        it2["ai_actions"].pop("option_candidates_instruction", None)
                        it2["ai_actions"].pop("applied_from_candidates", None)

                        st.session_state.payload = new_payload

                        if cand_key in st.session_state:
                            del st.session_state[cand_key]

                        st.rerun()
                else:
                    st.info("No candidates stored yet. Click **Suggest options** to generate candidates.")




        st.divider()

        # ==========================================================
        # B) Power-user command input (text)
        # ==========================================================
        # st.markdown("## Command input (power-user)")
        # st.caption("Use this if you prefer typed commands or want advanced operations (insert/delete, etc.).")

        # c1, c2 = st.columns([2, 1])

        # with c1:
        #     cmd = st.text_input(
        #         "Command",
        #         placeholder=(
        #             "Examples:\n"
        #             "  addq after 2 | What matters most? | Price; Style; Install time; Other\n"
        #             "  delq 4\n"
        #             "  tuneq 1 | Make it shorter"
        #         ),
        #         key="cmd_console_power",
        #     )
        #     run = st.button("Run command", key="run_cmd_console_power")
        #     if run and cmd.strip():
        #         new_payload, msg = apply_command(
        #             st.session_state.payload,
        #             st.session_state.ctx,
        #             st.session_state.client,
        #             cmd,
        #         )
        #         st.session_state.payload = new_payload
        #         st.session_state.history.append(cmd)
        #         st.session_state.logs.append(msg)

        #     if st.session_state.logs:
        #         st.code("\n\n".join(st.session_state.logs[-10:]), language="text")

        # with c2:
        #     st.markdown("**History**")
        #     for h in reversed(st.session_state.history[-25:]):
        #         st.code(h, language="text")

        # st.markdown("**Updated preview**")
        # st.code(neat_preview(st.session_state.payload, show_keys=False), language="text")




# -----------------------
# ADVANCED TAB
# -----------------------
with tab_advanced:
    st.subheader("Advanced (optional)")
    if st.session_state.payload is None:
        st.info("Build a survey in the **Build** tab first.")
    else:
        st.json(st.session_state.payload.get("meta", {}).get("construct_plan", {}))
        st.code(neat_preview(st.session_state.payload, show_keys=True), language="text")
        st.json(st.session_state.payload)