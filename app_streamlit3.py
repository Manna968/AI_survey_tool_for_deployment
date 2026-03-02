import os
import json
import streamlit as st
import re
import copy
from typing import List, Optional, Any, Dict

# ------------------------------------------------------------
# Streamlit config (MUST be the first st.* call)
# ------------------------------------------------------------
st.set_page_config(page_title="Arrival Survey Builder", layout="wide")
st.title("Arrival Survey Builder (Internal)")

# Import from your existing script file:
from survey_template_0209 import (
    load_library, build_arrival, neat_preview, BuilderContext, make_client,
    rewrite_question_text_openai, rewrite_answer_options_openai,
    var_name_for_slot, canonical_option_key,
    plan_l2_followups_openai, build_l2_condition, _ensure_closed_ended,
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

VALID_QTYPES = {
    "SingleSelection", "MultiSelection",
    "SingleSelectionWithOther", "MultiSelectionWithOther",
    "OpenText",
}

def normalize_qtype(qtype: str) -> str:
    qt = (qtype or "").strip()
    if not qt:
        return "SingleSelection"
    m = {
        "singleselect": "SingleSelection",
        "single_select": "SingleSelection",
        "single": "SingleSelection",
        "multiselect": "MultiSelection",
        "multi_select": "MultiSelection",
        "multi": "MultiSelection",
        "opentext": "OpenText",
        "open_text": "OpenText",
        "open": "OpenText",
        "text": "OpenText",
    }
    qt2 = m.get(qt.lower(), qt)
    return qt2 if qt2 in VALID_QTYPES else "SingleSelection"

def l2_fingerprint(parent_qid: str, draft: dict) -> str:
    trig = ",".join(sorted([str(x) for x in (draft.get("trigger_answer_keys") or [])]))
    qtext = (draft.get("question_text") or "").strip()
    qtype = (draft.get("question_type") or "").strip()
    base = f"{parent_qid}||{trig}||{qtype}||{qtext}"
    return slugify(base)[:80]


def generate_l2_from_user_requirements(
    client: Any,
    *,
    ctx: Any,                       # BuilderContext
    parent_it: Dict[str, Any],      # parent item dict
    trigger_keys: List[str],        # keys selected by user
    requirements: str,              # user text
    desired_qtype: str,             # normalized canonical qtype
) -> Dict[str, Any]:
    """
    Returns a 'draft' dict compatible with your build_l2_item_from_draft():
      {
        "trigger_answer_keys": [...],
        "question_text": "...",
        "question_type": "SingleSelection|MultiSelection|OpenText",
        "answer_options": [...],
        "why": "..."
      }
    """
    parent_q = (parent_it.get("question_text") or "").strip()

    # Map keys -> labels for context
    key_to_label = {
        o.get("key"): o.get("label")
        for o in (parent_it.get("answer_options") or [])
        if isinstance(o, dict) and (o.get("key") or "").strip()
    }
    trigger_labels = [key_to_label.get(k, k) for k in trigger_keys]

    # 1) Draft question text (use rewrite_question_text_openai to keep consistent w your system)
    prompt_seed = (
        f"Create an L2 follow-up question shown only when the user selected: {', '.join(trigger_labels)}.\n"
        f"Parent question: {parent_q}\n"
        f"Requirements: {requirements}\n"
        f"Desired question type: {desired_qtype}\n"
    )
    instruction_q = (
        "Write ONE concise follow-up (L2) survey question.\n"
        "- Must be answerable in <=10 seconds.\n"
        "- Must directly relate to the selected parent answer(s).\n"
        "- Must NOT drift into a different construct than the parent question.\n"
        "- Avoid analytics/trackable topics (device/referrer/browser/time on site/pages/clicks).\n"
        "- Do not mention internal keys or 'selected answer'.\n"
    )

    l2_text = rewrite_question_text_openai(
        client,
        site_purpose=ctx.site_purpose,
        survey_goal=ctx.survey_goal,
        site_category=ctx.site_category,
        original_question_text=prompt_seed,
        instruction=instruction_q,
    ).strip()

    # 2) Draft options (if closed-ended)
    if desired_qtype == "OpenText":
        return {
            "trigger_answer_keys": trigger_keys,
            "question_text": l2_text,
            "question_type": "OpenText",
            "answer_options": [],
            "why": f"user_requirements: {requirements}",
        }

    seed_opts = ["Option A", "Option B", "Other"]
    instruction_o = (
        "Generate answer options for the L2 follow-up question.\n"
        "- Return 2–6 options.\n"
        "- Short labels only.\n"
        "- Mutually exclusive where possible.\n"
        "- Include 'Other' only if useful.\n"
        f"Follow-up question: {l2_text}\n"
    )

    l2_opts = rewrite_answer_options_openai(
        client,
        site_purpose=ctx.site_purpose,
        survey_goal=ctx.survey_goal,
        site_category=ctx.site_category,
        question_text=l2_text,
        original_options=seed_opts,
        instruction=instruction_o,
        keep_other_if_present=True,
    )
    l2_opts = _dedupe_labels_preserve_order([str(x) for x in (l2_opts or [])])[:8]

    return {
        "trigger_answer_keys": trigger_keys,
        "question_text": l2_text,
        "question_type": desired_qtype,
        "answer_options": l2_opts,
        "why": f"user_requirements: {requirements}",
    }

def do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

def slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "opt"

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

    generated: List[str] = []
    selected: List[str] = []
    max_select = 8

    # Case A: legacy list[str]
    if isinstance(raw, list):
        generated = [str(x).strip() for x in raw if x and str(x).strip()]
        return {"generated": generated, "selected": [], "max_select": max_select}

    # Case B: new dict
    if isinstance(raw, dict):
        generated = raw.get("generated") or []
        selected = raw.get("selected") or []
        max_select = int(raw.get("max_select") or 8)

        generated = [str(x).strip() for x in generated if x and str(x).strip()]
        selected = [str(x).strip() for x in selected if x and str(x).strip()]
        return {"generated": generated, "selected": selected, "max_select": max_select}

    return {"generated": [], "selected": [], "max_select": max_select}

def _set_candidates_block(it: dict, *, generated: list, selected: list = None, max_select: int = 8) -> None:
    it.setdefault("ai_actions", {})
    it["ai_actions"]["option_candidates"] = {
        "generated": [x for x in (generated or []) if x and str(x).strip()],
        "selected": [x for x in (selected or []) if x and str(x).strip()] if selected is not None else [],
        "max_select": int(max_select),
    }

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

def build_l2_item_from_draft(
    *,
    parent_item: Dict[str, Any],
    draft: Dict[str, Any],
    new_id: str,
) -> Dict[str, Any]:
    """
    Convert one L2 draft (from ai_actions["l2_suggestions"]["drafts"][i]) into a real payload item.
    """
    pslot = str(parent_item.get("slot") or "")
    plevel = str(parent_item.get("level") or "L1")
    parent_var = var_name_for_slot(pslot, plevel)
    parent_qtype = str(parent_item.get("question_type") or "SingleSelect")

    trigger_keys = [str(x).strip() for x in (draft.get("trigger_answer_keys") or []) if str(x).strip()]

    # condition JSON for L2 display
    cond = build_l2_condition(parent_var, parent_qtype, trigger_keys)

    qtype = str(draft.get("question_type") or "SingleSelect")
    # normalize common variants
    qtype = qtype.replace("SingleSelection", "SingleSelect").replace("MultiSelection", "MultiSelect")

    qtext = str(draft.get("question_text") or "").strip()
    opts = list(draft.get("answer_options") or [])

    # ensure closed-ended questions have options (uses template helper)
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
            "source": "l2_suggestions_ui",
            "trigger_answer_keys": trigger_keys,
            "why": str(draft.get("why") or ""),
        },
    }

    if qtype != "OpenText":
        opts = _dedupe_labels_preserve_order([str(x) for x in opts])
        _set_options_from_labels(item, opts[:8])

    return item

def render_reorder_ui(payload: dict, *, selected_qid: str) -> None:
    it = _find_item(payload, selected_qid)
    if not it:
        st.error("Reorder: question not found.")
        return

    labels = [
        o.get("label")
        for o in (it.get("answer_options") or [])
        if isinstance(o, dict) and (o.get("label") or "").strip()
    ]
    labels = _dedupe_labels_preserve_order(labels)

    if len(labels) < 2:
        st.info("Reorder: not enough options to reorder.")
        return

    st.markdown("### Reorder options")

    state_key = f"reorder_working_{selected_qid}"
    if state_key not in st.session_state:
        st.session_state[state_key] = labels

    working = st.session_state[state_key]

    for i, lab in enumerate(working):
        c1, c2, c3, c4 = st.columns([6, 1, 1, 2])
        with c1:
            st.write(lab)
        with c2:
            up = st.button("⬆️", key=f"up_{selected_qid}_{i}", disabled=(i == 0))
        with c3:
            down = st.button("⬇️", key=f"down_{selected_qid}_{i}", disabled=(i == len(working) - 1))
        with c4:
            pass

        if up:
            working[i - 1], working[i] = working[i], working[i - 1]
            st.session_state[state_key] = working
            st.rerun()

        if down:
            working[i + 1], working[i] = working[i], working[i + 1]
            st.session_state[state_key] = working
            st.rerun()

    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        if st.button("✅ Save order", key=f"save_reorder_{selected_qid}"):
            new_payload = copy.deepcopy(st.session_state.payload)
            it2 = _find_item(new_payload, selected_qid)
            _set_options_from_labels(it2, st.session_state[state_key])

            it2.setdefault("ai_actions", {})
            it2["ai_actions"]["reordered_by_user"] = True

            st.session_state.payload = new_payload
            st.session_state.logs.append(f"✅ Reordered options in Q{selected_qid}.")
            st.rerun()

    with colB:
        if st.button("↩️ Reset", key=f"reset_reorder_{selected_qid}"):
            st.session_state[state_key] = labels
            st.rerun()

    with colC:
        st.caption("Use ⬆️ / ⬇️ to reorder. Click **Save order** to apply.")

def suggest_options_for_item(
    client: Any,
    *,
    ctx: Any,              # BuilderContext
    it: Dict[str, Any],    # item dict from payload
    instruction: str = "",
    n: int = 12,
) -> List[str]:
    qtext = (it.get("question_text") or "").strip()
    if not qtext:
        return []

    orig_opts = [o.get("label") for o in (it.get("answer_options") or []) if isinstance(o, dict)]
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

# ------------------------------------------------------------
# Cached library load
# ------------------------------------------------------------
LIB_PATH = "Question_Rates_Merged_backbone_ready.xlsx"

@st.cache_data(show_spinner=False)
def cached_load_library(path: str):
    return load_library(path)

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
    st.session_state.option_suggestions = {}
if "last_suggest_qid" not in st.session_state:
    st.session_state.last_suggest_qid = None
if "last_site_category" not in st.session_state:
    st.session_state.last_site_category = None

# -----------------------
# UI options
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
    cur = st.session_state.get(state_key, [])
    if not cur:
        return
    cur2 = [x for x in cur if x in valid_options]
    if cur2 != cur:
        st.session_state[state_key] = cur2

def hard_reset_on_category_change(new_category: str):
    prev = st.session_state.get("last_site_category")
    if prev is None:
        st.session_state.last_site_category = new_category
        return

    if prev != new_category:
        st.session_state.primary_audience = []
        st.session_state.primary_actions = []
        st.session_state.last_site_category = new_category

# ------------------------------------------------------------
# Command console (apply_command)
# ------------------------------------------------------------
def apply_command(payload, ctx, client, cmd: str):
    cmd = (cmd or "").strip()
    if not cmd:
        return payload, "Empty command."

    low = cmd.lower().strip()

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

    # legacy alias
    if low.startswith("genl2 "):
        cmd = "l2 " + cmd.split(" ", 1)[1]
        low = cmd.lower()

    if low.startswith("addq "):
        parts = _split_pipe(cmd, 3)
        if not parts:
            return payload, (
                "Usage:\n"
                "  addq after <qid> | <question_text> | <opt1; opt2; ...>\n"
                "  addq end | <question_text> | <opt1; opt2; ...>"
            )
        head = parts[0].strip()
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

# ------------------------------------------------------------
# Sidebar: Config
# ------------------------------------------------------------
st.sidebar.header("Config")
st.sidebar.markdown(
    "- Fill the structured **Build** form\n"
    "- Click **Build survey**\n"
    "- View output in **Survey Preview**\n"
    "- Refine in **Command Console**\n"
)

lib_path = st.sidebar.text_input(
    "Library path (xlsx)",
    value=os.getenv("LIB_PATH", LIB_PATH),
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

# Tabs
tab_build, tab_preview, tab_console, tab_advanced = st.tabs(
    ["Build", "Survey Preview", "Command Console", "Advanced"]
)

# ------------------------------------------------------------
# BUILD TAB
# ------------------------------------------------------------
with tab_build:
    st.subheader("Website basics")

    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        site_name = st.text_input("Site name (optional)", placeholder="e.g., Paragon Stairs", key="site_name")

    with c2:
        site_category = st.selectbox(
            "Site category",
            ["Pharma", "Ecommerce", "Education", "SaaS", "Content", "University"],
            index=1,
            key="site_category",
        )

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

# ------------------------------------------------------------
# PREVIEW TAB
# ------------------------------------------------------------
with tab_preview:
    st.subheader("Neat preview")
    if st.session_state.payload is None:
        st.info("Build a survey in the **Build** tab first.")
    else:
        st.code(neat_preview(st.session_state.payload, show_keys=False), language="text")

        blob = json.dumps(st.session_state.payload, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            label=f"Download {out_name}",
            data=blob,
            file_name=out_name,
            mime="application/json",
        )

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
                st.markdown(qtext if qtext else "_(no question text)_")

                selected = st.multiselect(
                    "Select the answer options to keep",
                    options=c["generated"],
                    default=c["selected"],
                    key=f"cand_select_{qid}",
                )

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

# ------------------------------------------------------------
# COMMAND CONSOLE TAB
# ------------------------------------------------------------
with tab_console:
    st.subheader("Command Console")

    # --- Persisted success message from actions that call st.rerun() ---
    msg = st.session_state.get("last_action_msg")
    if msg:
        st.success(msg)
        # Toast is harder to miss if you're scrolled down
        try:
            st.toast(msg)
        except Exception:
            pass
        del st.session_state["last_action_msg"]

    RULE_BASED = [
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

        # 1) Command input
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
                    new_payload, msg2 = apply_command(
                        st.session_state.payload,
                        st.session_state.ctx,
                        st.session_state.client,
                        cmd,
                    )
                    st.session_state.payload = new_payload
                    st.session_state.history.append(cmd)
                    st.session_state.logs.append(msg2)

                    if msg2.lower().startswith("unknown command"):
                        st.error(msg2)
                    else:
                        st.success(msg2)

        with hist_col:
            st.markdown("**History**")
            for h in reversed(st.session_state.history[-25:]):
                st.code(h, language="text")

        if st.session_state.logs:
            st.markdown("**Recent logs**")
            st.code("\n\n".join(st.session_state.logs[-10:]), language="text")

        st.divider()
        st.markdown("## Live preview (after changes)")
        if st.session_state.payload is not None:
            with st.expander("Show neat preview", expanded=True):
                st.code(neat_preview(st.session_state.payload, show_keys=False), language="text")

        st.divider()

        # 2) AI option recommender
        st.markdown("## AI option recommender (interactive)")
        st.caption("Suggest candidate answer options with AI, then append selected ones. You can also delete current options inline.")

        items = payload.get("items", []) or []

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

                                labels2[i] = new_lab.strip() if new_lab.strip() else labels2[i]
                                _set_options_from_labels(it2, labels2)

                                st.session_state.payload = new_payload
                                st.session_state.logs.append(f"✅ Renamed option {i+1} in Q{selected_qid}.")
                                st.rerun()

                        with c_del:
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

                render_reorder_ui(payload, selected_qid=selected_qid)

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

                        cands2 = suggest_options_for_item(
                            st.session_state.client,
                            ctx=st.session_state.ctx,
                            it=it2,
                            instruction=suggest_instruction,
                            n=12,
                        )

                        if not cands2:
                            st.warning("No candidates generated. Try a different instruction.")
                        else:
                            _set_candidates_block(it2, generated=cands2[:20], selected=[], max_select=8)
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

        # ✅ L2 follow-ups block (properly OUTSIDE the selected_qid flow)
        st.divider()
        st.markdown("## L2 follow-ups (user-driven)")

        payload = st.session_state.payload
        items = payload.get("items", []) or []

        # 1) choose parent
        parent_choices = []
        for it in items:
            if str(it.get("level", "")).upper() != "L1":
                continue
            opts = it.get("answer_options") or []
            if not any(isinstance(o, dict) and (o.get("key") or "").strip() for o in opts):
                continue
            qid = str(it.get("id", "")).strip()
            qtext = (it.get("question_text") or "").strip()
            parent_choices.append((qid, f"Q{qid}: {qtext[:90]}{'…' if len(qtext) > 90 else ''}"))

        if not parent_choices:
            st.info("No eligible L1 questions with keyed answer options found.")
        else:
            parent_qid = st.selectbox(
                "Parent question (L1)",
                options=[qid for qid, _ in parent_choices],
                format_func=lambda qid: dict(parent_choices).get(qid, qid),
                key="l2_parent_qid_user",
            )

            # Clear any stale drafts from other parents
            draft_state_key = f"l2_draft_{parent_qid}"
            for k in list(st.session_state.keys()):
                if k.startswith("l2_draft_") and k != draft_state_key:
                    del st.session_state[k]

            parent_it = _find_item(payload, parent_qid)
            if not parent_it:
                st.error("Parent question not found.")
            else:
                st.markdown("### Step 1 — Choose trigger answer option(s)")

                parent_opts = [
                    o for o in (parent_it.get("answer_options") or [])
                    if isinstance(o, dict) and (o.get("key") or "").strip()
                ]

                # Display "Label (key)" to avoid the duplicated "answer: answer" look and avoid collisions on duplicate labels.
                display_opts = [f"{(o.get('label') or '').strip()}  ({o.get('key')})" for o in parent_opts]
                display_to_key = {f"{(o.get('label') or '').strip()}  ({o.get('key')})": o.get("key") for o in parent_opts}

                selected_display = st.multiselect(
                    "Show L2 only when the user selects these parent answers",
                    options=display_opts,
                    default=[],
                    key="l2_trigger_display_user",
                )
                trigger_keys = [display_to_key[x] for x in selected_display if x in display_to_key and display_to_key[x]]

                st.markdown("### Step 2 — Requirements (what should this L2 accomplish?)")
                requirements = st.text_area(
                    "Requirements sent to AI",
                    placeholder="e.g., If they chose 'Contractor', ask trade type; keep it short; 4 options max.",
                    height=100,
                    key="l2_requirements_user",
                )

                st.markdown("### Step 3 — L2 question type")
                desired_qtype_ui = st.selectbox(
                    "Desired L2 question type",
                    ["SingleSelection", "MultiSelection", "OpenText"],
                    index=0,
                    key="l2_desired_qtype_user",
                )
                desired_qtype = normalize_qtype(desired_qtype_ui)

                colA, colB = st.columns([1, 2])
                with colA:
                    gen_btn = st.button("Generate L2 draft", key="btn_generate_l2_user")
                with colB:
                    st.caption("Generates one L2 draft from your selected triggers + requirements.")

                if gen_btn:
                    if st.session_state.client is None:
                        st.error("OpenAI client unavailable (enable key + LLM).")
                    elif not trigger_keys:
                        st.warning("Pick at least one trigger answer key.")
                    elif not requirements.strip():
                        st.warning("Please enter requirements for the L2.")
                    else:
                        draft = generate_l2_from_user_requirements(
                            st.session_state.client,
                            ctx=st.session_state.ctx,
                            parent_it=parent_it,
                            trigger_keys=trigger_keys,
                            requirements=requirements.strip(),
                            desired_qtype=desired_qtype,
                        )
                        st.session_state[draft_state_key] = draft
                        st.success("Generated L2 draft. Review below.")

                draft = st.session_state.get(draft_state_key)

                if draft:
                    st.divider()
                    st.markdown("### Generated L2 draft (preview)")
                    st.write(f"**Triggers:** {draft.get('trigger_answer_keys')}")
                    st.write(f"**Type:** {draft.get('question_type')}")
                    st.write(f"**Question:** {draft.get('question_text')}")
                    if draft.get("question_type") != "OpenText":
                        st.write("**Options:**")
                        for opt in (draft.get("answer_options") or []):
                            st.write(f"- {opt}")

                    add_after, add_end = st.columns([1, 1])

                    # Use a set for robust anti-double-click / anti-duplicate behavior
                    st.session_state.setdefault("l2_added_fps", set())
                    fp = l2_fingerprint(parent_qid, draft)

                    with add_after:
                        add_after_clicked = st.button("➕ Add after parent", key="btn_add_l2_after_user")
                        if add_after_clicked:
                            if fp in st.session_state["l2_added_fps"]:
                                st.warning("This L2 was just added — ignoring duplicate click.")
                            else:
                                new_payload = copy.deepcopy(st.session_state.payload)
                                items2 = new_payload.get("items", []) or []

                                parent_idx = next(
                                    (ix for ix, itx in enumerate(items2) if str(itx.get("id")) == str(parent_qid)),
                                    None,
                                )
                                if parent_idx is None:
                                    st.error("Could not locate parent in items.")
                                else:
                                    parent_item_live = _find_item(new_payload, parent_qid)
                                    parent_var = var_name_for_slot(str(parent_item_live.get("slot")), str(parent_item_live.get("level")))
                                    parent_qtype = normalize_qtype(str(parent_item_live.get("question_type") or "SingleSelection"))

                                    expected_cond = build_l2_condition(
                                        parent_var, parent_qtype, list(draft.get("trigger_answer_keys") or [])
                                    )
                                    expected_text = (draft.get("question_text") or "").strip()

                                    def _is_identical_l2(x: dict) -> bool:
                                        if str(x.get("level", "")).upper() != "L2":
                                            return False
                                        if (x.get("question_text") or "").strip() != expected_text:
                                            return False
                                        if x.get("display_condition_json") != expected_cond:
                                            return False
                                        return True

                                    if any(_is_identical_l2(x) for x in items2):
                                        st.warning("An identical L2 already exists. Not adding another.")
                                    else:
                                        new_item = build_l2_item_from_draft(
                                            parent_item=parent_item_live,
                                            draft=draft,
                                            new_id=str(parent_idx + 2),  # placeholder
                                        )
                                        items2.insert(parent_idx + 1, new_item)

                                        for k, itx in enumerate(items2, start=1):
                                            itx["id"] = str(k)

                                        new_payload["items"] = items2
                                        st.session_state.payload = new_payload

                                        st.session_state["l2_added_fps"].add(fp)
                                        st.session_state["last_action_msg"] = "✅ Added L2 after parent."
                                        st.rerun()

                    with add_end:
                        add_end_clicked = st.button("➕ Add at end", key="btn_add_l2_end_user")
                        if add_end_clicked:
                            if fp in st.session_state["l2_added_fps"]:
                                st.warning("This L2 was just added — ignoring duplicate click.")
                            else:
                                new_payload = copy.deepcopy(st.session_state.payload)
                                items2 = new_payload.get("items", []) or []

                                parent_item_live = _find_item(new_payload, parent_qid)
                                parent_var = var_name_for_slot(str(parent_item_live.get("slot")), str(parent_item_live.get("level")))
                                parent_qtype = normalize_qtype(str(parent_item_live.get("question_type") or "SingleSelection"))

                                expected_cond = build_l2_condition(
                                    parent_var, parent_qtype, list(draft.get("trigger_answer_keys") or [])
                                )
                                expected_text = (draft.get("question_text") or "").strip()

                                def _is_identical_l2(x: dict) -> bool:
                                    if str(x.get("level", "")).upper() != "L2":
                                        return False
                                    if (x.get("question_text") or "").strip() != expected_text:
                                        return False
                                    if x.get("display_condition_json") != expected_cond:
                                        return False
                                    return True

                                if any(_is_identical_l2(x) for x in items2):
                                    st.warning("An identical L2 already exists. Not adding another.")
                                else:
                                    new_item = build_l2_item_from_draft(
                                        parent_item=parent_item_live,
                                        draft=draft,
                                        new_id=str(len(items2) + 1),
                                    )
                                    items2.append(new_item)

                                    for k, itx in enumerate(items2, start=1):
                                        itx["id"] = str(k)

                                    new_payload["items"] = items2
                                    st.session_state.payload = new_payload

                                    st.session_state["l2_added_fps"].add(fp)
                                    st.session_state["last_action_msg"] = "✅ Added L2 at end."
                                    st.rerun()
# ------------------------------------------------------------
# ADVANCED TAB
# ------------------------------------------------------------
with tab_advanced:
    st.subheader("Advanced (optional)")
    if st.session_state.payload is None:
        st.info("Build a survey in the **Build** tab first.")
    else:
        st.json(st.session_state.payload.get("meta", {}).get("construct_plan", {}))
        st.code(neat_preview(st.session_state.payload, show_keys=True), language="text")
        st.json(st.session_state.payload)