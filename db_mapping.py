"""
db_mappings.py

Contract:
- Payload sends DB question IDs (ints, as strings in JSON) and answers as strings.
- We must map DB IDs -> NHANES variable codes.
- We must remap DB answer codes -> NHANES codes (even if identity), explicitly.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


# -----------------------------
# DB ID -> NHANES variable codes
# -----------------------------
lab_mapping: Dict[int, str] = {
    # Vitals / anthropometry
    99001: "BPXPLS",   # Pulse rate (beats/min, 60-sec pulse)
    11810: "BPXSAR",   # Systolic blood pressure (mmHg)
    11820: "BPXDAR",   # Diastolic blood pressure (mmHg)

    # Kidney (urine)
    99002: "URXUMASI",  # Urine albumin (SI)
    99003: "URXUCRSI",  # Urine creatinine (SI)

    # Iron panel
    10270: "LBDIRNSI",   # Serum iron (note: DB may be mg/dL; see UNIT_SCALE)
    99004: "LBDTIBSI",   # TIBC
    99005: "LBXPCT",     # Transferrin saturation (%)
    10570: "LBDFERSI",   # Ferritin

    # Folate / B12 / cotinine
    10610: "LBDFOLSI",   # Folate
    99006: "LBDB12SI",   # Vitamin B12 (warning: you noted DB might be dietary intake)
    13350: "LBXCOT",     # Cotinine / smoking-related

    # CBC (white cells)
    10350: "LBXWBCSI",   # WBC
    99007: "LBXLYPCT",
    99008: "LBXMOPCT",
    10370: "LBXNEPCT",
    10380: "LBXEOPCT",
    13170: "LBXBAPCT",
    99009: "LBDLYMNO",
    99010: "LBDMONO",
    10391: "LBDNENO",
    99011: "LBDEONO",
    99012: "LBDBANO",

    # CBC (red cells / platelets)
    13210: "LBXRBCSI",
    10400: "LBXHGB",
    13220: "LBXHCT",
    99013: "LBXMCVSI",  # MCV
    99014: "LBXMCHSI",  # MCH
    99015: "LBXMC",     # MCHC
    10430: "LBXRDW",
    13240: "LBXPLTSI",
    13250: "LBXMPSI",

    # Inflammation / glycemia / cardiac
    13440: "LBXCRP",
    99016: "LBXGH",     # HbA1c
    99017: "SSBNP",     # NT-proBNP

    # Basic chem (SI set)
    13280: "LBDSALSI",  # Albumin
    10170: "LBXSATSI",  # ALT
    10190: "LBXSASSI",  # AST
    10180: "LBXSAPSI",  # ALP
    10200: "LBDSBUSI",  # BUN (SI) (note scaling)
    13330: "LBDSCASI",  # Calcium (SI) (note scaling)
    13290: "LBXSC3SI",  # Bicarbonate
    10641: "LBDSGLSI",  # Glucose (SI) (note scaling)
    13300: "LBXSLDSI",  # LDH
    13310: "LBDSPHSI",  # Phosphorus (note scaling)
    10290: "LBDSTBSI",  # Total bilirubin (note scaling)
    10320: "LBDSTPSI",  # Total protein (note scaling)
    10340: "LBDSUASI",  # Uric acid (note scaling)
    10230: "LBDSCRSI",  # Creatinine (note scaling)
    10280: "LBXSNASI",  # Sodium
    13320: "LBXSKSI",   # Potassium
    10210: "LBXSCLSI",  # Chloride
    10240: "LBDSGBSI",  # Globulin (note scaling)
}

ques_mapping: Dict[int, str] = {
    14190: "BPQ020",   # Ever told you had high blood pressure?
    14199: "DIQ010",   # Doctor told you have diabetes?
    14254: "KIQ020",   # Weak/failing kidneys?
    14219: "MCQ010",   # Asthma?
    99018: "MCQ053",

    14222: "MCQ160A",
    14223: "MCQ160B",
    14224: "MCQ160C",
    99019: "MCQ160D",
    14225: "MCQ160E",
    14226: "MCQ160F",
    14227: "MCQ160G",
    99020: "MCQ160I",
    99021: "MCQ160J",
    99022: "MCQ160K",
    99023: "MCQ160L",
    99024: "MCQ220",

    99025: "OSQ010A",
    99026: "OSQ010B",
    99027: "OSQ010C",
    99028: "OSQ060",

    99030: "PFQ056",

    # Self-rated health
    99031: "HUQ010",   # (not yet in DB; still reserve mapping)
    14211: "HUQ020",   # matches NHANES already (1/2/3)
    99032: "HUQ050",   # (not yet in DB; still reserve mapping)
}


# -----------------------------
# Unit scaling: DB -> NHANES SI
# -----------------------------
# Multiply DB-provided numeric lab values by these factors to convert into the
# units expected by the LinAge2 pipeline (NHANES-style).
UNIT_SCALE: Dict[str, float] = {
    "LBDIRNSI": 0.179,   # db mg/dL -> NHANES µmol/L (per your note)
    "LBDSALSI": 0.1,     # db g/dL -> NHANES g/L
    "LBDSTPSI": 0.1,     # db g/dL -> NHANES g/L
    "LBDSGBSI": 0.1,     # db g/dL -> NHANES g/L
    "LBDSCRSI": 88.4,    # db mg/dL -> µmol/L
    "LBDSUASI": 59.48,   # db mg/dL -> µmol/L
    "LBDSTBSI": 17.104,  # db mg/dL -> µmol/L
    "LBDSGLSI": 0.0555,  # db mg/dL -> mmol/L
    "LBDSCASI": 0.2495,  # db mg/dL -> mmol/L
    "LBDSPHSI": 0.323,   # db mg/dL -> mmol/L
    "LBDSBUSI": 0.357,   # db mg/dL -> mmol/L
}


# -----------------------------
# Answer code transforms (DB -> NHANES)
# -----------------------------
def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(str(x).strip())
    except Exception:
        return None


def _identity_int(x: Any) -> Optional[int]:
    return _to_int(x)


def _yesno_db01_to_nhanes12(x: Any) -> Optional[int]:
    """
    DB codebook: 0=Yes, 1=No
    NHANES:      1=Yes, 2=No
    """
    v = _to_int(x)
    if v is None:
        return None
    if v == 0:
        return 1
    if v == 1:
        return 2
    # If backend accidentally sends NHANES already, allow pass-through for safety:
    if v in (1, 2):
        return v
    return None


# Which NHANES variables are expected to come from DB as 0/1 yes-no?
# (Explicit is better than guessing.)
YES_NO_DB01_VARS = {
    "BPQ020",
    "KIQ020",
    "MCQ010",
    "MCQ053",
    "MCQ160A", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F",
    "MCQ160G", "MCQ160I", "MCQ160J", "MCQ160K", "MCQ160L",
    "MCQ220",
    "OSQ010A", "OSQ010B", "OSQ010C",
    "OSQ060",
    "PFQ056",
    "HUQ070",
    # Note: DIQ010 is *not* pure yes/no (borderline=3 in NHANES), so we do NOT include it.
}

# Per-variable transform table (explicit even if identity)
QUESTION_VALUE_TRANSFORMS: Dict[str, Callable[[Any], Optional[int]]] = {
    # yes/no group:
    **{var: _yesno_db01_to_nhanes12 for var in YES_NO_DB01_VARS},

    # explicitly identity transforms (matches NHANES):
    "HUQ020": _identity_int,  # Better=1, Worse=2, Same=3 matches NHANES
    "HUQ010": _identity_int,  # not in DB yet; expect NHANES 1..5 when it arrives
    "HUQ050": _identity_int,  # not in DB yet; expect numeric when it arrives

    # special cases:
    "DIQ010": _identity_int,  # expect NHANES-style 1/2/3 when DB sends it
}
