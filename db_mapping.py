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
    11800: "BPXPLS",  # Pulse rate (beats/min, 60-sec pulse), 11800
    11810: "BPXSAR",  # Systolic blood pressure (mmHg)
    11820: "BPXDAR",  # Diastolic blood pressure (mmHg)
    # Kidney (urine), check with this 14010
    99002: "URXUMASI",  # Urine albumin (SI)
    99003: "URXUCRSI",  # Urine creatinine (SI)
    # Iron panel
    10270: "LBDIRNSI",  # Serum iron (note: DB may be mg/dL; see UNIT_SCALE)
    99004: "LBDTIBSI",  # TIBC, Total iron-binding capacity
    99005: "LBXPCT",  # Transferrin saturation (%)
    10570: "LBDFERSI",  # Ferritin
    # Folate / B12 / cotinine
    10610: "LBDFOLSI",  # Folate
    99006: "LBDB12SI",  # Vitamin B12 (warning: you noted DB might be dietary intake)
    13350: "LBXCOT",  # Cotinine / smoking-related
    # CBC (white cells)
    10350: "LBXWBCSI",  # White blood cell count (×10⁹/L)  also WBC
    10360: "LBXLYPCT",  # Lymphocyte %, 10360 99007
    13160: "LBXMOPCT",  # Monocyte %, 13160 99008
    10370: "LBXNEPCT",  # Neutrophil %
    10380: "LBXEOPCT",  # Eosinophil %
    13170: "LBXBAPCT",  # Basophil %
    10390: "LBDLYMNO",  # Lymphocyte absolute #, 10390 99009
    13180: "LBDMONO",  # Monocyte absolute #, 13180 99010
    10391: "LBDNENO",  # Neutrophil absolute #
    13190: "LBDEONO",  # Eosinophil absolute #, 13190 99011
    13200: "LBDBANO",  # Basophil absolute #, 13200 99012
    # CBC (red cells / platelets)
    13210: "LBXRBCSI", #Red blood cell count (×10¹²/L).
    10400: "LBXHGB", #Hemoglobin (g/dL).
    13220: "LBXHCT", #Hematocrit (%).
    10410: "LBXMCVSI",  # Mean corpuscular volume, MCV (fL), 10410 99013
    13230: "LBXMCHSI",  # Mean corpuscular hemoglobin, MCH (pg)., 13230 99014
    10420: "LBXMC",  #Mean corpuscular hemoglobin concentration, MCHC (g/dL)., 10420 99015
    10430: "LBXRDW", #Red cell distribution width (%)
    13240: "LBXPLTSI", #Platelet count (×10⁹/L)
    13250: "LBXMPSI", #Mean platelet volume, MPV (fL).
    # Inflammation / glycemia / cardiac
    13440: "LBXCRP", #C-reactive protein (mg/L)
    10640: "LBXGH",  # Glycohemoglobin (HbA1c, %), 10640 99016
    99017: "SSBNP",  # "N-terminal pro-B-type natriuretic peptide (NT-proBNP, pg/mL).,
    # Basic chem (SI set)
    13280: "LBDSALSI",   # Albumin (g/L).
    10170: "LBXSATSI",   # Alanine aminotransferase, ALT (U/L).
    10190: "LBXSASSI",   # Aspartate aminotransferase, AST (U/L).
    10180: "LBXSAPSI",   # Alkaline phosphatase (U/L).
    10200: "LBDSBUSI",   # Urea nitrogen (BUN), SI (mmol/L).
    13330: "LBDSCASI",   # Calcium, SI (mmol/L).
    13290: "LBXSC3SI",   # Bicarbonate (total CO₂), SI (mmol/L).
    10641: "LBDSGLSI",   # Glucose, SI (mmol/L).
    13300: "LBXSLDSI",   # Lactate dehydrogenase, LDH (U/L).
    13310: "LBDSPHSI",   # Phosphorus (mmol/L).
    10290: "LBDSTBSI",   # Total bilirubin (µmol/L).
    10320: "LBDSTPSI",   # Total protein (g/L).
    10340: "LBDSUASI",   # Uric acid (µmol/L).
    10230: "LBDSCRSI",   # Creatinine (µmol/L).
    10280: "LBXSNASI",   # Sodium (mmol/L).
    13320: "LBXSKSI",    # Potassium (mmol/L).
    10210: "LBXSCLSI",   # Chloride (mmol/L).
    10240: "LBDSGBSI",   # Globulin (g/L).
    # LDL-cholesterol : 13960
}

ques_mapping: Dict[int, str] = {
    14190: "BPQ020",  # Ever told you had high blood pressure?
    14199: "DIQ010",  # Doctor told you have diabetes?
    14254: "KIQ020",  # Weak/failing kidneys?
    14219: "MCQ010",  # Asthma?
    99018: "MCQ053",  # Anemia?, 
    14222: "MCQ160A", #Doctor ever said you had arthritis
    14223: "MCQ160B", #Ever told had congestive heart failure
    14224: "MCQ160C", #Ever told you had coronary heart disease
    99019: "MCQ160D", #Ever told you had angina/angina pectoris
    14225: "MCQ160E", #Ever told you had heart attack
    14226: "MCQ160F", #Ever told you had a stroke
    14227: "MCQ160G", #emphysema
    99020: "MCQ160I",  # thyroid_other_diseases_of, (yes/no/ 0/1)
    99021: "MCQ160J",  # obesity_and_overweight
    99022: "MCQ160K", #Ever told you had chronic bronchitis
    10850: "MCQ160L",  #Ever told you had any liver condition 10850 99023
    12160: "MCQ220",  #had cancer or a malignancy 12160 99024
    12030: "OSQ010A",  #Broken or fractured a hip 12030 99025
    99026: "OSQ010B", #Broken or fractured a wrist 
    12060: "OSQ010C",  #Broken or fractured spine 12060 99027
    10860: "OSQ060",  #Ever told had osteoporosis/brittle bones 10860 99028
    12340: "PFQ056",  #Experience confusion/memory problems 12100, 12340   99030
    # Self-rated health
    99031: "HUQ010",  #General health condition (not yet in DB; still reserve mapping),
    14211: "HUQ020",  #Health now compared with 1 year ago matches NHANES already (1/2/3), 14211
    10740: "HUQ050",  #Times received healthcare over past yr (not yet in DB; still reserve mapping), 10740 99032
    99033: "HUQ070", #Overnight hospital patient in last year
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

    return None

def _smoke_db012_to_nhanes_lbx_cot(x: Any) -> Optional[int]:
    """
    DB smoking question:
      0 = Yes, every day
      1 = Yes, some days
      2 = No
    LinAge2/NHANES LBXCOT bucket used in this project:
      0 = Non-smoker
      1 = Light/Recent
      2 = Heavy/Current

    Mapping:
      0 -> 2
      1 -> 1
      2 -> 0
    """
    v = _to_int(x)
    if v is None:
        return None
        
    if v in (0, 1, 2):
        return 2-v

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

LAB_VALUE_TRANSFORMS: Dict[str, Callable[[Any], Optional[int]]] = {
    "LBXCOT": _smoke_db012_to_nhanes_lbx_cot,
}
