import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from db_mappings import (
    lab_mapping,
    ques_mapping,
    UNIT_SCALE,
    QUESTION_VALUE_TRANSFORMS,
)

logger = logging.getLogger("linage2_service")


# -----------------------------
# External dependencies from your existing codebase
# -----------------------------
# These are required to reproduce the exact LinAge2 pipeline you run in ui.py.
# Keep the imports explicit so failures are obvious.
from src import (  # type: ignore
    boxCoxTransform,
    foldOutliers,
    normAsZscores_99_young_mf,
    projectToSVD,
    populateLDL,
    popPCFIfs1,
    popPCFIfs2,
    popPCFIfs3,
)

from imputation import impute_missing_values  # type: ignore


# -----------------------------
# Bundle
# -----------------------------
@dataclass(frozen=True)
class LinAge2Bundle:
    # sex-specific Cox models
    cox_full_F: Any
    cox_null_F: Any
    cox_full_M: Any
    cox_null_M: Any

    # matrices / training stats needed for inference path
    vMatDat99_F: np.ndarray
    vMatDat99_M: np.ndarray

    boxCox_lam: pd.DataFrame
    dataMat_trans_ref: pd.DataFrame
    qDataMat_R: pd.DataFrame

    coxCovsTrainM: pd.DataFrame
    coxCovsTrainF: pd.DataFrame

    zScoreMax: float = 6.0
    unit_scale: Dict[str, float] = None  # assigned in loader


def load_linage2_bundle(artifacts_dir: str = "artifacts") -> LinAge2Bundle:
    """
    Loads all artifacts needed for LinAge2 inference.

    IMPORTANT:
    It’s not just the 4 joblib models — your current pipeline also requires:
    - vMatDat99_F/M
    - boxCox_lam (logNoLog.csv slice)
    - dataMat_trans reference
    - qDataMat_R reference
    - coxCovsTrainM/F
    """
    def p(*parts: str) -> str:
        return os.path.join(artifacts_dir, *parts)

    # Models
    cox_full_F = joblib.load(p("cox_full_F.joblib"))
    cox_null_F = joblib.load(p("cox_null_F.joblib"))
    cox_full_M = joblib.load(p("cox_full_M.joblib"))
    cox_null_M = joblib.load(p("cox_null_M.joblib"))

    # Matrices / refs (paths match your ui.py)
    vMatDat99_F = pd.read_csv(p("vMatDat99_F_pre.csv")).values
    vMatDat99_M = pd.read_csv(p("vMatDat99_M_pre.csv")).values

    boxCox_lam = pd.read_csv(p("logNoLog.csv")).iloc[1:2, :]
    dataMat_trans_ref = pd.read_csv(p("dataMat_trans.csv"))

    # qDataMat_R location differs in your ui.py (root), so allow override
    qdatamat_r_path = os.getenv("QDATAMAT_R_PATH", p("qDataMat_R.csv"))
    qDataMat_R = pd.read_csv(qdatamat_r_path)

    coxCovsTrainM = pd.read_csv(p("coxCovsTrainM.csv"))
    coxCovsTrainF = pd.read_csv(p("coxCovsTrainF.csv"))

    return LinAge2Bundle(
        cox_full_F=cox_full_F,
        cox_null_F=cox_null_F,
        cox_full_M=cox_full_M,
        cox_null_M=cox_null_M,
        vMatDat99_F=vMatDat99_F,
        vMatDat99_M=vMatDat99_M,
        boxCox_lam=boxCox_lam,
        dataMat_trans_ref=dataMat_trans_ref,
        qDataMat_R=qDataMat_R,
        coxCovsTrainM=coxCovsTrainM,
        coxCovsTrainF=coxCovsTrainF,
        zScoreMax=float(os.getenv("Z_SCORE_MAX", "6")),
        unit_scale=dict(UNIT_SCALE),
    )


# -----------------------------
# Parsing helpers
# -----------------------------
def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(str(x).strip())
    except Exception:
        return None


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _extract_age_months_and_sex(payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[int], List[str]]:
    warnings: List[str] = []
    bio = payload.get("biometrics") or {}

    age_years = _to_float(bio.get("age"))
    if age_years is None:
        warnings.append("Missing biometrics.age")
        age_months = None
    else:
        age_months = age_years * 12.0

    sex = _to_int(bio.get("gender"))
    if sex not in (1, 2):
        warnings.append("Missing/invalid biometrics.gender (expected 1=Male, 2=Female)")
        sex = None

    return age_months, sex, warnings


def _apply_unit_scaling(labs: Dict[str, float], unit_scale: Dict[str, float]) -> Dict[str, float]:
    out = dict(labs)
    for k, v in list(out.items()):
        factor = unit_scale.get(k)
        if factor is not None:
            out[k] = v * factor
    return out


def _split_and_remap_survey_items(payload: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, int], List[str]]:
    """
    Reads payload['surveys'] which contains DB ids + DB-coded answers and returns:

    - labs_by_nhanes: {"LBDSGLSI": 5.1, ...} (floats)
    - ques_by_nhanes: {"BPQ020": 2, "HUQ020": 3, ...} (ints, already NHANES-coded)
    """
    warnings: List[str] = []
    labs: Dict[str, float] = {}
    ques: Dict[str, int] = {}

    items = payload.get("surveys") or []
    if not isinstance(items, list):
        return labs, ques, ["payload.surveys must be a list"]

    for it in items:
        qid = _to_int((it or {}).get("ques_id"))
        ans_raw = (it or {}).get("answer")

        if qid is None:
            warnings.append("Survey item with missing/invalid ques_id (skipped)")
            continue

        # Lab?
        if qid in lab_mapping:
            nh = lab_mapping[qid]
            v = _to_float(ans_raw)
            if v is None:
                warnings.append(f"Lab {qid}->{nh}: invalid numeric value {ans_raw!r} (skipped)")
                continue
            labs[nh] = float(v)
            continue

        # Questionnaire?
        if qid in ques_mapping:
            nh = ques_mapping[qid]
            transform = QUESTION_VALUE_TRANSFORMS.get(nh)
            if transform is None:
                # Explicitness: still do a formal "int cast" transform
                v_int = _to_int(ans_raw)
            else:
                v_int = transform(ans_raw)

            if v_int is None:
                warnings.append(f"Question {qid}->{nh}: invalid/unmappable value {ans_raw!r} (skipped)")
                continue

            ques[nh] = int(v_int)
            continue

        warnings.append(f"Unmapped DB ques_id {qid} (ignored)")

    return labs, ques, warnings


# -----------------------------
# Core inference
# -----------------------------
def process_payload(payload: Dict[str, Any], bundle: LinAge2Bundle) -> Dict[str, Any]:
    """
    Returns:
      {
        "success": bool,
        "data": {...} or None,
        "errors": [...],
        "warnings": [...]
      }
    """
    errors: List[str] = []
    warnings: List[str] = []

    age_months, sex, w0 = _extract_age_months_and_sex(payload)
    warnings.extend(w0)

    if age_months is None:
        errors.append("biometrics.age is required")
    if sex is None:
        errors.append("biometrics.gender is required (1=Male,2=Female)")

    labs_raw, ques_raw, w1 = _split_and_remap_survey_items(payload)
    warnings.extend(w1)

    if errors:
        return {"success": False, "data": None, "errors": errors, "warnings": warnings}

    # Scale units to NHANES conventions
    labs = _apply_unit_scaling(labs_raw, bundle.unit_scale or {})

    # -----------------------------
    # Questionnaire defaults (mirror Gradio CODEBOOK defaults)
    # -----------------------------
    q_defaults: Dict[str, Any] = {
        "SEQN": 1,
        "RIAGENDR": int(sex),
        "RIDAGEEX": float(age_months),  # months

        "BPQ020": 2, "DIQ010": 2, "KIQ020": 2, "MCQ010": 2, "MCQ053": 2,
        "MCQ160A": 2, "MCQ160B": 2, "MCQ160C": 2, "MCQ160D": 2, "MCQ160E": 2,
        "MCQ160F": 2, "MCQ160G": 2, "MCQ160I": 2, "MCQ160J": 2, "MCQ160K": 2, "MCQ160L": 2,
        "MCQ220": 2,
        "OSQ010A": 2, "OSQ010B": 2, "OSQ010C": 2, "OSQ060": 2,
        "PFQ056": 2, "HUQ070": 2,

        "HUQ010": 3, "HUQ020": 3,
        "HUQ050": 0,
    }
    q_defaults.update(ques_raw)
    q_df = pd.DataFrame([q_defaults])

    # -----------------------------
    # Lab features: create vector aligned to LAB_VARIABLES order (from ui_sliders)
    # -----------------------------
    # The imputer expects values+flags in LAB_VARIABLES order, same as your Gradio app.
    # We import LAB_VARIABLES indirectly through imputation.py; so we reconstruct raw/flag
    # using whatever labs were provided.
    #
    # NOTE: imputation.py already zips LAB_VARIABLES with raw_vals/flags, so we must pass
    # raw_vals/flags in that same order.
    from ui_sliders import LAB_VARIABLES  # local import to keep module load lighter

    raw_vals: List[float] = []
    flags: List[bool] = []
    for code in LAB_VARIABLES:
        if code in labs and labs[code] is not None and np.isfinite(labs[code]):
            raw_vals.append(float(labs[code]))
            flags.append(False)
        else:
            raw_vals.append(np.nan)
            flags.append(True)

    imputed_features = [c for c, miss in zip(LAB_VARIABLES, flags) if miss]

    # Impute
    try:
        lab_vals_imputed = impute_missing_values(raw_vals, flags, int(sex), float(age_months))
    except Exception as e:
        logger.exception("Imputation failed")
        return {
            "success": False,
            "data": None,
            "errors": [f"Imputation failed: {e}"],
            "warnings": warnings,
        }

    # Build dataMat_user exactly like ui.py: labs + SEQN
    dataMat_user = pd.DataFrame({name: [val] for name, val in zip(LAB_VARIABLES, lab_vals_imputed)})
    dataMat_user.insert(0, "SEQN", q_df["SEQN"])

    # -----------------------------
    # Derived features (mirror ui.py)
    # -----------------------------
    try:
        dataMat_user["fs1Score"] = popPCFIfs1(q_df)
        dataMat_user["fs2Score"] = popPCFIfs2(q_df)
        dataMat_user["fs3Score"] = popPCFIfs3(q_df)

        dataMat_user["LDLV"] = populateLDL(dataMat_user, q_df)

        crea = np.asarray(dataMat_user["URXUCRSI"].values, dtype=float)
        albu = np.asarray(dataMat_user["URXUMASI"].values, dtype=float)
        dataMat_user["crAlbRat"] = albu / (crea * 1.1312 * 10 ** -4)
    except Exception as e:
        logger.exception("Derived feature computation failed")
        return {
            "success": False,
            "data": None,
            "errors": [f"Derived feature computation failed: {e}"],
            "warnings": warnings,
        }

    # -----------------------------
    # Transform / normalize / fold / project / compute BA + contributions
    # -----------------------------
    try:
        initAge_user = np.asarray([float(age_months)], dtype=float)

        # BoxCox transform
        dataMat_trans_user = boxCoxTransform(bundle.boxCox_lam, dataMat_user)

        # ui.py drops these before normAsZscores
        for c in ["LBDTCSI", "LBDHDLSI", "LBDSTRSI"]:
            if c in dataMat_trans_user.columns:
                dataMat_trans_user = dataMat_trans_user.drop(columns=[c])

        # IMPORTANT: mirror ui.py call shape (they pass SEQN-containing DF)
        dataMatNorm_user = normAsZscores_99_young_mf(
            dataMat_trans_user,       # includes SEQN
            q_df,
            bundle.dataMat_trans_ref,
            bundle.qDataMat_R,
        )

        # Fold outliers then drop SEQN col for matrix
        dataMatUser_folded = foldOutliers(dataMatNorm_user, float(bundle.zScoreMax))

        if dataMatUser_folded.shape[1] < 2:
            return {
                "success": False,
                "data": None,
                "errors": ["Post-fold matrix has no feature columns (unexpected artifact mismatch)."],
                "warnings": warnings,
            }

        feature_order = list(dataMatUser_folded.columns[1:])  # drop SEQN
        inputMat_user = dataMatUser_folded.iloc[:, 1:].values.astype(float)  # (1, n_features)

        # Sex-specific selection
        if int(sex) == 1:
            vMatDat99 = bundle.vMatDat99_M
            coxModel = bundle.cox_full_M
            nullModel = bundle.cox_null_M
            coxCovsTrain = bundle.coxCovsTrainM
        else:
            vMatDat99 = bundle.vMatDat99_F
            coxModel = bundle.cox_full_F
            nullModel = bundle.cox_null_F
            coxCovsTrain = bundle.coxCovsTrainF

        pcMat_user = projectToSVD(inputMat_user, vMatDat99)
        pcMat_user = pd.DataFrame(pcMat_user, columns=[f"PC{i+1}" for i in range(pcMat_user.shape[1])])

        sex_user = np.asarray([float(sex)], dtype=float)

        coxCovs_user = np.column_stack([initAge_user, pcMat_user.values, sex_user])
        coxCovs_user = pd.DataFrame(
            coxCovs_user,
            columns=["chronAge"] + list(pcMat_user.columns) + ["sex_user"],
        )

        # Contribution weights (mirror ui.py)
        pc_indices = [int(x[2:]) - 1 for x in coxModel.feature_names_in_ if x.startswith("PC")]

        beta_full = np.zeros(pcMat_user.shape[1], dtype=float)
        beta_full[pc_indices] = np.asarray(coxModel.coef_[1:], dtype=float)

        beta_age_null = float(nullModel.coef_[0])
        beta_age_full = float(coxModel.coef_[0])

        w_feature_years = (vMatDat99 @ beta_full) / beta_age_null
        w_age = (beta_age_full / beta_age_null) - 1.0

        mu_PC = np.zeros(pcMat_user.shape[1], dtype=float)
        mu_PC[pc_indices] = (
            coxCovsTrain.mean()
            .loc[coxModel.feature_names_in_]
            .iloc[1:]
            .values
        )
        mu_age = float(coxCovsTrain["chronAge"].mean())

        mu_Z = mu_PC @ vMatDat99.T
        Z_centered = inputMat_user - mu_Z  # (1, n_features)

        term_features = float(Z_centered @ w_feature_years)
        term_age = float((initAge_user - mu_age) * w_age)

        delta_ba_years = (term_features + term_age) / 12.0
        chrono_years = float(initAge_user[0] / 12.0)
        bio_years = chrono_years + float(delta_ba_years)

        # Per-feature contributions to delta in years
        contrib_vec = (Z_centered.reshape(-1) * np.asarray(w_feature_years).reshape(-1)) / 12.0

        n = min(len(feature_order), contrib_vec.shape[0])
        feature_contributions = [
            {
                "feature": feature_order[i],
                "contribution_years": float(contrib_vec[i]),
            }
            for i in range(n)
            if np.isfinite(contrib_vec[i])
        ]
        feature_contributions.sort(key=lambda d: d["contribution_years"], reverse=True)

        return {
            "success": True,
            "data": {
                "biological_age": float(bio_years),
                "chronological_age": float(chrono_years),
                "delta_ba_ca": float(delta_ba_years),
                "feature_contributions": feature_contributions,
                "imputed_features": imputed_features,
                "features_used": len(feature_order),
                "total_features": len(feature_order),
            },
            "errors": [],
            "warnings": warnings,
        }

    except Exception as e:
        logger.exception("Inference failed")
        return {
            "success": False,
            "data": None,
            "errors": [f"Inference failed: {e}"],
            "warnings": warnings,
        }
