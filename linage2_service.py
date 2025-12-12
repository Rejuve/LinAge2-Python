from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from db_mapping import (
    lab_mapping,
    ques_mapping,
    UNIT_SCALE,
    QUESTION_VALUE_TRANSFORMS,
    LAB_VALUE_TRANSFORMS
)

logger = logging.getLogger("linage2_service")

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

from ui_sliders import LAB_VARIABLES  # type: ignore
from imputation import impute_missing_values  # type: ignore


@dataclass(frozen=True)
class LinAge2Bundle:
    cox_full_F: Any
    cox_null_F: Any
    cox_full_M: Any
    cox_null_M: Any

    vMatDat99_F: np.ndarray
    vMatDat99_M: np.ndarray

    boxCox_lam: pd.DataFrame
    dataMat_trans_ref: pd.DataFrame
    qDataMat_R: pd.DataFrame

    coxCovsTrainM: pd.DataFrame
    coxCovsTrainF: pd.DataFrame

    zScoreMax: float = 6.0
    unit_scale: Dict[str, float] = None


def load_linage2_bundle(artifacts_dir: str = "artifacts") -> LinAge2Bundle:
    def p(*parts: str) -> str:
        return os.path.join(artifacts_dir, *parts)

    cox_full_F = joblib.load(p("cox_full_F.joblib"))
    cox_null_F = joblib.load(p("cox_null_F.joblib"))
    cox_full_M = joblib.load(p("cox_full_M.joblib"))
    cox_null_M = joblib.load(p("cox_null_M.joblib"))

    vMatDat99_F = pd.read_csv(p("vMatDat99_F_pre.csv")).values
    vMatDat99_M = pd.read_csv(p("vMatDat99_M_pre.csv")).values

    boxCox_lam = pd.read_csv(p("logNoLog.csv")).iloc[1:2, :]
    dataMat_trans_ref = pd.read_csv(p("dataMat_trans.csv"))

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


def _extract_biometrics(payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[int], Dict[str, Optional[float]], List[str]]:
    """
    Returns:
      age_months, sex, biometrics_numeric, warnings
    biometrics_numeric contains weight_kg, height_cm, waist_cm if present.
    """
    warnings: List[str] = []
    bio = payload.get("biometrics") or {}

    age_years = _to_float(bio.get("age"))
    age_months = None if age_years is None else age_years * 12.0
    if age_months is None:
        warnings.append("Missing biometrics.age")

    sex = _to_int(bio.get("gender"))
    if sex not in (1, 2):
        warnings.append("Missing/invalid biometrics.gender (expected 1=Male, 2=Female)")
        sex = None

    biom = {
        "weight_kg": _to_float(bio.get("weight")),
        "height_cm": _to_float(bio.get("height")),
        "waist_cm": _to_float(bio.get("waist_circumference")),
        "bmi": _to_float(bio.get("bmi")),
    }
    return age_months, sex, biom, warnings


def _derive_bmi(weight_kg: Optional[float], height_cm: Optional[float]) -> Optional[float]:
    if weight_kg is None or height_cm is None:
        return None
    if not np.isfinite(weight_kg) or not np.isfinite(height_cm):
        return None
    if weight_kg <= 0 or height_cm <= 0:
        return None
    h_m = height_cm / 100.0
    if h_m <= 0:
        return None
    bmi = weight_kg / (h_m * h_m)
    # sanity bounds (keeps garbage payloads from poisoning output)
    if bmi < 5 or bmi > 120:
        return None
    return float(bmi)


def _apply_unit_scaling(labs: Dict[str, float], unit_scale: Dict[str, float]) -> Dict[str, float]:
    out = dict(labs)
    for k, v in list(out.items()):
        factor = unit_scale.get(k)
        if factor is not None:
            out[k] = float(v) * float(factor)
    return out


def _split_and_remap_survey_items(payload: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, int], List[str]]:
    warnings: List[str] = []
    labs: Dict[str, float] = {}
    ques: Dict[str, int] = {}

    items = payload.get("surveys") or []
    if not isinstance(items, list):
        return labs, ques, ["payload.surveys must be a list"]

    for it in items:
        it = it or {}
        qid = _to_int(it.get("ques_id"))
        ans_raw = it.get("answer")

        if qid is None:
            warnings.append("Survey item with missing/invalid ques_id (skipped)")
            continue

        # ---------- Labs ----------
        if qid in lab_mapping:
            nh = lab_mapping[qid]

            # Apply optional per-lab transform first (e.g., LBXCOT mapping)
            transformed = ans_raw
            if nh in LAB_VALUE_TRANSFORMS:
                try:
                    transformed = LAB_VALUE_TRANSFORMS[nh](ans_raw)
                except Exception as e:
                    warnings.append(f"Lab {qid}->{nh}: transform failed for value {ans_raw!r}: {e} (skipped)")
                    continue

            v = _to_float(transformed)
            if v is None:
                warnings.append(f"Lab {qid}->{nh}: invalid numeric value {ans_raw!r} (skipped)")
                continue

            labs[nh] = float(v)
            continue

        # ---------- Questions ----------
        if qid in ques_mapping:
            nh = ques_mapping[qid]
            transform = QUESTION_VALUE_TRANSFORMS.get(nh)
            try:
                v_int = transform(ans_raw) if transform is not None else _to_int(ans_raw)
            except Exception as e:
                warnings.append(f"Question {qid}->{nh}: transform failed for value {ans_raw!r}: {e} (skipped)")
                continue

            if v_int is None:
                warnings.append(f"Question {qid}->{nh}: invalid/unmappable value {ans_raw!r} (skipped)")
                continue

            ques[nh] = int(v_int)
            continue

        warnings.append(f"Unmapped DB ques_id {qid} (ignored)")

    return labs, ques, warnings


def process_payload(payload: Dict[str, Any], bundle: LinAge2Bundle) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    age_months, sex, biom, w0 = _extract_biometrics(payload)
    warnings.extend(w0)

    if age_months is None:
        errors.append("biometrics.age is required")
    if sex is None:
        errors.append("biometrics.gender is required (1=Male,2=Female)")

    labs_raw, ques_raw, w1 = _split_and_remap_survey_items(payload)
    warnings.extend(w1)

    if errors:
        return {"success": False, "data": None, "errors": errors, "warnings": warnings}

    labs = _apply_unit_scaling(labs_raw, bundle.unit_scale or {})

    # Derive BMI from biometrics if missing
    derived_bmi = _derive_bmi(biom.get("weight_kg"), biom.get("height_cm"))
    if biom.get('bmi') is not None:
        labs["BMXBMI"] = biom.get('bmi')
    else:
        if "BMXBMI" not in labs and derived_bmi is not None:
            labs["BMXBMI"] = derived_bmi
        elif "BMXBMI" not in labs and derived_bmi is None:
            warnings.append("BMXBMI not provided and could not be derived from biometrics (weight/height) — will be imputed")

    # Questionnaire defaults
    q_row: Dict[str, Any] = {
        "SEQN": 1,
        "RIAGENDR": int(sex),
        "RIDAGEEX": float(age_months),

        "BPQ020": 2,
        "DIQ010": 2,
        "KIQ020": 2,
        "MCQ010": 2,
        "MCQ053": 2,

        "MCQ160A": 2, "MCQ160B": 2, "MCQ160C": 2, "MCQ160D": 2, "MCQ160E": 2,
        "MCQ160F": 2, "MCQ160G": 2, "MCQ160I": 2, "MCQ160J": 2, "MCQ160K": 2, "MCQ160L": 2,
        "MCQ220": 2,

        "OSQ010A": 2, "OSQ010B": 2, "OSQ010C": 2, "OSQ060": 2,
        "PFQ056": 2,
        "HUQ070": 2,

        "HUQ010": 3,
        "HUQ020": 3,
        "HUQ050": 0,
    }
    q_row.update(ques_raw)
    q_df = pd.DataFrame([q_row])

    # Build raw lab vector + missing flags
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
        return {"success": False, "data": None, "errors": [f"Imputation failed: {e}"], "warnings": warnings}

    dataMat_user = pd.DataFrame({name: [val] for name, val in zip(LAB_VARIABLES, lab_vals_imputed)})
    dataMat_user.insert(0, "SEQN", q_df["SEQN"])

    # Derived features
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
        return {"success": False, "data": None, "errors": [f"Derived feature computation failed: {e}"], "warnings": warnings}

    # Inference pipeline
    try:
        initAge_user = np.asarray([float(age_months)], dtype=float)

        dataMat_trans_user = boxCoxTransform(bundle.boxCox_lam, dataMat_user)
        dataMatNorm_user = normAsZscores_99_young_mf(
            dataMat_trans_user.drop(['LBDTCSI', 'LBDHDLSI', 'LBDSTRSI'], axis=1),
            q_df,
            bundle.dataMat_trans_ref,
            bundle.qDataMat_R,
        )
        dataMatUser_folded = foldOutliers(dataMatNorm_user, float(bundle.zScoreMax))

        feature_order = list(dataMatUser_folded.columns[1:])
        inputMat_user = dataMatUser_folded.iloc[:, 1:].to_numpy(dtype=float)

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

        pc_indices = [int(x[2:]) - 1 for x in coxModel.feature_names_in_ if x.startswith("PC")]

        beta_full = np.zeros(pcMat_user.shape[1], dtype=float)
        beta_full[pc_indices] = np.asarray(coxModel.coef_[1:], dtype=float)

        beta_age_null = float(nullModel.coef_[0])
        beta_age_full = float(coxModel.coef_[0])

        w_feature_months_per_sd = (vMatDat99 @ beta_full) / beta_age_null
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
        Z_centered = inputMat_user - mu_Z

        term_features = float(Z_centered @ w_feature_months_per_sd)  # months
        term_age = float((initAge_user - mu_age) * w_age)            # months

        delta_ba_years = (term_features + term_age) / 12.0
        chrono_years = float(initAge_user[0] / 12.0)
        bio_years = chrono_years + float(delta_ba_years)

        contrib_years = (Z_centered.reshape(-1) * np.asarray(w_feature_months_per_sd).reshape(-1)) / 12.0
        n = min(len(feature_order), contrib_years.shape[0])
        imputed_set = set(imputed_features)

        feature_contributions = [
            {
                "feature": feature_order[i],
                "contribution_years": float(contrib_years[i]),
                "is_imputed": feature_order[i] in imputed_set,
            }
            for i in range(n)
            if np.isfinite(contrib_years[i])
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
        return {"success": False, "data": None, "errors": [f"Inference failed: {e}"], "warnings": warnings}
