lab_mapping = {
  # Vitals / anthropometry
  99001: "BPXPLS", #"Pulse rate (beats/min, 60-sec pulse).",   # CDC BPX doc
  11810: "BPXSAR",# "Systolic blood pressure — average reported to examinee (mmHg).",
  11820: "BPXDAR",# "Diastolic blood pressure — average reported to examinee (mmHg).",
  #"BMXBMI": "Body mass index (kg/m²).",

  # Kidney (urine)
  99002: "URXUMASI", # "Urine albumin (microalbumin), SI units (e.g., mg/L→mg/L; often used for UACR).",
  99003: "URXUCRSI", # "Urine creatinine, SI units (mmol/L).",

  # Iron panel
  10270: "LBDIRNSI", #"Serum iron (µmol/L).", #must be multiplied by 0.179 as its in "mg/dL" in db
  99004: "LBDTIBSI", # "Total iron binding capacity, TIBC (µmol/L).",
  99005: "LBXPCT", #"Transferrin saturation (%) = (serum iron / TIBC)×100.",
  10570: "LBDFERSI", # "Ferritin (µg/L).",

  # Folate / B12 / cotinine
  10610: "LBDFOLSI", #"Serum folate (nmol/L).",
  99006: "LBDB12SI": "Vitamin B12 (pmol/L).", #the one in db is a dietary intake
  13350: "LBXCOT",  #  "Smoking status: 0 - Non-smoker, 1 - Light/Recent, 2 - Heavy/Current",

  # CBC (white cells)
  10350: "LBXWBCSI" #"White blood cell count (×10⁹/L).",
  99007: "LBXLYPCT", # "Lymphocytes (%).",
  99008: "LBXMOPCT": "Monocytes (%).",
  10370 : "LBXNEPCT", #"Neutrophils (%).",
  10380: "LBXEOPCT", #"Eosinophils (%).",
  13170: "LBXBAPCT", #"Basophils (%).",
  99009: "LBDLYMNO": "Lymphocytes (×10⁹/L).",
  99010: "LBDMONO":  "Monocytes (×10⁹/L).",
  10391: "LBDNENO", #"Neutrophils (×10⁹/L).",
  99011: "LBDEONO":  "Eosinophils (×10⁹/L).",
  99012: "LBDBANO":  "Basophils (×10⁹/L).",

  # CBC (red cells / platelets)
  13210: "LBXRBCSI", #"Red blood cell count (×10¹²/L).",
  10400: "LBXHGB", #"Hemoglobin (g/dL).",
  13220: "LBXHCT", #"Hematocrit (%).",
  99013: "LBXMCVSI": "Mean corpuscular volume, MCV (fL).",
  99014: "LBXMCHSI": "Mean corpuscular hemoglobin, MCH (pg).",
  99015: "LBXMC":    "Mean corpuscular hemoglobin concentration, MCHC (g/dL).",
  10430: "LBXRDW",  #"Red cell distribution width (%).",
  13240: "LBXPLTSI", #"Platelet count (×10⁹/L).",
  13250: "LBXMPSI", #"Mean platelet volume, MPV (fL).",

  # Inflammation / glycemia / cardiac
  13440: "LBXCRP", #"C-reactive protein (mg/L).",
  99016: "LBXGH":  "Glycohemoglobin (HbA1c, %).",
  99017: "SSBNP":  "N-terminal pro-B-type natriuretic peptide (NT-proBNP, pg/mL).",

  # Basic chem (SI set)
  13280: "LBDSALSI", #"Albumin (g/L).", #must be divided by 10 as its in g/dL in database!
  10170: "LBXSATSI", # "Alanine aminotransferase, ALT (U/L).",
  10190: "LBXSASSI", # "Aspartate aminotransferase, AST (U/L).",
  10180: "LBXSAPSI", #"Alkaline phosphatase (U/L).",
  10200: "LBDSBUSI", # "Urea nitrogen (BUN), SI (mmol/L).", !must be multiplied by 0.357 as its in "mg/dL" in db
  13330: "LBDSCASI", # "Calcium, SI (mmol/L).", #must be multiplied by 0.2495 as its in "mg/dL" in db
  13290: "LBXSC3SI", # "Bicarbonate (total CO₂), SI (mmol/L).",
  10641: "LBDSGLSI", # "Glucose, SI (mmol/L).", #must be multiplied by 0.0555 as its in "mg/dL" in db
  13300: "LBXSLDSI", # "Lactate dehydrogenase, LDH (U/L).",
  13310: "LBDSPHSI", #"Phosphorus (mmol/L).", #must be multiplied by 0.323 as its in "mg/dL" in db
  10290: "LBDSTBSI", #"Total bilirubin (µmol/L).",#must be multiplied by 17.104 as its in "mg/dL" in db
  10320: "LBDSTPSI", #"Total protein (g/L).", # must be divided by 10 as its in g/dL in database
  10340: "LBDSUASI", #"Uric acid (µmol/L).", #must be multiplied by 59.48 as its in "mg/dL" in db
  10230: "LBDSCRSI", #"Creatinine (µmol/L).",#must be multiplied by 88.4 as its in "mg/dL" in db
  10280: "LBXSNASI", #"Sodium (mmol/L).",
  13320: "LBXSKSI", #"Potassium (mmol/L).",
  10210: "LBXSCLSI", #"Chloride (mmol/L).",
  10240: "LBDSGBSI", # "Globulin (g/L).",# must be divided by 10 as its in g/dL in database

  # Derived / study-specific
  #"fs1Score": "Comorbidity/Frailty index.",
  #"fs2Score": "Self-rated health × trajectory.",
  #"fs3Score": "Healthcare use (past year).",
  #"LDLV":     "Calculated LDL cholesterol (Friedewald or NHANES calc; mg/dL).",
  #"crAlbRat": "Urine albumin-to-creatinine ratio (UACR).",
}

ques_mapping = {
    14190: "BPQ020", #Ever told you had high blood pressure?
    14199: "DIQ010", #"Doctor told you have diabetes?"
    14254: "KIQ020", #"Ever told you had weak/failing kidneys (excl. stones/cancer)?" #NB its KIQ022 in the db which is the same thing
    14219: "MCQ010", #Has a doctor or other health professional ever told you that you have asthma?
    99018: "MCQ053",
    14222: "MCQ160A", 	#Has a doctor or other health professional ever told you that you had arthritis?
    14223: 'MCQ160B',
    14224: 'MCQ160C',
    99019: "MCQ160D", #Ever told you had angina/angina pectoris
    14225: 'MCQ160E',
    14226: 'MCQ160F',
    14227: 'MCQ160G',
    99020: "MCQ160I", #{"label": "Ever told you had stroke?", "choices": [("Yes",1),("No",2)], "value": 2},
    99021: "MCQ160J", #{"label": "Ever told you had chronic bronchitis?", "choices": [("Yes",1),("No",2)], "value": 2},
    99022: "MCQ160K", #{"label": "Ever told you had liver condition?", "choices": [("Yes",1),("No",2)], "value": 2},
    99023: "MCQ160L", #{"label": "Ever told you had thyroid problem?", "choices": [("Yes",1),("No",2)], "value": 2},
    99024: "MCQ220", #{"label": "Ever diagnosed with any other serious illness?"
    99025: "OSQ010A", #{"label": "Mouth pain — aching in mouth (past year)?", "choices": [("Yes",1),("No",2)], "value": 2},
    99026: "OSQ010B", #{"label": "Mouth pain — tooth sensitive to hot/cold/sweets?", "choices": [("Yes",1),("No",2)], "value": 2},
    99027: "OSQ010C", #{"label": "Mouth pain — toothache (past 6 months)?", "choices": [("Yes",1),("No",2)], "value": 2},
    99028: "OSQ060",  #{"label": "Difficulty sleeping because of teeth/gums?", "choices": [("Yes",1),("No",2)], "value": 2},
    99030: "PFQ056", #{"label": "Any difficulty walking/using steps without equipment?"
    99031: "HUQ010", #"General health condition"
    14211: 'HUQ020', #"Compared with 1 year ago, your health is…"
    99032: "HUQ050", #"Times received healthcare
]
}
    

