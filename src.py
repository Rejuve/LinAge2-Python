import pandas as pd
import numpy as np
from scipy import stats
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

## Load libraries
## For survival analysis - using scikit-survival
## To make figures - using matplotlib

## Clear workspace
# (Not needed in Python - variables are local to script)

## Fix seed - maintain direction of PCs
np.random.seed(111222333)


def markIncsFromCodeBook(codeBook):
    """Selects all columns marked as data ("1" in Data column) in codebook for inclusion in master dataset"""
    incFlags = codeBook["Data"].values  ## This column is "1" for include in dataset, 0 for not (comments or demo data)
    incNames = codeBook.iloc[:, 0].astype(str).values
    incList = np.column_stack((incNames, incFlags))
    return incList


def dropCols(dataSet, incList):
    """Simply drops everything not flagged with 1 in incList (Data)"""
    incInds = np.where(incList[:, 1].astype(int) == 1)[0]
    incTerms = incList[:, 0][incInds]
    nTerms = len(incTerms)
    
    ## The first column is always the subject number (SEQN) - add that back
    outData = dataSet.iloc[:, 0:1].copy()
    
    for i in range(1, nTerms):
        ## loop over terms that have a "1" in column 2 of the incList, find those in
        ## dataSet and include in output dataSet
        nextTerm = incTerms[i]
        if nextTerm in dataSet.columns:
            nextCol = dataSet[nextTerm]
            outData = pd.concat([outData, nextCol], axis=1)
    
    ## Name first column appropriately and return resulting dataset
    outData.columns = ['SEQN'] + list(outData.columns[1:])
    return outData


def dropNAcolumns(dataSet, pNAcut, incSwitch, verbose):
    """This takes a single cutoff fraction and drops all columns (features)
    that contain more NAs than allowed by the cutoff.
    However, if force include flag is set (==1) for a column, we will force the inclusion of feature"""
    
    nRows = dataSet.shape[0]
    nCols = dataSet.shape[1]
    forceFlags = np.zeros(nCols)
    
    ## If incSwitch IS set, we read them from the codebook
    if incSwitch == 1:
        ## Read include flags from codebook
        codeBookFlags = codeBook['ForceInc'].values
        ## Identify column terms that we cannot drop
        nForced = np.sum(codeBookFlags)
        forceIncTerms = codeBook['Var'][codeBookFlags == 1].values
        ## Now identify columns in dataset that need to be retained
        for i in range(nForced):
            nextTerm = forceIncTerms[i]
            if nextTerm in dataSet.columns:
                forceThis = dataSet.columns.get_loc(nextTerm)
                ## Flip the respective forceFlag to 1 - this column cannot be dropped
                nrOfNAs = dataSet.iloc[:, forceThis].isna().sum()
                if verbose:
                    print(f"Applying force flag to: {nextTerm}")
                    print(f"\t - this will include:\t {nrOfNAs}\t NAs in:\t {nRows}  {round(nrOfNAs/nRows*100, 2)}%")
                forceFlags[forceThis] = 1
    
    ## Now drop all columns with too many NAs
    naColSum = dataSet.isna().sum()
    naColP = naColSum / nRows
    
    ## Keep only those columns (features) for which naColP (number of NAs) is smaller than pNAcut
    keepCols = naColP < pNAcut
    
    ## Finally, recover all columns that we decided to force (retain)
    ## Merge keepCols (columns that will be kept due to cutoff) and forceFlags
    keepCols = keepCols | (forceFlags == 1)
    dataSet = dataSet.loc[:, keepCols]
    
    ## Print dimension of surviving matrix and list of surviving variables
    nrows = dataSet.shape[0]
    ncols = dataSet.shape[1]
    varNames = dataSet.columns
    humNames = varNames
    
    for i in range(1, len(varNames)):
        varName = varNames[i]
    
    return dataSet


def qDataMatGen(masterData, incList):
    """Loop over masterData and keep any column that has a zero in the "data" column"""
    allTerms = masterData.columns
    nTerms = masterData.shape[1]  ## number of total terms (columns) in masterData
    nIncFlags = incList.shape[0]  ## number of terms in codebook - for which we know include flags
    
    ## The first column of the qDataMatrix has to be SEQN number - add these first
    qDataNames = ["SEQN"]
    qDataMatrix = masterData.iloc[:, 0:1].copy()
    
    ## Loop over all terms (columns) in masterData - extract one term (column) at a time
    for i in range(1, nTerms):
        ## look at the next term in the data and get the respective flag from the incList
        nextTerm = allTerms[i]
        
        ## Now loop over all terms in the incList and get the flag for the current term
        flag = 0  ## Graceful default is 0 - not include
        for j in range(nIncFlags):
            if incList[j, 0] == nextTerm:
                ## Read the inc flag (second entry of that column) and return it
                flag = int(incList[j, 1])
        
        if flag == 0:
            ## If include == 0, we will include that parameter in the qDataMatrix
            qDataColumn = masterData.iloc[:, i]  ## Keep the current column for inclusion to qDataMatrix
            qDataMatrix = pd.concat([qDataMatrix, qDataColumn], axis=1)  ## Add current column to qDataMatrix
            qDataNames.append(nextTerm)  ## Also keep the current column name (nextTerm) as column name
    
    qDataMatrix.columns = qDataNames  ## Update all column names
    return qDataMatrix  ## Return the matrix


def getNonNARows(dataSet):
    """Identify rows that contain NAs and drop them by only retaining those that do not
    sums over NAs are NA so only rows with no (zero) NAs return !is.na"""
    keepRows = (dataSet.isna().sum(axis=1) == 0)
    return keepRows


#########################  < CALCULATING DERIVED FEATURES FROM DATA >  #####################################

def popPCFIfs1(qDataMat):
    """This will calculate our frailty index / disease and comorbidity index for each subject
    and populate the matrix
    
    NOTE: we will allow NAs here - so check that the variables are all there"""
    
    BPQ020 = qDataMat["BPQ020"].copy()
    DIQ010 = qDataMat["DIQ010"].copy()
    HUQ010 = qDataMat["HUQ010"].copy()
    HUQ020 = qDataMat["HUQ020"].copy()
    HUQ050 = qDataMat["HUQ050"].copy()
    HUQ070 = qDataMat["HUQ070"].copy()
    KIQ020 = qDataMat["KIQ020"].copy()
    MCQ010 = qDataMat["MCQ010"].copy()
    MCQ053 = qDataMat["MCQ053"].copy()
    MCQ160A = qDataMat["MCQ160A"].copy()
    MCQ160B = qDataMat["MCQ160B"].copy()
    MCQ160C = qDataMat["MCQ160C"].copy()
    MCQ160D = qDataMat["MCQ160D"].copy()
    MCQ160E = qDataMat["MCQ160E"].copy()
    MCQ160F = qDataMat["MCQ160F"].copy()
    MCQ160G = qDataMat["MCQ160G"].copy()
    MCQ160I = qDataMat["MCQ160I"].copy()
    MCQ160J = qDataMat["MCQ160J"].copy()
    MCQ160K = qDataMat["MCQ160K"].copy()
    MCQ160L = qDataMat["MCQ160L"].copy()
    MCQ220 = qDataMat["MCQ220"].copy()
    OSQ010A = qDataMat["OSQ010A"].copy()
    OSQ010B = qDataMat["OSQ010B"].copy()
    OSQ010C = qDataMat["OSQ010C"].copy()
    OSQ060 = qDataMat["OSQ060"].copy()
    PFQ056 = qDataMat["PFQ056"].copy()
    
    ## Give "safe" value to all NAs ...
    BPQ020.fillna(2, inplace=True)
    DIQ010.fillna(2, inplace=True)
    HUQ010.fillna(3, inplace=True)
    HUQ020.fillna(3, inplace=True)
    HUQ050.fillna(0, inplace=True)
    HUQ070.fillna(2, inplace=True)
    KIQ020.fillna(2, inplace=True)
    MCQ010.fillna(2, inplace=True)
    MCQ053.fillna(2, inplace=True)
    MCQ160A.fillna(2, inplace=True)
    MCQ160B.fillna(2, inplace=True)
    MCQ160C.fillna(2, inplace=True)
    MCQ160D.fillna(2, inplace=True)
    MCQ160E.fillna(2, inplace=True)
    MCQ160F.fillna(2, inplace=True)
    MCQ160G.fillna(2, inplace=True)
    MCQ160I.fillna(2, inplace=True)
    MCQ160J.fillna(2, inplace=True)
    MCQ160K.fillna(2, inplace=True)
    MCQ160L.fillna(2, inplace=True)
    MCQ220.fillna(2, inplace=True)
    OSQ010A.fillna(2, inplace=True)
    OSQ010B.fillna(2, inplace=True)
    OSQ010C.fillna(2, inplace=True)
    OSQ060.fillna(2, inplace=True)
    PFQ056.fillna(2, inplace=True)
    
    ## Binary yes/no decision vector
    binVec = np.column_stack([
        (BPQ020 == 1), ((DIQ010 == 1) | (DIQ010 == 3)), (KIQ020 == 1), (MCQ010 == 1), (MCQ053 == 1),
        (MCQ160A == 1), (MCQ160C == 1), (MCQ160D == 1), (MCQ160E == 1), (MCQ160F == 1),
        (MCQ160G == 1), (MCQ160I == 1), (MCQ160J == 1), (MCQ160K == 1), (MCQ160L == 1),
        (MCQ220 == 1), (OSQ010A == 1), (OSQ010B == 1), (OSQ010C == 1), (OSQ060 == 1),
        (PFQ056 == 1), (HUQ070 == 1)
    ])
    
    sumOverBinVec = binVec.sum(axis=1) / 22
    return sumOverBinVec


def popPCFIfs2(qDataMat):
    
    HUQ010 = qDataMat["HUQ010"].copy()
    HUQ020 = qDataMat["HUQ020"].copy()
    HUQ010.fillna(3, inplace=True)
    HUQ020.fillna(3, inplace=True)
    
    ## If sick/feeling bad, get score of 2 to 4 - if getting worse -> get 2x modifier
    ## if getting better -> 1/2 modifier
    aVec = ((HUQ010 == 4) * 2 + (HUQ010 == 5) * 4)
    dVec = (1 - (HUQ020 == 1) * 0.5 + (HUQ020 == 2))
    fScore = aVec * dVec
    
    return fScore


def popPCFIfs3(qDataMat):
    """This basically codes NHANES HUQ050: "Number times received healthcare over past year" """
    HUQ050 = qDataMat["HUQ050"].copy()
    HUQ050.fillna(0, inplace=True)
    HUQ050[HUQ050 == 77] = 0  ## Comment codes ("Refused")
    HUQ050[HUQ050 == 99] = 0  ## Comment codes ("Do not know")
    return HUQ050


def populateLDL(dataMat, qDataMat):
    """This function will calculate LDL and adds it to the dataMatrix
    LDL - calculated from:
       Variable: LBDTCSI	        Total Cholesterol (mmol/L)
       Variable: LBDHDLSI	HDL (mmol/L)
       Variable: LBDSTRSI	Triglycerides (mmol/L)
    Formula:  LDL-C=(TC)–(triglycerides/5)– (HDL-C).
    NOTES: Can be inaccurate if triglycerides are very high (above 150 mg/dL)"""
    
    nSubs = dataMat.shape[0]
    
    ## Extract all relevant variables from data matrix
    totCv = dataMat["LBDTCSI"].values
    HDLv = dataMat["LBDHDLSI"].values
    triGv = dataMat["LBDSTRSI"].values
    seqVec = dataMat["SEQN"].values
    LDLvec = np.zeros(nSubs)
    
    ## Loop over all subjects and update LDL
    for i in range(nSubs):
        totC = totCv[i]
        HDL = HDLv[i]
        TG = triGv[i]
        LDL = 0
        
        ## Check that we do not have any NAs here
        # actual condition is supposed to be  ~(np.isnan(totC) or np.isnan(HDL) or np.isnan(TG))
        # but we have buggy R prototype and have to replicate that
        condition = (not np.isnan(totC)) * (not np.isnan(HDL)) * (not np.isnan(TG))
        if condition:
            ## Calculate LDL from triglycerides and total cholesterol
            LDL = (totC - (TG / 5) - (HDL))
        
        LDLvec[i] = LDL
    
    return LDLvec


#############################  < DATA SELECTION - ROWS / SUBJECTS >  ####################################

def selectAgeBracket(qMat, ageCutLower, ageCutUpper):
    """Apply a age bracket to dataset - only retain samples between upper and lower age limit"""
    keepRows = ((qMat["RIDAGEYR"] >= ageCutLower) & (qMat["RIDAGEYR"] <= ageCutUpper))
    return keepRows


def nonAccidDeathFlags(qMat):
    """Here we will return keep flags for all subjects who die of non-accidental deaths
    The cause of death (leading) is recorded (if known) in the questionnaire data matrix
    qDatMat in the "UCOD_LEADING" column
    Possible values in "UCOD_LEADING" are:
    001 = Disease of the heart
    002 = Malignant neoplasm
    003 = Chronic lower respiratory disease
    004 = Accidents and unintentional injuries
    005 = Cerebrovascular disease
    007 = Diabetes
    008 = Influenza and pneumonia
    009 = Nephritis, kidney issues
    010 = All other causes (residuals)
    NA  = no info (the vast majority of cases)"""
    
    ## Extract cause of deaths
    causeOfDeath = qMat["UCOD_LEADING"].copy()
    ## Then drop NAs (turn into zeros)
    causeOfDeath.fillna(0, inplace=True)
    keepFlags = causeOfDeath != 4
    
    return keepFlags


def foldOutliers(dataMatNorm, zScoreMax):
    """Fold in outlier z-scores"""
    
    print(f"> Folding in outliers at maximum total zScore: {zScoreMax}", end="")
    ## Now truncate / fold outliers and show boxplots
    
    dataMatNorm_folded = dataMatNorm.copy()
    allTerms = dataMatNorm.columns[1:]
    
    for nextTerm in allTerms:
        colVals = dataMatNorm[nextTerm].values
        if np.isinf(colVals).any():
            print(f"Infinite value in: {nextTerm}")
        ## boxplot(colVals,main=paste(nextTerm,"-before"))
        foldThese = np.abs(colVals) > zScoreMax
        colVals[foldThese] = np.sign(colVals[foldThese]) * zScoreMax
        ## boxplot(colVals,main=paste(nextTerm,"-after"))
        ## readline()
        dataMatNorm_folded[nextTerm] = colVals
    
    print(" ... Done")
    return dataMatNorm_folded


def digiCot(dataMat):
    """Digitize continine to turn into smoking intensity
    Most clinics do not routinely measure cotinine - so here we will
    bin cot as follows:
    0  <= cot < 10 are non smokers (0)
    10 >= cot < 100 are light smokers (1)
    100 >= cot < 200 are moderate smokers (2)
    anything above 200 is a heavy smoker (3)"""
    
    print("> Digitizing cotinine data ... ", end="")
    cot = dataMat["LBXCOT"].copy()
    dataMat_out = dataMat.copy()
    dataMat_out.loc[cot < 10, "LBXCOT"] = 0
    dataMat_out.loc[(cot >= 10) & (cot < 100), "LBXCOT"] = 1
    dataMat_out.loc[(cot >= 100) & (cot < 200), "LBXCOT"] = 2
    dataMat_out.loc[cot >= 200, "LBXCOT"] = 3
    print("Done\n")
    return dataMat_out


def normAsZscores_99_young_mf(dataSet, qDataMat, dataSet_ref, qDataMat_ref):
    """Normalize by training set (1999–2000 only), sex-specific median & MAD (R-equivalent)."""

    # --- reference subset: yearsNHANES == 9900 and RIDAGEYR <= 50 ---
    years = qDataMat_ref["yearsNHANES"]
    seqSel = (years == 9900) | (years.astype(str) == "9900")
    ageSel = qDataMat_ref["RIDAGEYR"] <= 50
    selVec = seqSel & ageSel

    dataSet_temp = dataSet_ref.loc[selVec].copy()
    sexSel_temp = (qDataMat_ref.loc[selVec, "RIAGENDR"] == 1)  # True for males in REF
    sexSel = (qDataMat["RIAGENDR"] == 1)                       # True for males in TARGET

    nRows, nCols = dataSet.shape
    dataMatN = dataSet.copy()

    # column 0 stays as-is (e.g., SEQN)
    dataMatN.iloc[:, 0] = dataSet.iloc[:, 0]

    skipCols = {"fs1Score", "fs2Score", "fs3Score", "LBXCOT", "LBDBANO"}

    def safe_z(x, med, mad_val):
        if np.isnan(med) or np.isnan(mad_val):
            return np.full_like(x, np.nan, dtype=float)
        if mad_val == 0 or np.isclose(mad_val, 0.0):
            # exact median -> 0, otherwise NaN (mirrors division-by-zero behavior without infs)
            return np.where(np.isfinite(x), np.where(x == med, 0.0, np.nan), np.nan)
        return (x - med) / mad_val

    for col in range(1, nCols):
        name = dataSet.columns[col]
        if name in skipCols:
            continue

        # reference data split by sex (coerce to numeric; NAs are handled downstream)
        ref_m = pd.to_numeric(dataSet_temp.loc[sexSel_temp, name], errors="coerce").to_numpy()
        ref_f = pd.to_numeric(dataSet_temp.loc[~sexSel_temp, name], errors="coerce").to_numpy()

        # NA-robust medians (align with MAD's nan omission)
        med_m = np.nanmedian(ref_m)
        med_f = np.nanmedian(ref_f)

        # MAD with normal-consistent scaling (matches R's mad default), omit NaNs
        mad_m = stats.median_abs_deviation(ref_m, scale="normal", nan_policy="omit")
        mad_f = stats.median_abs_deviation(ref_f, scale="normal", nan_policy="omit")

        # target column values
        x = pd.to_numeric(dataSet.iloc[:, col], errors="coerce").to_numpy()

        # compute male/female z-scores then merge by target sex
        z_m = safe_z(x, med_m, mad_m)
        z_f = safe_z(x, med_f, mad_f)
        z = np.where(sexSel.to_numpy(), z_m, z_f)

        dataMatN.iloc[:, col] = z

    return dataMatN

def boxCoxTransform(boxCox_lam, dataMat):
    """Apply box cox transforms based on lambda given"""
    dataMat_out = dataMat.copy()
    allTerms = dataMat.columns[1:]
    print("> Applying boxCox transformed  ... ", end="")
    for nextTerm in allTerms:
        ## Get column number
        dataColNr = dataMat.columns.get_loc(nextTerm)
        if nextTerm in boxCox_lam.columns:
            lamNr = boxCox_lam.columns.get_loc(nextTerm)
            ## Get next transformation
            nextLam = boxCox_lam.iloc[0, lamNr]
            ## Get next data item (column)
            colVals = dataMat[nextTerm].copy()
            ## Selection of transformation is based on lambda value
            if not pd.isna(nextLam):  ## If NA, do nothing
                if nextLam == 0:
                    colVals = np.log(colVals)  ## If the lambda value is zero, we log the data column
                else:
                    colVals = (colVals**nextLam - 1) / nextLam  ## If it is neither NA nor zero - boxCox formula for lambda
            dataMat_out[nextTerm] = colVals
    
    print("Done")
    return dataMat_out


def projectToSVD(inputMat, svdCoordMat):
    """Project inputMat data matrix into the same PC coordinates provided by svdCoordMat"""
    
    print("> Projecting data into PC coordinates  ... ", end="")
    mSamples = inputMat.shape[0]
    nSVs = svdCoordMat.shape[1]
    pcMat = np.zeros((mSamples, nSVs))  ## Empty data matrix in PC coordinates
    
    ## Doing loop to calculate coordinates for samples in terms of PCs - could do matrix mult instead
    for sample in range(mSamples):
        ## Current sample is current row of data (input) matrix
        curSample = inputMat[sample, :]
        
        ## Now loop over all nSVs and determine
        for pcNr in range(nSVs):
            ## current PC vector is the column
            curPC = svdCoordMat[:, pcNr]
            coord = curSample @ curPC
            pcMat[sample, pcNr] = coord
    
    print("Done")
    return pcMat


def calcBioAge_R_equiv(coxModelNew, nullModel, dataTable, trainTable):
    """
    Reproduce R's calcBioAge:
      risk = exp( (X - colMeans_train) @ beta )
      logRiskRatio = log(risk_full / risk_null) = [(X - mu_full)@b_full] - [(X - mu_null)@b_null]
      delta = (logRiskRatio / log(2)) * MRDT, where MRDT = log(2) / beta_age(null)
    Arguments:
      coxModelNew: fitted CoxPHSurvivalAnalysis (full model)
      nullModel:   fitted CoxPHSurvivalAnalysis (chronAge-only)
      dataTable:   DataFrame of covariates for which to compute bioage (test or train block)
      trainTable:  DataFrame used to fit the models (same columns, training rows)
    """
    # coefficients
    b_full = coxModelNew.coef_
    b_null = nullModel.coef_

    # MRDT from null model’s age coefficient (do NOT round)
    beta_age = b_null[0]
    MRDT = np.log(2) / beta_age

    # columns used by each model (scikit-survival stores this)
    cols_full = list(coxModelNew.feature_names_in_)
    cols_null = ["chronAge"]

    # training means used for centering (R: model$means)
    mu_full = trainTable[cols_full].mean().to_numpy()
    mu_null = trainTable[cols_null].mean().to_numpy()

    # centered linear predictors (no exp needed since we take log-risk ratio)
    Xf = dataTable[cols_full].to_numpy()
    Xn = dataTable[cols_null].to_numpy()

    lp_full = (Xf - mu_full) @ b_full
    lp_null = (Xn - mu_null) @ b_null

    logRiskRatio = lp_full - lp_null
    delta = (logRiskRatio / np.log(2)) * MRDT
    return delta