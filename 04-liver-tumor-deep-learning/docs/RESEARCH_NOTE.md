# Research note — from segmentation to longitudinal tumour progression

## Research motivation

The clinically interesting problem is not merely to delineate a liver lesion. A useful computational system should transform imaging into quantitative, reproducible representations that can be combined with patient outcomes and potentially support prognosis or treatment follow-up.

## Working hypothesis

A representation combining learned imaging features with explicit tumour morphology and clinical variables may provide more useful longitudinal information than any single representation alone.

## Proposed pipeline

1. CT preprocessing and spatial standardisation.
2. Liver/tumour segmentation.
3. Quantitative morphology and intensity descriptors.
4. Deep feature extraction from the imaging encoder.
5. Fusion with appropriately available clinical variables/outcomes.
6. Patient-level longitudinal prediction and calibrated evaluation.
7. Error analysis, interpretability and external validation.

## Evaluation plan for real data

### Segmentation
- Dice similarity coefficient
- IoU/Jaccard
- HD95
- lesion-wise sensitivity, especially for small tumours

### Prediction
Metric selection depends on the endpoint. For a binary progression endpoint: AUROC, AUPRC, sensitivity/specificity and calibration. For time-to-event outcomes: concordance index and survival-specific calibration. Longitudinal experiments must use patient-level temporal definitions established before model fitting.

## Methodological safeguards

- split by patient, never by slice;
- fit preprocessing only on the training cohort;
- report confidence intervals where feasible;
- explicitly separate internal validation from external validation;
- prevent target leakage from post-outcome variables;
- treat synthetic-demo results as software validation only.

## Next research-grade extension

The next milestone is a 3-D MONAI/nnU-Net baseline on an authorised public liver CT cohort, followed by encoder-feature extraction and outcome modelling on a dataset that genuinely contains longitudinal endpoints. The segmentation and prognosis claims must remain separate until both datasets and evaluation protocols justify their linkage.
