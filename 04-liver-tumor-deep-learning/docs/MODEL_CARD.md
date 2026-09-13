# Model card — LiverAI U-Net baseline

## Intended use

Research and education prototype for studying liver-tumour segmentation pipelines and downstream quantitative analysis.

## Not intended for

- diagnosis, treatment planning or clinical decision-making;
- estimating real-world cancer prognosis from the included synthetic demo;
- deployment in a hospital or medical device.

## Baseline

A compact 2-D U-Net maps one CT slice to a binary tumour mask. Training uses a combined BCE and soft-Dice loss. Evaluation reports Dice, IoU and 95th-percentile Hausdorff distance.

## Limitations

The default demo trains on procedurally generated 2-D data. Real liver cancer imaging is 3-D, heterogeneous across scanners/protocols and affected by domain shift, lesion size imbalance, contrast phase, annotation variability and clinically meaningful patient-level confounding.

A credible research extension should therefore evaluate 3-D or 2.5-D architectures, patient-level splits, external validation, calibration, subgroup robustness and uncertainty.
