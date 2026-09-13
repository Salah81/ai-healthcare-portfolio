# Data card

## Synthetic validation data

The repository includes a deterministic generator for CT-like 2-D images with liver and tumour masks. Its sole purpose is to make the complete software pipeline executable without distributing patient data.

**Important:** these images are not anatomically or clinically realistic. Any metrics obtained from them are software smoke-test metrics, not medical evidence.

## Intended real-data extension

The project is designed to be extended to public, appropriately licensed liver CT datasets such as the **Liver Tumor Segmentation (LiTS) challenge dataset**, subject to the dataset's access conditions and licence. Real images are deliberately not committed to GitHub.

For a real-data experiment, preserve patient-level train/validation/test separation and document the cohort, acquisition characteristics, inclusion criteria and licence. Do not mix slices from the same patient across splits.

## Privacy

No patient-identifiable data are included in this repository. Never commit DICOM headers or other protected health information.
