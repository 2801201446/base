# Data directories

Keep datasets and experiment outputs outside Git:

- `nnUNet_raw/`: raw nnU-Net datasets.
- `nnUNet_preprocessed/`: fingerprints, plans, and preprocessed cases.
- `nnUNet_results/`: checkpoints, validation predictions, and logs for this baseline.

The main README shows environment variables that reuse the existing raw and
preprocessed PARSE2022 data while writing new results into this project.

