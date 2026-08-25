# Data directories

Keep datasets and experiment outputs outside Git:

- `nnUNet_raw/`: raw nnU-Net datasets.
- `nnUNet_preprocessed/`: fingerprints, plans, and preprocessed cases.
- `nnUNet_results/`: checkpoints, validation predictions, and logs for this baseline.

This project uses `Dataset502_PARSE2022`. Its raw and preprocessed directories
live under this folder, and new checkpoints are written to this project's
`nnUNet_results/` directory.
