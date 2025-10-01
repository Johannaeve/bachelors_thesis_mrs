# Bachelor’s Thesis
My bachelor's thesis on magnetic resonance spectroscopy with and without J-difference editing.

This repository contains:

- `final/thesis.pdf` — the final thesis document
- `code/` — evaluation functions and an example usage script
- `data/example_spectrum.nii.gz` — one example spectrum for demonstration

## Environment (conda)
Create the environment from the file in this repo:

```bash
conda env create -f environment.yml -n MRS_env
conda activate MRS_env

## How to run the example
```bash
python code/example_usage.py --input data/example_spectrum.nii.gz
