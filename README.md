# MRS at 7T With and Without J-Difference Editing

This repository accompanies the bachelor's thesis and provides minimal, runnable code to reproduce the evaluation on an example.

## Contents
- `BachelorsThesis_MRS_at_7T_With_And_Without_J-Difference_Editing.pdf` — final thesis document  
- `mrs/` — evaluation code (importable package)  
- `example/example_usage.py` — runnable demo script  
- `data/` — example input from J-difference editing of GABA (`example_1.nii.gz`, `example_2.nii.gz` - edit-OFF and edit-ON)  
  > Note: parallelization only makes sense for multiple spectra; one is included for demonstration.
- `environment.yml` — conda environment for reproducibility

## Environment (conda)
```bash
conda env create -f environment.yml -n MRS_env
conda activate MRS_env
