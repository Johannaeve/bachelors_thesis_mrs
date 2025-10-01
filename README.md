# Bachelor’s Thesis
My bachelor's thesis on magnetic resonance spectroscopy with and without J-difference editing.

This repository contains:
- `BachelorsThesis_MRS_at_7T_With_And_Without_J-Difference_Editing.pdf` — final thesis document  
- `mrs/` — evaluation code 
- `example/example_usage.py` — runnable demo script  
- `data/` — small example inputs (`example_1.nii.gz`, `example_2.nii.gz`)  
- `environment.yml` — conda environment for reproducibility

## Environment (conda)
```bash
conda env create -f environment.yml -n MRS_env
conda activate MRS_env
