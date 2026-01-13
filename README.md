# cVEP speller – experiment and decoding code

This repository contains the code used for the experiment and decoding analyses for my thesis on c-VEP-based BCI spelling.

The repository includes:
- experiment code for stimulus presentation and data collection
- decoding and analysis code used after data acquisition

---

## Repository structure

### Experiment code
The experiment code used for stimulus presentation and data collection is
included in this repository.

### DecodingAnalysis
All post-experiment analyses are located in the `DecodingAnalysis/` folder.

The notebooks implement decoding, permutation tests, statistical analyses,
and figure generation used in the thesis.

---

## Data and results

Raw data and analysis outputs are **not included** in this repository.
The folders `DecodingAnalysis/Data/` and `DecodingAnalysis/Results/` are expected
to exist locally.

---

## Running the experiment

The experiment code is based on and adapted from an existing c-VEP speller
implementation.

To obtain the code:

```bash
git clone https://github.com/donja-schipper/cvep-speller-thesis-code.git
conda create --name thesis_code python=3.10
conda activate thesis_code
pip install -r requirements.txt
```

To run the experiment code in this repository:

```bash
python -m speller.cvep_speller.speller
```

---

## Running the analysis

These notebooks were developed using a local folder structure on Windows.  
To run them on your machine, change the `ROOT = r"..."` variable at the top of each notebook to point to your local thesis folder.

Expected folders under `ROOT`:
- `DecodingAnalysis/code/`                 (contains `decoding_utils.py`)
- `DecodingAnalysis/Data/derivatives/`     (contains `sub-*/sub-*_cvep_*.npz`)
- `DecodingAnalysis/Results/decoding/`     (generated outputs)
- `DecodingAnalysis/Results/permutations/` (generated outputs)
- `DecodingAnalysis/Results/questionnaire/` (input CSV for comfort analysis)

Recommended notebook run order:
1. `01_permutations.ipynb`
2. `02_decoding.ipynb`
3. `03_anova_performance.ipynb`
4. `04_anova_comfort.ipynb`
5. `05_vep_reconvolution.ipynb`
6. `06_figures.ipynb`

## Dependencies and acknowledgements

The experiment code in this repository is adapted from the open-source
`dp-cvep-speller`, which provides a c-VEP speller
implementation for stimulus presentation and data collection.

- dp-cvep-speller: https://github.com/thijor/dp-cvep-speller

Please refer to the original repository for licensing information and citation
requirements, and cite the corresponding work if you use or extend the
experiment code.

Decoding analyses in this repository make use of the `pyntbci` Python toolbox,
including its rCCA-based decoding implementation.

- pyntbci: https://github.com/thijor/pyntbci

If you use or extend this code, please also cite the corresponding `pyntbci`
software and publications.

In addition, analyses rely on standard scientific Python libraries, including:
NumPy, pandas, scikit-learn, SciPy, pingouin, statsmodels, Matplotlib, and seaborn.



