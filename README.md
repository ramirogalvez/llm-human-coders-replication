# Replication Package: LLMs outperform outsourced human coders on complex textual analysis

This repository contains the input data and code needed to reproduce the figures and tables for:

Bermejo, V. J., Gago, A., Galvez, R. H., & Harari, N. (2025). "LLMs outperform outsourced human coders on complex textual analysis." Scientific Reports, 15, 40122. https://doi.org/10.1038/s41598-025-23798-y

## Contents

- `src/full_analysis.py`: Python script that creates the figures, most tables, and the input file used by Stata.
- `src/Replication_Package_Stata.do`: Stata script that creates the regression tables and political-bias appendix table.
- `data/main_data.csv`: cleaned analysis dataset.
- `data/corpus/`: 210 article text files used in the study.
- `data/political_party_mayor.dta`: auxiliary Stata dataset used by the political-bias analysis.
- `data/variable_dictionary.md`: variable definitions.
- `data/output_data/`: empty output folders where the scripts save generated files.

Generated figures and tables are not included. They are created when the scripts are run.

## Requirements

Python:

- Python 3.10 or newer
- Python packages listed in `requirements.txt`

Stata:

- Stata 16 or newer is recommended.
- Stata packages `outreg2` and `estout`. In Stata, install them with:

```stata
ssc install outreg2
ssc install estout
```

## How to Run

First install the Python packages. From the main replication package folder:

```
python -m pip install -r requirements.txt
```

Then open the `src` folder in your terminal and run the Python script:

```
cd src
python full_analysis.py
```

After the Python script finishes, run the Stata script from the same `src` folder:

```
do Replication_Package_Stata.do
```

The Python script must be run before the Stata script because it creates `../data/output_data/data_for_regs.txt`, which is used by Stata.

## Outputs

The scripts write generated files to:

- `data/output_data/plots/`
- `data/output_data/tables/`
- `data/output_data/data_for_regs.txt`

Running the scripts again overwrites generated files in `data/output_data/`.
