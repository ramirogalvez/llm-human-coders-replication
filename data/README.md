# Data

This folder contains the input data used by the replication scripts.

- `main_data.csv`: cleaned analysis dataset used by `../src/full_analysis.py`. The file is pipe-delimited and contains article identifiers, expert labels, outsourced-human labels, LLM labels, disagreement flags, respondent metadata, and article-level metadata.

- `variable_dictionary.md`: variable definitions and coding rules for `main_data.csv`.

- `corpus/`: 210 Markdown files with the article texts analyzed in the study. File names follow the `NewsId` identifier, for example `N21.md`.

  **Source note:** The article texts in `corpus/` come from Factiva and are included for replication purposes, with Factiva identified as the database source.

- `political_party_mayor.dta`: auxiliary Stata dataset mapping news items to the mayor's party for the political-bias appendix table.

- `output_data/`: output folders where the scripts save generated figures, tables, and Stata input data. The scripts do not download external data.

The scripts do not download external data.
