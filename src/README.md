# Source Code

Run both scripts from this `src` folder.

First run:

```
python full_analysis.py
```

Then run in Stata:

```
do Replication_Package_Stata.do
```

`full_analysis.py` creates the figures, most tables, and `../data/output_data/data_for_regs.txt`.

`Replication_Package_Stata.do` uses `../data/output_data/data_for_regs.txt` and `../data/political_party_mayor.dta` to create the regression and political-bias appendix tables. It requires the Stata packages `outreg2` and `estout`.
