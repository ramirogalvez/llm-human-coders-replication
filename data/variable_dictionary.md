# Variable Dictionary (`concat.csv` subset used by `full_analysis.py`)

## 1) Naming scheme

Task suffix (same meaning across all sources):
- `Q1`: Municipalities mentioned in the news.
- `Q2`: Number of municipalities mentioned.
- `Q3`: Whether the news criticizes municipal management.
- `Q4`: Who emits the criticism.
- `Q5`: Who is the target of the criticism.

Source prefixes:
- `G`: Golden standard.
- `QQ`: Outsourced human coders (cleaned Qualtrics responses).
- `C35`: GPT-3.5-turbo (April 2024 run family).
- `C4`: GPT-4-turbo.
- `CL`: Claude 3 Opus.
- `CS`: Claude 3.5 Sonnet.
- `C35FS`: Few-shot GPT-3.5-turbo.
- `C4FS`: Few-shot GPT-4-turbo.
- `C35ZS`: Zero-shot GPT-3.5-turbo.
- `C4ZS`: Zero-shot GPT-4-turbo.
- `orig_C35`: Original GPT-3.5 output from the previous paper (October 2023 baseline).

## 2) Shared coding for Q3-Q5

`Q3` codes:
- `1`: Yes, there is criticism.
- `0`: No criticism.
- `99`: Not sure.

`Q4` codes:
- `1`: Opposition councilor.
- `2`: Governing-party councilor.
- `3`: Mayor.
- `4`: PP.
- `5`: PSOE.
- `0`: No criticism.
- `98`: Does not fit.
- `99`: Not sure.

`Q5` codes:
- `1`: Criticism targets current municipal government.
- `2`: Criticism targets previous municipal government.
- `3`: Criticism targets national government.
- `4`: Criticism targets municipal opposition.
- `5`: Criticism targets PP.
- `6`: Criticism targets PSOE.
- `0`: No criticism.
- `98`: Does not fit.
- `99`: Not sure.

## 3) Variable-by-variable dictionary

### A) Identifiers, timing, and demographics
- `NewsId`: News/article identifier.
- `ResponseId`: Qualtrics respondent identifier (outsourced coder id).
- `RecordedDate`: Timestamp when the Qualtrics response row was recorded.
- `word_count`: Number of words in the news article.
- `D1`: Respondent birth date (used with `RecordedDate` to compute age).
- `D2`: Respondent gender self-identification (`Hombre`, `Mujer`, `Prefiero identificarme como:`, `Prefiero no decir`).
- `D3`: Respondent program/degree (`Bachelor in Transformational Leadership and Social Impact`, `Bachelor of Business Administration`, `Double degree in Business Administration and AI for Business`, `MSc in Marketing Management`).
- `D4`: Spanish nationality indicator (`Si`/`No`).
- `D5.3`: If `D4 == No`, years lived in Spain (categorical bins, from `<1 year` to `>10 years`).
- `Duration`: Time spent on the survey block (stored as duration/timedelta).

### B) Intercoder disagreement flags (difficulty)
- `IA_Q1`: `True` if coders disagreed on task Q1 in intermediate GS versions.
- `IA_Q2`: `True` if coders disagreed on task Q2.
- `IA_Q3`: `True` if coders disagreed on task Q3.
- `IA_Q4`: `True` if coders disagreed on task Q4.
- `IA_Q5`: `True` if coders disagreed on task Q5.
- `IA_Any`: `True` if any of `IA_Q1`...`IA_Q5` is `True`.

### C) Golden standard labels
- `GQ1`: Golden-standard answer to Q1.
- `GQ2`: Golden-standard answer to Q2.
- `GQ3`: Golden-standard answer to Q3.
- `GQ4`: Golden-standard answer to Q4.
- `GQ5`: Golden-standard answer to Q5.

### D) Outsourced human labels (Qualtrics)
- `QQ1`: Outsourced-human answer to Q1.
- `QQ2`: Outsourced-human answer to Q2.
- `QQ3`: Outsourced-human answer to Q3.
- `QQ4`: Outsourced-human answer to Q4.
- `QQ5`: Outsourced-human answer to Q5.

### E) GPT-3.5-turbo (April 2024 family)
- `C35Q1`: GPT-3.5 answer to Q1.
- `C35Q2`: GPT-3.5 answer to Q2.
- `C35Q3`: GPT-3.5 answer to Q3.
- `C35Q4`: GPT-3.5 answer to Q4.
- `C35Q5`: GPT-3.5 answer to Q5.

### F) GPT-4-turbo
- `C4Q1`: GPT-4 answer to Q1.
- `C4Q2`: GPT-4 answer to Q2.
- `C4Q3`: GPT-4 answer to Q3.
- `C4Q4`: GPT-4 answer to Q4.
- `C4Q5`: GPT-4 answer to Q5.

### G) Claude 3 Opus
- `CLQ1`: Claude Opus answer to Q1.
- `CLQ2`: Claude Opus answer to Q2.
- `CLQ3`: Claude Opus answer to Q3.
- `CLQ4`: Claude Opus answer to Q4.
- `CLQ5`: Claude Opus answer to Q5.

### H) Claude 3.5 Sonnet
- `CSQ1`: Claude Sonnet answer to Q1.
- `CSQ2`: Claude Sonnet answer to Q2.
- `CSQ3`: Claude Sonnet answer to Q3.
- `CSQ4`: Claude Sonnet answer to Q4.
- `CSQ5`: Claude Sonnet answer to Q5.

### I) Few-shot GPT-3.5
- `C35FSQ1`: Few-shot GPT-3.5 answer to Q1.
- `C35FSQ2`: Few-shot GPT-3.5 answer to Q2.
- `C35FSQ3`: Few-shot GPT-3.5 answer to Q3.
- `C35FSQ4`: Few-shot GPT-3.5 answer to Q4.
- `C35FSQ5`: Few-shot GPT-3.5 answer to Q5.

### J) Few-shot GPT-4
- `C4FSQ1`: Few-shot GPT-4 answer to Q1.
- `C4FSQ2`: Few-shot GPT-4 answer to Q2.
- `C4FSQ3`: Few-shot GPT-4 answer to Q3.
- `C4FSQ4`: Few-shot GPT-4 answer to Q4.
- `C4FSQ5`: Few-shot GPT-4 answer to Q5.

### K) Zero-shot GPT-3.5
- `C35ZSQ1`: Zero-shot GPT-3.5 answer to Q1.
- `C35ZSQ2`: Zero-shot GPT-3.5 answer to Q2.
- `C35ZSQ3`: Zero-shot GPT-3.5 answer to Q3.
- `C35ZSQ4`: Zero-shot GPT-3.5 answer to Q4.
- `C35ZSQ5`: Zero-shot GPT-3.5 answer to Q5.

### L) Zero-shot GPT-4
- `C4ZSQ1`: Zero-shot GPT-4 answer to Q1.
- `C4ZSQ2`: Zero-shot GPT-4 answer to Q2.
- `C4ZSQ3`: Zero-shot GPT-4 answer to Q3.
- `C4ZSQ4`: Zero-shot GPT-4 answer to Q4.
- `C4ZSQ5`: Zero-shot GPT-4 answer to Q5.

### M) Panel-position variables
- `Q_block_number`: Position of this news item in the Qualtrics questionnaire (`1`, `2`, or `3`).
- `tagged_order`: Repetition/run index for a given `NewsId` (typically first/second pass, values `1` or `2` in the balanced panel).

### N) Legacy GPT-3.5 baseline (October 2023)
- `orig_C35Q1`: Original GPT-3.5 answer to Q1 from prior paper baseline.
- `orig_C35Q2`: Original GPT-3.5 answer to Q2 from prior paper baseline.
- `orig_C35Q3`: Original GPT-3.5 answer to Q3 from prior paper baseline.
- `orig_C35Q4`: Original GPT-3.5 answer to Q4 from prior paper baseline.
- `orig_C35Q5`: Original GPT-3.5 answer to Q5 from prior paper baseline.
