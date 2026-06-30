"""Reproduce the analysis outputs reported in the paper.

Run this script from the ``src/`` directory. It expects the cleaned analysis
dataset at ``../data/main_data.csv`` and writes figures, tables, and regression
inputs under ``../data/output_data/``.
"""

import ast
import os
import re
from collections import Counter

DATA_DIR = "../data"
OUTPUT_DIR = "../data/output_data"
PLOTS_DIR = "../data/output_data/plots"
TABLES_DIR = "../data/output_data/tables"

import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

pd.options.display.max_columns = 300
pd.options.display.max_rows = 300

NAME_MAPPING = {
    "Q": "Outsourced humans",
    "QH": "High-performing humans",
    "C35": "GPT-3.5-turbo",
    "C4": "GPT-4-turbo",
    "C35ZS": "Zero-Shot GPT-3.5-turbo (2025-08)",
    "C4ZS": "Zero-Shot GPT-4-turbo (2025-08)",
    "C35FS": "Few-Shot GPT-3.5-turbo (2025-08)",
    "C4FS": "Few-Shot GPT-4-turbo (2025-08)",
    "CL": "Claude 3 Opus",
    "CS": "Claude 3.5 Sonnet",
}

# Canonical model sets used throughout the paper.
CORE_TAGGERS = ["Q", "C35", "C4", "CL", "CS"]
ALL_TAGGERS = CORE_TAGGERS + ["C35FS", "C4FS", "C35ZS", "C4ZS"]
# Shared panel titles keep figure labels consistent across all plots.
PANEL_TITLES = [
    r"T1: Cities names (Macro F$_1$)",
    "T2: Named cities count (MAE)",
    "T3: Presence of criticism (Accuracy)",
    "T4: Source of criticism (Accuracy)",
    "T5: Target of criticism (Accuracy)",
    "All correct (Proportion)",
]


def load_main_data():
    """Load the cleaned analysis dataset used throughout the script.

    The raw CSV stores several date columns as strings. This helper centralizes
    the parsing logic so downstream functions can assume those fields are ready
    for arithmetic and filtering.

    Returns:
        pandas.DataFrame: Main analysis dataset with normalized date columns.
    """
    # Read the pipe-delimited file exactly as used in the original analysis.
    df = pd.read_csv(f"{DATA_DIR}/main_data.csv", sep="|", encoding="utf-8")
    # Normalize dates once here instead of repeating this conversion downstream.
    df['RecordedDate'] = pd.to_datetime(df['RecordedDate'])
    df['D1'] = pd.to_datetime(df["D1"].str.replace(".", "/", regex=False), format='%d/%m/%Y')
    return df


def ensure_output_dirs():
    """Create output directories used by the replication scripts."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)


def analyze_outsourced_coders_demographics(df):
    """Print descriptive statistics for the outsourced human coders.

    Args:
        df (pandas.DataFrame): Main analysis dataset at the article-response
            level.
    """
    # Demographic fields are respondent-level, so collapse repeated article rows.
    df_dem = df.copy()
    df_dem = df_dem[["ResponseId", "RecordedDate", "D1", "D2", "D3", "D4", "D5.3", "Duration"]].drop_duplicates().reset_index(drop=True)

    print(f"Number of outsourced coders: {df_dem['ResponseId'].shape[0]}")
    print(f"Percentage self-identified as female: {100 * (df_dem['D2'] == 'Mujer').mean():.1f}%")
    print(f"Percentage self-identified as male: {100 * (df_dem['D2'] == 'Hombre').mean():.1f}%")

    # Convert birth dates into ages at response time.
    age_in_years = (df_dem["RecordedDate"] - pd.to_datetime(df_dem["D1"])).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    # Any negative age is a data-entry issue; mark it missing before summarizing.
    age_in_years[age_in_years < 0] = np.nan
    df_dem.loc[age_in_years < 0, "D1"] = np.nan

    print(f"Average age in years: {age_in_years.mean():.1f}")
    print(f"Standard deviation of age (in years): {age_in_years.std():.1f}")

    print("Program to which the participant belonged:")
    print(df_dem["D3"].value_counts())

    print("\nParticipants' nationality:")
    print(df_dem["D4"].value_counts())
    print(df_dem["D4"].value_counts(normalize=True))

    print(f"\nShare of foreigners having lived at least one year: {(df_dem.loc[df_dem['D4'] == 'No', 'D5.3'] != 'Menos de 1 año').mean():.2f}")

    # Qualtrics durations are stored as timedeltas and reported here in minutes.
    duration_minutes = pd.to_timedelta(df_dem['Duration']).dt.total_seconds() / 60
    print(f"Median duration (in minutes): {duration_minutes.median():.2f}")
    print(f"Percentile 90 of duration (in minutes): {duration_minutes.quantile(0.9):.2f}")


def analyze_gold_standard_labels(df):
    """Print consistency checks and summary statistics for the gold labels.

    This function is intentionally verbose because it serves as a lightweight
    audit of the expert benchmark before any model or human comparison is run.

    Args:
        df (pandas.DataFrame): Main analysis dataset at the article-response
            level.
    """

    print("\nAnalyzing and verifying the gold standard labels")

    # Gold labels are article-level, so deduplicate away repeated coder rows.
    df = df[["NewsId", "GQ1", "GQ2", "GQ3", "GQ4", "GQ5"]].drop_duplicates()

    # These columns are stored as string representations of sets/lists in the CSV.
    for col in ["GQ1", "GQ4", "GQ5"]:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    print("\nCheck 1: Consistency between the length GQ1 (mentioned cities) and GQ2 (number of cities)")
    mismatches = df[
        ((df["GQ1"].apply(len) != df["GQ2"]) & (df["GQ1"] != {"ninguno"})) |
        ((df["GQ1"] == {"ninguno"}) & (df["GQ2"] != 0))
    ][["NewsId", "GQ1", "GQ2"]]

    if mismatches.empty:
        print("No inconsistencies found between GQ1 length and GQ2. Verification successful!")
    else:
        print(f"Found {len(mismatches)} inconsistent entries:")
        print(mismatches)

    print("\nGQ1 Analysis: Frequency of mentioned cities")
    # Flatten all municipality mentions to inspect the empirical distribution.
    city_counts = Counter([city for cities in df["GQ1"] for city in cities])
    city_df = pd.DataFrame(city_counts.items(), columns=["City", "Freq"])
    city_df["Rel_freq"] = city_df["Freq"] / city_df["Freq"].sum()
    print(city_df.sort_values(by="Freq", ascending=False).head(10))
    print(f"Number of articles mentioning just a single municipality: {(city_df['Freq'] == 1).sum()}")

    print("\nGQ2 Analysis: Summary statistics for number of cities mentioned")
    print(f"Mean GQ2: {df['GQ2'].mean():.2f}")
    print(f"Standard deviation GQ2: {df['GQ2'].std():.2f}")
    print(f"Min GQ2: {df['GQ2'].min():.2f}")
    print(f"Max GQ2: {df['GQ2'].max():.2f}")

    print('\nCheck 2: Every time GQ2 is 0, GQ1 is {"ninguno"}, and vice versa')
    inconsistencies_gq1_gq2 = df[
        ((df['GQ2'] == 0) & (df['GQ1'] != {"ninguno"})) |
        ((df['GQ1'] == {"ninguno"}) & (df['GQ2'] != 0))
    ][["NewsId", "GQ1", "GQ2"]]

    if inconsistencies_gq1_gq2.empty:
        print('Every time GQ1 is {"ninguno"}, GQ2 is 0, and vice versa. Verification successful!')
    else:
        print(f"Found {len(inconsistencies_gq1_gq2)} inconsistent entries:")
        print(inconsistencies_gq1_gq2)

    print("\nCheck 3: Observations with 99 in GQ3")
    # Code 99 means "not sure" and should not appear in the final expert labels.
    gq3_equal_99 = df[df["GQ3"].apply(lambda x: "99" in x)]
    if gq3_equal_99.empty:
        print("No observations with 99 in GQ3. Verification successful!")
    else:
        print(f"Found {len(gq3_equal_99)} observations with 99 in GQ3:")
        print(gq3_equal_99[["NewsId", "GQ3", "GQ4", "GQ5"]])

    print("\nGQ3 Analysis: Distribution of responses (0 = No, 1 = Yes, 99 = Not sure)")
    print((df["GQ3"].value_counts(normalize=True) * 100).round(2))

    print("\nCheck 4: Observations with 99 in GQ4 and GQ5")
    gq4_gq5_equal_99 = df[df["GQ4"].apply(lambda x: "99" in x) | df["GQ5"].apply(lambda x: "99" in x)]
    if gq4_gq5_equal_99.empty:
        print("No observations with 99 in GQ4 or GQ5. Verification successful!")
    else:
        print(f"Found {len(gq4_gq5_equal_99)} observations with 99 in GQ4 or GQ5:")
        print(gq4_gq5_equal_99[["NewsId", "GQ3", "GQ4", "GQ5"]])

    print("\nCheck 5: Inconsistencies between GQ3 being yes and GQ4 containing 0")
    inconsistencies_gq3_gq4 = df[(df["GQ3"] == "1") & (df["GQ4"].apply(lambda x: "0" in x))]
    if inconsistencies_gq3_gq4.empty:
        print("No inconsistencies found between GQ3 and GQ4. Verification successful!")
    else:
        print(f"Found {len(inconsistencies_gq3_gq4)} inconsistent entries:")
        print(inconsistencies_gq3_gq4[["NewsId", "GQ3", "GQ4"]])

    print("\nGQ4 Analysis: Distribution of sources of criticism")
    Q4_label_mapping = {
        "0": "No criticism",
        "1": "Opposition councilor",
        "2": "Governing party councilor",
        "3": "Mayor",
        "4": "PP",
        "5": "PSOE",
        "98": "Does not fit",
        "99": "Not sure"
    }
    # The published appendix reports these category frequencies in percent.
    GQ4_table = pd.DataFrame([
        {"value": Q4_label_mapping[v], "freq": round(100 * sum(v in e for e in df["GQ4"]) / df.shape[0], 2)}
        for v in set([e2 for e1 in df["GQ4"] for e2 in e1])
    ])
    print(GQ4_table.sort_values(by="freq", ascending=False))

    print("\nCheck 6: Inconsistencies between GQ3 being yes and GQ5 containing 0")
    inconsistencies_gq3_gq5 = df[(df["GQ3"] == "1") & (df["GQ5"].apply(lambda x: "0" in x))]
    if inconsistencies_gq3_gq5.empty:
        print("No inconsistencies found between GQ3 and GQ5. Verification successful!")
    else:
        print(f"Found {len(inconsistencies_gq3_gq5)} inconsistent entries:")
        print(inconsistencies_gq3_gq5[["NewsId", "GQ3", "GQ5"]])

    print("\nGQ5 Analysis: Distribution of sources of criticism")
    Q5_label_mapping = {
        "0": "No criticism",
        "1": "Current municipal government",
        "2": "Previous municipal government",
        "3": "National government",
        "4": "Municipal opposition",
        "5": "PP",
        "6": "PSOE",
        "98": "Does not fit",
        "99": "Not sure"
    }
    GQ5_table = pd.DataFrame([
        {"value": Q5_label_mapping[v], "freq": round(100 * sum(v in e for e in df["GQ5"]) / df.shape[0], 2)}
        for v in set([e2 for e1 in df["GQ5"] for e2 in e1])
    ])
    print(GQ5_table.sort_values(by="freq", ascending=False))


def get_summary_stats_word_counts(df, word_count_column='word_count', id_column='NewsId'):
    """Print summary statistics for article word counts.

    Args:
        df (pandas.DataFrame): Main analysis dataset.
        word_count_column (str): Name of the word-count column.
        id_column (str): Column used to identify unique articles.
    """
    # Word counts are article-level, so remove duplicate coder rows first.
    df = df[[id_column, word_count_column]].drop_duplicates().copy()
    
    print(f"Mean word count: {df[word_count_column].mean():.2f}")
    print(f"10th percentile word count: {df[word_count_column].quantile(0.1):.2f}")
    print(f"Median word count: {df[word_count_column].quantile(0.5):.2f}")
    print(f"90th percentile word count: {df[word_count_column].quantile(0.9):.2f}")


def check_match(real_y, predicted_y):
    """Return whether a prediction matches any admissible gold-standard label.

    Args:
        real_y: Gold-standard label or collection of admissible labels.
        predicted_y: Single predicted label.

    Returns:
        bool: ``True`` when the prediction matches any allowed label.
    """
    try:
        # Most labels are stored as stringified sets in the source data.
        real_y = ast.literal_eval(real_y)
    except (ValueError, SyntaxError):
        # Fall back to a singleton label when the cell is already scalar-like.
        real_y = {real_y}

    if not isinstance(real_y, (set, list, tuple)):
        real_y = {real_y}

    predicted_y = str(predicted_y)
    return predicted_y in map(str, real_y)


def abs_error(real_y, predicted_y):
    """Return the absolute counting error, treating code 99 as zero.

    The analysis treats "not sure" as equivalent to a failed count, which is
    operationalized here by converting code 99 into zero before computing the
    absolute error.
    """
    if predicted_y == 99:
        predicted_y = 0
    if real_y == 99:
        real_y = 0
    return abs(real_y - predicted_y)


def f1_score(correct, predicted):
    """Return the article-level F1 score for municipality extraction.

    Municipality names are compared after lowercasing and accent removal. A
    fuzzy Levenshtein similarity threshold is used to tolerate minor spelling
    differences in coder or model outputs.

    Args:
        correct: Gold-standard collection of municipalities.
        predicted: Predicted collection of municipalities.

    Returns:
        float: Per-article F1 score.
    """
    # Parse the stored string representations into comparable Python sets.
    correct = ast.literal_eval(correct)
    correct = set([lower_and_remove_accents(e).strip() for e in correct])

    predicted = ast.literal_eval(predicted)
    predicted = set([lower_and_remove_accents(e).strip() for e in predicted])

    tp = 0
    for cor in correct:
        for pred in predicted:
            # Fuzzy matching avoids penalizing small spelling or accent mistakes.
            distance = Levenshtein.distance(cor, pred)  
            max_length = max(len(cor), len(pred))
            similarity = 1 - (distance / max_length)

            if 0.75 <= similarity <= 1:
                tp += 1
                break

    fp = len(predicted) - tp
    fn = len(correct) - tp
 
    if tp == 0:
        return 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return f1


def lower_and_remove_accents(old):
    """Lowercase a string and strip common accented Latin vowels.

    Args:
        old (str): Raw text value.

    Returns:
        str: Normalized string used in fuzzy municipality matching.
    """
    new = old.lower()
    new = re.sub(r"[àáâãäå]", "a", new)
    new = re.sub(r"[èéêë]", "e", new)
    new = re.sub(r"[ìíîï]", "i", new)
    new = re.sub(r"[òóôõö]", "o", new)
    new = re.sub(r"[ùúûü]", "u", new)
    return new


def process_Q(df_orig, real_y, predicted_y, n_shuffles, score_func):
    """Score one task and optionally compute a permutation null distribution.

    Args:
        df_orig (pandas.DataFrame): Evaluation sample.
        real_y (str): Gold-standard column.
        predicted_y (str): Prediction column.
        n_shuffles (int): Number of permutation draws for the null distribution.
        score_func (callable): Row-level scoring function.

    Returns:
        dict: Aggregate score and, when requested, permutation statistics.
    """
    df = df_orig.copy()
    # Score each row first, then average across the requested evaluation sample.
    df["res_metric"] =  df.apply(lambda x: score_func(x[real_y], x[predicted_y]), axis=1)

    if n_shuffles > 0:
        shuffled_res = []
        for _ in tqdm(range(n_shuffles), desc=f"Shuffling {predicted_y}"):
            shuffled_df = df_orig.copy()
            # The null keeps marginal distributions fixed while breaking alignment.
            shuffled_df[predicted_y] = shuffled_df[predicted_y].sample(frac=1).reset_index(drop=True)
            shuffled_res.append(shuffled_df.apply(lambda x: score_func(x[real_y], x[predicted_y]), axis=1).mean())
        shuffled_mean = np.mean(shuffled_res)
        shuffled_q025 = np.quantile(shuffled_res, 0.025)
        shuffled_q975 = np.quantile(shuffled_res, 0.975)
    else:
        shuffled_mean = None
        shuffled_q025 = None
        shuffled_q975 = None

    # Strip the historical prefix so old GPT-3.5 runs map back to the same coder id.
    if predicted_y.startswith("orig_"):
        predicted_y = predicted_y.replace("orig_", "")

    return {"coder": predicted_y[:-2],
            "question": predicted_y.split("_")[0][-2:],
            "score": df["res_metric"].mean(),
            "shuffled_mean": shuffled_mean,
            "shuffled_q025": shuffled_q025,
            "shuffled_q975": shuffled_q975}


def perfect_score(df_orig, tagger):
    """Return the share of rows with all five tasks answered correctly.

    Args:
        df_orig (pandas.DataFrame): Evaluation sample.
        tagger (str): Tagger prefix used in the dataset.

    Returns:
        dict: Aggregate "all tasks correct" rate.
    """
    df = df_orig.copy()

    # Convert each task into a boolean exact-correct indicator.
    Q1 = df.apply(lambda x: f1_score(x["GQ1"], x[f"{tagger}Q1"]), axis=1) == 1
    Q2 = df.apply(lambda x: abs_error(x["GQ2"], x[f"{tagger}Q2"]), axis=1) == 0
    Q3 = df.apply(lambda x: check_match(x["GQ3"], x[f"{tagger}Q3"]), axis=1) == 1
    Q4 = df.apply(lambda x: check_match(x["GQ4"], x[f"{tagger}Q4"]), axis=1) == 1
    Q5 = df.apply(lambda x: check_match(x["GQ5"], x[f"{tagger}Q5"]), axis=1) == 1

    all_questions = pd.concat([Q1, Q2, Q3, Q4, Q5], axis=1)

    if tagger.startswith("orig_"):
        tagger = tagger.replace("orig_", "")

    return {"coder": tagger,
            "question": "all_perfect",
            "score": (all_questions.mean(axis=1) == 1).mean(),
            "shuffled_mean": None,
            "shuffled_q025": None,
            "shuffled_q975": None}


def calculate_intercoder_agreement(df):
    """Print the intercoder agreement statistics reported in the paper.

    Args:
        df (pandas.DataFrame): Main analysis dataset containing disagreement
            indicator columns such as ``IA_Q1``.
    """
    print("\nCalculating intercoder agreement:")
    print("Overall intercoder agreement:")
    for i in range(1, 6):
        agreement = round(100 * (~df[f"IA_Q{i}"]).mean(), 2)
        print(f"Q{i}: {agreement}%")

    print("\nIntercoder Agreement for T3 = Yes:")
    t3_yes_mask = df["GQ3"] == "{'1'}"
    for i in range(4, 6):
        agreement = round(100 * (~df.loc[t3_yes_mask, f"IA_Q{i}"]).mean(), 2)
        print(f"Q{i}: {agreement}%")


def gen_scores_overall(df, coder, n_shuffles):
    """Compute overall performance metrics for one tagging strategy.

    Args:
        df (pandas.DataFrame): Evaluation sample.
        coder (str): Tagger prefix in the dataset.
        n_shuffles (int): Number of permutation draws for null benchmarks.

    Returns:
        pandas.DataFrame: One row per task plus the "all correct" metric.
    """
    coder_res = []
    coder_res.append(process_Q(df, "GQ1", f"{coder}Q1", n_shuffles, f1_score))
    coder_res.append(process_Q(df, "GQ2", f"{coder}Q2", n_shuffles, abs_error))
    coder_res.append(process_Q(df, "GQ3", f"{coder}Q3", n_shuffles, check_match))
    coder_res.append(process_Q(df, "GQ4", f"{coder}Q4", n_shuffles, check_match))
    coder_res.append(process_Q(df, "GQ5", f"{coder}Q5", n_shuffles, check_match))
    coder_res.append(perfect_score(df, coder))
    coder_res = pd.DataFrame(coder_res)
    coder_res["type"] = "Overall"

    return coder_res


def plot_overall(result_table, output_file, reverse_model_order=["Q", "C35", "C4", "CL", "CS"]):
    """Plot the six-panel overall comparison for a set of tagging strategies.

    Args:
        result_table (pandas.DataFrame): Output from ``gen_scores_overall`` or a
            dataframe with the same schema.
        output_file (str): Destination PDF path.
        reverse_model_order (list[str]): Coder ids in paper order, before the
            plot flips them for top-to-bottom display.
    """
    model_order = list(reversed(reverse_model_order))
    unique_questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'all_perfect']
    colors = ["silver"]
    fig, axes = plt.subplots(3, 2, figsize=(20*0.5, 16*0.5))
    axes = axes.flatten()

    for i, question in enumerate(unique_questions):
        ax = axes[i]
        # Each panel isolates one task, then orders bars exactly as in the paper.
        question_data = result_table[result_table['question'] == question]
        scores = [question_data[question_data['coder'] == model]['score'].values[0] if len(question_data[question_data['coder'] == model]) > 0 else 0 
                  for model in model_order]

        y_pos = range(len(model_order))
        ax.barh(y_pos, scores, align='center', color=colors)
        ax.set_title(PANEL_TITLES[i], fontsize=16, pad=20)
        # All overall panels are bounded on [0, 1] in the published figure.
        ax.set_xlim(0, 1)

        # Only left-hand panels show y-axis labels to reduce clutter.
        if i % 2 == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels([NAME_MAPPING[model] for model in model_order], fontsize=14)
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=14)
        if i % 2 != 0:
            ax.tick_params(axis='y', which='both', left=False)
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.tight_layout(h_pad=3.5, w_pad=2.5)
    plt.subplots_adjust(left=0.13, right=0.95, top=0.95, bottom=0.05)
    with PdfPages(output_file) as pdf:
        pdf.savefig(fig, bbox_inches='tight')


def gen_scores_difficulty(df, coder, n_shuffles):
    """Compute task metrics separately for regular and difficult articles.

    Difficulty is defined article-by-task using the expert disagreement flags.

    Args:
        df (pandas.DataFrame): Main evaluation sample.
        coder (str): Tagger prefix.
        n_shuffles (int): Number of permutation draws.

    Returns:
        pandas.DataFrame: Scores split into regular and difficult subsets.
    """
    coder_regular = []
    # For task-specific panels, the "regular" subset uses only rows marked easy for that task.
    coder_regular.append(process_Q(df[df["IA_Q1"] == False], "GQ1", f"{coder}Q1", n_shuffles, f1_score))
    coder_regular.append(process_Q(df[df["IA_Q2"] == False], "GQ2", f"{coder}Q2", n_shuffles, abs_error))
    coder_regular.append(process_Q(df[df["IA_Q3"] == False], "GQ3", f"{coder}Q3", n_shuffles, check_match))
    coder_regular.append(process_Q(df[df["IA_Q4"] == False], "GQ4", f"{coder}Q4", n_shuffles, check_match))
    coder_regular.append(process_Q(df[df["IA_Q5"] == False], "GQ5", f"{coder}Q5", n_shuffles, check_match))
    coder_regular.append(perfect_score(df[df["IA_Any"] == False], coder))
    coder_regular = pd.DataFrame(coder_regular)
    coder_regular["type"] = "Regular"

    # The "all correct" panel instead uses the article-level difficulty flag.
    coder_res_diff = []
    coder_res_diff.append(process_Q(df[df["IA_Q1"] == True], "GQ1", f"{coder}Q1", n_shuffles, f1_score))
    coder_res_diff.append(process_Q(df[df["IA_Q2"] == True], "GQ2", f"{coder}Q2", n_shuffles, abs_error))
    coder_res_diff.append(process_Q(df[df["IA_Q3"] == True], "GQ3", f"{coder}Q3", n_shuffles, check_match))
    coder_res_diff.append(process_Q(df[df["IA_Q4"] == True], "GQ4", f"{coder}Q4", n_shuffles, check_match))
    coder_res_diff.append(process_Q(df[df["IA_Q5"] == True], "GQ5", f"{coder}Q5", n_shuffles, check_match))
    coder_res_diff.append(perfect_score(df[df["IA_Any"] == True], coder))
    coder_res_diff = pd.DataFrame(coder_res_diff)
    coder_res_diff["type"] = "Difficult"

    coder_res = pd.concat([coder_regular, coder_res_diff], axis=0)

    return coder_res


def plot_model_comparison(result_table, 
                          output_file, 
                          reverse_model_order=["Q", "C35", "C4", "CL", "CS"]):
    """Plot the six-panel comparison split by article difficulty or length.

    Args:
        result_table (pandas.DataFrame): Output from ``gen_scores_difficulty``
            or ``gen_scores_by_length``.
        output_file (str): Destination PDF path.
        reverse_model_order (list[str]): Coder ids in paper order.
    """
    model_order = list(reversed(reverse_model_order))
    unique_questions = result_table['question'].unique()
    unique_types = result_table['type'].unique()
    colors = ["silver", "black"]
    fig, axes = plt.subplots(3, 2, figsize=(20*0.5, 22*0.5))
    axes = axes.flatten()

    for i, question in enumerate(unique_questions):
        ax = axes[i]
        # Each row in ``type_scores`` holds the subset-specific values for one model.
        question_data = result_table[result_table['question'] == question]
        type_scores = []
        for model in model_order:
            model_data = question_data[question_data['coder'] == model]
            type_scores.append([model_data[model_data['type'] == t]['score'].values[0] if len(model_data[model_data['type'] == t]) > 0 else 0 
                                for t in unique_types])

        # Transpose so we can draw one horizontal layer per subset type.
        scores_transposed = list(zip(*type_scores))
        bar_width = 0.25
        bar_spacing = 0.01
        group_spacing = 0.1
        y_pos = [p * (1 + group_spacing) for p in range(len(model_order))]

        for j, scores in enumerate(scores_transposed):
            ax.barh([p - j * (bar_width + bar_spacing) for p in y_pos], scores, bar_width, label=unique_types[j], color=colors[j])

        ax.set_title(PANEL_TITLES[i], fontsize=16, pad=20)
        # T2 is an MAE and needs a wider axis to accommodate counts above 1.
        ax.set_xlim(0, 2 if question == "Q2" else 1)
        ax.set_yticks([p - bar_width for p in y_pos])
        if i % 2 == 0:
            ax.set_yticklabels([NAME_MAPPING[model] for model in model_order], fontsize=14)
        else:
            ax.set_yticklabels([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=14)
        if i % 2 != 0:
            ax.tick_params(axis='y', which='both', left=False)
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.tight_layout(h_pad=3.5, w_pad=2.5)
    plt.subplots_adjust(left=0.13, right=0.95, top=0.95, bottom=0.1)
    # Build a single shared legend beneath all panels.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
               ncol=len(unique_types), fontsize=16, frameon=False)
    with PdfPages(output_file) as pdf:
        pdf.savefig(fig, bbox_inches='tight')


def gen_scores_by_length(df, coder, n_shuffles):
    """Compute task metrics separately for regular and long articles.

    The "long" subset is defined as articles above the 90th percentile of word
    count, matching the definition used in the paper.

    Args:
        df (pandas.DataFrame): Main evaluation sample.
        coder (str): Tagger prefix.
        n_shuffles (int): Number of permutation draws.

    Returns:
        pandas.DataFrame: Scores split into regular and long subsets.
    """
    # Use unique articles when computing the percentile threshold.
    threshold = df[["NewsId", "word_count"]].drop_duplicates()["word_count"].quantile(0.9)

    coder_short = []
    df_short = df[df["word_count"] <= threshold].copy()
    coder_short.append(process_Q(df_short, "GQ1", f"{coder}Q1", n_shuffles, f1_score))
    coder_short.append(process_Q(df_short, "GQ2", f"{coder}Q2", n_shuffles, abs_error))
    coder_short.append(process_Q(df_short, "GQ3", f"{coder}Q3", n_shuffles, check_match))
    coder_short.append(process_Q(df_short, "GQ4", f"{coder}Q4", n_shuffles, check_match))
    coder_short.append(process_Q(df_short, "GQ5", f"{coder}Q5", n_shuffles, check_match))
    coder_short.append(perfect_score(df_short, coder))
    coder_short = pd.DataFrame(coder_short)
    coder_short["type"] = "Regular"

    coder_long = []
    df_long = df[df["word_count"] > threshold].copy()
    coder_long.append(process_Q(df_long, "GQ1", f"{coder}Q1", n_shuffles, f1_score))
    coder_long.append(process_Q(df_long, "GQ2", f"{coder}Q2", n_shuffles, abs_error))
    coder_long.append(process_Q(df_long, "GQ3", f"{coder}Q3", n_shuffles, check_match))
    coder_long.append(process_Q(df_long, "GQ4", f"{coder}Q4", n_shuffles, check_match))
    coder_long.append(process_Q(df_long, "GQ5", f"{coder}Q5", n_shuffles, check_match))
    coder_long.append(perfect_score(df_long, coder))
    coder_long = pd.DataFrame(coder_long)
    coder_long["type"] = "Long"

    coder_res = pd.concat([coder_short, coder_long], axis=0)

    return coder_res


def gen_scores_humans_by_block(df, n_shuffles, seed=None):
    """Compute human performance overall and by article order in the survey.

    Args:
        df (pandas.DataFrame): Main evaluation sample.
        n_shuffles (int): Number of permutation draws for the overall rows.
        seed (int | None): Optional seed for reproducible permutation results.

    Returns:
        pandas.DataFrame: Human scores for the full sample and for each block.
    """
    if seed is not None:
        np.random.seed(seed)

    coder_overall = []
    # Only the overall rows receive permutation statistics for the significance bars.
    coder_overall.append(process_Q(df, "GQ1", "QQ1", n_shuffles, f1_score))
    coder_overall.append(process_Q(df, "GQ2", "QQ2", n_shuffles, abs_error))
    coder_overall.append(process_Q(df, "GQ3", "QQ3", n_shuffles, check_match))
    coder_overall.append(process_Q(df, "GQ4", "QQ4", n_shuffles, check_match))
    coder_overall.append(process_Q(df, "GQ5", "QQ5", n_shuffles, check_match))
    coder_overall.append(perfect_score(df, "Q"))
    coder_overall = pd.DataFrame(coder_overall)
    coder_overall["type"] = "Overall"

    # Block-specific rows are descriptive only, so no permutation benchmark is needed.
    df_1st_block = df[df["Q_block_number"]==1].copy()
    coder_first_block = []
    coder_first_block.append(process_Q(df_1st_block, "GQ1", "QQ1", 0, f1_score))
    coder_first_block.append(process_Q(df_1st_block, "GQ2", "QQ2", 0, abs_error))
    coder_first_block.append(process_Q(df_1st_block, "GQ3", "QQ3", 0, check_match))
    coder_first_block.append(process_Q(df_1st_block, "GQ4", "QQ4", 0, check_match))
    coder_first_block.append(process_Q(df_1st_block, "GQ5", "QQ5", 0, check_match))
    coder_first_block.append(perfect_score(df_1st_block, "Q"))
    coder_first_block = pd.DataFrame(coder_first_block)
    coder_first_block["type"] = "First article"
    coder_first_block = coder_first_block.drop(columns=["shuffled_mean", "shuffled_q025", "shuffled_q975"])

    df_2nd_block = df[df["Q_block_number"]==2].copy()
    coder_second_block = []
    coder_second_block.append(process_Q(df_2nd_block, "GQ1", "QQ1", 0, f1_score))
    coder_second_block.append(process_Q(df_2nd_block, "GQ2", "QQ2", 0, abs_error))
    coder_second_block.append(process_Q(df_2nd_block, "GQ3", "QQ3", 0, check_match))
    coder_second_block.append(process_Q(df_2nd_block, "GQ4", "QQ4", 0, check_match))
    coder_second_block.append(process_Q(df_2nd_block, "GQ5", "QQ5", 0, check_match))
    coder_second_block.append(perfect_score(df_2nd_block, "Q"))
    coder_second_block = pd.DataFrame(coder_second_block)
    coder_second_block["type"] = "Second article"
    coder_second_block = coder_second_block.drop(columns=["shuffled_mean", "shuffled_q025", "shuffled_q975"])

    df_3rd_block = df[df["Q_block_number"]==3].copy()
    coder_third_block = []
    coder_third_block.append(process_Q(df_3rd_block, "GQ1", "QQ1", 0, f1_score))
    coder_third_block.append(process_Q(df_3rd_block, "GQ2", "QQ2", 0, abs_error))
    coder_third_block.append(process_Q(df_3rd_block, "GQ3", "QQ3", 0, check_match))
    coder_third_block.append(process_Q(df_3rd_block, "GQ4", "QQ4", 0, check_match))
    coder_third_block.append(process_Q(df_3rd_block, "GQ5", "QQ5", 0, check_match))
    coder_third_block.append(perfect_score(df_3rd_block, "Q"))
    coder_third_block = pd.DataFrame(coder_third_block)
    coder_third_block["type"] = "Third article"
    coder_third_block = coder_third_block.drop(columns=["shuffled_mean", "shuffled_q025", "shuffled_q975"])

    coder_res = pd.concat([coder_overall, coder_first_block, coder_second_block, coder_third_block], axis=0)

    return coder_res


def plot_human_perf_by_block(result_table, output_file):
    """Plot outsourced-human performance by article order and permutation cutoff.

    Args:
        result_table (pandas.DataFrame): Output from
            ``gen_scores_humans_by_block``.
        output_file (str): Destination PDF path.
    """
    unique_questions = result_table['question'].unique()
    unique_blocks = ["Third article", "Second article", "First article", "Overall"]
    fig, axes = plt.subplots(3, 2, figsize=(20*0.5, 14*0.5))
    axes = axes.flatten()

    for i, question in enumerate(unique_questions):
        ax = axes[i]
        question_data = result_table[result_table['question'] == question]
        block_scores = [question_data[question_data['type'] == block]['score'].values[0] 
                        if len(question_data[question_data['type'] == block]) > 0 else 0 
                        for block in unique_blocks]

        y_pos = range(len(unique_blocks))
        colors = ['lightgray' if block != 'Overall' else 'silver' for block in unique_blocks]
        ax.barh(y_pos, block_scores, align='center', color=colors)

        # The black marker reproduces the 97.5th percentile permutation cutoff in Figure 4.
        overall_data = question_data[question_data['type'] == 'Overall']
        if len(overall_data) > 0 and 'shuffled_q975' in overall_data.columns:
            q975 = overall_data['shuffled_q975'].values[0]
            ax.plot([q975, q975], [y_pos[-1] - 0.4, y_pos[-1] + 0.4], color='black', linewidth=2)

        ax.set_title(PANEL_TITLES[i], fontsize=16, pad=20)
        ax.set_xlim(0, 2.2 if question == "Q2" else 1)
        ax.set_yticks(y_pos)
        if i % 2 == 0:
            ax.set_yticklabels(unique_blocks, fontsize=14)
        else:
            ax.set_yticklabels([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=14)
        if i % 2 != 0:
            ax.tick_params(axis='y', which='both', left=False)
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.tight_layout(h_pad=3.5, w_pad=2.5)
    plt.subplots_adjust(left=0.13, right=0.95, top=0.95, bottom=0.05)
    with PdfPages(output_file) as pdf:
        pdf.savefig(fig, bbox_inches='tight')


def get_high_performing_human_subset(df):
    """Return the respondent and row subsets used for the Figure 5 comparison.

    Respondents are ranked by the mean of all binary task-correctness indicators
    across the articles they answered. Models are then evaluated only on the
    ``NewsId`` and ``tagged_order`` pairs represented in the high-performing
    human sample.

    Args:
        df (pandas.DataFrame): Main evaluation sample.

    Returns:
        dict: Selected respondent ids, human rows, and matched article/order pairs.
    """
    human_correct = pd.DataFrame(index=df.index)
    # Build respondent-level correctness indicators one task at a time.
    human_correct["Q1"] = df.apply(lambda x: f1_score(x["GQ1"], x["QQ1"]) == 1, axis=1)
    human_correct["Q2"] = df.apply(lambda x: abs_error(x["GQ2"], x["QQ2"]) == 0, axis=1)
    human_correct["Q3"] = df.apply(lambda x: check_match(x["GQ3"], x["QQ3"]) == 1, axis=1)
    human_correct["Q4"] = df.apply(lambda x: check_match(x["GQ4"], x["QQ4"]) == 1, axis=1)
    human_correct["Q5"] = df.apply(lambda x: check_match(x["GQ5"], x["QQ5"]) == 1, axis=1)
    human_correct["ResponseId"] = df["ResponseId"]

    # Stack the five task indicators so each respondent score is the mean over all answered questions.
    aggregate_scores = human_correct.melt(id_vars="ResponseId", value_name="correct").groupby("ResponseId")[
        "correct"
    ].mean()
    threshold = aggregate_scores.median()
    high_perf_ids = aggregate_scores[aggregate_scores >= threshold].index

    print(
        f"Selected {len(high_perf_ids)} high-performing human coders with aggregate score above the median "
        f"({threshold:.3f})."
    )

    # Keep the selected human rows for the human bar in Figure 5.
    df_humans = df[df["ResponseId"].isin(high_perf_ids)].copy()
    selected_news_ids = df_humans["NewsId"].drop_duplicates()
    # Keep the exact article-draw pairs represented by the selected human rows.
    selected_pairs = df_humans[["NewsId", "tagged_order"]].drop_duplicates()

    print(
        f"Selected {selected_news_ids.shape[0]} unique news articles answered by high-performing human coders."
    )
    print(f"Selected {selected_pairs.shape[0]} NewsId-tagged_order pairs for model evaluation.")

    return {
        "high_perf_ids": high_perf_ids,
        "df_humans": df_humans,
        "selected_pairs": selected_pairs,
    }


def generate_high_performing_humans_figure(df, output_file):
    """Generate the Figure 5 comparison against high-performing human coders.

    Args:
        df (pandas.DataFrame): Main evaluation sample.
        output_file (str): Destination PDF path.

    Returns:
        pandas.DataFrame: Scored result table used to draw the figure.
    """
    subset_info = get_high_performing_human_subset(df)
    df_humans = subset_info["df_humans"]
    selected_pairs = subset_info["selected_pairs"]

    # Human scores use only responses from the selected high-performing coders.
    coder_res = [gen_scores_overall(df_humans, "Q", 0)]

    # Model scores are evaluated on the same article/order support seen in the selected human sample.
    df_models = df.merge(selected_pairs, on=["NewsId", "tagged_order"], how="inner")
    for tagger in ["C35", "C4", "CL", "CS"]:
        coder_res.append(gen_scores_overall(df_models, tagger, 0))

    coder_res = pd.concat(coder_res, axis=0)
    coder_res.loc[coder_res["coder"] == "Q", "coder"] = "QH"

    plot_overall(coder_res, output_file, reverse_model_order=["QH", "C35", "C4", "CL", "CS"])

    return coder_res


def create_results_latex_table(res_df, caption, label, row_order, filepath):
    """Write a LaTeX summary table from a scored result dataframe.

    Args:
        res_df (pandas.DataFrame): Score table with ``coder``, ``question``,
            ``type``, and ``score`` columns.
        caption (str): LaTeX caption.
        label (str): LaTeX label.
        row_order (list[str]): Display order after applying ``NAME_MAPPING``.
        filepath (str): Output path for the generated ``.tex`` file.

    Returns:
        str: Rendered LaTeX table.
    """
    new_titles = [
        "T1 (Macro F$_1$)",
        "T2 (MAE)",
        "T3 (Acc)",
        "T4 (Acc)",
        "T5 (Acc)",
        "All (Prop)"
    ]

    latex_table = r"\begin{table}[ht!]" + "\n"
    latex_table += r"\centering" + "\n"
    latex_table += r"\setlength{\tabcolsep}{4pt}" + "\n"
    latex_table += r"\footnotesize" + "\n"
    latex_table += r"\caption{" + caption + "}" + "\n"
    latex_table += r"\begin{tabular}{|>{\raggedright\arraybackslash}p{3.0cm}|" + "c|" * len(new_titles) + "}" + "\n"
    latex_table += r"\hline" + "\n"

    header = r"\multicolumn{1}{|c|}{\textbf{Coding Strategy}} & " + " & ".join([rf"\textbf{{{title}}}" for title in new_titles]) + r" \\"
    latex_table += header + "\n"
    latex_table += r"\hline" + "\n"

    # When the input has multiple subsets (for example Regular/Difficult), each
    # subset becomes its own block inside the LaTeX table.
    By_values = res_df["type"].drop_duplicates().values
    for v in By_values:
        res_df_tmp = res_df[res_df["type"] == v]
        pivot_df = res_df_tmp.pivot(index='coder', columns='question', values='score')
        pivot_df.columns = new_titles
        pivot_df = pivot_df.rename(index=NAME_MAPPING)
        pivot_df = pivot_df.loc[row_order]
        pivot_df = pivot_df.map(lambda x: f"{x:.3f}")
        if len(By_values) > 1:
            latex_table += r"\multicolumn{7}{|l|}{\textbf{" + v + "}}" + r" \\" + "\n"
            latex_table += r"\hline" + "\n"
        for index, row in pivot_df.iterrows():
            latex_table += f"{index} & " + " & ".join(row) + r" \\" + "\n"
        latex_table += r"\hline" + "\n"

    latex_table += r"\end{tabular}" + "\n"
    latex_table += r"\label{" + label + "}" + "\n"
    latex_table += r"\end{table}"

    if filepath is not None:
        with open(filepath, "w") as f:
            f.write(latex_table)

    return latex_table


def create_significance_latex_table(df, filepath=None):
    """Write the LaTeX table used for the Figure 4 permutation cutoff values.

    Args:
        df (pandas.DataFrame): Human block-performance table with permutation
            statistics.
        filepath (str | None): Optional output path for the generated ``.tex`` file.

    Returns:
        str: Rendered LaTeX table.
    """
    df_filtered = df[df["shuffled_mean"].notna()][['question', 'score', 'shuffled_mean', 'shuffled_q025', 'shuffled_q975']]
    new_titles = [
        "T1 (Macro F$_1$)",
        "T2 (MAE)",
        "T3 (Acc)",
        "T4 (Acc)",
        "T5 (Acc)",
    ]

    latex_table = r"\begin{table}[ht!]" + "\n"
    latex_table += r"\centering" + "\n"
    latex_table += r"\setlength{\tabcolsep}{4pt}" + "\n"
    latex_table += r"\footnotesize" + "\n"
    latex_table += r"\caption{Performance metrics for different tasks}" + "\n"
    latex_table += r"\begin{tabular}{|l|c|c|c|c|}" + "\n"
    latex_table += r"\hline" + "\n"
    header = r"\textbf{Task} & \textbf{Observed value} & \textbf{Null mean} & \textbf{Null $2.5^{th}$ perc} & \textbf{Null $97.5^{th}$ perc} \\"
    latex_table += header + "\n"
    latex_table += r"\hline" + "\n"

    for (_, row), new_title in zip(df_filtered.iterrows(), new_titles):
        latex_row = (f"{new_title} & "
                     f"{row['score']:.3f} & "
                     f"{row['shuffled_mean']:.3f} & "
                     f"{row['shuffled_q025']:.3f} & "
                     f"{row['shuffled_q975']:.3f} \\\\")
        latex_table += latex_row + "\n"
    
    latex_table += r"\hline" + "\n"
    latex_table += r"\end{tabular}" + "\n"
    latex_table += r"\label{tab:significance}" + "\n"
    latex_table += r"\end{table}"

    if filepath is not None:
        with open(filepath, "w") as f:
            f.write(latex_table)

    return latex_table


def generate_data_for_regs(df):
    """Build the long-format dataset used in the regression appendix.

    The output stacks all coder/article draws into a common schema and adds
    derived performance metrics plus article-level covariates such as
    difficulty, article length, and outsourced-coder characteristics.

    Args:
        df (pandas.DataFrame): Main evaluation sample.

    Returns:
        pandas.DataFrame: Long-format regression dataset.
    """
    df = df.copy()
    # Start from one article-level gold-standard row per NewsId.
    gold_standard = df[["NewsId", "GQ1", "GQ2", "GQ3", "GQ4", "GQ5"]].drop_duplicates().reset_index(drop=True)
    gold_standard = gold_standard.rename(columns= {
            "GQ1": "gold_Q1",
            "GQ2": "gold_Q2",
            "GQ3": "gold_Q3",
            "GQ4": "gold_Q4",
            "GQ5": "gold_Q5",      
    })
    data_for_regs = []

    # Re-express each coder's answers under common column names so all coders
    # can be stacked into one long table.
    for coder in CORE_TAGGERS:
        df_coder = df[
            ["NewsId",
            f"{coder}Q1",
            f"{coder}Q2",
            f"{coder}Q3",
            f"{coder}Q4",
            f"{coder}Q5",
            "tagged_order"
            ]
        ].reset_index(drop=True)
        df_coder = df_coder.rename(columns= {
            f"{coder}Q1": "coder_Q1",
            f"{coder}Q2": "coder_Q2",
            f"{coder}Q3": "coder_Q3",
            f"{coder}Q4": "coder_Q4",
            f"{coder}Q5": "coder_Q5",      
        })
        df_coder["coder"] = coder
        data_for_regs.append(df_coder)

    data_for_regs = pd.concat(data_for_regs, axis=0).reset_index(drop=True)
    data_for_regs = gold_standard.merge(data_for_regs, on="NewsId", how="left")

    # Reuse the same scoring functions used in the main analysis.
    task_metric = {
        "Q1": f1_score,
        "Q2": abs_error,
        "Q3": check_match,
        "Q4": check_match,
        "Q5": check_match
    }

    for task in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        data_for_regs[f'score_{task}'] = data_for_regs.apply(
            lambda x: task_metric[task](x[f"gold_{task}"], x[f"coder_{task}"]), 
            axis=1
        )

    # Cast exact-match tasks to integers so the exported file is easier to use in regressions.
    data_for_regs['score_Q3'] = data_for_regs[f'score_Q3'].astype(int)
    data_for_regs['score_Q4'] = data_for_regs[f'score_Q4'].astype(int)
    data_for_regs['score_Q5'] = data_for_regs[f'score_Q5'].astype(int)

    data_for_regs["score_all_perfect"] = (data_for_regs[
        ['score_Q1',
         'score_Q2',
         'score_Q3',
         'score_Q4',
         'score_Q5']
    ] == [1, 0, 1, 1, 1]).all(axis=1)
    data_for_regs['score_all_perfect'] = data_for_regs[f'score_all_perfect'].astype(int)

    # Append article-level difficulty flags used in the paper's heterogeneity analysis.
    diff_tasks = df[["NewsId", "IA_Q1", "IA_Q2", "IA_Q3", "IA_Q4", "IA_Q5", "IA_Any"]].drop_duplicates().reset_index(drop=True)
    diff_tasks = diff_tasks.rename(columns={
        "IA_Q1": "difficult_Q1",
        "IA_Q2": "difficult_Q2",
        "IA_Q3": "difficult_Q3",
        "IA_Q4": "difficult_Q4",
        "IA_Q5": "difficult_Q5",
        "IA_Any": "difficult_Any"
    })
    data_for_regs = data_for_regs.merge(diff_tasks, on="NewsId", how="left")

    # Add the long-article indicator used in the length-based comparisons.
    wc_threshold = df[["NewsId", "word_count"]].drop_duplicates().reset_index(drop=True)
    wc_threshold["long_article"] = (wc_threshold["word_count"] > wc_threshold["word_count"].quantile(0.9)).astype(int)
    wc_threshold = wc_threshold.drop(columns="word_count")
    data_for_regs = data_for_regs.merge(wc_threshold, on="NewsId", how="left")

    # Outsourced-human rows receive respondent metadata needed in the appendix regressions.
    df_outsourced = df[["NewsId", "ResponseId", "tagged_order", "D4", "Q_block_number"]].copy()
    df_outsourced["D4"] = (df_outsourced["D4"] == "Si").astype(int)
    df_outsourced = df_outsourced.rename(columns={"D4": "is_spanish"})
    df_outsourced["coder"] = "Q"
    data_for_regs = data_for_regs.merge(df_outsourced, on=["coder", "NewsId", "tagged_order"], how="left")
    
    return data_for_regs


def gen_internal_consistency(df, coder, n_shuffles):
    """Measure within-coder consistency across the two draws per article.

    Args:
        df (pandas.DataFrame): Main evaluation sample.
        coder (str): Tagger prefix.
        n_shuffles (int): Number of permutation draws, typically zero here.

    Returns:
        pandas.DataFrame: Internal-consistency scores by task.
    """
    question_cols = df.columns[df.columns.str.startswith(coder + "Q")].tolist()

    # Split the two draws, then align them article by article.
    df_1 = df.copy()[df["tagged_order"] == 1].reset_index(drop=True)
    df_1 = df_1[["NewsId"] + question_cols]
    rename_dict = {col: col + "_1" for col in question_cols}
    df_1 = df_1.rename(columns=rename_dict)

    df_2 = df.copy()[df["tagged_order"] == 2].reset_index(drop=True)
    df_2 = df_2[["NewsId"] + question_cols]
    rename_dict = {col: col + "_2" for col in question_cols}
    df_2 = df_2.rename(columns=rename_dict)

    df_1_2 = df_1.merge(df_2, on="NewsId", how="left")

    coder_res = []
    coder_res.append(process_Q(df_1_2, f"{coder}Q1_2", f"{coder}Q1_1", n_shuffles, f1_score))
    coder_res.append(process_Q(df_1_2, f"{coder}Q2_2", f"{coder}Q2_1", n_shuffles, abs_error))
    coder_res.append(process_Q(df_1_2, f"{coder}Q3_2", f"{coder}Q3_1", n_shuffles, check_match))
    coder_res.append(process_Q(df_1_2, f"{coder}Q4_2", f"{coder}Q4_1", n_shuffles, check_match))
    coder_res.append(process_Q(df_1_2, f"{coder}Q5_2", f"{coder}Q5_1", n_shuffles, check_match))
    coder_res = pd.DataFrame(coder_res)
    coder_res["type"] = "Consistency"
    return coder_res


def generate_consistency_analysis(df, filepath=None, model_order=["Q", "C35", "C4", "CL", "CS"]):
    """Write the internal-consistency table used in the main paper.

    Args:
        df (pandas.DataFrame): Main evaluation sample.
        filepath (str | None): Optional output path.
        model_order (list[str]): Display order for rows.
    """
    print(f"\nGenerating internal consistency table:")
    coder_res_overall = []
    for tagger in CORE_TAGGERS:
        coder_res_overall.append(gen_internal_consistency(df, tagger, 0))
    coder_res_overall = pd.concat(coder_res_overall, axis=0)

    # ``process_Q`` returns ids like ``C35Q1``; strip the task suffix before pivoting.
    coder_res_overall['base_coder'] = coder_res_overall['coder'].str.extract(r'(.*?)Q\d+')[0]
    pivot_table = coder_res_overall.pivot(index='base_coder', 
                                          columns='question', 
                                          values='score')
    pivot_table.index = pivot_table.index.map(NAME_MAPPING)
    pivot_table = pivot_table.reindex([NAME_MAPPING[model] for model in model_order])
    pivot_table = pivot_table.round(3)
    formatted_table = pivot_table.reset_index()
    formatted_table.columns = [
        'Coding strategy',
        '\\shortstack{T1\\\\(Macro F$_1$)}',
        '\\shortstack{T2\\\\(MAE)}',
        '\\shortstack{T3\\\\(Accuracy)}',
        '\\shortstack{T4\\\\(Accuracy)}',
        '\\shortstack{T5\\\\(Accuracy)}'
    ]
    latex_table = formatted_table.to_latex(
        float_format=lambda x: '{:.3f}'.format(x),
        caption='Internal consistency by tagging strategy and task',
        label='tab:internal_consistency',
        position='htbp',
        column_format='lccccc',
        bold_rows=False,
        index=False,
        escape=False
    )
    if filepath is not None:
        with open(filepath, "w") as f:
            f.write(latex_table)
    
    print("All internal consistency scores concatenated and saved into a table.")


def generate_perfect_agreement_table(df, filepath=None, model_order=["Q", "C35", "C4", "CL", "CS"]):
    """Write the exact-match table for T1 to T3.

    Args:
        df (pandas.DataFrame): Main evaluation sample.
        filepath (str | None): Optional output path.
        model_order (list[str]): Display order for rows.
    """
    print(f"\nGenerating perfect agreement table:")

    def gen_perfect_Q1_Q2_Q3_score(df, coder):
        coder_res = []
        coder_res.append({"coder": coder, "question": "T1", "score": (df.apply(lambda x: f1_score(x["GQ1"], x[f"{coder}Q1"]), axis=1) == 1).mean()})
        coder_res.append({"coder": coder, "question": "T2", "score": (df.apply(lambda x: abs_error(x["GQ2"], x[f"{coder}Q2"]), axis=1) == 0).mean()})
        coder_res.append({"coder": coder, "question": "T3", "score": (df.apply(lambda x: check_match(x["GQ3"], x[f"{coder}Q3"]), axis=1) == 1).mean()})
        return pd.DataFrame(coder_res)

    coder_res_perfect_Q1_Q2_Q3 = []
    for tagger in CORE_TAGGERS:
        coder_res_perfect_Q1_Q2_Q3.append(gen_perfect_Q1_Q2_Q3_score(df, tagger))
    coder_res_perfect_Q1_Q2_Q3 = pd.concat(coder_res_perfect_Q1_Q2_Q3, axis=0)

    pivot_table = coder_res_perfect_Q1_Q2_Q3.pivot(index='coder', 
                                                columns='question', 
                                                values='score')
    pivot_table.index = pivot_table.index.map(NAME_MAPPING)
    pivot_table = pivot_table.reindex([NAME_MAPPING[model] for model in model_order])
    pivot_table = pivot_table.round(3)
    formatted_table = pivot_table.reset_index()
    formatted_table.columns = [
        'Coding strategy',
        '\\shortstack{T1}',
        '\\shortstack{T2}',
        '\\shortstack{T3}'
    ]
    latex_table = formatted_table.to_latex(
        float_format=lambda x: '{:.3f}'.format(x),
        caption='Perfect agreement with the gold standard in T1, T2, and T3 by tagging strategy and task',
        label='tab:perfect_Q1_Q2_Q3',
        position='htbp',
        column_format='lccccc',
        bold_rows=False,
        index=False,
        escape=False
    )
    if filepath is not None:
        with open(filepath, "w") as f:
            f.write(latex_table)

    print("Perfect agreement table created and saved.")


def calculate_intertemporal_consistency(df, filepath):
    """Write the GPT-3.5 intertemporal-consistency table.

    Args:
        df (pandas.DataFrame): Main evaluation sample.
        filepath (str): Output path for the generated ``.tex`` file.
    """
    print(f"\nGenerating intertemporal consistency table:")
    question_cols = df.columns[df.columns.str.startswith("C35" + "Q")].tolist()

    df_1 = df.copy()[df["tagged_order"] == 1].reset_index(drop=True)
    df_1 = df_1[["NewsId"] + question_cols]
    rename_dict = {col: col + "_1" for col in question_cols}
    df_1 = df_1.rename(columns=rename_dict)

    df_2 = df.copy()[df["tagged_order"] == 2].reset_index(drop=True)
    df_2 = df_2[["NewsId"] + question_cols]
    rename_dict = {col: col + "_2" for col in question_cols}
    df_2 = df_2.rename(columns=rename_dict)

    # Start from the two April 2024 draws, which anchor the comparison.
    df_1_2 = df_1.merge(df_2, on="NewsId", how="left")

    # Merge in the October 2023 runs and the August 2025 zero-shot runs.
    question_cols = df.columns[df.columns.str.startswith("orig_C35" + "Q")].tolist()
    df_orig = df.copy()[df["tagged_order"] == 1].reset_index(drop=True)
    df_orig = df_orig[["NewsId"] + question_cols]
    df_orig = df_orig.rename(columns=rename_dict)
    df_1_2 = df_1_2.merge(df_orig, on="NewsId", how="left")

    question_cols = df.columns[df.columns.str.startswith("C35ZS" + "Q")].tolist()
    df_2025 = df.copy()[df["tagged_order"] == 1].reset_index(drop=True)
    df_2025 = df_2025[["NewsId"] + question_cols]
    df_1_2 = df_1_2.merge(df_2025, on="NewsId", how="left")

    coder_res_former = []
    coder_res_former.append(process_Q(df_1_2, "C35Q1_2", "orig_C35Q1", 0, f1_score))
    coder_res_former.append(process_Q(df_1_2, "C35Q2_2", "orig_C35Q2", 0, abs_error))
    coder_res_former.append(process_Q(df_1_2, "C35Q3_2", "orig_C35Q3", 0, check_match))
    coder_res_former.append(process_Q(df_1_2, "C35Q4_2", "orig_C35Q4", 0, check_match))
    coder_res_former.append(process_Q(df_1_2, "C35Q5_2", "orig_C35Q5", 0, check_match))
    coder_res_former = pd.DataFrame(coder_res_former)
    coder_res_former["type"] = "October 2023"

    coder_res_latter = []
    coder_res_latter.append(process_Q(df_1_2, "C35Q1_2", "C35Q1_1", 0, f1_score))
    coder_res_latter.append(process_Q(df_1_2, "C35Q2_2", "C35Q2_1", 0, abs_error))
    coder_res_latter.append(process_Q(df_1_2, "C35Q3_2", "C35Q3_1", 0, check_match))
    coder_res_latter.append(process_Q(df_1_2, "C35Q4_2", "C35Q4_1", 0, check_match))
    coder_res_latter.append(process_Q(df_1_2, "C35Q5_2", "C35Q5_1", 0, check_match))
    coder_res_latter = pd.DataFrame(coder_res_latter)
    coder_res_latter["type"] = "April 2024"

    coder_res_2025 = []
    coder_res_2025.append(process_Q(df_1_2, "C35Q1_2", "C35ZSQ1", 0, f1_score))
    coder_res_2025.append(process_Q(df_1_2, "C35Q2_2", "C35ZSQ2", 0, abs_error))
    coder_res_2025.append(process_Q(df_1_2, "C35Q3_2", "C35ZSQ3", 0, check_match))
    coder_res_2025.append(process_Q(df_1_2, "C35Q4_2", "C35ZSQ4", 0, check_match))
    coder_res_2025.append(process_Q(df_1_2, "C35Q5_2", "C35ZSQ5", 0, check_match))
    coder_res_2025 = pd.DataFrame(coder_res_2025)
    coder_res_2025["type"] = "August 2025"

    coder_res = pd.concat([coder_res_latter, coder_res_former, coder_res_2025], axis=0)

    # Pivot into the exact wide format used by the paper table.
    pivoted_table = coder_res.pivot(index='question', columns='type', values='score')
    pivoted_table = pivoted_table.reset_index()
    pivoted_table = pivoted_table.rename(columns={'question': 'Task'})

    pivoted_table["Task"] = [
        'T1 (Macro F$_1$)',
        'T2 (MAE)',
        'T3 (Accuracy)',
        'T4 (Accuracy)',
        'T5 (Accuracy)'
    ]

    pivoted_table = pivoted_table[["Task", "October 2023", "April 2024", "August 2025"]]

    latex_table = pivoted_table.to_latex(
        float_format=lambda x: '{:.3f}'.format(x) if isinstance(x, float) else x,
        caption='Intertemporal consistency of GPT-3.5-turbo',
        label='tab:intertemporal_consistency',
        position='htbp',
        column_format='lcc',
        bold_rows=False,
        index=False,
        escape=False
    )
    if filepath is not None:
        with open(filepath, "w") as f:
            f.write(latex_table)

    print("All intertemporal consistency scores concatenated and saved into a table.")


def calculate_gpt35_perf_across_time(df, filepath):
    """Write the GPT-3.5 performance-by-release table.

    Args:
        df (pandas.DataFrame): Main evaluation sample.
        filepath (str): Output path for the generated ``.tex`` file.
    """
    print(f"\nGenerating GPT-3.5 performance across time:")

    coder_res_2024 = []
    coder_res_2024.append(process_Q(df, "GQ1", "C35Q1", 0, f1_score))
    coder_res_2024.append(process_Q(df, "GQ2", "C35Q2", 0, abs_error))
    coder_res_2024.append(process_Q(df, "GQ3", "C35Q3", 0, check_match))
    coder_res_2024.append(process_Q(df, "GQ4", "C35Q4", 0, check_match))
    coder_res_2024.append(process_Q(df, "GQ5", "C35Q5", 0, check_match))
    coder_res_2024.append(perfect_score(df, "C35"))
    coder_res_2024 = pd.DataFrame(coder_res_2024)
    coder_res_2024["type"] = "April 2024"

    # The historical runs are only available once per article, so keep tagged_order 1.
    df_1 = df.copy()[df["tagged_order"] == 1].reset_index(drop=True)
    coder_res_2023 = []
    coder_res_2023.append(process_Q(df_1, "GQ1", "orig_C35Q1", 0, f1_score))
    coder_res_2023.append(process_Q(df_1, "GQ2", "orig_C35Q2", 0, abs_error))
    coder_res_2023.append(process_Q(df_1, "GQ3", "orig_C35Q3", 0, check_match))
    coder_res_2023.append(process_Q(df_1, "GQ4", "orig_C35Q4", 0, check_match))
    coder_res_2023.append(process_Q(df_1, "GQ5", "orig_C35Q5", 0, check_match))
    coder_res_2023.append(perfect_score(df_1, "orig_C35"))
    coder_res_2023 = pd.DataFrame(coder_res_2023)
    coder_res_2023["type"] = "October 2023"

    coder_res = pd.concat([coder_res_2023, coder_res_2024], axis=0)

    # Present the two vintages side by side in the published order.
    pivoted_table = coder_res.pivot(index='question',
                                    columns='type',
                                    values='score')
    pivoted_table = pivoted_table[["October 2023", "April 2024"]]
    pivoted_table = pivoted_table.reset_index()
    pivoted_table = pivoted_table.rename(columns={'question': 'Task'})

    pivoted_table["Task"] = [
        'T1 (Macro F$_1$)',
        'T2 (MAE)',
        'T3 (Accuracy)',
        'T4 (Accuracy)',
        'T5 (Accuracy)',
        'All correct (Proportion)'
    ]

    latex_table = pivoted_table.to_latex(
        float_format=lambda x: '{:.3f}'.format(x) if isinstance(x, float) else x,
        caption='GPT-3.5-turbo performance across time',
        label='tab:gpt35_across_time',
        position='htbp',
        column_format='lcc',
        bold_rows=False,
        index=False,
        escape=False
    )
    if filepath is not None:
        with open(filepath, "w") as f:
            f.write(latex_table)
    
    print("All GPT-3.5 scores across time concatenated and saved into a table.")


def main():
    """Run the full analysis pipeline and write all public outputs.

    The pipeline follows the paper structure: descriptive checks first, then
    main figures and tables, then appendix-oriented exports.
    """
    ensure_output_dirs()
    df = load_main_data()
    print("Main data loaded successfully.")

    # Report the field period for the outsourced-coder exercise.
    study_duration = df["RecordedDate"].max() - df["RecordedDate"].min()
    days = study_duration.days
    print(f"\nStudy duration: {days:,} days ({study_duration})")

    print("\nGenerating summary statistics for word counts.\n")
    get_summary_stats_word_counts(df)

    print("\nGenerating summary statistics for outsourced coder demographics.\n")
    analyze_outsourced_coders_demographics(df)

    calculate_intercoder_agreement(df)
    print("Intercoder agreement calculated.")

    analyze_gold_standard_labels(df)
    print("Gold standard labels analyzed.")

    print("\nGenerating overall scores:")
    coder_res_overall = []
    taggers = ALL_TAGGERS
    for tagger in taggers:
        coder_res_overall.append(gen_scores_overall(df, tagger, 0))
    coder_res_overall = pd.concat(coder_res_overall, axis=0)
    print("All overall scores concatenated.")

    # The row order is set once here and reused in all main-text LaTeX tables.
    plot_overall(coder_res_overall, f"{PLOTS_DIR}/Fig_01_-_Overall.pdf")
    print("\nOverall results plotted and saved as Fig_01_-_Overall.pdf")
    print("\nCalculating overall detailed results")
    row_order = [
        "Outsourced humans",
        "GPT-3.5-turbo",
        "GPT-4-turbo",
        "Claude 3 Opus",
        "Claude 3.5 Sonnet"
    ]
    create_results_latex_table(coder_res_overall,
                               "Overall performance, across tasks and coding strategies",
                               "tab:overall",
                               row_order,
                               f"{TABLES_DIR}/Tab_A1_-_Overall.tex")

    print("\nCalculating perfect agreement results")
    generate_perfect_agreement_table(df, f"{TABLES_DIR}/Tab_D5_-_Perfect_agreement.tex")

    print("\nGenerating scores by difficulty:")
    coder_res_diff = []
    for tagger in taggers:
        coder_res_diff.append(gen_scores_difficulty(df, tagger, 0))
    coder_res_diff = pd.concat(coder_res_diff, axis=0)
    print("All difficulty scores concatenated.")

    plot_model_comparison(coder_res_diff, f"{PLOTS_DIR}/Fig_02_-_By_difficulty.pdf")
    print("Model comparison by difficulty plotted and saved as Fig_02_-_By_difficulty.pdf")
    print("\nCalculating by difficulty detailed results")
    create_results_latex_table(coder_res_diff,
                               "Performance by article difficulty, across tasks and coding strategies",
                               "tab:By_difficulty",
                               row_order,
                               f"{TABLES_DIR}/Tab_A2_-_By_difficulty.tex")

    print("\nGenerating scores by length:")
    coder_res_length = []
    for tagger in taggers:
        coder_res_length.append(gen_scores_by_length(df, tagger, 0))
    coder_res_length = pd.concat(coder_res_length, axis=0)
    print("All length scores concatenated.")

    plot_model_comparison(coder_res_length, f"{PLOTS_DIR}/Fig_03_-_By_length.pdf")
    print("Model comparison by length plotted and saved as Fig_03_-_By_length.pdf")
    print("\nCalculating by length detailed results")
    create_results_latex_table(coder_res_length,
                               "Performance by article length, across tasks and coding strategies",
                               "tab:By_length",
                               row_order,
                               f"{TABLES_DIR}/Tab_A3_-_By_length.tex")

    # Figure 4 needs permutation benchmarks to draw the significance markers.
    print("Calculating the permutations for Figure 4")
    data_by_block = gen_scores_humans_by_block(df, 2000, 2345)
    print("Scores for humans by block generated.")

    plot_human_perf_by_block(data_by_block, f"{PLOTS_DIR}/Fig_04_-_By_block.pdf")
    print("Human performance by block plotted and saved as Fig_04_-_By_block.pdf")
    print("\nCalculating human performance detailed results")
    data_by_block["coder"] = data_by_block["type"]
    data_by_block["type"] = "Human"
    create_results_latex_table(data_by_block,
                               "Human coders' performance task progression",
                               "tab:task_progression",
                               ['Overall', 'First article', 'Second article', 'Third article'],
                               f"{TABLES_DIR}/Tab_A5_-_By_block.tex")

    print("\nCalculating human performance significance results")
    create_significance_latex_table(data_by_block, f"{TABLES_DIR}/Tab_A4_-_significance.tex")

    print("\nGenerating scores for Figure 5")
    generate_high_performing_humans_figure(
        df, f"{PLOTS_DIR}/Fig_05_-_High_performing_humans.pdf"
    )
    print("High-performing human comparison plotted and saved as Fig_05_-_High_performing_humans.pdf")

    generate_consistency_analysis(df, f"{TABLES_DIR}/Tab_1_-_Internal_consistency.tex")
    calculate_intertemporal_consistency(df, f"{TABLES_DIR}/Tab_C4_-_Intertemporal_consistency.tex")
    calculate_gpt35_perf_across_time(df, f"{TABLES_DIR}/Tab_C4_-_gpt35_perf_across_time.tex")

    # Export a regression-ready long file for the appendix analysis.
    data_for_regs = generate_data_for_regs(df)
    print("\nGenerating data for regs")
    data_for_regs.to_csv(f"{OUTPUT_DIR}/data_for_regs.txt", sep="\t", index=False)

    # The prompt-sensitivity appendix reuses the score tables already computed above.
    print("\nCalculating overall detailed results for prompt sensitivity")
    create_results_latex_table(
        coder_res_overall,
        "Overall performance across prompts.",
        "tab:overall_sens",
        row_order=[
            "Zero-Shot GPT-3.5-turbo (2025-08)",
            "Few-Shot GPT-3.5-turbo (2025-08)",
            "Zero-Shot GPT-4-turbo (2025-08)",
            "Few-Shot GPT-4-turbo (2025-08)",
        ],
        filepath=f"{TABLES_DIR}/Tab_E1_-_Overall_prompts_sensitivity.tex",
    )

    print("\nCalculating by difficulty detailed results")
    create_results_latex_table(
        coder_res_diff,
        "Performance by article difficulty, across tasks and coding strategies",
        "tab:By_difficulty_sens",
        row_order=[
            "Zero-Shot GPT-3.5-turbo (2025-08)",
            "Few-Shot GPT-3.5-turbo (2025-08)",
            "Zero-Shot GPT-4-turbo (2025-08)",
            "Few-Shot GPT-4-turbo (2025-08)",
        ],
        filepath=f"{TABLES_DIR}/Tab_E3_-_By_difficulty_sensitivity.tex",
    )

    print("\nCalculating by length detailed results")
    create_results_latex_table(
        coder_res_length,
        "Performance by article length, across tasks and coding strategies",
        "tab:By_length_sens",
        row_order=[
            "Zero-Shot GPT-3.5-turbo (2025-08)",
            "Few-Shot GPT-3.5-turbo (2025-08)",
            "Zero-Shot GPT-4-turbo (2025-08)",
            "Few-Shot GPT-4-turbo (2025-08)",
        ],
        filepath=f"{TABLES_DIR}/Tab_E3_-_By_length_sensitivity.tex",
    )


if __name__ == "__main__":
    main()
