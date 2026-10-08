# -*- coding: utf-8 -*-
"""
DOTA 2 MATCH OUTCOME PREDICTION USING HERO COMPOSITION

Research Question:
Can Dota 2 match outcomes be predicted based on the hero composition
of the Radiant and Dire teams?

IMPORTANT CLEANING RULES USED IN THIS VERSION
---------------------------------------------
STEP 1 - Check duplicate values in match_id.
STEP 2 - Check missing values in each column.
STEP 3 - Check invalid radiant_win values:
         1 = Radiant win
         0 = Dire win
STEP 4 - Check and REMOVE a match only when Radiant OR Dire has
         the complete hero composition [0, 0, 0, 0, 0].

Steps 1-3 are checks/reports only. They do not silently delete rows.
Step 4 is the only automatic row-removal rule in the cleaning section.

The model keeps Radiant and Dire hero features separate.
"""

# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import gc
from collections import Counter, defaultdict
from itertools import combinations, chain

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import hstack

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    log_loss,
    ConfusionMatrixDisplay
)

try:
    from IPython.display import display
except ImportError:
    display = None


def show(obj):
    """Display DataFrames in notebooks and print them in normal Python."""
    if display is not None:
        display(obj)
    else:
        print(obj)


RANDOM_STATE = 42


# ============================================================
# STEP 2: CONFIGURATION
# ============================================================

# Spyder / Windows paths.
# Change BASE_FOLDER only if your dataset is stored elsewhere.
BASE_FOLDER = (
    r"C:\Users\Rex\Desktop\Working File\Masteral Files"
    r"\Data 501 - Machine Learning\Dataset\Dota 2"
)

HERO_INFO = (
    BASE_FOLDER
    + r"\dota_heroes.csv"
)

MATCHES = (
    BASE_FOLDER
    + r"\dota2_500k_matches_ids.csv"
)

CLEAN_FILE = (
    BASE_FOLDER
    + r"\dota_matches_clean.csv"
)

RESULTS_FILE = (
    BASE_FOLDER
    + r"\dota2_model_results.csv"
)

HERO_EDA_FILE = (
    BASE_FOLDER
    + r"\dota2_hero_eda.csv"
)

RF_IMPORTANCE_FILE = (
    BASE_FOLDER
    + r"\dota2_rf_feature_importance.csv"
)

# Increase only if your computer has enough RAM/CPU.
MODEL_N_JOBS = 2



# ============================================================
# STEP 3: LOAD EXISTING DOTA 2 MATCH DATA
# ============================================================

print("=" * 70)
print("STEP 3: LOAD DATASET")
print("=" * 70)

# Only load the columns needed by the research.
# This reduces memory usage, especially in Google Colab.
required_columns = [
    "match_id",
    "radiant_win",
    "radiant_team",
    "dire_team"
]

df_matches = pd.read_csv(
    MATCHES,
    usecols=lambda col: col in required_columns
)

print("Rows:", len(df_matches))
print("Columns:", len(df_matches.columns))
print("\nColumns:")
print(df_matches.columns.tolist())

show(df_matches.head())


# ============================================================
# STEP 4: CHECK REQUIRED COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("STEP 4: CHECK REQUIRED COLUMNS")
print("=" * 70)

missing_required = [
    col
    for col in required_columns
    if col not in df_matches.columns
]

if missing_required:
    print("Missing required columns:")
    print(missing_required)
    raise SystemExit(
        "Project stopped because required columns are missing."
    )
else:
    print("All required columns are present.")


# ============================================================
# STEP 5: PARSE HERO TEAM COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("STEP 5: PARSE HERO TEAM COLUMNS")
print("=" * 70)


def parse_heroes(value):
    """
    Convert a comma-separated team into a list of integer hero IDs.

    Examples:
        "1,2,3,4,5" -> [1, 2, 3, 4, 5]
        missing value -> []

    The function does not delete rows.
    """
    if pd.isna(value):
        return []

    try:
        return [
            int(hero.strip())
            for hero in str(value).split(",")
            if hero.strip() != ""
        ]
    except (ValueError, TypeError):
        return []


df_matches["radiant_team"] = (
    df_matches["radiant_team"]
    .apply(parse_heroes)
)

df_matches["dire_team"] = (
    df_matches["dire_team"]
    .apply(parse_heroes)
)

print("Example Radiant team:")
print(df_matches.loc[0, "radiant_team"])

print("\nExample Dire team:")
print(df_matches.loc[0, "dire_team"])


# ============================================================
# STEP 6: CHECK TEAM SIZE
# ============================================================
# This is a descriptive check only.
# No rows are deleted based on team size.

df_matches["radiant_count"] = (
    df_matches["radiant_team"].str.len()
)

df_matches["dire_count"] = (
    df_matches["dire_team"].str.len()
)

print("\n" + "=" * 70)
print("STEP 6: TEAM SIZE CHECK")
print("=" * 70)

print("Radiant team-size distribution:")
print(
    df_matches["radiant_count"]
    .value_counts()
    .sort_index()
)

print("\nDire team-size distribution:")
print(
    df_matches["dire_count"]
    .value_counts()
    .sort_index()
)

print(
    "\nMatches where both teams have exactly 5 heroes:",
    (
        (df_matches["radiant_count"] == 5)
        & (df_matches["dire_count"] == 5)
    ).sum()
)


# ============================================================
# DATA CLEANING
# ============================================================
# The following section follows the user's four requested rules.


# ============================================================
# CLEANING STEP 1:
# CHECK DUPLICATE VALUES IN MATCH_ID
# ============================================================
# CHECK ONLY - duplicate rows are not automatically deleted.

print("\n" + "=" * 70)
print("CLEANING STEP 1: CHECK DUPLICATE MATCH IDs")
print("=" * 70)

duplicate_match_ids = df_matches[
    df_matches["match_id"].duplicated(
        keep=False
    )
].copy()

print(
    "Number of rows belonging to duplicate match IDs:",
    len(duplicate_match_ids)
)

print(
    "Number of unique duplicate match IDs:",
    duplicate_match_ids["match_id"].nunique()
)

if len(duplicate_match_ids) > 0:
    print("\nSample duplicate match IDs:")
    show(
        duplicate_match_ids[
            [
                "match_id",
                "radiant_win",
                "radiant_team",
                "dire_team"
            ]
        ]
        .sort_values("match_id")
        .head(20)
    )
else:
    print("No duplicate match IDs found.")


# ============================================================
# CLEANING STEP 2:
# CHECK MISSING VALUES IN EACH COLUMN
# ============================================================
# CHECK ONLY - missing rows are not automatically deleted.

print("\n" + "=" * 70)
print("CLEANING STEP 2: CHECK MISSING VALUES")
print("=" * 70)

missing_report = pd.DataFrame({
    "Missing_Count":
        df_matches.isnull().sum(),
    "Missing_Percentage":
        (
            df_matches.isnull().sum()
            / len(df_matches)
        ) * 100
})

show(missing_report)

missing_rows = df_matches[
    df_matches.isnull().any(axis=1)
].copy()

print(
    "\nNumber of rows with at least one missing value:",
    len(missing_rows)
)

if len(missing_rows) > 0:
    print("\nSample rows with missing values:")
    show(missing_rows.head(20))
else:
    print("No missing values found.")


# ============================================================
# CLEANING STEP 3:
# CHECK INVALID INPUTS IN RADIANT_WIN
# ============================================================
# 1 = Radiant Win
# 0 = Dire Win
#
# CHECK ONLY - invalid rows are not automatically deleted.

print("\n" + "=" * 70)
print("CLEANING STEP 3: CHECK RADIANT_WIN")
print("=" * 70)

invalid_radiant_win = df_matches[
    df_matches["radiant_win"].isna()
    | ~df_matches["radiant_win"].isin(
        [0, 1]
    )
].copy()

print(
    "Number of invalid/missing radiant_win rows:",
    len(invalid_radiant_win)
)

print("\nradiant_win value counts:")
print(
    df_matches["radiant_win"]
    .value_counts(dropna=False)
)

print("\nMeaning:")
print("1 = Radiant Win")
print("0 = Dire Win")

if len(invalid_radiant_win) > 0:
    print("\nSample invalid radiant_win rows:")
    show(
        invalid_radiant_win[
            [
                "match_id",
                "radiant_win",
                "radiant_team",
                "dire_team"
            ]
        ].head(20)
    )
else:
    print("No invalid radiant_win values found.")


# ============================================================
# CLEANING STEP 4:
# CHECK AND REMOVE [0, 0, 0, 0, 0] COMPOSITIONS
# ============================================================
# This is the ONLY automatic row deletion in the cleaning section.
#
# Remove the complete match row if:
#   Radiant = [0, 0, 0, 0, 0]
# OR
#   Dire    = [0, 0, 0, 0, 0]

print("\n" + "=" * 70)
print("CLEANING STEP 4: REMOVE ALL-ZERO TEAM COMPOSITIONS")
print("=" * 70)

all_zero_team = [
    0,
    0,
    0,
    0,
    0
]

radiant_zero_mask = (
    df_matches["radiant_team"]
    .apply(
        lambda team:
            team == all_zero_team
    )
)

dire_zero_mask = (
    df_matches["dire_team"]
    .apply(
        lambda team:
            team == all_zero_team
    )
)

zero_composition_mask = (
    radiant_zero_mask
    | dire_zero_mask
)

zero_composition_rows = df_matches[
    zero_composition_mask
].copy()

print(
    "Radiant [0, 0, 0, 0, 0] rows:",
    int(radiant_zero_mask.sum())
)

print(
    "Dire [0, 0, 0, 0, 0] rows:",
    int(dire_zero_mask.sum())
)

print(
    "Total match rows to remove:",
    int(zero_composition_mask.sum())
)

if len(zero_composition_rows) > 0:
    print("\nSample rows that will be removed:")
    show(
        zero_composition_rows[
            [
                "match_id",
                "radiant_win",
                "radiant_team",
                "dire_team"
            ]
        ].head(20)
    )

original_count = len(df_matches)

df_matches_clean = df_matches[
    ~zero_composition_mask
].copy()

df_matches_clean.reset_index(
    drop=True,
    inplace=True
)

print("\n" + "=" * 70)
print("DATA CLEANING SUMMARY")
print("=" * 70)

print("Original rows:", original_count)
print(
    "Duplicate match-ID rows found:",
    len(duplicate_match_ids)
)
print(
    "Rows with missing values found:",
    len(missing_rows)
)
print(
    "Invalid/missing radiant_win rows found:",
    len(invalid_radiant_win)
)
print(
    "Rows removed for [0, 0, 0, 0, 0] composition:",
    len(zero_composition_rows)
)
print(
    "Final rows after Step 4:",
    len(df_matches_clean)
)


# ============================================================
# STEP 7: LOAD HERO ID -> HERO NAME MAPPING
# ============================================================

print("\n" + "=" * 70)
print("STEP 7: LOAD HERO MAPPING")
print("=" * 70)

try:
    heroes_df = pd.read_csv(
        HERO_INFO,
        usecols=lambda col:
            col in ["id", "localized_name"]
    )

    heroes_df = heroes_df.dropna(
        subset=[
            "id",
            "localized_name"
        ]
    ).copy()

    heroes_df["id"] = (
        heroes_df["id"]
        .astype(int)
    )

    hero_names = dict(
        zip(
            heroes_df["id"],
            heroes_df["localized_name"]
        )
    )

    full_hero_ids = sorted(
        heroes_df["id"]
        .unique()
        .tolist()
    )

    print(
        "Hero mapping loaded:",
        len(hero_names),
        "heroes"
    )

except Exception as e:
    print(
        "Could not load hero mapping."
    )
    print(e)

    hero_names = {}

    # Fallback: use hero IDs observed in the match data.
    full_hero_ids = sorted(
        set(
            chain.from_iterable(
                df_matches_clean[
                    "radiant_team"
                ]
            )
        )
        |
        set(
            chain.from_iterable(
                df_matches_clean[
                    "dire_team"
                ]
            )
        )
    )


# ============================================================
# STEP 8: CREATE HERO NAME COLUMNS
# ============================================================


def hero_ids_to_names(hero_list):
    return [
        hero_names.get(
            hero_id,
            f"Unknown ({hero_id})"
        )
        for hero_id in hero_list
    ]


df_matches_clean[
    "radiant_hero_names"
] = (
    df_matches_clean[
        "radiant_team"
    ].apply(
        hero_ids_to_names
    )
)

df_matches_clean[
    "dire_hero_names"
] = (
    df_matches_clean[
        "dire_team"
    ].apply(
        hero_ids_to_names
    )
)

print("\nExample hero names:")
print(
    "Radiant:",
    df_matches_clean.loc[
        0,
        "radiant_hero_names"
    ]
)

print(
    "Dire:",
    df_matches_clean.loc[
        0,
        "dire_hero_names"
    ]
)


# ============================================================
# STEP 9: SAVE CLEAN DATA
# ============================================================
# The saved clean file reflects exactly the four cleaning rules above.

df_matches_clean.to_csv(
    CLEAN_FILE,
    index=False
)

print("\nClean data saved to:")
print(CLEAN_FILE)


# ============================================================
# MODEL-READINESS CHECK
# ============================================================
# Steps 1-3 were check-only, as requested.
# For valid model evaluation, the target must still contain only 0 and 1.
# Duplicate IDs and missing required values are also reported because they
# can affect the scientific validity of the train/test evaluation.
#
# We do NOT silently delete these rows.

required_missing_after_cleaning = (
    df_matches_clean[
        required_columns
    ].isna().any(axis=1)
)

invalid_target_after_cleaning = (
    df_matches_clean[
        "radiant_win"
    ].isna()
    |
    ~df_matches_clean[
        "radiant_win"
    ].isin([0, 1])
)

duplicate_after_cleaning = (
    df_matches_clean[
        "match_id"
    ].duplicated(
        keep=False
    )
)

model_ready = (
    required_missing_after_cleaning.sum() == 0
    and invalid_target_after_cleaning.sum() == 0
    and duplicate_after_cleaning.sum() == 0
)

print("\n" + "=" * 70)
print("MODEL-READINESS CHECK")
print("=" * 70)

print(
    "Duplicate match-ID rows still present:",
    int(
        duplicate_after_cleaning.sum()
    )
)

print(
    "Rows with missing required values still present:",
    int(
        required_missing_after_cleaning.sum()
    )
)

print(
    "Rows with invalid radiant_win still present:",
    int(
        invalid_target_after_cleaning.sum()
    )
)

if model_ready:
    print(
        "Dataset passed the model-readiness checks."
    )
else:
    print(
        "\nWARNING:"
        "\nThe requested cleaning rules do not delete problems found "
        "in Steps 1-3. Therefore the modeling section will be skipped "
        "until those source-data issues are resolved."
    )


# ============================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================
# Objective:
# Explore match outcomes, hero pick rates, hero performance,
# hero combinations, and team composition patterns before modeling.


# ============================================================
# EDA 1: DATASET OVERVIEW
# ============================================================

print("\n" + "=" * 70)
print("EDA 1: DATASET OVERVIEW")
print("=" * 70)

print(
    "Number of matches:",
    len(df_matches_clean)
)

print(
    "Number of columns:",
    len(df_matches_clean.columns)
)

print(
    "Duplicate match IDs:",
    df_matches_clean[
        "match_id"
    ].duplicated().sum()
)

print("\nMissing values:")
print(
    df_matches_clean
    .isnull()
    .sum()
    .sort_values(
        ascending=False
    )
)

# Avoid describe(include="all"), which is slow for list/object columns.
numeric_columns = (
    df_matches_clean
    .select_dtypes(
        include=np.number
    )
    .columns
)

show(
    df_matches_clean[
        numeric_columns
    ].describe().T
)


# ============================================================
# EDA 2: WIN DISTRIBUTION
# ============================================================
# This analysis is meaningful only for valid 0/1 outcomes.

print("\n" + "=" * 70)
print("EDA 2: WIN DISTRIBUTION")
print("=" * 70)

if invalid_target_after_cleaning.sum() == 0:

    win_counts = (
        df_matches_clean[
            "radiant_win"
        ]
        .value_counts()
        .sort_index()
    )

    radiant_win_rate = (
        df_matches_clean[
            "radiant_win"
        ].mean() * 100
    )

    print(
        "Radiant wins:",
        int(
            win_counts.get(
                1,
                0
            )
        )
    )

    print(
        "Dire wins:",
        int(
            win_counts.get(
                0,
                0
            )
        )
    )

    print(
        "Radiant win rate:",
        round(
            radiant_win_rate,
            2
        ),
        "%"
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        [
            "Dire Win",
            "Radiant Win"
        ],
        [
            win_counts.get(
                0,
                0
            ),
            win_counts.get(
                1,
                0
            )
        ]
    )

    plt.title(
        "Radiant vs Dire Match Outcomes"
    )

    plt.ylabel(
        "Number of Matches"
    )

    plt.tight_layout()
    plt.show()

else:
    print(
        "Skipped because invalid radiant_win values are present."
    )


# ============================================================
# EDA 3: HERO PICK AND WIN STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("EDA 3: HERO PICK AND WIN STATISTICS")
print("=" * 70)

hero_pick_counter = Counter()

for radiant_team, dire_team in (
    df_matches_clean[
        [
            "radiant_team",
            "dire_team"
        ]
    ].itertuples(
        index=False,
        name=None
    )
):
    hero_pick_counter.update(
        radiant_team
    )

    hero_pick_counter.update(
        dire_team
    )


hero_eda = pd.DataFrame({
    "hero_id":
        full_hero_ids
})

hero_eda[
    "hero_name"
] = (
    hero_eda[
        "hero_id"
    ].map(
        lambda hero_id:
            hero_names.get(
                hero_id,
                f"Unknown ({hero_id})"
            )
    )
)

hero_eda[
    "pick_count"
] = (
    hero_eda[
        "hero_id"
    ].map(
        hero_pick_counter
    )
    .fillna(0)
    .astype(int)
)

total_hero_slots = sum(
    len(team)
    for team in df_matches_clean[
        "radiant_team"
    ]
) + sum(
    len(team)
    for team in df_matches_clean[
        "dire_team"
    ]
)

hero_eda[
    "pick_rate"
] = np.where(
    total_hero_slots > 0,
    hero_eda[
        "pick_count"
    ] / total_hero_slots * 100,
    0
)

print("\nTop 10 heroes by pick count:")

show(
    hero_eda
    .sort_values(
        "pick_count",
        ascending=False
    )
    .head(10)
)


# Win statistics require a valid target.
if invalid_target_after_cleaning.sum() == 0:

    hero_games = Counter()
    hero_wins = Counter()

    for (
        radiant_team,
        dire_team,
        radiant_win
    ) in (
        df_matches_clean[
            [
                "radiant_team",
                "dire_team",
                "radiant_win"
            ]
        ].itertuples(
            index=False,
            name=None
        )
    ):

        for hero_id in radiant_team:
            hero_games[
                hero_id
            ] += 1

            if radiant_win == 1:
                hero_wins[
                    hero_id
                ] += 1

        for hero_id in dire_team:
            hero_games[
                hero_id
            ] += 1

            if radiant_win == 0:
                hero_wins[
                    hero_id
                ] += 1


    hero_eda[
        "wins"
    ] = (
        hero_eda[
            "hero_id"
        ].map(
            hero_wins
        )
        .fillna(0)
        .astype(int)
    )

    hero_eda[
        "win_rate"
    ] = np.where(
        hero_eda[
            "pick_count"
        ] > 0,

        hero_eda[
            "wins"
        ]
        /
        hero_eda[
            "pick_count"
        ]
        * 100,

        np.nan
    )

    min_hero_games = max(
        30,
        int(
            len(
                df_matches_clean
            ) * 0.01
        )
    )

    print(
        "\nTop 10 heroes by win rate "
        f"(minimum {min_hero_games} appearances):"
    )

    show(
        hero_eda[
            hero_eda[
                "pick_count"
            ] >= min_hero_games
        ]
        .sort_values(
            "win_rate",
            ascending=False
        )
        .head(10)
    )


# ============================================================
# EDA 4: HERO WIN RATE BY SIDE
# ============================================================

print("\n" + "=" * 70)
print("EDA 4: HERO WIN RATE BY SIDE")
print("=" * 70)

if invalid_target_after_cleaning.sum() == 0:

    radiant_games_counter = Counter()
    radiant_wins_counter = Counter()

    dire_games_counter = Counter()
    dire_wins_counter = Counter()

    for (
        radiant_team,
        dire_team,
        radiant_win
    ) in (
        df_matches_clean[
            [
                "radiant_team",
                "dire_team",
                "radiant_win"
            ]
        ].itertuples(
            index=False,
            name=None
        )
    ):

        for hero_id in radiant_team:
            radiant_games_counter[
                hero_id
            ] += 1

            if radiant_win == 1:
                radiant_wins_counter[
                    hero_id
                ] += 1

        for hero_id in dire_team:
            dire_games_counter[
                hero_id
            ] += 1

            if radiant_win == 0:
                dire_wins_counter[
                    hero_id
                ] += 1


    side_records = []

    for hero_id in full_hero_ids:

        r_games = (
            radiant_games_counter[
                hero_id
            ]
        )

        d_games = (
            dire_games_counter[
                hero_id
            ]
        )

        side_records.append({
            "hero_id":
                hero_id,

            "hero_name":
                hero_names.get(
                    hero_id,
                    f"Unknown ({hero_id})"
                ),

            "radiant_games":
                r_games,

            "radiant_win_rate":
                (
                    radiant_wins_counter[
                        hero_id
                    ]
                    / r_games
                    * 100

                    if r_games > 0
                    else np.nan
                ),

            "dire_games":
                d_games,

            "dire_win_rate":
                (
                    dire_wins_counter[
                        hero_id
                    ]
                    / d_games
                    * 100

                    if d_games > 0
                    else np.nan
                )
        })


    hero_side_eda = pd.DataFrame(
        side_records
    )

    show(
        hero_side_eda
        .sort_values(
            "radiant_games",
            ascending=False
        )
        .head(10)
    )

else:
    print(
        "Skipped because invalid radiant_win values are present."
    )


# ============================================================
# EDA 5: HERO PICK CONCENTRATION
# ============================================================

print("\n" + "=" * 70)
print("EDA 5: HERO PICK CONCENTRATION")
print("=" * 70)

plt.figure(
    figsize=(9, 5)
)

plt.hist(
    hero_eda[
        "pick_rate"
    ].dropna(),
    bins=20
)

plt.title(
    "Distribution of Hero Pick Rates"
)

plt.xlabel(
    "Pick Rate (%)"
)

plt.ylabel(
    "Number of Heroes"
)

plt.tight_layout()
plt.show()


# ============================================================
# EDA 6: HERO PAIR COMBINATIONS
# ============================================================

print("\n" + "=" * 70)
print("EDA 6: HERO PAIR COMBINATIONS")
print("=" * 70)

if invalid_target_after_cleaning.sum() == 0:

    pair_wins = defaultdict(
        lambda: {
            "games": 0,
            "wins": 0
        }
    )

    for (
        radiant_team,
        dire_team,
        radiant_win
    ) in (
        df_matches_clean[
            [
                "radiant_team",
                "dire_team",
                "radiant_win"
            ]
        ].itertuples(
            index=False,
            name=None
        )
    ):

        for pair in combinations(
            sorted(
                set(
                    radiant_team
                )
            ),
            2
        ):

            pair_wins[
                pair
            ]["games"] += 1

            if radiant_win == 1:
                pair_wins[
                    pair
                ]["wins"] += 1


        for pair in combinations(
            sorted(
                set(
                    dire_team
                )
            ),
            2
        ):

            pair_wins[
                pair
            ]["games"] += 1

            if radiant_win == 0:
                pair_wins[
                    pair
                ]["wins"] += 1


    MIN_PAIR_GAMES = 30

    pair_records = []

    for (
        h1,
        h2
    ), data in pair_wins.items():

        if (
            data[
                "games"
            ]
            >= MIN_PAIR_GAMES
        ):

            pair_records.append({
                "hero_1":
                    hero_names.get(
                        h1,
                        f"Unknown ({h1})"
                    ),

                "hero_2":
                    hero_names.get(
                        h2,
                        f"Unknown ({h2})"
                    ),

                "games":
                    data[
                        "games"
                    ],

                "wins":
                    data[
                        "wins"
                    ],

                "win_rate":
                    (
                        data[
                            "wins"
                        ]
                        /
                        data[
                            "games"
                        ]
                        * 100
                    )
            })


    pair_eda = pd.DataFrame(
        pair_records
    )

    if not pair_eda.empty:
        show(
            pair_eda
            .sort_values(
                "games",
                ascending=False
            )
            .head(20)
        )
    else:
        print(
            "No pair reached the minimum-game threshold."
        )

else:
    print(
        "Skipped because invalid radiant_win values are present."
    )


# ============================================================
# EDA 7: HERO COMPOSITION DIVERSITY
# ============================================================

print("\n" + "=" * 70)
print("EDA 7: HERO COMPOSITION DIVERSITY")
print("=" * 70)

unique_heroes_in_match = np.fromiter(
    (
        len(
            set(
                radiant_team
            )
            |
            set(
                dire_team
            )
        )

        for (
            radiant_team,
            dire_team
        ) in (
            df_matches_clean[
                [
                    "radiant_team",
                    "dire_team"
                ]
            ].itertuples(
                index=False,
                name=None
            )
        )
    ),

    dtype=np.int16,
    count=len(
        df_matches_clean
    )
)

print(
    pd.Series(
        unique_heroes_in_match,
        name="unique_heroes_in_match"
    ).describe()
)


# ============================================================
# EDA 8: COMMON EXACT 5-HERO COMPOSITIONS
# ============================================================
# This EDA uses only exact 5-hero teams.
# It does NOT delete non-5v5 rows from df_matches_clean.

print("\n" + "=" * 70)
print("EDA 8: COMMON EXACT 5-HERO COMPOSITIONS")
print("=" * 70)

df_5v5 = df_matches_clean[
    (
        df_matches_clean[
            "radiant_count"
        ] == 5
    )
    &
    (
        df_matches_clean[
            "dire_count"
        ] == 5
    )
].copy()

print(
    "Exact 5v5 matches used in this EDA:",
    len(df_5v5)
)

composition_counts = Counter()

for (
    radiant_team,
    dire_team
) in (
    df_5v5[
        [
            "radiant_team",
            "dire_team"
        ]
    ].itertuples(
        index=False,
        name=None
    )
):

    radiant_composition = tuple(
        sorted(
            radiant_team
        )
    )

    dire_composition = tuple(
        sorted(
            dire_team
        )
    )

    composition_counts[
        radiant_composition
    ] += 1

    composition_counts[
        dire_composition
    ] += 1


# Only convert the 100 most common compositions into a DataFrame.
# This avoids a large memory-heavy table when most compositions are unique.
top_compositions = (
    composition_counts
    .most_common(100)
)

composition_stats = pd.DataFrame(
    top_compositions,
    columns=[
        "composition",
        "match_count"
    ]
)

composition_stats[
    "hero_names"
] = (
    composition_stats[
        "composition"
    ].apply(
        lambda comp:
            ", ".join(
                hero_names.get(
                    hero_id,
                    f"Unknown ({hero_id})"
                )
                for hero_id in comp
            )
    )
)

show(
    composition_stats[
        [
            "hero_names",
            "match_count"
        ]
    ].head(10)
)

del composition_counts
gc.collect()


# ============================================================
# MODELING SECTION
# ============================================================
# The model is run only if the check-only issues in Steps 1-3
# are absent. This avoids silently changing the user's data rules.

if model_ready:

    # ========================================================
    # STEP 10: CREATE SEPARATE RADIANT AND DIRE HERO FEATURES
    # ========================================================
    #
    # The teams are NOT combined into one feature set.
    #
    # Radiant produces:
    #   radiant_hero_1, radiant_hero_2, ...
    #
    # Dire produces:
    #   dire_hero_1, dire_hero_2, ...
    #
    # sparse_output=True keeps memory usage low in Colab.

    print("\n" + "=" * 70)
    print("STEP 10: CREATE HERO FEATURES")
    print("=" * 70)

    model_heroes = sorted(
        {
            hero_id
            for team in df_matches_clean[
                "radiant_team"
            ]
            for hero_id in team
            if hero_id != 0
        }
        |
        {
            hero_id
            for team in df_matches_clean[
                "dire_team"
            ]
            for hero_id in team
            if hero_id != 0
        }
    )

    mlb = MultiLabelBinarizer(
        classes=model_heroes,
        sparse_output=True
    )

    radiant_matrix = (
        mlb.fit_transform(
            df_matches_clean[
                "radiant_team"
            ]
        )
    )

    # Use the same fitted hero classes for Dire.
    dire_matrix = (
        mlb.transform(
            df_matches_clean[
                "dire_team"
            ]
        )
    )

    X = hstack(
        [
            radiant_matrix,
            dire_matrix
        ],
        format="csr"
    )

    feature_names = (
        [
            f"radiant_hero_{hero_id}"
            for hero_id in model_heroes
        ]
        +
        [
            f"dire_hero_{hero_id}"
            for hero_id in model_heroes
        ]
    )

    y = (
        df_matches_clean[
            "radiant_win"
        ]
        .astype(np.int8)
        .to_numpy()
    )

    print(
        "Unique heroes used for modeling:",
        len(model_heroes)
    )

    print(
        "Number of model features:",
        X.shape[1]
    )

    print(
        "Feature matrix shape:",
        X.shape
    )

    print(
        "Non-zero feature values:",
        X.nnz
    )


    # ========================================================
    # EDA 9: HERO FEATURE SPARSITY
    # ========================================================

    print("\n" + "=" * 70)
    print("EDA 9: HERO FEATURE SPARSITY")
    print("=" * 70)

    feature_non_zero = np.asarray(
        X.sum(
            axis=0
        )
    ).ravel()

    feature_usage = pd.DataFrame({
        "feature":
            feature_names,

        "non_zero_count":
            feature_non_zero
    })

    feature_usage[
        "non_zero_percent"
    ] = (
        feature_usage[
            "non_zero_count"
        ]
        /
        len(
            df_matches_clean
        )
        * 100
    )

    show(
        feature_usage
        .sort_values(
            "non_zero_percent",
            ascending=False
        )
        .head(20)
    )


    # ========================================================
    # STEP 11: TARGET VARIABLE
    # ========================================================

    print("\n" + "=" * 70)
    print("STEP 11: TARGET VARIABLE")
    print("=" * 70)

    print(
        "1 = Radiant Win"
    )

    print(
        "0 = Dire Win"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        pd.Series(
            y
        ).value_counts()
    )


    # ========================================================
    # STEP 12: TRAIN / TEST SPLIT
    # ========================================================
    #
    # No match_time is available, so use a stratified random
    # 80/20 split rather than a chronological split.

    print("\n" + "=" * 70)
    print("STEP 12: TRAIN / TEST SPLIT")
    print("=" * 70)

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(
        "Training samples:",
        X_train.shape[0]
    )

    print(
        "Testing samples:",
        X_test.shape[0]
    )


    # ========================================================
    # STEP 13: BASELINE MODEL
    # ========================================================
    #
    # The baseline helps determine whether the ML models provide
    # useful predictive information beyond simple class frequency.

    baseline_model = DummyClassifier(
        strategy="prior",
        random_state=RANDOM_STATE
    )

    baseline_model.fit(
        X_train,
        y_train
    )

    y_pred_baseline = (
        baseline_model
        .predict(
            X_test
        )
    )

    y_prob_baseline = (
        baseline_model
        .predict_proba(
            X_test
        )[:, 1]
    )


    # ========================================================
    # STEP 14: LOGISTIC REGRESSION
    # ========================================================

    print(
        "\nTraining Logistic Regression..."
    )

    logistic_model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    logistic_model.fit(
        X_train,
        y_train
    )

    y_pred_lr = (
        logistic_model
        .predict(
            X_test
        )
    )

    y_prob_lr = (
        logistic_model
        .predict_proba(
            X_test
        )[:, 1]
    )

    print(
        "Logistic Regression completed."
    )


    # ========================================================
    # STEP 15: DECISION TREE
    # ========================================================

    print(
        "\nTraining Decision Tree..."
    )

    tree_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=10,
        random_state=RANDOM_STATE
    )

    tree_model.fit(
        X_train,
        y_train
    )

    y_pred_tree = (
        tree_model
        .predict(
            X_test
        )
    )

    y_prob_tree = (
        tree_model
        .predict_proba(
            X_test
        )[:, 1]
    )

    print(
        "Decision Tree completed."
    )


    # ========================================================
    # STEP 16: RANDOM FOREST
    # ========================================================
    #
    # The number/depth of trees is limited to reduce Colab RAM usage.

    print(
        "\nTraining Random Forest..."
    )

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=MODEL_N_JOBS
    )

    rf_model.fit(
        X_train,
        y_train
    )

    y_pred_rf = (
        rf_model
        .predict(
            X_test
        )
    )

    y_prob_rf = (
        rf_model
        .predict_proba(
            X_test
        )[:, 1]
    )

    print(
        "Random Forest completed."
    )


    # ========================================================
    # STEP 17: BERNOULLI NAIVE BAYES
    # ========================================================
    #
    # BernoulliNB is appropriate for binary 0/1 hero-presence
    # features and is computationally light for large sparse data.

    print(
        "\nTraining Bernoulli Naive Bayes..."
    )

    nb_model = BernoulliNB()

    nb_model.fit(
        X_train,
        y_train
    )

    y_pred_nb = (
        nb_model
        .predict(
            X_test
        )
    )

    y_prob_nb = (
        nb_model
        .predict_proba(
            X_test
        )[:, 1]
    )

    print(
        "Bernoulli Naive Bayes completed."
    )


    # ========================================================
    # STEP 18: SGD CLASSIFIER
    # ========================================================
    #
    # SGDClassifier with log-loss is a memory-efficient linear
    # classifier suitable for large sparse datasets.

    print(
        "\nTraining SGD Classifier..."
    )

    sgd_model = SGDClassifier(
        loss="log_loss",
        max_iter=1000,
        tol=1e-3,
        random_state=RANDOM_STATE
    )

    sgd_model.fit(
        X_train,
        y_train
    )

    y_pred_sgd = (
        sgd_model
        .predict(
            X_test
        )
    )

    y_prob_sgd = (
        sgd_model
        .predict_proba(
            X_test
        )[:, 1]
    )

    print(
        "SGD Classifier completed."
    )


    # ========================================================
    # STEP 19: EVALUATION FUNCTION
    # ========================================================

    def evaluate_model(
        model_name,
        y_true,
        y_pred,
        y_prob
    ):

        return {
            "Model":
                model_name,

            "Accuracy":
                accuracy_score(
                    y_true,
                    y_pred
                ),

            "Precision":
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

            "Recall":
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

            "F1 Score":
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

            "ROC-AUC":
                roc_auc_score(
                    y_true,
                    y_prob
                ),

            "Log Loss":
                log_loss(
                    y_true,
                    y_prob
                )
        }


    # ========================================================
    # STEP 20: COMPARE MODELS
    # ========================================================

    results = pd.DataFrame([
        evaluate_model(
            "Baseline",
            y_test,
            y_pred_baseline,
            y_prob_baseline
        ),

        evaluate_model(
            "Logistic Regression",
            y_test,
            y_pred_lr,
            y_prob_lr
        ),

        evaluate_model(
            "Decision Tree",
            y_test,
            y_pred_tree,
            y_prob_tree
        ),

        evaluate_model(
            "Random Forest",
            y_test,
            y_pred_rf,
            y_prob_rf
        ),

        evaluate_model(
            "Bernoulli Naive Bayes",
            y_test,
            y_pred_nb,
            y_prob_nb
        ),

        evaluate_model(
            "SGD Classifier",
            y_test,
            y_pred_sgd,
            y_prob_sgd
        )
    ])

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    show(
        results.round(4)
    )


    # ========================================================
    # STEP 21: CLASSIFICATION REPORTS
    # ========================================================

    for (
        model_name,
        prediction
    ) in [
        (
            "Logistic Regression",
            y_pred_lr
        ),
        (
            "Decision Tree",
            y_pred_tree
        ),
        (
            "Random Forest",
            y_pred_rf
        ),
        (
            "Bernoulli Naive Bayes",
            y_pred_nb
        ),
        (
            "SGD Classifier",
            y_pred_sgd
        )
    ]:

        print(
            "\n===",
            model_name,
            "==="
        )

        print(
            classification_report(
                y_test,
                prediction,
                target_names=[
                    "Dire Win",
                    "Radiant Win"
                ],
                zero_division=0
            )
        )


    # ========================================================
    # STEP 22: CONFUSION MATRICES
    # ========================================================

    for (
        model_name,
        prediction
    ) in [
        (
            "Logistic Regression",
            y_pred_lr
        ),
        (
            "Decision Tree",
            y_pred_tree
        ),
        (
            "Random Forest",
            y_pred_rf
        ),
        (
            "Bernoulli Naive Bayes",
            y_pred_nb
        ),
        (
            "SGD Classifier",
            y_pred_sgd
        )
    ]:

        cm = confusion_matrix(
            y_test,
            prediction
        )

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Dire Win",
                "Radiant Win"
            ]
        )

        disp.plot()

        plt.title(
            f"{model_name} Confusion Matrix"
        )

        plt.tight_layout()
        plt.show()


    # ========================================================
    # STEP 23: FEATURE INTERPRETATION
    # ========================================================

    def get_hero_name_from_feature(
        feature
    ):

        hero_id = int(
            feature
            .split("_")[-1]
        )

        return hero_names.get(
            hero_id,
            f"Unknown ({hero_id})"
        )


    # Logistic Regression uses coefficients.
    # It does NOT have feature_importances_.
    logistic_coefficients = pd.DataFrame({
        "feature":
            feature_names,

        "coefficient":
            logistic_model.coef_[0]
    })

    logistic_coefficients[
        "absolute_coefficient"
    ] = (
        logistic_coefficients[
            "coefficient"
        ].abs()
    )

    logistic_coefficients[
        "hero_name"
    ] = (
        logistic_coefficients[
            "feature"
        ].apply(
            get_hero_name_from_feature
        )
    )

    logistic_coefficients = (
        logistic_coefficients
        .sort_values(
            "absolute_coefficient",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\nTop Logistic Regression coefficients:"
    )

    show(
        logistic_coefficients[
            [
                "feature",
                "hero_name",
                "coefficient"
            ]
        ].head(20)
    )


    # Decision Tree feature importance.
    tree_importance = pd.DataFrame({
        "feature":
            feature_names,

        "importance":
            tree_model.feature_importances_
    })

    tree_importance[
        "hero_name"
    ] = (
        tree_importance[
            "feature"
        ].apply(
            get_hero_name_from_feature
        )
    )

    tree_importance = (
        tree_importance
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\nTop Decision Tree features:"
    )

    show(
        tree_importance[
            [
                "feature",
                "hero_name",
                "importance"
            ]
        ].head(20)
    )


    # Random Forest feature importance.
    rf_importance = pd.DataFrame({
        "feature":
            feature_names,

        "importance":
            rf_model.feature_importances_
    })

    rf_importance[
        "hero_name"
    ] = (
        rf_importance[
            "feature"
        ].apply(
            get_hero_name_from_feature
        )
    )

    rf_importance = (
        rf_importance
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\nTop Random Forest features:"
    )

    show(
        rf_importance[
            [
                "feature",
                "hero_name",
                "importance"
            ]
        ].head(20)
    )


    # SGD Classifier is also a linear model, so coefficients can
    # be inspected in the same way as Logistic Regression.
    sgd_coefficients = pd.DataFrame({
        "feature":
            feature_names,

        "coefficient":
            sgd_model.coef_[0]
    })

    sgd_coefficients[
        "absolute_coefficient"
    ] = (
        sgd_coefficients[
            "coefficient"
        ].abs()
    )

    sgd_coefficients[
        "hero_name"
    ] = (
        sgd_coefficients[
            "feature"
        ].apply(
            get_hero_name_from_feature
        )
    )

    sgd_coefficients = (
        sgd_coefficients
        .sort_values(
            "absolute_coefficient",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\nTop SGD Classifier coefficients:"
    )

    show(
        sgd_coefficients[
            [
                "feature",
                "hero_name",
                "coefficient"
            ]
        ].head(20)
    )

    print(
        "\nNote: Bernoulli Naive Bayes does not use "
        "tree-style feature_importances_. Its role in this "
        "project is primarily predictive comparison."
    )


    # ========================================================
    # STEP 24: TEST ONE NEW MATCH
    # ========================================================

    def create_match_features(
        radiant_team,
        dire_team
    ):
        """
        Convert one match into the same separate
        Radiant/Dire hero-feature format used during training.
        """

        radiant_new = (
            mlb.transform(
                [radiant_team]
            )
        )

        dire_new = (
            mlb.transform(
                [dire_team]
            )
        )

        return hstack(
            [
                radiant_new,
                dire_new
            ],
            format="csr"
        )


    # Example:
    #
    # radiant_team = [1, 2, 3, 4, 5]
    # dire_team = [6, 7, 8, 9, 10]
    #
    # new_match = create_match_features(
    #     radiant_team,
    #     dire_team
    # )
    #
    # prediction = rf_model.predict(
    #     new_match
    # )[0]
    #
    # probability = rf_model.predict_proba(
    #     new_match
    # )[0]
    #
    # print(
    #     "Predicted Winner:",
    #     "Radiant"
    #     if prediction == 1
    #     else "Dire"
    # )
    #
    # print(
    #     "Dire probability:",
    #     round(
    #         probability[0] * 100,
    #         2
    #     ),
    #     "%"
    # )
    #
    # print(
    #     "Radiant probability:",
    #     round(
    #         probability[1] * 100,
    #         2
    #     ),
    #     "%"
    # )


    # ========================================================
    # STEP 25: MODEL QUALITY INTERPRETATION
    # ========================================================

    baseline_row = (
        results[
            results[
                "Model"
            ] == "Baseline"
        ].iloc[0]
    )

    model_rows = (
        results[
            results[
                "Model"
            ] != "Baseline"
        ]
        .copy()
    )

    # Select only for reporting which tested model had
    # the highest held-out ROC-AUC.
    best_row = (
        model_rows
        .sort_values(
            "ROC-AUC",
            ascending=False
        )
        .iloc[0]
    )

    accuracy_gain = (
        best_row[
            "Accuracy"
        ]
        -
        baseline_row[
            "Accuracy"
        ]
    )

    auc = (
        best_row[
            "ROC-AUC"
        ]
    )

    print("\n" + "=" * 70)
    print("MODEL QUALITY INTERPRETATION")
    print("=" * 70)

    print(
        "Model with highest test ROC-AUC:",
        best_row[
            "Model"
        ]
    )

    print(
        "Accuracy:",
        round(
            best_row[
                "Accuracy"
            ],
            4
        )
    )

    print(
        "Baseline Accuracy:",
        round(
            baseline_row[
                "Accuracy"
            ],
            4
        )
    )

    print(
        "Accuracy improvement over baseline:",
        round(
            accuracy_gain,
            4
        )
    )

    print(
        "ROC-AUC:",
        round(
            auc,
            4
        )
    )

    print(
        "Log Loss:",
        round(
            best_row[
                "Log Loss"
            ],
            4
        )
    )

    print(
        "\nHOW TO INTERPRET THE RESULT:"
    )

    if (
        auc < 0.55
        and accuracy_gain < 0.02
    ):
        print(
            "The model shows weak predictive signal. "
            "Hero composition alone provides little improvement "
            "over the baseline in this dataset."
        )

    elif (
        auc < 0.65
        or accuracy_gain < 0.05
    ):
        print(
            "The model shows some predictive signal, but the "
            "improvement is modest. The result may support the "
            "research question, but it should not be described "
            "as highly accurate."
        )

    else:
        print(
            "The model shows a meaningful predictive signal "
            "relative to the baseline. Additional validation "
            "should still be performed before describing it "
            "as a strong general-purpose predictor."
        )

    print(
        "\nDo not judge the model using Accuracy alone. "
        "Also evaluate Precision, Recall, F1 Score, ROC-AUC, "
        "Log Loss, and the confusion matrices."
    )


    # ========================================================
    # STEP 26: SAVE MODEL RESULTS
    # ========================================================

    results.to_csv(
        RESULTS_FILE,
        index=False
    )

    hero_eda.to_csv(
        HERO_EDA_FILE,
        index=False
    )

    rf_importance.to_csv(
        RF_IMPORTANCE_FILE,
        index=False
    )

    print(
        "\nModel results saved to:",
        RESULTS_FILE
    )

    print(
        "Hero EDA saved to:",
        HERO_EDA_FILE
    )

    print(
        "Random Forest importance saved to:",
        RF_IMPORTANCE_FILE
    )


else:

    print("\n" + "=" * 70)
    print("MODELING NOT RUN")
    print("=" * 70)

    print(
        "The cleaning rules you requested only REMOVE "
        "[0, 0, 0, 0, 0] compositions."
    )

    print(
        "One or more check-only issues from Steps 1-3 are "
        "still present. They were reported but not deleted."
    )

    print(
        "Resolve those source-data issues before training "
        "the prediction models."
    )


# ============================================================
# FINAL RESEARCH IMPROVEMENT NOTES
# ============================================================

print("\n" + "=" * 70)
print("HOW TO IMPROVE THE RESEARCH")
print("=" * 70)

print(
    """
1. ADD PATCH AND MATCH-TIME DATA
   The current dataset does not contain patch or match_time.
   Dota 2 hero strength changes between patches. Adding these
   fields would allow chronological validation and analysis of
   model performance over time.

2. USE CROSS-VALIDATION AFTER THE BASIC PIPELINE WORKS
   A single 80/20 split is useful for the first experiment,
   but repeated or stratified cross-validation can show whether
   the result is stable.

3. TUNE THE MODELS USING TRAINING DATA ONLY
   Useful parameters include:
   - Logistic Regression: C
   - Decision Tree: max_depth, min_samples_leaf
   - Random Forest: n_estimators, max_depth,
     min_samples_leaf, max_features

4. ADD HERO INTERACTION FEATURES
   Individual hero indicators do not fully represent synergy
   or counters. Future work can add:
   - same-team hero pairs,
   - Radiant-vs-Dire hero matchup features,
   - role combinations.

5. ADD PLAYER AND MATCH CONTEXT
   If available, include player rank/MMR, roles, draft order,
   team identity, tournament context, or other pre-match data.

6. VALIDATE ON LATER DATA
   A stronger practical test is to train on earlier data and
   evaluate on matches from a later time period or later patch.

7. REPORT ASSOCIATION, NOT CAUSATION
   Feature coefficients/importances show relationships learned
   by the model. They do not prove that a particular hero causes
   a match to be won.
"""
)

print("\nPROJECT COMPLETE.")
