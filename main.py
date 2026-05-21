"""
============================================================
SISTEM REKOMENDASI MAKANAN KANTIN
WEIGHTED KNN + FILTER PER VENDOR + PER KATEGORI
============================================================

Alur:
  1. Load dataset
  2. Input user
  3. Hitung favorite profile
  4. Weighted Euclidean Distance
  5. Similarity bonus
  6. Rekomendasi PER VENDOR
  7. Di dalam vendor:
        - Karbo
        - Lauk
        - Sayur
        - Lainnya
        - Minuman
        - Standalone
  8. Evaluasi per vendor
============================================================
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 1. CONFIG
# ============================================================

FILE_PATH = "_kantin_filled.csv"

RASA_COLS = ['Manis', 'Pahit', 'Asin', 'Asam', 'Pedas']

KOLOM_VENDOR_ASLI = "Kode_Vendor"

K = 5

SLOT_CONFIG = {
    'Karbo': {'required': True},
    'Lauk': {'required': True},
    'Sayur': {'required': True},
    'Lainnya': {'required': False},
    'Minuman': {'required': False},
}

# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File tidak ditemukan: {file_path}"
        )

    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)

    elif ext in [".xlsx", ".xlsm"]:
        df = pd.read_excel(file_path, engine='openpyxl')

    elif ext == ".xls":
        df = pd.read_excel(file_path, engine='xlrd')

    else:
        raise ValueError("Format file tidak didukung")

    df.columns = df.columns.str.strip()

    # Rename vendor
    if KOLOM_VENDOR_ASLI in df.columns:
        df = df.rename(
            columns={KOLOM_VENDOR_ASLI: "Vendor"}
        )

    required_cols = [
        'Nama Menu',
        'Kategori',
        'Tipe_Makanan_Simplified',
        'Vendor'
    ] + RASA_COLS

    missing = [
        c for c in required_cols
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom tidak ditemukan:\n{missing}"
        )

    for col in RASA_COLS:
        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        ).fillna(0)

    print("=" * 60)
    print("[OK] DATASET LOADED")
    print("=" * 60)
    print(f"Total menu   : {len(df)}")
    print(f"Total vendor : {df['Vendor'].nunique()}")
    print()

    return df


# ============================================================
# 3. INPUT USER
# ============================================================

USER_INTENSITY = np.array(
    [2, 1, 2, 5, 3],
    dtype=float
)

USER_DESIRE = np.array(
    [2, 1, 4, 5, 2],
    dtype=float
)

USER_FAVORITES = [
    "Mie Kering",
    "Mie Bakso",
    "Nasi Gila"
]

FAVORITES_WEIGHT = 0.3


# ============================================================
# 4. FAVORITE PROFILE
# ============================================================

def compute_fav_profile(favorites, df):

    matched = df[
        df['Nama Menu'].isin(favorites)
    ]

    if matched.empty:
        print("[WARN] Tidak ada favorit cocok")
        return None

    fav_profile = (
        matched[RASA_COLS]
        .mean(axis=0)
        .values
    )

    print("=" * 60)
    print("FAVORITE PROFILE")
    print("=" * 60)

    for col, val in zip(RASA_COLS, fav_profile):
        print(f"{col:<10}: {val:.2f}")

    print()

    return fav_profile


# ============================================================
# 5. WEIGHTED EUCLIDEAN
# ============================================================

def weighted_euclidean(
    food_matrix,
    user_intensity,
    user_desire
):

    diff = food_matrix - user_intensity

    weighted_diff = (
        (diff ** 2) * user_desire
    )

    distances = np.sqrt(
        np.sum(weighted_diff, axis=1)
    )

    return distances


# ============================================================
# 6. SIMILARITY BONUS
# ============================================================

def similarity_bonus_score(
    food_matrix,
    fav_profile,
    weight
):

    if fav_profile is None:
        return np.zeros(len(food_matrix))

    diff = food_matrix - fav_profile

    distances = np.sqrt(
        np.sum(diff ** 2, axis=1)
    )

    bonus = weight * (
        1.0 / (1.0 + distances)
    )

    return bonus


# ============================================================
# 7. RECOMMEND FUNCTION
# ============================================================

def recommend_food(
    user_intensity,
    user_desire,
    fav_profile,
    df,
    favorites_weight=0.3,
    k=5
):

    if df.empty:
        return pd.DataFrame()

    food_matrix = (
        df[RASA_COLS]
        .values
        .astype(float)
    )

    base_dist = weighted_euclidean(
        food_matrix,
        user_intensity,
        user_desire
    )

    bonus = similarity_bonus_score(
        food_matrix,
        fav_profile,
        favorites_weight
    )

    result = df.copy()

    result['base_distance'] = base_dist
    result['similarity_bonus'] = bonus

    result['final_score'] = (
        base_dist - bonus
    )

    # Match %
    max_score = result['final_score'].max()
    min_score = result['final_score'].min()

    score_range = (
        max_score - min_score
        if max_score != min_score
        else 1
    )

    result['match_pct'] = (
        100 * (
            1 - (
                (result['final_score'] - min_score)
                / score_range
            )
        )
    ).round(1)

    return (
        result
        .sort_values('final_score')
        .head(k)
    )


# ============================================================
# 8. REKOMENDASI PER SLOT
# ============================================================

def recommend_component_slots(
    user_intensity,
    user_desire,
    fav_profile,
    df_vendor,
    favorites_weight,
    k
):
    """
    Rekomendasi:
      - Karbo
      - Lauk
      - Sayur
      - Lainnya
      - Minuman

    Khusus dalam 1 vendor saja.
    """

    component_df = df_vendor[
        df_vendor['Kategori'] == 'condiment'
    ]

    results = {}

    for slot, cfg in SLOT_CONFIG.items():

        slot_df = component_df[
            component_df['Tipe_Makanan_Simplified'] == slot
        ]

        if slot_df.empty:
            continue

        recs = recommend_food(
            user_intensity,
            user_desire,
            fav_profile,
            slot_df,
            favorites_weight,
            k
        )

        results[slot] = {
            'required': cfg['required'],
            'recommendations': recs
        }

    return results


# ============================================================
# 9. REKOMENDASI STANDALONE
# ============================================================

def recommend_standalone(
    user_intensity,
    user_desire,
    fav_profile,
    df_vendor,
    favorites_weight,
    k
):

    standalone_df = df_vendor[
        df_vendor['Kategori'] == 'standalone'
    ]

    return recommend_food(
        user_intensity,
        user_desire,
        fav_profile,
        standalone_df,
        favorites_weight,
        k
    )


# ============================================================
# 10. DISPLAY
# ============================================================

def print_separator(title=""):

    line = "=" * 60

    print("\n" + line)

    if title:
        print(title)
        print(line)


def display_slot_recommendations(slot_results):

    for slot, data in slot_results.items():

        status = (
            "WAJIB"
            if data['required']
            else "OPSIONAL"
        )

        print_separator(
            f"SLOT: {slot} [{status}]"
        )

        cols = [
            'Nama Menu',
            'base_distance',
            'similarity_bonus',
            'final_score',
            'match_pct'
        ]

        print(
            data['recommendations'][cols]
            .to_string(index=False)
        )


def display_standalone(recs):

    if recs.empty:
        print("Tidak ada menu standalone")
        return

    print_separator(
        "MENU STANDALONE"
    )

    cols = [
        'Nama Menu',
        'base_distance',
        'similarity_bonus',
        'final_score',
        'match_pct'
    ]

    print(
        recs[cols]
        .to_string(index=False)
    )


# ============================================================
# 11. EVALUATION
# ============================================================

def evaluate_recommender_discovery(
    df_full,
    user_intensity,
    user_desire,
    recommendations,
    vendor_name
):

    df_pool = df_full[
        df_full['Vendor'] == vendor_name
    ].copy()

    if df_pool.empty:
        return

    food_matrix = (
        df_pool[RASA_COLS]
        .values
        .astype(float)
    )

    actual_distances = weighted_euclidean(
        food_matrix,
        user_intensity,
        user_desire
    )

    df_pool['pure_dist'] = actual_distances

    threshold_count = max(
        1,
        int(len(df_pool) * 0.20)
    )

    top_relevant_names = (
        df_pool
        .nsmallest(threshold_count, 'pure_dist')
        ['Nama Menu']
        .tolist()
    )

    y_true = df_pool['Nama Menu'].apply(
        lambda x:
        1 if x in top_relevant_names else 0
    ).values

    recommended_names = (
        recommendations['Nama Menu']
        .tolist()
    )

    y_pred = df_pool['Nama Menu'].apply(
        lambda x:
        1 if x in recommended_names else 0
    ).values

    cm = confusion_matrix(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    print_separator(
        f"EVALUASI VENDOR: {vendor_name}"
    )

    print(f"Precision : {precision:.2f}")
    print(f"Recall    : {recall:.2f}")
    print(f"F1 Score  : {f1:.2f}")

    plt.figure(figsize=(4,3))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Greens',
        xticklabels=['Not Pred', 'Pred'],
        yticklabels=['Not Rel', 'Rel']
    )

    plt.title(f'Confusion Matrix - {vendor_name}')
    # plt.show()

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    plt.title(f'Confusion Matrix - {vendor_name}')

    # plt.show()

    plt.savefig(f"{output_dir}/confusion_matrix_{vendor_name}.png")
    plt.close()


# ============================================================
# 12. MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_dataset(FILE_PATH)

    # --------------------------------------------------------
    # USER INPUT
    # --------------------------------------------------------

    print_separator("INPUT USER")

    print(f"Intensity : {USER_INTENSITY}")
    print(f"Desire    : {USER_DESIRE}")
    print(f"Favorites : {USER_FAVORITES}")

    # --------------------------------------------------------
    # FAVORITE PROFILE
    # --------------------------------------------------------

    fav_profile = compute_fav_profile(
        USER_FAVORITES,
        df
    )

    # --------------------------------------------------------
    # LOOP PER VENDOR
    # --------------------------------------------------------

    list_vendor = sorted(
        df['Vendor'].unique()
    )

    for vendor in list_vendor:

        print_separator(
            f"VENDOR: {vendor}"
        )

        # Filter vendor
        df_vendor = df[
            df['Vendor'] == vendor
        ]

        # ----------------------------------------------------
        # COMPONENT / PRASMANAN
        # ----------------------------------------------------

        slot_results = recommend_component_slots(
            USER_INTENSITY,
            USER_DESIRE,
            fav_profile,
            df_vendor,
            FAVORITES_WEIGHT,
            K
        )

        display_slot_recommendations(
            slot_results
        )

        # ----------------------------------------------------
        # STANDALONE
        # ----------------------------------------------------

        standalone_recs = recommend_standalone(
            USER_INTENSITY,
            USER_DESIRE,
            fav_profile,
            df_vendor,
            FAVORITES_WEIGHT,
            K
        )

        display_standalone(
            standalone_recs
        )

        # ----------------------------------------------------
        # EVALUASI
        # ----------------------------------------------------

        # Gabungkan semua rekomendasi
        all_recs = []

        for slot, data in slot_results.items():
            all_recs.append(
                data['recommendations']
            )

        if not standalone_recs.empty:
            all_recs.append(
                standalone_recs
            )

        if len(all_recs) > 0:

            final_recs = pd.concat(
                all_recs
            ).drop_duplicates(
                subset=['Nama Menu']
            )

            evaluate_recommender_discovery(
                df,
                USER_INTENSITY,
                USER_DESIRE,
                final_recs,
                vendor
            )

    print_separator("SELESAI")