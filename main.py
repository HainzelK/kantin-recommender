from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn
import requests
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Dict

# ============================================================
# KONFIGURASI SUPABASE
# ============================================================
SUPABASE_URL = "https://iwoiolguqbkyjssyifqr.supabase.co/rest/v1/food_ml"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml3b2lvbGd1cWJreWpzc3lpZnFyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzc2MTYxODIsImV4cCI6MjA5MzE5MjE4Mn0.Ufz1cWschQbKbdyG3VGOPb_c_4B4UzJfrGeUCBthiWA"

# ============================================================
# CONFIG
# ============================================================
RASA_COLS = ['Manis', 'Pahit', 'Asin', 'Asam', 'Pedas']

SLOT_CONFIG = {
    'Karbo': {'required': True},
    'Lauk': {'required': True},
    'Sayur': {'required': True},
    'Lainnya': {'required': False},
    'Minuman': {'required': False},
}

# ============================================================
# FASTAPI
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CLEAN JSON
# ============================================================
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# ============================================================
# ML CORE (TIDAK DIUBAH)
# ============================================================
def weighted_euclidean(food_matrix, user_intensity, user_desire):
    diff = food_matrix - user_intensity
    weighted_diff = (diff ** 2) * user_desire
    return np.sqrt(np.sum(weighted_diff, axis=1))

def similarity_bonus_score(food_matrix, fav_profile, weight):
    if fav_profile is None:
        return np.zeros(len(food_matrix))
    diff = food_matrix - fav_profile
    distances = np.sqrt(np.sum(diff ** 2, axis=1))
    return weight * (1.0 / (1.0 + distances))

def recommend_food(user_intensity, user_desire, fav_profile, df, favorites_weight=0.3, k=5):
    if df.empty:
        return pd.DataFrame()

    food_matrix = df[RASA_COLS].values.astype(float)

    base_dist = weighted_euclidean(food_matrix, user_intensity, user_desire)
    bonus = similarity_bonus_score(food_matrix, fav_profile, favorites_weight)

    result = df.copy()
    result['base_distance'] = base_dist
    result['similarity_bonus'] = bonus
    result['final_score'] = base_dist - bonus

    max_s, min_s = result['final_score'].max(), result['final_score'].min()
    score_range = max_s - min_s if max_s != min_s else 1
    result['match_pct'] = (100 * (1 - ((result['final_score'] - min_s) / score_range))).round(1)

    cols = ['Nama Menu', 'base_distance', 'similarity_bonus', 'final_score', 'match_pct']
    return result.sort_values('final_score').head(k)[cols]

def calculate_metrics(df_pool, user_intensity, user_desire, recommended_names):
    if df_pool.empty or not recommended_names:
        return {"precision": 0, "recall": 0, "f1": 0}

    food_matrix = df_pool[RASA_COLS].values.astype(float)
    actual_distances = weighted_euclidean(food_matrix, user_intensity, user_desire)
    df_pool = df_pool.copy()
    df_pool['pure_dist'] = actual_distances

    threshold_count = max(1, int(len(df_pool) * 0.20))
    top_relevant_names = df_pool.nsmallest(threshold_count, 'pure_dist')['Nama Menu'].tolist()

    y_true = df_pool['Nama Menu'].apply(lambda x: 1 if x in top_relevant_names else 0).values
    y_pred = df_pool['Nama Menu'].apply(lambda x: 1 if x in recommended_names else 0).values

    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 2),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 2),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 2)
    }

# ============================================================
# SUPABASE
# ============================================================
def fetch_data_from_supabase():
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }

    response = requests.get(f"{SUPABASE_URL}?select=*", headers=headers)

    if response.status_code != 200:
        raise Exception(response.text)

    df = pd.DataFrame(response.json())
    df.columns = df.columns.str.strip()

    if "Kode Vendor" in df.columns:
        df = df.rename(columns={"Kode Vendor": "Vendor"})

    return df

# ============================================================
# REQUEST SCHEMA (FIX YANG HILANG)
# ============================================================
class PreferenceBlock(BaseModel):
    desire: Dict[str, float]
    intensity: Dict[str, float]
    categories: List[str] = []

class PreferenceRequest(BaseModel):
    preferences: PreferenceBlock

# ============================================================
# GET TEST (TETAP ORIGINAL)
# ============================================================
@app.get("/test-recommend")
async def test_recommend():
    try:
        df_global = fetch_data_from_supabase()

        u_intensity = np.array([2, 1, 2, 5, 3], dtype=float)
        u_desire = np.array([2, 1, 4, 5, 2], dtype=float)
        u_favorites = ['Mie Kering', 'Mie Bakso', 'Nasi Gila']

        matched = df_global[df_global['Nama Menu'].isin(u_favorites)]
        fav_profile = matched[RASA_COLS].mean(axis=0).values if not matched.empty else None

        final_result = {
            "dataset_info": {
                "total_menu": len(df_global),
                "total_vendor": int(df_global['Vendor'].nunique())
            },
            "user_input": {
                "intensity": u_intensity.tolist(),
                "desire": u_desire.tolist(),
                "favorites": u_favorites
            },
            "favorite_profile": (
                {col: round(val, 2) for col, val in zip(RASA_COLS, fav_profile)}
                if fav_profile is not None else "None"
            ),
            "vendors": []
        }

        for vendor in sorted(df_global['Vendor'].unique()):
            df_vendor = df_global[df_global['Vendor'] == vendor]
            all_rec_names = []

            comp_results = {}
            comp_df = df_vendor[df_vendor['Kategori'].str.lower() == 'condiment']

            for slot in SLOT_CONFIG:
                slot_df = comp_df[comp_df['Tipe_Makanan_Simplified'] == slot]
                recs = recommend_food(u_intensity, u_desire, fav_profile, slot_df)

                if not recs.empty:
                    all_rec_names.extend(recs['Nama Menu'].tolist())
                    comp_results[slot] = recs.to_dict(orient='records')

            standalone_df = df_vendor[df_vendor['Kategori'].str.lower() == 'standalone']
            standalone_recs = recommend_food(u_intensity, u_desire, fav_profile, standalone_df)

            if not standalone_recs.empty:
                all_rec_names.extend(standalone_recs['Nama Menu'].tolist())

            metrics = calculate_metrics(
                df_vendor,
                u_intensity,
                u_desire,
                list(set(all_rec_names))
            )

            final_result["vendors"].append({
                "vendor_id": vendor,
                "slots": comp_results,
                "standalone": standalone_recs.to_dict(orient='records') if not standalone_recs.empty else [],
                "evaluation": metrics
            })

        return clean_for_json(final_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# POST (FIX UTAMA: INPUT JSON DIPAKAI KE ML)
# ============================================================
@app.post("/recommend")
async def recommend(req: PreferenceRequest):
    try:
        df_global = fetch_data_from_supabase()

        pref = req.preferences

        # ======================================================
        # INI FIX UTAMA: JSON → VECTOR (BENERAN DIPAKAI)
        # ======================================================
        u_intensity = np.array([
            pref.intensity.get("Manis", 0),
            pref.intensity.get("Pahit", 0),
            pref.intensity.get("Asin / Gurih", 0),
            pref.intensity.get("Asam / Segar", 0),
            pref.intensity.get("Pedas", 0),
        ], dtype=float)

        u_desire = np.array([
            pref.desire.get("Manis", 0),
            pref.desire.get("Pahit", 0),
            pref.desire.get("Asin / Gurih", 0),
            pref.desire.get("Asam / Segar", 0),
            pref.desire.get("Pedas", 0),
        ], dtype=float)

        u_favorites = []

        matched = df_global[df_global['Nama Menu'].isin(u_favorites)]
        fav_profile = matched[RASA_COLS].mean(axis=0).values if not matched.empty else None

        final_result = {
            "dataset_info": {
                "total_menu": len(df_global),
                "total_vendor": int(df_global['Vendor'].nunique())
            },
            "user_input": pref.dict(),
            "favorite_profile": "None",
            "vendors": []
        }

        for vendor in sorted(df_global['Vendor'].unique()):
            df_vendor = df_global[df_global['Vendor'] == vendor]
            all_rec_names = []

            comp_results = {}
            comp_df = df_vendor[df_vendor['Kategori'].str.lower() == 'condiment']

            for slot in SLOT_CONFIG:
                slot_df = comp_df[comp_df['Tipe_Makanan_Simplified'] == slot]
                recs = recommend_food(u_intensity, u_desire, fav_profile, slot_df)

                if not recs.empty:
                    all_rec_names.extend(recs['Nama Menu'].tolist())
                    comp_results[slot] = recs.to_dict(orient='records')

            standalone_df = df_vendor[df_vendor['Kategori'].str.lower() == 'standalone']
            standalone_recs = recommend_food(u_intensity, u_desire, fav_profile, standalone_df)

            if not standalone_recs.empty:
                all_rec_names.extend(standalone_recs['Nama Menu'].tolist())

            metrics = calculate_metrics(
                df_vendor,
                u_intensity,
                u_desire,
                list(set(all_rec_names))
            )

            final_result["vendors"].append({
                "vendor_id": vendor,
                "slots": comp_results,
                "standalone": standalone_recs.to_dict(orient='records') if not standalone_recs.empty else [],
                "evaluation": metrics
            })

        return clean_for_json(final_result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)