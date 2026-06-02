from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import uvicorn
import requests
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Dict, Any

# ============================================================
# KONFIGURASI
# ============================================================
SUPABASE_URL = "https://iwoiolguqbkyjssyifqr.supabase.co/rest/v1/food_ml"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml3b2lvbGd1cWJreWpzc3lpZnFyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Nzc2MTYxODIsImV4cCI6MjA5MzE5MjE4Mn0.Ufz1cWschQbKbdyG3VGOPb_c_4B4UzJfrGeUCBthiWA"
EXTERNAL_PREF_URL = "https://api.moodbites.qzz.io/api/v1/external"

RASA_COLS = ['Manis', 'Pahit', 'Asin', 'Asam', 'Pedas']
SLOT_CONFIG = ['Karbo', 'Lauk', 'Sayur', 'Lainnya', 'Minuman']
K_DEFAULT = 5  # Top 5 Rekomendasi

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# UTILS
# ============================================================
def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# ============================================================
# ML CORE
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

def recommend_items(user_intensity, user_desire, fav_profile, df, favorites_weight=0.3, k=5):
    if df.empty: return pd.DataFrame()
    
    food_matrix = df[RASA_COLS].values.astype(float)
    base_dist = weighted_euclidean(food_matrix, user_intensity, user_desire)
    bonus = similarity_bonus_score(food_matrix, fav_profile, favorites_weight)

    result = df.copy()
    result['final_score'] = base_dist - bonus

    max_s, min_s = result['final_score'].max(), result['final_score'].min()
    score_range = max_s - min_s if max_s != min_s else 1
    result['match_pct'] = (100 * (1 - ((result['final_score'] - min_s) / score_range))).round(1)

    return result.sort_values('final_score').head(k)

# --- FUNGSI EVALUASI FLEKSIBEL ---
def calculate_metrics(df_pool, user_intensity, user_desire, recommended_names):
    if df_pool.empty or not recommended_names:
        return {"precision": 0, "recall": 0, "f1_score": 0}

    # 1. Hitung Jarak Murni (Ground Truth)
    food_matrix = df_pool[RASA_COLS].values.astype(float)
    actual_distances = weighted_euclidean(food_matrix, user_intensity, user_desire)
    df_temp = df_pool.copy()
    df_temp['pure_dist'] = actual_distances

    # 2. Threshold Dinamis: 
    # Mengambil mana yang lebih besar antara 25% dari total menu atau jumlah yang direkomendasikan
    n_total = len(df_pool)
    n_rec = len(recommended_names)
    threshold_count = max(min(n_rec, n_total), int(n_total * 0.25))
    threshold_count = max(1, threshold_count)

    top_relevant_names = df_temp.nsmallest(threshold_count, 'pure_dist')['Nama Menu'].tolist()

    # 3. Hitung Skor
    y_true = df_temp['Nama Menu'].apply(lambda x: 1 if x in top_relevant_names else 0).values
    y_pred = df_temp['Nama Menu'].apply(lambda x: 1 if x in recommended_names else 0).values

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2)
    }

# ============================================================
# DATA FETCHERS
# ============================================================
def fetch_data_from_supabase():
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    response = requests.get(f"{SUPABASE_URL}?select=*", headers=headers)
    if response.status_code != 200: raise Exception(f"Supabase Error: {response.text}")
    df = pd.DataFrame(response.json())
    if "Kode Vendor" in df.columns: df = df.rename(columns={"Kode Vendor": "Vendor"})
    return df

def fetch_external_preferences(mood: str, user_id: str):
    url = f"{EXTERNAL_PREF_URL}/{mood}/{user_id}"
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=404, detail="Gagal menarik data dari Moodbites API")
    return response.json()

# ============================================================
# ENDPOINT
# ============================================================
@app.get("/recommend-external/{mood}/{user_id}")
async def recommend_external(mood: str, user_id: str):
    try:
        # 1. Tarik Data
        full_data = fetch_external_preferences(mood, user_id)
        df_global = fetch_data_from_supabase()

        # 2. Extract Preferences
        pref_data = full_data.get("preferences", {})
        int_map = pref_data.get("intensity", {})
        des_map = pref_data.get("desire", {})

        def get_v(d, keys):
            for k in keys:
                if k in d: return d[k]
            return 0

        u_intensity = np.array([
            get_v(int_map, ["Manis"]),
            get_v(int_map, ["Pahit"]),
            get_v(int_map, ["Asin / Gurih", "Asin", "Gurih"]),
            get_v(int_map, ["Asam / Segar", "Asam", "Segar"]),
            get_v(int_map, ["Pedas"])
        ], dtype=float)

        u_desire = np.array([
            get_v(des_map, ["Manis"]),
            get_v(des_map, ["Pahit"]),
            get_v(des_map, ["Asin / Gurih", "Asin", "Gurih"]),
            get_v(des_map, ["Asam / Segar", "Asam", "Segar"]),
            get_v(des_map, ["Pedas"])
        ], dtype=float)

        # 3. Favorite Profile
        u_favorites = pref_data.get("categories", [])
        matched = df_global[df_global['Nama Menu'].isin(u_favorites)]
        fav_profile = matched[RASA_COLS].mean(axis=0).values if not matched.empty else None

        final_result = {
            "metadata": {
                "mood": mood, 
                "user_id": user_id, 
                "k_limit": K_DEFAULT,
                "user_intensity": u_intensity,
                "user_desire": u_desire
            },
            "vendors": []
        }

        # 4. Filter Per Vendor
        for vendor in sorted(df_global['Vendor'].unique()):
            df_vendor = df_global[df_global['Vendor'] == vendor]
            all_rec_names = []

            # --- CONDIMENTS ---
            cond_results = {}
            df_cond = df_vendor[df_vendor['Kategori'].str.upper() == 'CONDIMENT']
            
            for slot in SLOT_CONFIG:
                mask = df_cond['Tipe_Makanan_Simplified'].str.upper() == slot.upper()
                slot_df = df_cond[mask]
                
                if not slot_df.empty:
                    recs = recommend_items(u_intensity, u_desire, fav_profile, slot_df, k=K_DEFAULT)
                    all_rec_names.extend(recs['Nama Menu'].tolist())
                    cond_results[slot] = recs.to_dict(orient='records')
                else:
                    cond_results[slot] = []

            # --- STANDALONE ---
            df_stand = df_vendor[df_vendor['Kategori'].str.upper() == 'STANDALONE']
            stand_recs = []
            if not df_stand.empty:
                recs_st = recommend_items(u_intensity, u_desire, fav_profile, df_stand, k=K_DEFAULT)
                all_rec_names.extend(recs_st['Nama Menu'].tolist())
                stand_recs = recs_st.to_dict(orient='records')

            # --- EVALUASI ---
            unique_rec_names = list(set(all_rec_names))
            metrics = calculate_metrics(df_vendor, u_intensity, u_desire, unique_rec_names)

            final_result["vendors"].append({
                "vendor_id": vendor,
                "condiment_categories": cond_results,
                "standalone_menus": stand_recs,
                "evaluation": metrics
            })

        return clean_for_json(final_result)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)