from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import uvicorn
import json
import requests  # Tambahkan ini
from sklearn.metrics import precision_score, recall_score, f1_score

# ============================================================
# KONFIGURASI SUPABASE
# ============================================================
SUPABASE_URL = "https://iwoiolguqbkyjssyifqr.supabase.co/rest/v1/food_ml"
# GANTI DENGAN ANON KEY ANDA (Bisa ditemukan di Dashboard Supabase > Settings > API)
SUPABASE_KEY = "MASUKKAN_SUPABASE_ANON_KEY_ANDA_DI_SINI"

# ============================================================
# FUNGSI PEMBERSIH DATA (Agar tidak error NumPy)
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
# 1. LOGIKA ML ASLI ANDA
# ============================================================
RASA_COLS = ['Manis', 'Pahit', 'Asin', 'Asam', 'Pedas']
SLOT_CONFIG = {
    'Karbo': {'required': True},
    'Lauk': {'required': True},
    'Sayur': {'required': True},
    'Lainnya': {'required': False},
    'Minuman': {'required': False},
}

def weighted_euclidean(food_matrix, user_intensity, user_desire):
    diff = food_matrix - user_intensity
    weighted_diff = (diff ** 2) * user_desire
    return np.sqrt(np.sum(weighted_diff, axis=1))

def similarity_bonus_score(food_matrix, fav_profile, weight):
    if fav_profile is None: return np.zeros(len(food_matrix))
    diff = food_matrix - fav_profile
    distances = np.sqrt(np.sum(diff ** 2, axis=1))
    return weight * (1.0 / (1.0 + distances))

def recommend_food(user_intensity, user_desire, fav_profile, df, favorites_weight=0.3, k=5):
    if df.empty: return pd.DataFrame()
    food_matrix = df[RASA_COLS].values.astype(float)
    base_dist = weighted_euclidean(food_matrix, user_intensity, user_desire)
    bonus = similarity_bonus_score(food_matrix, fav_profile, favorites_weight)
    
    result = df.copy()
    result['base_distance'] = base_dist
    result['similarity_bonus'] = bonus
    result['final_score'] = base_dist - bonus
    
    max_score, min_score = result['final_score'].max(), result['final_score'].min()
    score_range = max_score - min_score if max_score != min_score else 1
    result['match_pct'] = (100 * (1 - ((result['final_score'] - min_score) / score_range))).round(1)
    
    return result.sort_values('final_score').head(k)

def calculate_metrics(df_pool, user_intensity, user_desire, recommended_names):
    if df_pool.empty:
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
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 2),
        "total_menu_in_vendor": len(df_pool),
        "relevant_menu_count": threshold_count
    }

# ============================================================
# 2. FRAMEWORK FASTAPI
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fungsi untuk mengambil data dari Supabase
def fetch_data_from_supabase():
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}"
    }
    response = requests.get(f"{SUPABASE_URL}?select=*", headers=headers)
    if response.status_code != 200:
        raise Exception(f"Gagal mengambil data Supabase: {response.text}")
    
    data = response.json()
    df = pd.DataFrame(data)
    
    # Pembersihan kolom seperti logika awal Anda
    df.columns = df.columns.str.strip()
    if "Kode_Vendor" in df.columns:
        df = df.rename(columns={"Kode_Vendor": "Vendor"})
    
    return df

@app.get("/test-recommend")
async def test_recommend():
    # Tarik data terbaru dari Supabase setiap kali request (atau bisa ditaruh di luar jika ingin cache)
    try:
        df_global = fetch_data_from_supabase()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # DATA HARDCODE UNTUK TEST
    u_intensity = np.array([2, 1, 2, 5, 3], dtype=float)
    u_desire = np.array([2, 1, 4, 5, 2], dtype=float)
    u_favorites = ["Mie Kering", "Mie Bakso", "Nasi Gila"]
    k_val = 5
    
    try:
        matched = df_global[df_global['Nama Menu'].isin(u_favorites)]
        fav_profile = matched[RASA_COLS].mean(axis=0).values if not matched.empty else None
        
        list_vendor = sorted(df_global['Vendor'].unique())
        final_response = []

        for vendor in list_vendor:
            df_vendor = df_global[df_global['Vendor'] == vendor]
            all_vendor_recs_df = [] 

            # 1. Slot logic
            formatted_slots = {}
            component_df = df_vendor[df_vendor['Kategori'] == 'condiment']
            for slot, cfg in SLOT_CONFIG.items():
                slot_df = component_df[component_df['Tipe_Makanan_Simplified'] == slot]
                if not slot_df.empty:
                    recs = recommend_food(u_intensity, u_desire, fav_profile, slot_df, 0.3, k_val)
                    all_vendor_recs_df.append(recs)
                    formatted_slots[slot] = {
                        "required": cfg['required'],
                        "items": recs[['Nama Menu', 'match_pct']].to_dict(orient='records')
                    }

            # 2. Standalone logic
            standalone_df = df_vendor[df_vendor['Kategori'] == 'standalone']
            standalone_recs = recommend_food(u_intensity, u_desire, fav_profile, standalone_df, 0.3, k_val)
            if not standalone_recs.empty:
                all_vendor_recs_df.append(standalone_recs)
            
            # --- HITUNG EVALUASI UNTUK VENDOR INI ---
            if all_vendor_recs_df:
                combined_recs = pd.concat(all_vendor_recs_df).drop_duplicates(subset=['Nama Menu'])
                rec_names = combined_recs['Nama Menu'].tolist()
                eval_metrics = calculate_metrics(df_vendor, u_intensity, u_desire, rec_names)
            else:
                eval_metrics = {"precision": 0, "recall": 0, "f1": 0}

            final_response.append({
                "vendor": vendor,
                "evaluation": eval_metrics,
                "recommendations": {
                    "components": formatted_slots,
                    "standalone": standalone_recs[['Nama Menu', 'match_pct']].to_dict(orient='records') if not standalone_recs.empty else []
                }
            })

        return clean_for_json({
            "status": "success",
            "data": final_response
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)