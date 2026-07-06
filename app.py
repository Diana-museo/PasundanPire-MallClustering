from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── paths ──────────────────────────────────────────────────────────
BASE   = os.path.dirname(__file__)
MODEL  = os.path.join(BASE, 'models', 'kmeans_model.pkl')
SCALER = os.path.join(BASE, 'models', 'scaler.pkl')
DATA_HASIL   = os.path.join(BASE, 'data', 'customer_clustering_hasil.csv')
DATA_EVAL    = os.path.join(BASE, 'data', 'hasil_evaluasi.csv')
DATA_TUNING  = os.path.join(BASE, 'data', 'hasil_tuning.csv')

# ── load model ─────────────────────────────────────────────────────
try:
    kmeans = joblib.load(MODEL)
    scaler = joblib.load(SCALER)
    MODEL_LOADED = True
except Exception as e:
    print(f"[WARN] Model tidak ditemukan: {e}")
    MODEL_LOADED = False

# ── cluster info ───────────────────────────────────────────────────
# Berdasarkan hasil evaluasi:
# Cluster 0: Usia 56.3 | Income 54.3 | Spending 49.1  → Middle-aged Moderate
# Cluster 1: Usia 26.8 | Income 57.1 | Spending 48.1  → Young Moderate
# Cluster 2: Usia 41.9 | Income 88.9 | Spending 17.0  → High Income Saver
# Cluster 3: Usia 32.7 | Income 86.5 | Spending 82.1  → VIP / Premium
# Cluster 4: Usia 25.0 | Income 25.3 | Spending 77.6  → Impulsive Spender
# Cluster 5: Usia 45.5 | Income 26.3 | Spending 19.4  → Budget Conscious

CLUSTER_INFO = {
    0: {
        "nama": "Mature Regular",
        "emoji": "🏬",
        "badge_color": "#6366f1",
        "deskripsi": "Pengunjung berusia matang dengan pendapatan dan pengeluaran seimbang. Pelanggan setia yang konsisten.",
        "karakteristik": ["Usia rata-rata 56 tahun", "Pendapatan menengah ~54k", "Spending seimbang ~49/100"],
        "rekomendasi": [
            "Perluas tenant kebutuhan sehari-hari dan apotek",
            "Program loyalitas & membership eksklusif",
            "Area duduk & fasilitas kenyamanan yang lebih banyak",
            "Tenant fashion mid-range dan aksesori rumah tangga"
        ]
    },
    1: {
        "nama": "Young Explorer",
        "emoji": "🎯",
        "badge_color": "#0ea5e9",
        "deskripsi": "Pengunjung muda dengan pendapatan cukup dan spending moderat. Potensi besar untuk ditingkatkan.",
        "karakteristik": ["Usia rata-rata 27 tahun", "Pendapatan menengah ~57k", "Spending moderat ~48/100"],
        "rekomendasi": [
            "Hadirkan tenant F&B kekinian dan food court yang instagramable",
            "Area hiburan: bioskop, arcade, dan zona gaming",
            "Brand fashion lokal dan internasional mid-range",
            "Event & pop-up store untuk engagement komunitas muda"
        ]
    },
    2: {
        "nama": "High Income Saver",
        "emoji": "💼",
        "badge_color": "#f59e0b",
        "deskripsi": "Pengunjung berpendapatan tinggi namun spending rendah. Pelanggan selektif yang menunggu nilai terbaik.",
        "karakteristik": ["Usia rata-rata 42 tahun", "Pendapatan tinggi ~89k", "Spending rendah ~17/100"],
        "rekomendasi": [
            "Kampanye promosi eksklusif & member-only deals",
            "Tenant premium dengan program cicilan atau reward points",
            "Pengalaman berbelanja premium: personal shopper, lounge VIP",
            "Komunikasi value proposition yang lebih kuat melalui digital"
        ]
    },
    3: {
        "nama": "VIP Premium",
        "emoji": "👑",
        "badge_color": "#10b981",
        "deskripsi": "Pelanggan terbaik mall. Pendapatan tinggi dan spending sangat tinggi. Aset terpenting bisnis.",
        "karakteristik": ["Usia rata-rata 33 tahun", "Pendapatan tinggi ~87k", "Spending tinggi ~82/100"],
        "rekomendasi": [
            "Prioritaskan tenant luxury dan brand internasional premium",
            "Bangun zona experience premium: spa, fine dining, butik mewah",
            "Program VIP membership dengan benefit eksklusif",
            "Customer service personal dan layanan concierge khusus"
        ]
    },
    4: {
        "nama": "Impulsive Spender",
        "emoji": "🛍️",
        "badge_color": "#ef4444",
        "deskripsi": "Pengunjung muda dengan pendapatan rendah tapi spending tinggi. Mudah tertarik penawaran dan tren.",
        "karakteristik": ["Usia rata-rata 25 tahun", "Pendapatan rendah ~25k", "Spending tinggi ~78/100"],
        "rekomendasi": [
            "Optimalkan window display & in-store promotion yang menarik",
            "Flash sale, limited offer, dan diskon time-limited",
            "Tenant fashion tren, accessories, dan lifestyle terjangkau",
            "Zona FOMO: produk edisi terbatas dan kolaborasi brand"
        ]
    },
    5: {
        "nama": "Budget Conscious",
        "emoji": "💰",
        "badge_color": "#8b5cf6",
        "deskripsi": "Pengunjung dengan pendapatan dan spending sama-sama rendah. Fokus pada kebutuhan dan nilai ekonomis.",
        "karakteristik": ["Usia rata-rata 46 tahun", "Pendapatan rendah ~26k", "Spending rendah ~19/100"],
        "rekomendasi": [
            "Perbanyak tenant pasar/supermarket dan kebutuhan pokok",
            "Program diskon reguler dan promo harga terjangkau",
            "Zona produk lokal & UMKM dengan harga kompetitif",
            "Layanan parkir terjangkau dan fasilitas dasar yang nyaman"
        ]
    }
}

# ── simulate prediction if model not loaded ────────────────────────
def simulate_cluster(age, income, spending):
    if income > 70 and spending > 60:
        return 3
    elif income > 70 and spending <= 60:
        return 2
    elif income <= 40 and spending > 60:
        return 4
    elif income <= 40 and spending <= 40:
        return 5
    elif age > 45:
        return 0
    else:
        return 1

# ── routes ─────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        gender  = int(data.get('gender', 0))
        age     = float(data.get('age', 30))
        income  = float(data.get('income', 60))
        spending= float(data.get('spending', 50))

        if MODEL_LOADED:
            X = np.array([[age, income, spending]])
            X_scaled = scaler.transform(X)
            cluster = int(kmeans.predict(X_scaled)[0])
        else:
            cluster = simulate_cluster(age, income, spending)

        info = CLUSTER_INFO[cluster]
        return jsonify({
            "success": True,
            "cluster": cluster,
            "nama": info["nama"],
            "emoji": info["emoji"],
            "badge_color": info["badge_color"],
            "deskripsi": info["deskripsi"],
            "karakteristik": info["karakteristik"],
            "rekomendasi": info["rekomendasi"],
            "model_loaded": MODEL_LOADED
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/evaluasi')
def evaluasi():
    try:
        df = pd.read_csv(DATA_EVAL)
        clusters = []
        for _, row in df.iterrows():
            c = int(row['Cluster'])
            clusters.append({
                "cluster": c,
                "nama": CLUSTER_INFO[c]["nama"],
                "emoji": CLUSTER_INFO[c]["emoji"],
                "badge_color": CLUSTER_INFO[c]["badge_color"],
                "jumlah": int(row['Jumlah_Pelanggan']),
                "usia": round(float(row['Rata_Rata_Usia']), 1),
                "income": round(float(row['Rata_Rata_Pendapatan']), 1),
                "spending": round(float(row['Rata_Rata_Spending']), 1),
                "silhouette": round(float(row['Silhouette_Score_Model']), 4)
            })
        return jsonify({"success": True, "clusters": clusters,
                        "k_optimal": 6, "silhouette_score": 0.4284})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/tuning')
def tuning():
    try:
        df = pd.read_csv(DATA_TUNING)
        records = df.to_dict(orient='records')
        best = df.loc[df['Silhouette'].idxmax()].to_dict()
        # Group by K for chart (best silhouette per K)
        chart = df.groupby('K')['Silhouette'].max().reset_index()
        chart_data = [{"k": int(r['K']), "silhouette": round(r['Silhouette'], 4)}
                      for _, r in chart.iterrows()]
        return jsonify({
            "success": True,
            "records": records,
            "best": {
                "k": int(best['K']),
                "init": best['Init'],
                "max_iter": int(best['Max_Iter']),
                "silhouette": round(float(best['Silhouette']), 4)
            },
            "chart_data": chart_data
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/distribusi')
def distribusi():
    try:
        df = pd.read_csv(DATA_HASIL)
        dist = df['Cluster'].value_counts().sort_index()
        result = [{"cluster": int(k), "nama": CLUSTER_INFO[int(k)]["nama"],
                   "emoji": CLUSTER_INFO[int(k)]["emoji"],
                   "color": CLUSTER_INFO[int(k)]["badge_color"],
                   "jumlah": int(v)} for k, v in dist.items()]
        return jsonify({"success": True, "distribusi": result, "total": len(df)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
