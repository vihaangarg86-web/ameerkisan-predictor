from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import timedelta

app = Flask(__name__)
CORS(app)

@app.get("/health")
def health():
    return jsonify({"ok": True, "msg": "Ameer Kisan predictor running"})

@app.post("/predict7")
def predict7():
    data = request.get_json() or {}
    rows = data.get("rates", [])
    if len(rows) < 30:
        return jsonify({"error": "Need at least 30 days of data"}), 400

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    def to_avg(rate_str):
        s = str(rate_str).replace("₹", "").replace(" ", "")
        parts = s.split("-")
        try:
            if len(parts) == 2:
                lo, hi = float(parts[0]), float(parts[1])
                return (lo + hi) / 2.0
            return float(parts[0])
        except Exception:
            return np.nan

    df["avg"] = df["rate"].apply(to_avg)
    df = df.dropna(subset=["avg"]).sort_values("date")
    df["t"] = (df["date"] - df["date"].min()).dt.days.astype(int)
    t = df["t"].values.reshape(-1, 1)
    y = df["avg"].values.reshape(-1, 1)
    X = np.hstack([t, np.ones_like(t)])
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    a = float(theta[0][0])
    b = float(theta[1][0])

    last_t = int(df["t"].max())
    last_date = pd.to_datetime(df["date"].max()).date()
    tail = df["avg"].tail(30)
    sd = float(tail.std()) if len(tail) >= 4 else float(df["avg"].std() or 0.5)
    band = max(0.5, sd * 0.75)

    out = []
    for i in range(1, 8):
        tt = last_t + i
        mid = a * tt + b
        low = max(0.0, mid - band)
        high = mid + band
        d = (last_date + timedelta(days=i)).strftime("%Y-%m-%d")
        out.append({"date": d, "low": round(low, 2), "mid": round(mid, 2), "high": round(high, 2)})

    return jsonify({"predictions": out})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
