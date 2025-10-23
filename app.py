import pandas as pd
import joblib
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Allows your Wix site to call this API

MODEL_FILE = 'maize_model.joblib'

# 1. Load the trained model ONCE when the app starts
try:
    model = joblib.load(MODEL_FILE)
    print(f"--- Model '{MODEL_FILE}' loaded successfully ---")
except FileNotFoundError:
    print(f"---")
    print(f"--- ERROR: Model file '{MODEL_FILE}' not found. ---")
    print(f"--- Please run 'python train.py' first to create it. ---")
    print(f"---")
    model = None

@app.get("/health")
def health():
    """A simple check to see if the server is running."""
    return jsonify({"ok": True, "msg": "Ameer Kisan Prophet predictor running"})

@app.get("/predict_maize")
def predict_maize():
    """The main prediction endpoint."""
    if model is None:
        return jsonify({"error": "Model is not loaded. Please check server logs."}), 500

    try:
        # 2. Create a dataframe for the next 7 days
        future = model.make_future_dataframe(periods=7)

        # 3. Use the model to predict those 7 days
        forecast = model.predict(future)
        
        # 4. Get *only* the last 7 rows (our predictions)
        # 'yhat' is the predicted price (mid)
        # 'yhat_lower' and 'yhat_upper' are the low/high confidence bounds
        predictions_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)

        # 5. Format the output to match your desired JSON
        output = []
        for _, row in predictions_df.iterrows():
            # Ensure predictions don't go below zero
            low_price = max(0.0, round(row['yhat_lower'], 2))
            mid_price = max(0.0, round(row['yhat'], 2))
            high_price = max(0.0, round(row['yhat_upper'], 2))

            output.append({
                "date": row['ds'].strftime('%Y-%m-%d'),
                "low": low_price,
                "mid": mid_price,
                "high": high_price
            })

        return jsonify({"predictions": output})

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    # This runs the app locally on your computer
    app.run(host="0.0.0.0", port=3000, debug=True)