from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('model_lstm_emas.h5')

PRICE_MIN = 765.45
PRICE_MAX = 775.42

def inverse_minmax(scaled_value, min_, max_):
    return scaled_value * (max_ - min_) + min_

def predict_sequence(model, input_sequence, days_ahead):
    predictions = []
    current_input = input_sequence.copy()

    for _ in range(days_ahead):
        pred = model.predict(current_input, verbose=0)[0][0]
        predictions.append(pred)
        current_input = np.append(current_input[:, 1:, :], [[[pred]]], axis=1)

    return predictions

def convert_numpy_types(obj):
    """
    Fungsi rekursif untuk mengubah numpy types ke tipe Python standar
    agar bisa diserialisasi ke JSON tanpa error.
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        days = int(data['days_ahead'])

        df = pd.read_csv('data.csv')
        last_60 = df['Price'].values[-60:]

        if len(last_60) != 60:
            return jsonify({'error': 'Data historis kurang dari 60 entri.'}), 400

        scaled = [(p - PRICE_MIN) / (PRICE_MAX - PRICE_MIN) for p in last_60]
        input_seq = np.array(scaled).reshape(1, 60, 1)

        pred_scaled = predict_sequence(model, input_seq, days)
        pred_actual = [round(float(inverse_minmax(p, PRICE_MIN, PRICE_MAX)), 2) for p in pred_scaled]

        average_prediction = round(float(np.mean(pred_actual)), 2)

        response_data = {
            'prediction': pred_actual,
            'average': average_prediction
        }

        # Pastikan data bebas dari numpy types
        response_data = convert_numpy_types(response_data)

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
