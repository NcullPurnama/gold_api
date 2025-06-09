from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import json
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
        pred_actual = [round(inverse_minmax(p, PRICE_MIN, PRICE_MAX), 2) for p in pred_scaled]

        average_prediction = round(float(np.mean(pred_actual)), 2)

        return jsonify({
            'prediction': pred_actual,
            'average': average_prediction
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
