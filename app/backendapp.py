import flask
from flask import Flask, render_template, request, jsonify
import pytest
import random
from datetime import datetime, timedelta

from ml.entry_LSTM import predict as LSTM_predict

flaskapp = Flask(__name__)

@flaskapp.route('/index')
@flaskapp.route('/')
def index():
    return render_template('index.html')

@flaskapp.route("/info")
def info():
  return render_template('info.html')

@flaskapp.route("/predict", methods=['GET'])
def predict():
  return render_template('predict.html')

@flaskapp.route('/predict', methods=['POST'])
def predict_post():
    data = request.get_json()
    
    model = data.get('model')
    ticker = data.get('ticker')
    days = int(data.get('days', 7))
    
    # Validation
    if not model or not ticker:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    # STUB: This is where you'll run your actual ML model
    # For now, generating fake prediction data
    
    model_names = {
        'lstm': 'LSTM Neural Network',
        'random_forest': 'Random Forest',
        'linear_regression': 'Linear Regression',
        'arima': 'ARIMA',
        'prophet': 'Prophet'
    }

    if model == 'lstm':
        current, pred_values = LSTM_predict(ticker, days)
        print("/PREDICT APP TICKER", ticker)
        predictions = []
        start_date = datetime.now().date() + timedelta(days=1)

        for i, price in enumerate(pred_values):
            prediction_date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")

            predictions.append({
                "date": prediction_date,
                "price": round(float(price), 2)
            })

        result = {
            'ticker': ticker,
            'model_name': model_names.get(model, model),
            'current_price': current, # RANDOM AHH STUB
            'predictions': predictions,
            'confidence': random.randint(65, 95),
            'accuracy': random.randint(70, 90)
            }
        return result
    
    # Generate current price
    current_price = round(random.uniform(50, 500), 2)
    
    # Generate predictions for each day
    predictions = []
    price = current_price
    
    for i in range(1, days + 1):
        # Simulate daily price change (random walk with slight upward bias)
        daily_change = random.uniform(-0.03, 0.04)  # -3% to +4% daily change
        price = price * (1 + daily_change)
        
        prediction_date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
        
        predictions.append({
            'date': prediction_date,
            'price': round(price, 2)
        })
    
    result = {
        'ticker': ticker,
        'model_name': model_names.get(model, model),
        'current_price': current_price,
        'predictions': predictions,
        'confidence': random.randint(65, 95),
        'accuracy': random.randint(70, 90)
    }
    
    return jsonify(result)

if __name__ == "__main__":
  flaskapp.run(debug=True)
