import flask
from flask import Flask, render_template, request, jsonify
import pytest
import random
from datetime import datetime, timedelta
import sys
import os

from ml.random_forest.entry import run_prediction as rf_predict
from ml.linear_reg.entry import run_prediction as lr_predict

# Add ml directory to path for imports
#sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ml'))

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

    # Model names mapping
    model_names = {
        'lstm': 'LSTM Neural Network',
        'random_forest': 'Random Forest',
        'linear_regression': 'Linear Regression',
        'arima': 'ARIMA',
        'prophet': 'Prophet'
    }

    # Run the actual model
    if model == 'random_forest':

        #from random_forest.entry import run_prediction as rf_predict

        try:
            # Import the random forest entry point

            # Run prediction
            result = rf_predict(ticker, days_ahead=days)
            return jsonify(result)

        except Exception as e:
            return jsonify({'error': f'Random Forest prediction failed: {str(e)}'}), 500

    elif model == 'linear_regression':

        #from linear_reg.entry import run_prediction as lr_predict

        try:
            # Import the linear regression entry point

            # Run prediction
            result = lr_predict(ticker, days_ahead=days)
            return jsonify(result)

        except Exception as e:
            return jsonify({'error': f'Linear Regression prediction failed: {str(e)}'}), 500

    # STUB: Other models not yet implemented
    # For now, generating fake prediction data for other models
    else:
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
