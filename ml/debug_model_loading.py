"""
Debug script to trace model loading issue
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("DEBUGGING MODEL LOADING")
print("="*60)

# Patch the load_model function to add debug output
import model_loader
original_load_model = model_loader.load_model

def debug_load_model(ticker, model_type):
    print(f"\n>>> load_model() called with ticker='{ticker}', model_type='{model_type}'")
    result = original_load_model(ticker, model_type)
    print(f">>> Returning model for type: {model_type}")
    return result

model_loader.load_model = debug_load_model

print("\n1. Testing Random Forest predict function:")
print("-" * 60)
from random_forest.predict import predict_stock_for_api_with_pretrained
result = predict_stock_for_api_with_pretrained('AAPL', days_ahead=1)
print(f"\nResult model_name: {result['model_name']}")
print(f"Result R²: {result['r2_score']}")
