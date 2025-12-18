"""
Test script to verify both models are loading their respective .pkl files
and generating different predictions.
"""

import sys
import os

# Add ml directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("Testing Model Predictions")
print("="*60)

# Test Linear Regression
print("\n1. Testing LINEAR REGRESSION model...")
import linear_reg.entry
lr_result = linear_reg.entry.run_prediction('AAPL', days_ahead=3)

print(f"   Model: {lr_result['model_name']}")
print(f"   Current Price: ${lr_result['current_price']:.2f}")
print(f"   R² Score: {lr_result['r2_score']:.4f}")
print(f"   Using pretrained: {lr_result.get('using_pretrained', 'N/A')}")
print(f"   Predictions:")
for i, pred in enumerate(lr_result['predictions'][:3], 1):
    print(f"      Day {i}: ${pred['price']:.2f}")

# Test Random Forest
print("\n2. Testing RANDOM FOREST model...")
import random_forest.entry
rf_result = random_forest.entry.run_prediction('AAPL', days_ahead=3)

print(f"   Model: {rf_result['model_name']}")
print(f"   Current Price: ${rf_result['current_price']:.2f}")
print(f"   R² Score: {rf_result['r2_score']:.4f}")
print(f"   Using pretrained: {rf_result.get('using_pretrained', 'N/A')}")
print(f"   Predictions:")
for i, pred in enumerate(rf_result['predictions'][:3], 1):
    print(f"      Day {i}: ${pred['price']:.2f}")

# Compare predictions
print("\n" + "="*60)
print("COMPARISON:")
print("="*60)
print(f"Linear Regression Day 1: ${lr_result['predictions'][0]['price']:.2f}")
print(f"Random Forest Day 1:     ${rf_result['predictions'][0]['price']:.2f}")
print(f"Difference:              ${abs(lr_result['predictions'][0]['price'] - rf_result['predictions'][0]['price']):.2f}")

print(f"\nLinear Regression Day 2: ${lr_result['predictions'][1]['price']:.2f}")
print(f"Random Forest Day 2:     ${rf_result['predictions'][1]['price']:.2f}")
print(f"Difference:              ${abs(lr_result['predictions'][1]['price'] - rf_result['predictions'][1]['price']):.2f}")

print(f"\nLinear Regression Day 3: ${lr_result['predictions'][2]['price']:.2f}")
print(f"Random Forest Day 3:     ${rf_result['predictions'][2]['price']:.2f}")
print(f"Difference:              ${abs(lr_result['predictions'][2]['price'] - rf_result['predictions'][2]['price']):.2f}")

if lr_result['predictions'][0]['price'] == rf_result['predictions'][0]['price']:
    print("\n⚠️  WARNING: Predictions are IDENTICAL!")
    print("This suggests both models might be loading the same .pkl file.")
else:
    print("\n✓ Predictions are DIFFERENT - models are working correctly!")

print("="*60)
