from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = pd.DataFrame(data)
    
    # Ensure that the input data has the correct columns and shape
    required_columns = [
        'base_price', 'total_price', 'diff', 'relative_diff_base', 
        'relative_diff_total', 'is_featured_sku', 'is_display_sku', 
        'store_encoded', 'sku_encoded', 'store_id', 'sku_id', 
        'month', 'end_month', 'year', 'weeknum', 'date', 
        'week_serial', 'end_date', 'weekday', 'end_weekday', 
        'end_weeknum', 'end_year', 'end_week_serial'
    ]
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in features.columns]
    if missing_columns:
        return jsonify({'error': f'Missing columns: {", ".join(missing_columns)}'}), 400

    # Predict
    predictions = model.predict(features)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
