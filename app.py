# Importing essential libraries
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

# Load the pre-trained LGBM model
filename = 'model.pkl'
model = joblib.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract data from form
        data = {
            'base_price': float(request.form['base_price']),
            'total_price': float(request.form['total_price']),
            'diff': float(request.form['diff']),
            'relative_diff_base': float(request.form['relative_diff_base']),
            'relative_diff_total': float(request.form['relative_diff_total']),
            'is_featured_sku': int(request.form['is_featured_sku']),
            'is_display_sku': int(request.form['is_display_sku']),
            'store_encoded': int(request.form['store_encoded']),
            'sku_encoded': int(request.form['sku_encoded']),
            'store_id': int(request.form['store_id']),
            'sku_id': int(request.form['sku_id']),
            'month': int(request.form['month']),
            'end_month': int(request.form['end_month']),
            'year': int(request.form['year']),
            'weeknum': int(request.form['weeknum']),
            'date': int(request.form['date']),
            'week_serial': float(request.form['week_serial']),
            'end_date': int(request.form['end_date']),
            'weekday': int(request.form['weekday']),
            'end_weekday': int(request.form['end_weekday']),
            'end_weeknum': int(request.form['end_weeknum']),
            'end_year': int(request.form['end_year']),
            'end_week_serial': float(request.form['end_week_serial'])
        }

        # Convert to DataFrame
        features = pd.DataFrame([data])
        
        # Predict
        prediction = model.predict(features)
        
        return render_template('result.html', prediction=prediction[0])

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if request.method == 'POST':
        data = request.get_json()

        # Convert JSON to DataFrame
        features = pd.DataFrame([data])

        # Predict
        prediction = model.predict(features)

        # Return prediction as JSON
        return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
