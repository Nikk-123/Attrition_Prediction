from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import shap  # For explanation

app = Flask(__name__)

# Load the trained model
with open('churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset used during training
customer_data = pd.read_csv('indian_customer_data.csv')

# Pre-load label encoders for categorical columns
label_encoders = {}
categorical_columns = ['Gender', 'City', 'State', 'Types of Products']  # Adjust based on your dataset
for col in categorical_columns:
    le = LabelEncoder()
    customer_data[col] = le.fit_transform(customer_data[col])
    label_encoders[col] = le

# Pre-load the scaler used for training
scaler = StandardScaler()
features = customer_data.drop(['Customer ID', 'Name', 'Attrition Risk'], axis=1)
scaler.fit(features)

@app.route('/')
def home():
    return render_template('index.html', customer_data=customer_data.to_dict(orient='records'), prediction_text=None, customer_details=None, explanation_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input Customer ID
        customer_id = int(request.form['customer_id'])

        # Fetch customer features by ID
        customer_row = customer_data[customer_data['Customer ID'] == customer_id]
        if customer_row.empty:
            return render_template(
                'index.html',
                customer_data=customer_data.to_dict(orient='records'),
                prediction_text='Customer ID not found!',
                customer_details=None,
                explanation_text=None
            )

        # Get customer details
        customer_details = customer_row.to_dict(orient='records')[0]

        # Drop non-feature columns
        customer_features = customer_row.drop(columns=['Customer ID', 'Name', 'Attrition Risk'])

        # Encode categorical columns
        for col in categorical_columns:
            encoder = label_encoders[col]
            customer_features[col] = customer_features[col].apply(
                lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1  # Handle unseen labels
            )

        # Scale the features
        customer_features_scaled = scaler.transform(customer_features)

        # Predict using the trained model
        probability = model.predict_proba(customer_features_scaled)[0][1]  # Probability of attrition
        status = "Likely to Stay" if probability < 0.5 else "Likely to Leave"
        likelihood = round(probability * 100, 2)

        # Explain the prediction using SHAP
        explainer = shap.LinearExplainer(model, features, feature_perturbation="interventional")
        shap_values = explainer.shap_values(customer_features_scaled)[0]  # Extract the SHAP values for class 0

        # Convert the SHAP values to a flat array for sorting
        shap_values_flat = shap_values.flatten()

        # Zip the feature names with their corresponding SHAP values and sort them by magnitude
        feature_importances = sorted(zip(features.columns, shap_values_flat), key=lambda x: -abs(x[1]))

        # Prepare explanation text
        explanation = "<br>".join([f"{feat}: {round(val, 2)}" for feat, val in feature_importances[:5]])

        # Return the prediction and customer details
        return render_template(
            'index.html',
            customer_data=customer_data.to_dict(orient='records'),
            prediction_text=f'{status} with {likelihood}% probability',
            customer_details=customer_details,
            explanation_text=explanation
        )

    except Exception as e:
        return render_template(
            'index.html',
            customer_data=customer_data.to_dict(orient='records'),
            prediction_text=f'Error: {str(e)}',
            customer_details=None,
            explanation_text=None
        )

@app.route('/predict', methods=['GET'])
def predict_get():
    customer_id = int(request.args.get('customer_id'))
    customer_row = customer_data[customer_data['Customer ID'] == customer_id]
    if customer_row.empty:
        return jsonify({'customer_details': {}, 'prediction_text': 'Customer ID not found!'})

    customer_details = customer_row.to_dict(orient='records')[0]
    customer_features = customer_row.drop(columns=['Customer ID', 'Name', 'Attrition Risk'])

    for col in categorical_columns:
        encoder = label_encoders[col]
        customer_features[col] = customer_features[col].apply(
            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
        )

    customer_features_scaled = scaler.transform(customer_features)
    probability = model.predict_proba(customer_features_scaled)[0][1]
    status = "Likely to Stay" if probability < 0.5 else "Likely to Leave"
    likelihood = round(probability * 100, 2)

    # Explain the prediction using SHAP
    explainer = shap.LinearExplainer(model, features, feature_perturbation="interventional")
    shap_values = explainer.shap_values(customer_features_scaled)[0]  # Extract the SHAP values for class 0

    # Convert the SHAP values to a flat array for sorting
    shap_values_flat = shap_values.flatten()

    # Zip the feature names with their corresponding SHAP values and sort them by magnitude
    feature_importances = sorted(zip(features.columns, shap_values_flat), key=lambda x: -abs(x[1]))

    # Prepare explanation text
    explanation = "<br>".join([f"{feat}: {round(val, 2)}" for feat, val in feature_importances[:5]])

    return jsonify({
        'customer_details': customer_details,
        'prediction_text': f'{status} with {likelihood}% probability',
        'explanation_text': explanation
    })

if __name__ == "__main__":
    app.run(debug=True)
