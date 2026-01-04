from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and feature order
model = pickle.load(open('bigmart_model.pkl', 'rb'))
feature_order = pickle.load(open('feature_order.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # Build features dynamically using training order
        features = [float(data[col]) for col in feature_order]

        final_features = np.array(features).reshape(1, -1)

        # Debug (optional)
        print("Feature order:", feature_order)
        print("Feature shape:", final_features.shape)

        prediction = model.predict(final_features)

        return render_template(
            'index.html',
            prediction_text=f"Predicted Item Outlet Sales: ₹ {prediction[0]:.2f}"
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)