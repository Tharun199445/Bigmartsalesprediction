from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained objects
model = pickle.load(open('bigmart_model.pkl', 'rb'))
feature_order = pickle.load(open('feature_order.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        features = []

        # Build input exactly like training
        for col in feature_order:
            value = data[col]

            # Try numeric conversion
            try:
                value = float(value)
            except:
                # Encode categorical values
                value = encoder.transform([value])[0]

            features.append(value)

        final_features = np.array(features).reshape(1, -1)

        # Debug (can remove later)
        print("Final features:", final_features)

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
