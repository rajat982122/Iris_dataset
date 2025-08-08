from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load pickle files
scaler = pickle.load(open("scalar.pkl", "rb"))
model = pickle.load(open("regression.pkl", "rb"))

# Mapping from numeric prediction to species name
species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract features from request
        features = np.array([[ 
            float(data["sepal_length"]), 
            float(data["sepal_width"]), 
            float(data["petal_length"]), 
            float(data["petal_width"]) 
        ]])

        # Scale features
        scaled_features = scaler.transform(features)

        # Predict species
        prediction = model.predict(scaled_features)[0]
        species_name = species_map.get(prediction, "Unknown")

        return jsonify({"prediction": species_name})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
