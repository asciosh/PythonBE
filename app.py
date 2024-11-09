from flask import Flask, request, jsonify
import joblib  # Per caricare il modello di ML

app = Flask(__name__)

# Carica il tuo modello
model_path = "model.pkl"
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Ottieni i dati dal corpo della richiesta
    data = request.json
    prediction = model.predict([data['input']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
