from flask import Flask, request, jsonify
import joblib
import os
import numpy as np

app = Flask(__name__)

# Definisci il percorso per il modello salvato
MODEL_PATH = '/model.pkl'

# Carica il modello all'avvio dell'app
try:
    model = joblib.load(MODEL_PATH)
    print("Modello caricato con successo.")
except Exception as e:
    print(f"Errore nel caricare il modello: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ricevi i dati dalla richiesta POST
        data = request.get_json()  # Supponiamo che i dati siano in formato JSON
        
        # Esegui la previsione con il modello
        prediction = model.predict(np.array(data['features']).reshape(1, -1))
        
        # Ritorna la previsione come risposta JSON
        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
