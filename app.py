import gradio as gr
import numpy as np
import pandas as pd
from joblib import load
from pathlib import Path
import sys

MODEL_PATH = Path(__file__).with_name("model.pkl")

try:
    model = load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}", file=sys.stderr)
    model = None

def predict_water(ph, hardness, solids, chloramines, sulfate,
                  conductivity, organic_carbon, trihalomethanes, turbidity):
    
    if model is None:
        return "Error: Model not loaded."

    feature_names = [
        "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
        "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
    ]
    
    input_data = pd.DataFrame([[
        ph, hardness, solids, chloramines, sulfate,
        conductivity, organic_carbon, trihalomethanes, turbidity
    ]], columns=feature_names)

    try:
        prediction = model.predict(input_data)[0]
        return "Potable Water" if prediction == 1 else "Not Potable Water"
    except Exception as e:
        print(f"Prediction Error: {e}")
        return f"Prediction Error: {str(e)}"

interface = gr.Interface(
    fn=predict_water,
    inputs=[
        gr.Number(label="pH (0-14)", value=7.0),
        gr.Number(label="Hardness", value=196.3),
        gr.Number(label="Solids", value=21469.4),
        gr.Number(label="Chloramines", value=7.3),
        gr.Number(label="Sulfate", value=333.7),
        gr.Number(label="Conductivity", value=426.2),
        gr.Number(label="Organic Carbon", value=14.2),
        gr.Number(label="Trihalomethanes", value=66.3),
        gr.Number(label="Turbidity", value=3.9)
    ],
    outputs="text",
    title="Water Potability Prediction",
    description="Enter water quality metrics to determine if the water is potable."
)

if __name__ == "__main__":
    if model is None:
        sys.exit(1)
    interface.launch()