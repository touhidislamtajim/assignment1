import gradio as gr
import numpy as np
import joblib

model = joblib.load("model.pkl")

def predict_water(ph, hardness, solids, chloramines, sulfate,
                  conductivity, organic_carbon, trihalomethanes, turbidity):
    
    data = np.array([[ph, hardness, solids, chloramines, sulfate,
                      conductivity, organic_carbon, trihalomethanes, turbidity]])
    
    prediction = model.predict(data)[0]
    return "Potable Water" if prediction == 1 else "Not Potable Water"

interface = gr.Interface(
    fn=predict_water,
    inputs=[gr.Number(label=feature) for feature in [
        "pH", "Hardness", "Solids", "Chloramines", "Sulfate",
        "Conductivity", "Organic Carbon", "Trihalomethanes", "Turbidity"]],
    outputs="text",
    title="Water Potability Prediction"
)

interface.launch()
