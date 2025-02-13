import streamlit as st
import joblib
import numpy as np

# Load trained models
models = {
    "Linear Regression": joblib.load("Linear_Regression.pkl"),
    "Decision Tree": joblib.load("Decision_Tree.pkl"),
    "Random Forest": joblib.load("Random_Forest.pkl"),
    "Support Vector Regression": joblib.load("SVR.pkl")
}

# Streamlit UI
st.title("AI-Powered Game Level Predictor")

# Select model
model_choice = st.selectbox("Select an AI model", list(models.keys()))

# User inputs
last_level_attempts = st.number_input("Last Level Attempts", min_value=1, max_value=10, step=1)
last_level_cleared = st.selectbox("Last Level Cleared", [0, 1])
difficulty = st.slider("Difficulty Level", min_value=1, max_value=6, step=1)
level_completed = st.number_input("Levels Completed", min_value=1, max_value=10, step=1)

# Prepare input data
input_data = np.array([[last_level_attempts, last_level_cleared, difficulty, level_completed]])

# Predict next level and difficulty
if st.button("Predict Next Level"):
    model = models[model_choice]
    prediction = model.predict(input_data)
    next_level = int(round(prediction[0]))
    next_difficulty = min(6, max(1, difficulty + (1 if next_level > level_completed else 0)))
    
    st.success(f"Predicted Next Level: {next_level}")
    st.success(f"Predicted Next Difficulty: {next_difficulty}")
