import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Regression': SVR()
}

# Streamlit app
st.title("AI-Powered Game Level Predictor")
st.markdown("""This app predicts the player's next level and its difficulty based on gameplay data using various AI models.""")

# Model selection
model_name = st.selectbox("Select an AI model", list(models.keys()))

# User input for prediction
st.subheader("Enter gameplay data:")
last_level_attempts = st.number_input("Last Level Attempts", min_value=0, value=5)
level_cleared = st.selectbox("Was the Latest Level Cleared?", ["No", "Yes"])
difficulty = st.number_input("Current Difficulty (1 to 6)", min_value=1, max_value=6, value=3)
level_completed = st.number_input("Levels Completed", min_value=0, value=1)

# Map input values
level_cleared_num = 1 if level_cleared == "Yes" else 0

difficulty_num = difficulty

# Prepare input data
input_data = np.array([[last_level_attempts, level_cleared_num, difficulty_num, level_completed]])

# Load trained models (dummy training here for demonstration)
X = np.random.rand(100, 4)  # Dummy features
y_next_level = np.random.randint(1, 11, 100)  # Dummy target for next level
y_next_difficulty = np.random.randint(1, 4, 100)  # Dummy target for next difficulty

selected_model = models[model_name]
selected_model.fit(X, y_next_level)  # Train the model for next level prediction
next_level_prediction = selected_model.predict(input_data)[0]

selected_model.fit(X, y_next_difficulty)  # Train the model for next difficulty prediction
next_difficulty_prediction = selected_model.predict(input_data)[0]

# Map numerical difficulty back to text
difficulty_reverse_mapping = {1: 'Very easy', 2: 'Easy', 3: 'Medium', 4: 'Challenging', 5: 'Hard', 6: 'Very hard'}
predicted_difficulty = difficulty_reverse_mapping.get(round(next_difficulty_prediction), 'unknown')

# Display predictions
st.subheader("Predicted Outcome:")
st.write(f"**Next Level:** {int(next_level_prediction)}")
st.write(f"**Next Difficulty:(1:Very easy, 2: Easy, 3: Medium, 4: Challenging, 5: Hard, 6: Very hard):**  {predicted_difficulty}")
