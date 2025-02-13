import streamlit as st
import pandas as pd
import pickle
import joblib  # Import joblib

# Load the trained models
def load_model(filename):
    with open(filename, 'rb') as file:
        return joblib.load(file)  # Use joblib.load

models = {
    "Decision Tree": load_model("Decision_Tree.pkl"),
    "Linear Regression": load_model("Linear_Regression.pkl"),
    "Random Forest": load_model("Random_Forest.pkl"),
    "Support Vector Regression": load_model("Support_Vector_Regression.pkl")
}


# Load the dataset
data = pd.read_csv("synthetic_game_data.csv")

st.title("Game Level Predictor App")

st.write("This app predicts the next game level and its level of difficulty using pre-trained AI models.")

# Display dataset preview
if st.checkbox("Show dataset preview"):
    st.write(data.head())

# User inputs for prediction
st.sidebar.header("Input Features")
last_level_attempts = st.sidebar.slider("Last Level Attempts", min_value=1, max_value=10, value=5)
last_level_cleared = st.sidebar.slider("Last Level Cleared", min_value=1, max_value=10, value=5)
difficulty = st.sidebar.slider("Difficulty", min_value=1, max_value=6, value=3)
level_completed = st.sidebar.radio("Level Completed", options=["Yes", "No"])
level_completed = 1 if level_completed == "Yes" else 0

# Select the model
model_name = st.selectbox("Select a model for prediction", list(models.keys()))

# Prepare input for prediction
input_features = [[last_level_attempts, last_level_cleared, difficulty, level_completed]]

# Make predictions
if st.button("Predict"):
    model = models[model_name]
    prediction = model.predict(input_features)
    st.write(f"**Prediction using {model_name}:**")
    st.write(f"- Next Level: {int(prediction[0])}")  # Assuming the next level is a whole number
    st.write(f"- Level of Difficulty: 1:Very easy, 2: Easy, 3: Medium, 4: Challenging, 5: Hard, 6: Very hard {round(prediction[0], 2)}")  # Example: treating the prediction as difficulty score

st.write("\n**Note:** Adjust the input features and select different models to compare predictions.")
