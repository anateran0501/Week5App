import streamlit as st
import pandas as pd
import joblib  # Use joblib for loading models
import numpy as np  # For reshaping input

# Load the trained models
def load_model(filename):
    try:
        return joblib.load(filename)
    except Exception as e:
        st.error(f"Failed to load model {filename}: {e}")
        return None

models = {
    "Decision Tree": load_model("Decision_Tree.pkl"),
    "Linear Regression": load_model("Linear_Regression.pkl"),
    "Random Forest": load_model("Random_Forest.pkl"),
    "Support Vector Regression": load_model("Support_Vector_Regression.pkl")
}

# Remove models that failed to load
models = {name: model for name, model in models.items() if model is not None}

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
if models:
    model_name = st.selectbox("Select a model for prediction", list(models.keys()))
    
    # Prepare input for prediction based on the selected model
    if model_name in ["Decision Tree", "Random Forest"]:
        input_features = np.array([[last_level_attempts, last_level_cleared, difficulty]])  # These models expect 3 features
    else:
        input_features = np.array([[last_level_attempts, last_level_cleared, difficulty, level_completed]])  # These models expect 4 features
    
    # Make predictions
    if st.button("Predict"):
        model = models[model_name]
        try:
            prediction = model.predict(input_features)
            st.write(f"**Prediction using {model_name}:**")
            st.write(f"- Next Level: {int(prediction[0])}")  # Assuming the next level is a whole number
            st.write(f"- Level of Difficulty: 1: Very easy, 2: Easy, 3: Medium, 4: Challenging, 5: Hard, 6: Very hard")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
else:
    st.error("No models available for prediction. Please check your model files.")

st.write("\n**Note:** Adjust the input features and select different models to compare predictions.")
