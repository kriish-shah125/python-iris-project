import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('iris_model.pkl')

# GUI for user input
st.title("Iris Species Predictor")
st.write("Enter the feature values:")

sepal_length = st.number_input("Sepal Length", min_value=0.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0)
petal_length = st.number_input("Petal Length", min_value=0.0)
petal_width = st.number_input("Petal Width", min_value=0.0)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    st.write(f"The predicted species is: {prediction[0]}")