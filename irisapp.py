import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load accuracy
with open("accuracy.txt", "r") as f:
    accuracy = f.read()


st.title("🌸 Iris Species Predictor")
st.write(f"Model Accuracy on Test Set: **{accuracy}%**")  # Display accuracy

st.image("iris_flower.jpg", caption="Iris Flower", width=200)
# Input fields
splen = st.text_input("Enter sepal length (cm):")
spwid = st.text_input("Enter sepal width (cm):")
ptlen = st.text_input("Enter petal length (cm):")
ptwid = st.text_input("Enter petal width (cm):")

if st.button("Predict"):
    if splen and spwid and ptlen and ptwid:
        data = np.array([[float(splen), float(spwid), float(ptlen), float(ptwid)]])
        data_scaled = scaler.transform(data)
        pred = model.predict(data_scaled)
        st.success(f"The predicted species is: {pred[0]}")
    else:
        st.warning("Please fill all input fields.")
