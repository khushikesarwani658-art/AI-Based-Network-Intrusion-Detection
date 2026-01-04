import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate Sample Network Data

def generate_data(samples=500):
    data = {
        'Packet_Length_Mean': np.random.uniform(20, 1500, samples),
        'Flow_Duration': np.random.uniform(1, 10000, samples),
        'Protocol': np.random.randint(0, 3, samples),
        'Flag': np.random.randint(0, 2, samples),
        'Label': np.random.randint(0, 2, samples)
    }
    return pd.DataFrame(data)

# Train Machine Learning Model

def train_model(df):
    X = df.drop('Label', axis=1)
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc

# Streamlit UI

st.title("AI-Based Network Intrusion Detection System")

df = generate_data()
model, accuracy = train_model(df)

st.success(f"Model trained successfully with accuracy: {accuracy:.2f}")

st.subheader("Enter Network Traffic Parameters")

packet_length = st.slider("Packet Length Mean", 20.0, 1500.0, 500.0)
flow_duration = st.slider("Flow Duration", 1.0, 10000.0, 2000.0)
protocol = st.selectbox("Protocol Type", [0, 1, 2])
flag = st.selectbox("Flag", [0, 1])

if st.button("Check Traffic"):
    input_data = np.array([[packet_length, flow_duration, protocol, flag]])
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("Normal Traffic")
    else:
        st.error("Intrusion Detected")
