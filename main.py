# ice_cream_app_enhanced.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

st.set_page_config(page_title="Ice Cream Sales Predictor", layout="centered")
st.title("🍦 Ice Cream Sales Predictor (Enhanced)")
st.subheader("Predict sales based on temperature using Machine Learning")

# Upload dataset
st.sidebar.header("📂 Dataset Options")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file:
    df = load_data(uploaded_file)
else:
    df = load_data("Ice_cream selling data.csv")

# Show dataset
if st.checkbox("Show Dataset"):
    st.write(df)

# Feature & target
X = df.iloc[:, [0]]
y = df.iloc[:, [1]]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Sidebar polynomial degree
degree = st.sidebar.slider("Polynomial Degree", 1, 5, 2)

# Build pipeline
model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=degree),
    LinearRegression()
)

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"*Degree:* {degree}")
st.write(f"*MSE:* {mse:.2f}")
st.write(f"*RMSE:* {rmse:.2f}")
st.write(f"*R² Score:* {r2:.4f}")

# Visualization
st.subheader("📊 Actual vs Predicted")
fig, ax = plt.subplots()
ax.scatter(X_test, y_test, color='blue', label='Actual')
ax.scatter(X_test, y_pred, color='red', label='Predicted')
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Ice Cream Sales (units)")
ax.legend()
st.pyplot(fig)

# User prediction input
st.subheader("🧮 Predict Sales")
user_temp = st.number_input("Enter Temperature (°C)", value=25.0)
predicted_sales = model.predict(np.array([[user_temp]]))[0][0]
st.success(f"📦 Predicted Sales at {user_temp}°C: *{predicted_sales:.2f} units*")

# Option to export predictions
if st.button("Export Test Predictions to CSV"):
    prediction_df = pd.DataFrame({
        "Temperature (°C)": X_test.values.flatten(),
        "Actual Sales": y_test.values.flatten(),
        "Predicted Sales": y_pred.flatten()
    })
    csv = prediction_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "predictions.csv", "text/csv")

# Weather API Feature
st.subheader("🌤 Get Real-Time Prediction (Live Weather)")
city = st.text_input("Enter your city for live temperature", "")

if st.button("Fetch Weather & Predict"):
    if city:
        api_key = "8679ed120f0bec3837c6727737f6363b"  # 🔐 Replace with your key
        try:
            response = requests.get(
                f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            )
            data = response.json()
            temp_live = data['main']['temp']
            pred_live = model.predict(np.array([[temp_live]]))[0][0]
            st.info(f"🌡 Current Temp in {city}: {temp_live}°C")
            st.success(f"📈 Predicted Sales: *{pred_live:.2f} units*")
        except:
            st.error("Failed to fetch weather data. Check city name or API key.")
    else:
        st.warning("Please enter a city name.")

# Footer
st.markdown("---")
st.markdown("🔧 Built with Streamlit | ✨ Enhanced with interactivity and real-time weather integration")
