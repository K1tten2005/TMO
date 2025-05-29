import streamlit as st
import pickle
import numpy as np
import pandas as pd

with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Прогнозирование диабета")

st.markdown("""
Введите значения признаков пациента для предсказания вероятности наличия диабета. 
Вы также можете изменить порог вероятности, чтобы управлять чувствительностью предсказания.
""")

def user_input_features():
    Pregnancies = st.slider('Беременности', 0, 15, 1)
    Glucose = st.slider('Глюкоза', 40, 200, 120)
    BloodPressure = st.slider('Давление', 40, 120, 70)
    SkinThickness = st.slider('Толщина кожной складки', 10, 70, 20)
    Insulin = st.slider('Инсулин', 15, 300, 80)
    BMI = st.slider('ИМТ', 15.0, 50.0, 25.0)
    DiabetesPedigreeFunction = st.slider('Наследственный фактор', 0.0, 2.5, 0.5, 0.01)
    Age = st.slider('Возраст', 20, 90, 30)
    
    data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

input_scaled = scaler.transform(input_df)

threshold = st.slider("Порог вероятности для положительного класса", 0.0, 1.0, 0.5, 0.05)

probability = model.predict_proba(input_scaled)[0, 1]
prediction = int(probability >= threshold)

st.markdown(f"### Вероятность диабета: {probability:.2f}")
st.markdown(f"### Предсказание: {'Есть диабет' if prediction == 1 else 'Нет диабета'}")
