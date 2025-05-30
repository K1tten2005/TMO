import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Загрузка данных (или предобученных X и y)
@st.cache_data
def load_data():
    df = pd.read_csv('diabetes.csv')  # Убедись, что этот файл лежит рядом
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

X_scaled, y, scaler = load_data()

st.title("Прогнозирование диабета (Random Forest с настройкой)")

st.markdown("Измените гиперпараметр `n_estimators` для переобучения модели Random Forest")

# Гиперпараметр
n_estimators = st.slider('Количество деревьев (n_estimators)', 10, 200, 100, 10)

# Обучение модели
model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_scaled, y)

# Форма ввода
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
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
input_scaled = scaler.transform(input_df)

threshold = st.slider("Порог вероятности", 0.0, 1.0, 0.5, 0.05)

probability = model.predict_proba(input_scaled)[0, 1]
prediction = int(probability >= threshold)

st.markdown(f"### Вероятность диабета: {probability:.2f}")
st.markdown(f"### Предсказание: {'Есть диабет' if prediction == 1 else 'Нет диабета'}")
