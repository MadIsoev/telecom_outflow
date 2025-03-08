import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Загрузка данных
data = pd.read_csv('telecom_users.csv')

# Предобработка данных
data = data.replace({'Yes': 1, 'No': 0})
data['SeniorCitizen'] = data['SeniorCitizen'].astype(int)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)
data.fillna(0, inplace=True)

# Выбор признаков
categorical_features = ['gender', 'Dependents', 'Contract', 'PhoneService', 'InternetService', 'StreamingTV', 'StreamingMovies']
numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges']

# Кодирование категориальных признаков
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)

# Разделение данных
X = data[categorical_features + numerical_features]
y = data['Churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

# Обучение модели
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Интерфейс Streamlit
st.set_page_config(page_title='Прогноз оттока клиентов', layout='wide')
st.title('📊 Прогнозирование оттока клиентов')
st.write('🔍 Анализ данных и предсказание оттока клиентов телекоммуникационной компании.')

# Боковая панель для ввода признаков
with st.sidebar:
    st.header("🔧 Введите признаки: ")

    gender = st.selectbox('Пол клиента', ['male', 'female'], index=0)
    SeniorCitizen = st.selectbox('Пенсионер?', ['Yes', 'No'], index=1)
    Dependents = st.selectbox('Есть ли иждивенцы?', ['Yes', 'No'], index=1)
    PhoneService = st.selectbox('Подключена ли услуга телефонной связи?', ['Yes', 'No'], index=0)

    Contract_options = ['Month-to-month', 'One year', 'Two year']
    Contract = st.selectbox('Тип контракта', Contract_options, index=0)

    tenure = st.slider('Длительность обслуживания (месяцы)', min_value=int(data['tenure'].min()), max_value=int(data['tenure'].max()), value=int(data['tenure'].mean()))

    InternetService_options = ['DSL', 'Fiber optic', 'No']
    InternetService = st.selectbox('Тип интернет-услуги', InternetService_options, index=0)

    StreamingTV_options = ['Yes', 'No', 'No internet service']
    StreamingTV = st.selectbox('Подключена ли услуга стримингового телевидения?', StreamingTV_options, index=0)

    StreamingMovies_options = ['Yes', 'No', 'No internet service']
    StreamingMovies = st.selectbox('Подключена ли услуга стримингового кинотеатра?', StreamingMovies_options, index=0)

    MonthlyCharges = st.slider('Ежемесячные платежи', min_value=float(data['MonthlyCharges'].min()), max_value=float(data['MonthlyCharges'].max()), value=float(data['MonthlyCharges'].mean()))

# Преобразование входных данных
input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [1 if SeniorCitizen == 'Yes' else 0],
    'Dependents': [Dependents],
    'Contract': [Contract],
    'tenure': [tenure],
    'PhoneService': [PhoneService],
    'InternetService': [InternetService],
    'StreamingTV': [StreamingTV],
    'StreamingMovies': [StreamingMovies],
    'MonthlyCharges': [MonthlyCharges]
})

# Прогнозирование
prediction = model.predict(input_data)
prediction_prob = model.predict_proba(input_data)

# Отображение результата
st.subheader("📌 Результат предсказания")
if prediction[0] == 1:
    st.error("Этот клиент, вероятно, уйдёт.")
else:
    st.success("Этот клиент, вероятно, останется.")

st.write(f'Вероятность оттока: {prediction_prob[0][1] * 100:.2f}%')

# Оценка модели
st.subheader('Оценка модели')
st.write(f'Точность модели: {accuracy * 100:.2f}%')

# Матрица путаницы
st.subheader('Матрица путаницы')
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Не отток", "Отток"], yticklabels=["Не отток", "Отток"])
plt.xlabel("Предсказанный")
plt.ylabel("Истинный")
st.pyplot(plt)

#  Детальная оценка
st.subheader('Детальная оценка модели')
st.text(classification_report(y_test, y_pred))
