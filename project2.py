import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Загрузка данных
data = pd.read_csv('telecom_users.csv')

# Предобработка данных
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)
data['SeniorCitizen'] = data['SeniorCitizen'].astype(int)
data.fillna(0, inplace=True)

# Кодирование бинарных признаков
binary_columns = ['Churn', 'PhoneService', 'Dependents']
for col in binary_columns:
    data[col] = data[col].map({'Yes': 1, 'No': 0})

# Кодирование категориальных признаков с OneHotEncoder
categorical_features = ['gender', 'Contract', 'InternetService', 'StreamingTV', 'StreamingMovies']
ohe = OneHotEncoder(drop='first', sparse=False)
categorical_encoded = pd.DataFrame(ohe.fit_transform(data[categorical_features]))
categorical_encoded.columns = ohe.get_feature_names_out(categorical_features)
data = data.drop(columns=categorical_features).reset_index(drop=True)
data = pd.concat([data, categorical_encoded], axis=1)

# Определение важных признаков
features = ['SeniorCitizen', 'Dependents', 'tenure', 'PhoneService', 'MonthlyCharges'] + list(categorical_encoded.columns)

# Масштабирование данных
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(data[features]), columns=features)
y = data['Churn']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Интерфейс Streamlit
st.set_page_config(page_title='Прогноз оттока клиентов', layout='wide')
st.title('📊 Прогнозирование оттока клиентов')

# Боковая панель
with st.sidebar:
    st.header("🔧 Введите признаки: ")
    SeniorCitizen = st.selectbox('Пенсионер?', ['Yes', 'No'], index=1)
    Dependents = st.selectbox('Есть ли иждивенцы?', ['Yes', 'No'], index=1)
    tenure = st.slider('Длительность обслуживания (месяцы)', min_value=int(data['tenure'].min()), max_value=int(data['tenure'].max()), value=int(data['tenure'].mean()))
    PhoneService = st.selectbox('Телефонная связь?', ['Yes', 'No'], index=0)
    MonthlyCharges = st.slider('Ежемесячные платежи', min_value=float(data['MonthlyCharges'].min()), max_value=float(data['MonthlyCharges'].max()), value=float(data['MonthlyCharges'].mean()))

# Преобразование входных данных
input_data = pd.DataFrame({
    'SeniorCitizen': [1 if SeniorCitizen == 'Yes' else 0],
    'Dependents': [1 if Dependents == 'Yes' else 0],
    'tenure': [tenure],
    'PhoneService': [1 if PhoneService == 'Yes' else 0],
    'MonthlyCharges': [MonthlyCharges]
})
input_data_scaled = scaler.transform(input_data)

# Прогнозирование
prediction = model.predict(input_data_scaled)
prediction_prob = model.predict_proba(input_data_scaled)
probability_of_churn = prediction_prob[0][1] if prediction_prob.shape[1] == 2 else None

# Отображение результата
st.subheader("📌 Результат предсказания")
st.write(f'Вероятность оттока: {probability_of_churn * 100:.2f}%')
if prediction == 1:
    st.error("Этот клиент, вероятно, уйдёт.")
else:
    st.success("Этот клиент, вероятно, останется.")

# Оценка модели
st.subheader('Оценка модели')
st.write(f'Точность модели: {accuracy * 100:.2f}%')
st.text(classification_report(y_test, y_pred))

# Матрица путаницы
st.subheader('Матрица путаницы')
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Не отток", "Отток"], yticklabels=["Не отток", "Отток"])
st.pyplot(plt)
