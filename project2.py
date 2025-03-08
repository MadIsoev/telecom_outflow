import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# Загрузка данных
data = pd.read_csv('telecom_users.csv')

# Предобработка данных
data = data.replace({'Yes': 1, 'No': 0})
data['SeniorCitizen'] = data['SeniorCitizen'].astype(int)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)
data.fillna(0, inplace=True)

# Определение важных признаков
categorical_features = ['gender', 'Dependents', 'Contract', 'PhoneService', 'InternetService', 'StreamingTV', 'StreamingMovies']
numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges']
features = categorical_features + numerical_features

# Масштабирование числовых данных
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Перевод целевой переменной
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
y = data['Churn']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(data[features], y, test_size=0.2, random_state=42)

# Обучение модели
model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='Logloss', verbose=0, cat_features=categorical_features)
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

# Масштабирование числовых данных
input_data[numerical_features] = scaler.transform(input_data[numerical_features])

# Прогнозирование
prediction = model.predict(input_data)

# Отображение результата
st.subheader("📌 Результат предсказания")
if prediction == 1:
    st.error("Этот клиент, вероятно, уйдёт.")
else:
    st.success("Этот клиент, вероятно, останется.")

# Матрица путаницы
st.subheader('Матрица путаницы')
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Не отток", "Отток"], yticklabels=["Не отток", "Отток"])
plt.xlabel("Предсказанный")
plt.ylabel("Истинный")
st.pyplot(plt)


# Обзор данных
st.subheader('Обзор данных')
st.write(data.head())

# Введенные данные
st.subheader('Введенные данные')
st.write(input_data)

# Churn – доля отток и не отток
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=data, palette='coolwarm', hue='Churn')
plt.title('Распределение оттока клиентов')
plt.xlabel('Отток клиентов')
plt.ylabel('Количество')
# Легенда цветов
plt.legend(title="Churn", labels=["Не отток (0)", "Отток (1)"], loc="upper right")
st.pyplot(plt)

# Доля пенсионеров и не пенсионеров
plt.figure(figsize=(6, 4))
sns.countplot(x='SeniorCitizen', data=data, hue='SeniorCitizen', palette='coolwarm', legend=False)
plt.title('Доля пенсионеров и не пенсионеров')
plt.xlabel('Пенсионеры')
plt.ylabel('Количество')
# Легенда цветов
plt.legend(title="SeniorCitizen", labels=["Не пенсионер (0)", "Пенсионер (1)"], loc="upper right")
st.pyplot(plt)

# Доля женских и мужских половых клиентов
plt.figure(figsize=(6, 4))
sns.countplot(x='gender', data=data, hue='gender', palette='coolwarm', legend=False)
plt.title('Доля женских и мужских половых клиентов')
plt.xlabel('Пол')
plt.ylabel('Количество')
# Легенда цветов
plt.legend(title="Gender", labels=["Женский (0)", "Мужской (1)"], loc="upper right")
st.pyplot(plt)

# Доля ежемесячной оплаты и общей суммы оплаты
plt.figure(figsize=(6, 6))
sns.boxplot(data=data[['MonthlyCharges', 'TotalCharges']], palette=["skyblue", "orange"])
plt.title('Распределение оплаты клиентов')
plt.xlabel('Тип оплаты')
plt.ylabel('Сумма оплаты')
plt.xticks(ticks=[0, 1], labels=['Ежемесячная оплата', 'Общая сумма оплаты'])
st.pyplot(plt)

# Доля PhoneService и InternetService 
phone_internet_data = data[['PhoneService', 'InternetService']].apply(pd.Series.value_counts, normalize=True).T
plt.figure(figsize=(6, 4))
phone_internet_data.plot(kind='bar', stacked=True, color=['skyblue', 'orange'], edgecolor='black')
plt.title('Доля PhoneService и InternetService')
plt.xlabel('Тип услуги')
plt.ylabel('Доля')
# Легенда цветов
plt.legend(title="Тип услуги", labels=["PhoneService", "InternetService"], loc="upper right")
st.pyplot(plt)

# Тип контракта клиента
plt.figure(figsize=(6, 4))
sns.countplot(x='Contract', data=data, hue='Contract', palette='coolwarm', legend=False)
plt.title('Тип контракта клиента')
plt.xlabel('Тип контракта')
plt.ylabel('Количество')
# Легенда цветов
plt.legend(title="Contract", labels=["Month-to-month", "One year", "Two year"], loc="upper right")
st.pyplot(plt)

# Оценка модели
#st.subheader('Оценка модели')
#st.write(f'Точность модели: {accuracy * 100:.2f}%')

# Матрица путаницы
st.subheader('Матрица путаницы')
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Не отток", "Отток"], yticklabels=["Не отток", "Отток"])
plt.xlabel("Предсказанный")
plt.ylabel("Истинный")
st.pyplot(plt)
