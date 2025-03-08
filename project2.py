import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Загрузка данных
data = pd.read_csv('telecom_users.csv')

# Предобработка данных
data = data.replace({'Yes': 1, 'No': 0})
data['SeniorCitizen'] = data['SeniorCitizen'].astype(int)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0)
data.fillna(0, inplace=True)

# Кодирование категориальных признаков
encoder = LabelEncoder()
categorical_features = ['gender', 'Dependents', 'Contract', 'PhoneService', 'InternetService', 'StreamingTV', 'StreamingMovies']
for col in categorical_features:
    data[col] = encoder.fit_transform(data[col].astype(str))  # Приводим к строковому типу и кодируем

# Определение важных признаков
features = ['gender', 'SeniorCitizen', 'Dependents', 'Contract', 'tenure', 'PhoneService', 
            'InternetService', 'StreamingTV', 'StreamingMovies', 'MonthlyCharges']

# Масштабирование данных
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(data[features]), columns=features)

# Перевод целевой переменной
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
y = data['Churn']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
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

    # Использование чекбоксов для выбора "Yes" или "No" значений
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
    'gender': [1 if gender == 'male' else 0],
    'SeniorCitizen': [1 if SeniorCitizen == 'Yes' else 0],
    'Dependents': [1 if Dependents == 'Yes' else 0],
    'Contract': [Contract_options.index(Contract)],
    'tenure': [tenure],
    'PhoneService': [1 if PhoneService == 'Yes' else 0],
    'InternetService': [InternetService_options.index(InternetService)],
    'StreamingTV': [StreamingTV_options.index(StreamingTV)],
    'StreamingMovies': [StreamingMovies_options.index(StreamingMovies)],
    'MonthlyCharges': [MonthlyCharges]
})

# Масштабирование данных
input_data_scaled = scaler.transform(input_data)

# Прогнозирование
prediction = model.predict(input_data_scaled)
prediction_prob = model.predict_proba(input_data_scaled)

# Проверяем размерность прогноза
if prediction_prob.shape[1] == 2:
    # Вероятность оттока
    probability_of_churn = prediction_prob[0][1]
else:
    probability_of_churn = None  # Если вероятности оттока нет, то присваиваем None

# Отображение результата
st.subheader("📌 Результат предсказания")
if prediction == 1:
    st.error("Этот клиент, вероятно, уйдёт.")
else:
    st.success("Этот клиент, вероятно, останется.")

# Если есть вероятность оттока
if probability_of_churn is not None:
    st.write(f'Вероятность оттока: {probability_of_churn * 100:.2f}%')
else:
    st.warning('Не удалось рассчитать вероятность оттока.')

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

# График: Доля ежемесячной оплаты и общей суммы оплаты
plt.figure(figsize=(6, 6))
sns.boxplot(data=data[['MonthlyCharges', 'TotalCharges']], palette=["skyblue", "orange"])
plt.title('Распределение оплаты клиентов')
plt.xlabel('Тип оплаты')
plt.ylabel('Сумма оплаты')
plt.xticks(ticks=[0, 1], labels=['Ежемесячная оплата', 'Общая сумма оплаты'])
st.pyplot(plt)

# Доля PhoneService и InternetService в одном гистаграмме
phone_internet_data = data[['PhoneService', 'InternetService']].apply(pd.Series.value_counts, normalize=True).T
plt.figure(figsize=(6, 4))
phone_internet_data.plot(kind='bar', stacked=True, color=['skyblue', 'orange'], edgecolor='black')
plt.title('Доля PhoneService и InternetService')
plt.xlabel('Тип услуги')
plt.ylabel('Доля')
# Легенда цветов
plt.legend(title="Тип услуги", labels=["PhoneService", "InternetService"], loc="upper right")
st.pyplot(plt)

# Contract – тип контракта клиента
plt.figure(figsize=(6, 4))
sns.countplot(x='Contract', data=data, hue='Contract', palette='coolwarm', legend=False)
plt.title('Тип контракта клиента')
plt.xlabel('Тип контракта')
plt.ylabel('Количество')
# Легенда цветов
plt.legend(title="Contract", labels=["Month-to-month", "One year", "Two year"], loc="upper right")
st.pyplot(plt)

# Оценка модели
st.subheader('Оценка модели')
st.write(f'Точность модели: {accuracy * 100:.2f}%')

#  Матрица путаницы
st.subheader('Матрица путаницы')
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Не отток", "Отток"], yticklabels=["Не отток", "Отток"])
plt.xlabel("Предсказанный")
plt.ylabel("Истинный")
st.pyplot(plt)

#  Детальная оценка
st.subheader('Детальная оценка модели')
st.text(classification_report(y_test, y_pred))

# Улучшенные параметры для графиков
sns.set_style("whitegrid")
plt.rcParams.update({'figure.figsize': (8, 5), 'axes.titlesize': 14, 'axes.labelsize': 12})

# Гистограмма распределения оттока клиентов
fig, ax = plt.subplots()
sns.countplot(x='Churn', data=data, palette='coolwarm', ax=ax)
ax.set_title('Распределение оттока клиентов')
ax.set_xlabel('Отток клиентов')
ax.set_ylabel('Количество')
ax.set_xticklabels(['Не отток (0)', 'Отток (1)'])
st.pyplot(fig)

# Гистограмма пенсионеров
fig, ax = plt.subplots()
sns.countplot(x='SeniorCitizen', data=data, palette='coolwarm', ax=ax)
ax.set_title('Доля пенсионеров и не пенсионеров')
ax.set_xlabel('Пенсионеры')
ax.set_ylabel('Количество')
ax.set_xticklabels(['Не пенсионер (0)', 'Пенсионер (1)'])
st.pyplot(fig)

# Гистограмма пола
fig, ax = plt.subplots()
sns.countplot(x='gender', data=data, palette='coolwarm', ax=ax)
ax.set_title('Распределение по полу')
ax.set_xlabel('Пол')
ax.set_ylabel('Количество')
ax.set_xticklabels(['Женский (0)', 'Мужской (1)'])
st.pyplot(fig)

# Boxplot оплаты
fig, ax = plt.subplots()
sns.boxplot(data=data[['MonthlyCharges', 'TotalCharges']], palette=["skyblue", "orange"], ax=ax)
ax.set_title('Распределение платежей')
ax.set_xticklabels(['Ежемесячная оплата', 'Общая сумма оплаты'])
ax.set_ylabel('Сумма оплаты')
st.pyplot(fig)

# Доля PhoneService и InternetService
fig, ax = plt.subplots()
phone_internet_data = data[['PhoneService', 'InternetService']].apply(pd.Series.value_counts, normalize=True).T
phone_internet_data.plot(kind='bar', stacked=True, color=['skyblue', 'orange'], edgecolor='black', ax=ax)
ax.set_title('Доля пользователей PhoneService и InternetService')
ax.set_xlabel('Тип услуги')
ax.set_ylabel('Доля')
st.pyplot(fig)

# Тип контракта
fig, ax = plt.subplots()
sns.countplot(x='Contract', data=data, palette='coolwarm', order=['Month-to-month', 'One year', 'Two year'], ax=ax)
ax.set_title('Распределение типов контрактов')
ax.set_xlabel('Тип контракта')
ax.set_ylabel('Количество')
st.pyplot(fig)

# Матрица путаницы
from sklearn.metrics import confusion_matrix
fig, ax = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Не отток", "Отток"], yticklabels=["Не отток", "Отток"], ax=ax)
ax.set_title('Матрица путаницы')
ax.set_xlabel("Предсказанный")
ax.set_ylabel("Истинный")
st.pyplot(fig)
