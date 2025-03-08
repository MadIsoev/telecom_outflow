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
st.subheader('Результат предсказания')
if prediction == 1:
    st.write("Клиент вероятно уйдет (отток).")
else:
    st.write("Клиент не уйдет (не будет оттока).")

# Если есть вероятность оттока
if probability_of_churn is not None:
    st.write(f'Вероятность оттока: {probability_of_churn:.2f}')
else:
    st.warning('Не удалось рассчитать вероятность оттока.')

# Обзор данных
st.subheader('Обзор данных')
st.write(data.head())

# Введенные данные
st.subheader('Введенные данные')
st.write(input_data)

# 2) Диаграмма: Churn – доля отток и не отток
st.subheader('Доля оттока и не оттока')
churn_data = data['Churn'].value_counts(normalize=True)
fig, ax = plt.subplots()
sns.barplot(x=churn_data.index, y=churn_data.values, ax=ax, palette="Blues_d")
ax.set_ylabel('Доля клиентов')
ax.set_xlabel('Отток (Churn)')
ax.set_title('Доля оттока и не оттока')
st.pyplot(fig)

# 3) Диаграмма1: Доля пенсионеров или не пенсионеров
st.subheader('Доля пенсионеров и не пенсионеров')
senior_data = data['SeniorCitizen'].value_counts(normalize=True)
fig, ax = plt.subplots()
sns.barplot(x=senior_data.index, y=senior_data.values, ax=ax, palette="Blues_d")
ax.set_ylabel('Доля клиентов')
ax.set_xlabel('Пенсионер')
ax.set_title('Доля пенсионеров и не пенсионеров')
st.pyplot(fig)

# 4) Диаграмма2: Доля женских и мужских половых клиентов
st.subheader('Доля женских и мужских половых клиентов')
gender_data = data['gender'].value_counts(normalize=True)
fig, ax = plt.subplots()
sns.barplot(x=gender_data.index, y=gender_data.values, ax=ax, palette="Blues_d")
ax.set_ylabel('Доля клиентов')
ax.set_xlabel('Пол клиента')
ax.set_title('Доля женских и мужских половых клиентов')
st.pyplot(fig)

# 5) График: оплата клиента за месяц и период
st.subheader('Оплата клиента за месяц и период')
fig, ax = plt.subplots()
sns.boxplot(x='SeniorCitizen', y='MonthlyCharges', data=data, ax=ax)
ax.set_title('Оплата клиента за месяц и период')
ax.set_xlabel('Пенсионер')
ax.set_ylabel('Ежемесячная оплата')
st.pyplot(fig)

# 6) Доля PhoneService и InternetService в одном гистограмме
st.subheader('Доля PhoneService и InternetService')
phone_internet_data = data[['PhoneService', 'InternetService']].apply(pd.Series.value_counts, normalize=True).T
fig, ax = plt.subplots()
phone_internet_data.plot(kind='bar', stacked=True, ax=ax, color=['skyblue', 'orange'])
ax.set_ylabel('Доля клиентов')
ax.set_xlabel('Сервисы')
ax.set_title('Доля клиентов с услугами PhoneService и InternetService')
st.pyplot(fig)

# 7) Гистограмма: Contract – тип контракта клиента
st.subheader('Тип контракта клиента')
contract_data = data['Contract'].value_counts(normalize=True)
fig, ax = plt.subplots()
sns.barplot(x=contract_data.index, y=contract_data.values, ax=ax, palette="Blues_d")
ax.set_ylabel('Доля клиентов')
ax.set_xlabel('Тип контракта')
ax.set_title('Доля клиентов по типу контракта')
st.pyplot(fig)
