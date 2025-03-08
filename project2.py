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
categorical_features = ['InternetService', 'Contract', 'PaymentMethod', 'MultipleLines']
for col in categorical_features:
    data[col] = data[col].astype(str)  # Приводим к строковому типу
    data[col] = encoder.fit_transform(data[col])

# Определение важных признаков
features = ['tenure', 'PhoneService', 'InternetService', 'MonthlyCharges', 'TotalCharges',
            'Contract', 'PaymentMethod']
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(data[features]), columns=features)
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
    
    # Длительность обслуживания (tenure)
    tenure = st.slider('Длительность обслуживания', min_value=int(data['tenure'].min()), max_value=int(data['tenure'].max()), value=int(data['tenure'].mean()))
    
    # Ежемесячные платежи (MonthlyCharges)
    MonthlyCharges = st.slider('Ежемесячные платежи', min_value=float(data['MonthlyCharges'].min()), max_value=float(data['MonthlyCharges'].max()), value=float(data['MonthlyCharges'].mean()))
    
    # Тип интернет-услуги (InternetService)
    InternetService_options = ['DSL', 'Fiber optic', 'No']
    InternetService = st.selectbox('Тип интернет-услуги', InternetService_options, index=InternetService_options.index('DSL'))  # По умолчанию выбрано 'DSL'
    
    # Общая сумма (TotalCharges)
    TotalCharges = st.slider('Общая сумма', min_value=float(data['TotalCharges'].min()), max_value=float(data['TotalCharges'].max()), value=float(data['TotalCharges'].mean()))
    
    # Сервис (PhoneService)
    PhoneService_options = ['Yes', 'No']
    PhoneService = st.selectbox('Сервис', PhoneService_options, index=PhoneService_options.index('Yes'))  # По умолчанию выбрано 'Yes'
    
    # Тип контракта (Contract)
    Contract_options = ['Month-to-month', 'One year', 'Two year']
    Contract = st.selectbox('Тип контракта', Contract_options, index=Contract_options.index('Month-to-month'))  # По умолчанию выбрано 'Month-to-month'
    
    # Метод оплаты (PaymentMethod)
    PaymentMethod_options = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    PaymentMethod = st.selectbox('Метод оплаты', PaymentMethod_options, index=PaymentMethod_options.index('Electronic check'))  # По умолчанию выбрано 'Electronic check'

# Создание данных для предсказания
input_data = pd.DataFrame({
    'tenure': [tenure],
    'PhoneService': [1 if PhoneService == 'Yes' else 0],
    'InternetService': [InternetService_options.index(InternetService)],
    'MonthlyCharges': [MonthlyCharges],
    'TotalCharges': [TotalCharges],
    'Contract': [Contract_options.index(Contract)],
    'PaymentMethod': [PaymentMethod_options.index(PaymentMethod)],
})

# Масштабирование данных
input_data_scaled = scaler.transform(input_data)

# Прогнозирование
prediction = model.predict(input_data_scaled)
prediction_prob = model.predict_proba(input_data_scaled)

# Проверка, что вероятности присутствуют
if prediction_prob.shape[1] > 1:
    # Вероятность оттока (для класса 1)
    probability_of_churn = prediction_prob[0][1]
else:
    # Если только один класс (например, для бинарной классификации с одним положительным классом)
    probability_of_churn = prediction_prob[0][0]

# Отображение результата
st.subheader('Результат предсказания')
if prediction == 1:
    st.write("Клиент вероятно уйдет (отток).")
else:
    st.write("Клиент не уйдет (не будет оттока).")

# Вероятность оттока
st.write(f'Вероятность оттока: {probability_of_churn:.2f}')

# Обзор данных
st.subheader('Обзор данных')
st.write(data.head())

# Визуализация важности признаков
st.subheader('Важность признаков')
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
st.pyplot(fig)

# Матрица ошибок
st.subheader('Матрица ошибок')
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Не ушел', 'Ушел'], yticklabels=['Не ушел', 'Ушел'])
st.pyplot(fig)
