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

# Боковая панель выбора признаков и их значений
st.sidebar.header('🔧 Выбор признаков и значений')

# Для каждого категориального признака создаем мультивыбор
internet_service_options = data['InternetService'].unique()
contract_options = data['Contract'].unique()
payment_method_options = data['PaymentMethod'].unique()

selected_internet_service = st.sidebar.multiselect('Выберите тип подключения к интернету', internet_service_options)
selected_contract = st.sidebar.multiselect('Выберите тип контракта', contract_options)
selected_payment_method = st.sidebar.multiselect('Выберите способ оплаты', payment_method_options)

# Применение фильтрации данных по выбранным значениям
filtered_data = data.copy()

if selected_internet_service:
    filtered_data = filtered_data[filtered_data['InternetService'].isin(selected_internet_service)]
if selected_contract:
    filtered_data = filtered_data[filtered_data['Contract'].isin(selected_contract)]
if selected_payment_method:
    filtered_data = filtered_data[filtered_data['PaymentMethod'].isin(selected_payment_method)]

# Отображение данных
st.subheader('Обзор данных')
st.write(filtered_data.head())

# Разделение данных для обучения модели с учетом фильтрации
X_filtered = pd.DataFrame(scaler.fit_transform(filtered_data[features]), columns=features)
y_filtered = filtered_data['Churn']

# Если есть отфильтрованные данные, обучаем модель и показываем результат
if not filtered_data.empty:
    X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)
    model.fit(X_train_filtered, y_train_filtered)
    y_pred_filtered = model.predict(X_test_filtered)
    accuracy_filtered = accuracy_score(y_test_filtered, y_pred_filtered)

    # Результат предсказания
    st.subheader('Результат предсказания для выбранных значений')
    st.write(f'Точность модели: {accuracy_filtered:.2f}')
    st.text(classification_report(y_test_filtered, y_pred_filtered))

    # Визуализации
    st.subheader('Гистограммы')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(filtered_data['MonthlyCharges'], kde=True, bins=30, ax=ax)
    st.pyplot(fig)

    # Важность признаков
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
    conf_matrix = confusion_matrix(y_test_filtered, y_pred_filtered)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Не ушел', 'Ушел'], yticklabels=['Не ушел', 'Ушел'])
    st.pyplot(fig)
else:
    st.warning('Нет данных, соответствующих выбранным значениям.')
