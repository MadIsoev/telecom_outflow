import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from category_encoders import TargetEncoder

# Установка страницы
st.set_page_config(page_title='Прогноз оттока клиентов', layout='wide')
st.title('📊 Прогнозирование оттока клиентов')
st.write('🔍 Анализ данных и предсказание оттока клиентов телекоммуникационной компании.')

# Загрузка данных
data = pd.read_csv('telecom_users.csv')

# Обзор данных
with st.expander('📊 Просмотр данных'):
    st.write(data.head())

# Обработка данных
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# Кодирование категориальных признаков
label_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
ohe_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
target_cols = ['InternetService']

# Используем LabelEncoder для категориальных признаков с бинарным кодированием
le = LabelEncoder()
for col in label_cols:
    data[col] = le.fit_transform(data[col])

# One-hot кодирование для переменных с несколькими категориями
data = pd.get_dummies(data, columns=ohe_cols, drop_first=True)

# Кодирование столбца InternetService с использованием TargetEncoder
te = TargetEncoder(cols=target_cols)
data[target_cols] = te.fit_transform(data[target_cols], data['Churn'])

data = data.apply(pd.to_numeric, errors='coerce')

# Разделение данных
X = data.drop(columns=['Churn'])
y = data['Churn']

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Разделение на тренировочные и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=0)
clf.fit(X_train, y_train)

# Прогнозы
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Форма для ввода данных
with st.sidebar:
    st.header("🔧 Введите признаки: ")
    
    # Длительность обслуживания (tenure)
    tenure = st.slider('Длительность обслуживания', min_value=int(data['tenure'].min()), max_value=int(data['tenure'].max()), value=int(data['tenure'].mean()))
    
    # Ежемесячные платежи (MonthlyCharges)
    MonthlyCharges = st.slider('Ежемесячные платежи', min_value=float(data['MonthlyCharges'].min()), max_value=float(data['MonthlyCharges'].max()), value=float(data['MonthlyCharges'].mean()))
    
    # Тип интернет-услуги (InternetService)
    InternetService_options = data['InternetService'].unique()
    InternetService = 
