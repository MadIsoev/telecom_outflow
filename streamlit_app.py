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
st.sidebar.header("🔧 Введите данные клиента")
input_data = {}
for col in X.columns:
    input_data[col] = st.sidebar.number_input(col, value=float(X[col].mean()))

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)
input_prediction = clf.predict(input_scaled)
input_proba = clf.predict_proba(input_scaled)[:, 1]

# Результат предсказания на главной странице (перед "Результаты модели")
st.subheader("📌 Результат предсказания")
if input_prediction == 1:
    st.error("Этот клиент, вероятно, уйдёт.")
else:
    st.success("Этот клиент, вероятно, останется.")
st.write(f"🔍 Вероятность оттока: {input_proba[0]:.2f}")

# Вывод результатов модели
st.subheader('📊 Результаты модели')
st.metric(label='Точность', value=f"{accuracy:.4f}")
st.metric(label='ROC AUC', value=f"{roc_auc:.4f}")

# Отчет по классификации
st.subheader('📌 Отчет по классификации')
st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

# Визуализация корреляции
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
st.pyplot(fig)

# Гистограмма оттока клиентов
fig1 = px.histogram(data, x='Churn', title='Распределение оттока клиентов')
st.plotly_chart(fig1)

# Важность признаков
importances = clf.get_feature_importance()
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
fig2 = plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances.index, y=feature_importances.values, palette='viridis')
plt.xticks(rotation=45)
st.pyplot(fig2)

# Распределение целевого признака
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=data, hue='Churn', palette='coolwarm', legend=False)
plt.title('Распределение оттока клиентов (0 - останется, 1 - уйдёт)')
st.pyplot(plt)
