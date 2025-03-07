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

st.title('📊 Прогнозирование оттока клиентов')

st.write('Анализ данных и предсказание оттока клиентов телекоммуникационной компании.')

# Загрузка данных
data = pd.read_csv('telecom_users.csv')

# Показать обзор данных
with st.expander('📊 Обзор данных'):
    st.write("**Данные**")
    st.dataframe(data.head())

# Предварительная обработка данных
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

if data.isnull().sum().any():
    st.write("В данных присутствуют пропущенные значения.")
else:
    st.write("Пропущенные значения отсутствуют.")

# Кодирование категориальных признаков
label_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
ohe_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
target_cols = ['InternetService']

le = LabelEncoder()
for col in label_cols:
    data[col] = le.fit_transform(data[col])

data = pd.get_dummies(data, columns=ohe_cols, drop_first=True)

data[target_cols] = data[target_cols].astype(str)  # Убедимся, что категориальные признаки имеют строковый тип перед кодированием
te = TargetEncoder(cols=target_cols)
data[target_cols] = te.fit_transform(data[target_cols], data['Churn'])

data = data.apply(pd.to_numeric, errors='coerce')

# Разделение на признаки и целевую переменную
X = data.drop(columns=['Churn'])
y = data['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Категориальные признаки для CatBoost
cat_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'InternetService']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, cat_features=[] , verbose=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
st.subheader('🔮 Результаты предсказания')
st.write(f"Точность модели: {accuracy:.4f}")
st.write(f"ROC AUC: {roc_auc:.4f}")

report = classification_report(y_test, y_pred, output_dict=True)
st.subheader("📊 Отчет по классификации")
st.write(pd.DataFrame(report).transpose())

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Корреляционная матрица')
st.pyplot(fig)

st.subheader('📊 Визуализация результатов')
fig1 = px.histogram(data, x="Churn", title="Распределение оттока клиентов")
st.plotly_chart(fig1)

importances = clf.get_feature_importance()
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
fig2 = plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances.index, y=feature_importances.values, palette='viridis')
plt.title('Важность признаков')
plt.xticks(rotation=45)
st.pyplot(fig2)

st.sidebar.header("🔧 Введите признаки клиента:")

input_data = {
    'age': st.slider('Возраст', 18, 100, 30),
    'TotalCharges': st.slider('Общие расходы', 0, 10000, 1000),
    'Tenure': st.slider('Стаж (месяцы)', 1, 72, 12),
    'MonthlyCharges': st.slider('Ежемесячные расходы', 0, 200, 50),
}

input_df = pd.DataFrame(input_data, index=[0])
input_scaled = scaler.transform(input_df)
input_prediction = clf.predict(input_scaled)

if input_prediction == 1:
    st.success("Этот клиент, вероятно, уйдет из компании (отток).")
else:
    st.success("Этот клиент, вероятно, останется в компании.")

input_proba = clf.predict_proba(input_scaled)[:, 1]
st.write(f"Вероятность оттока для этого клиента: {input_proba[0]:.2f}")
