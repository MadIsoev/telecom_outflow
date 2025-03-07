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
# Преобразуем 'TotalCharges' в числовой формат, заменяя ошибки на NaN
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Заполнение пропущенных значений медианой
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())

# Проверка на наличие пустых значений в других столбцах перед масштабированием
if data.isnull().sum().any():
    st.write("В данных присутствуют пропущенные значения.")
else:
    st.write("Пропущенные значения отсутствуют.")

# Кодирование категориальных признаков
label_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
ohe_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
target_cols = ['InternetService']

# Label Encoding
le = LabelEncoder()
for col in label_cols:
    data[col] = le.fit_transform(data[col])

# One-Hot Encoding
data = pd.get_dummies(data, columns=ohe_cols, drop_first=True)

# Target Encoding
te = TargetEncoder(cols=target_cols)
data[target_cols] = te.fit_transform(data[target_cols], data['Churn'])

# Преобразование всех данных в числовой формат
data = data.apply(pd.to_numeric, errors='coerce')

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.drop(columns=['Churn'])) 
X = pd.DataFrame(X_scaled, columns=data.drop(columns=['Churn']).columns) 
y = data['Churn']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели CatBoost
clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, cat_features=[data.columns.get_loc(col) for col in target_cols], verbose=0)
clf.fit(X_train, y_train)

# Прогнозирование
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
st.subheader('🔮 Результаты предсказания')
st.write(f"Точность модели: {accuracy:.4f}")
st.write(f"ROC AUC: {roc_auc:.4f}")

# Отображение отчета по классификации
report = classification_report(y_test, y_pred, output_dict=True)
st.subheader("📊 Отчет по классификации")
st.write(pd.DataFrame(report).transpose())

# Визуализация ROC-кривой
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Корреляционная матрица')
st.pyplot(fig)

# Визуализация результатов
st.subheader('📊 Визуализация результатов')

# Гистограмма распределения оттока
fig1 = px.histogram(data, x="Churn", title="Распределение оттока клиентов")
st.plotly_chart(fig1)

# Важность признаков
importances = clf.get_feature_importance()
feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
fig2 = plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances.index, y=feature_importances.values, palette='viridis')
plt.title('Важность признаков')
plt.xticks(rotation=45)
st.pyplot(fig2)

# Предсказание для нового клиента
st.sidebar.header("🔧 Введите признаки клиента:")

# Ввод данных нового клиента
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

# Прогноз вероятности оттока
input_pro
