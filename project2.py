import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import numpy as np

# Загрузка данных с обработкой ошибок
try:
    data = pd.read_csv('telecom_users.csv')
except FileNotFoundError:
    st.error('Файл с данными не найден. Пожалуйста, загрузите его заново.')
    st.stop()

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

# Боковая панель выбора признаков
st.sidebar.header('🔧 Выбор признаков')
selected_features = st.sidebar.multiselect('Выберите признаки', features, default=features)

# Отображение данных
st.subheader('Обзор данных')
st.write(data.head())

# Выбранные признаки
st.subheader('Выбранные признаки')
if selected_features:
    st.write(X[selected_features].head())
else:
    st.warning('Выберите хотя бы один признак.')

# Визуализация распределения оттока клиентов
st.subheader('Распределение оттока клиентов')
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='Churn', data=data, ax=ax)
ax.set_xticklabels(['Не ушел', 'Ушел'])
st.pyplot(fig)

# Гистограмма
st.subheader('Гистограммы')
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(data['MonthlyCharges'], kde=True, bins=30, ax=ax)
st.pyplot(fig)

# Визуализация важности признаков
st.subheader('Важность признаков')
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
st.pyplot(fig)

# Результат предсказания
st.subheader('Результат предсказания')
st.write(f'Точность модели: {accuracy:.2f}')
st.text(classification_report(y_test, y_pred))

# Матрица ошибок
st.subheader('Матрица ошибок')
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Не ушел', 'Ушел'], yticklabels=['Не ушел', 'Ушел'])
st.pyplot(fig)

# ROC-кривая
st.subheader('ROC-кривая')
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='red', linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC-Кривая')
ax.legend(loc='lower right')
st.pyplot(fig)
