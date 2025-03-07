import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Загружаем данные
data = pd.read_csv('telecom_users.csv')

# Покажем первые строки данных
st.title('Прогнозирование оттока клиентов')
st.write('Данные о клиентах:')
st.dataframe(data.head())

# Предобработка данных
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')  # Преобразуем TotalCharges в числовой формат
data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].median())  # Заполняем NaN медианой

# Кодирование категориальных признаков
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
data['Partner'] = le.fit_transform(data['Partner'])
data['Dependents'] = le.fit_transform(data['Dependents'])
data['PhoneService'] = le.fit_transform(data['PhoneService'])
data['PaperlessBilling'] = le.fit_transform(data['PaperlessBilling'])
data['Churn'] = le.fit_transform(data['Churn'])

# Разделяем данные на признаки и целевую переменную
X = data.drop(columns=['Churn'])
y = data['Churn']

# Масштабируем признаки
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Обучаем модель
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Прогнозирование
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
st.write(f"Точность модели: {accuracy:.4f}")
st.write(f"ROC AUC: {roc_auc:.4f}")

# Отчет по классификации
report = classification_report(y_test, y_pred, output_dict=True)
st.write(pd.DataFrame(report).transpose())

# Прогнозирование для нового клиента
st.sidebar.header("Ввод данных нового клиента")

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
input_proba = clf.predict_proba(input_scaled)[:, 1]
st.write(f"Вероятность оттока для этого клиента: {input_proba[0]:.2f}")
