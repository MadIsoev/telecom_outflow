import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

# Загружаем данные
data = pd.read_csv('telecom_users.csv')

# Покажем первые строки данных
st.title('Прогнозирование оттока клиентов')
st.write('Данные о клиентах:')
st.dataframe(data.head())

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

# Указываем индексы категориальных признаков (если они имеются в DataFrame)
cat_features = [0, 1, 2, 3, 4]  # Индексы категориальных признаков в X

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель CatBoost
clf = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, cat_features=cat_features, verbose=200)
clf.fit(X_train, y_train)

# Прогнозирование
y_pred = clf.predict(X_test)

# Оценка модели
accuracy = (y_pred == y_test).mean()
st.write(f"Точность модели: {accuracy:.4f}")

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
input_prediction = clf.predict(input_df)

if input_prediction == 1:
    st.success("Этот клиент, вероятно, уйдет из компании (отток).")
else:
    st.success("Этот клиент, вероятно, останется в компании.")

