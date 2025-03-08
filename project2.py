import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

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
    data[col] = encoder.fit_transform(data[col].astype(str))

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

# Обучение модели CatBoost
cat_features = ['gender', 'Dependents', 'Contract', 'PhoneService', 'InternetService', 'StreamingTV', 'StreamingMovies']
model = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.1, loss_function='Logloss', cat_features=cat_features, verbose=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Интерфейс Streamlit
st.set_page_config(page_title='Прогноз оттока клиентов', layout='wide')
st.title('📊 Прогнозирование оттока клиентов')
st.write('🔍 Анализ данных и предсказание оттока клиентов телекоммуникационной компании.')

# Оценка модели
st.subheader('Оценка модели')
st.write(f'Точность модели: {accuracy * 100:.2f}%')




# Обзор данных
st.subheader('Обзор данных')
st.write(data.head())

# Введенные данные
st.subheader('Введенные данные')
st.write(input_data)

# Churn – доля отток и не отток
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=data, palette='coolwarm', hue='Churn')
plt.title('Распределение оттока клиентов')
plt.xlabel('Отток клиентов')
plt.ylabel('Количество')
# Легенда цветов
plt.legend(title="Churn", labels=["Не отток (0)", "Отток (1)"], loc="upper right")
st.pyplot(plt)

# Доля пенсионеров и не пенсионеров
plt.figure(figsize=(6, 4))
sns.countplot(x='SeniorCitizen', data=data, hue='SeniorCitizen', palette='coolwarm', legend=False)
plt.title('Доля пенсионеров и не пенсионеров')
plt.xlabel('Пенсионеры')
plt.ylabel('Количество')
# Легенда цветов
plt.legend(title="SeniorCitizen", labels=["Не пенсионер (0)", "Пенсионер (1)"], loc="upper right")
st.pyplot(plt)

# Доля женских и мужских половых клиентов
plt.figure(figsize=(6, 4))
sns.countplot(x='gender', data=data, hue='gender', palette='coolwarm', legend=False)
plt.title('Доля женских и мужских половых клиентов')
plt.xlabel('Пол')
plt.ylabel('Количество')
# Легенда цветов
plt.legend(title="Gender", labels=["Женский (0)", "Мужской (1)"], loc="upper right")
st.pyplot(plt)

# Доля ежемесячной оплаты и общей суммы оплаты
plt.figure(figsize=(6, 6))
sns.boxplot(data=data[['MonthlyCharges', 'TotalCharges']], palette=["skyblue", "orange"])
plt.title('Распределение оплаты клиентов')
plt.xlabel('Тип оплаты')
plt.ylabel('Сумма оплаты')
plt.xticks(ticks=[0, 1], labels=['Ежемесячная оплата', 'Общая сумма оплаты'])
st.pyplot(plt)

# Доля PhoneService и InternetService 
phone_internet_data = data[['PhoneService', 'InternetService']].apply(pd.Series.value_counts, normalize=True).T
plt.figure(figsize=(6, 4))
phone_internet_data.plot(kind='bar', stacked=True, color=['skyblue', 'orange'], edgecolor='black')
plt.title('Доля PhoneService и InternetService')
plt.xlabel('Тип услуги')
plt.ylabel('Доля')
# Легенда цветов
plt.legend(title="Тип услуги", labels=["PhoneService", "InternetService"], loc="upper right")
st.pyplot(plt)

# Тип контракта клиента
plt.figure(figsize=(6, 4))
sns.countplot(x='Contract', data=data, hue='Contract', palette='coolwarm', legend=False)
plt.title('Тип контракта клиента')
plt.xlabel('Тип контракта')
plt.ylabel('Количество')
# Легенда цветов
plt.legend(title="Contract", labels=["Month-to-month", "One year", "Two year"], loc="upper right")
st.pyplot(plt)

# Оценка модели
#st.subheader('Оценка модели')
#st.write(f'Точность модели: {accuracy * 100:.2f}%')

# Матрица путаницы
st.subheader('Матрица путаницы')
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Не отток", "Отток"], yticklabels=["Не отток", "Отток"])
plt.xlabel("Предсказанный")
plt.ylabel("Истинный")
st.pyplot(plt)
