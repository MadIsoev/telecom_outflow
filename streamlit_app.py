import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Установка страницы
st.set_page_config(page_title='Прогноз оттока клиентов', layout='wide')
st.title('📊 Прогнозирование оттока клиентов')
st.write('🔍 Анализ данных и предсказание оттока клиентов телекоммуникационной компании.')

# Загрузка данных
data = pd.read_csv('telecom_users.csv')

# Обработка данных
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# Категориальные признаки
categorical_columns = ['PhoneService', 'Contract', 'PaymentMethod', 'InternetService']

# Кодирование категориальных признаков
le = LabelEncoder()

# Словарь для хранения соответствий категорий и преобразования
le_dict = {
    'PhoneService': ['Yes', 'No'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaymentMethod': ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'],
    'InternetService': ['DSL', 'Fiber optic', 'No']
}

# Применяем LabelEncoder для категориальных признаков с фиксированными категориями
for col in categorical_columns:
    le.fit(le_dict[col])  # fit для каждой категории в словаре
    data[col] = le.transform(data[col])

# One-hot кодирование для переменных с несколькими категориями
ohe_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
data = pd.get_dummies(data, columns=ohe_cols, drop_first=True)

# Преобразуем все данные в числовые
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

# Указываем категориальные признаки для CatBoost
cat_features = ['PhoneService', 'Contract', 'PaymentMethod', 'InternetService']

# Проверка на наличие категориальных признаков в данных
cat_feature_indices = [X.columns.get_loc(col) for col in cat_features if col in X.columns]
if len(cat_feature_indices) != len(cat_features):
    st.error("Некоторые категориальные признаки не найдены в данных!")

# Обучение модели CatBoost
clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, cat_features=cat_feature_indices, verbose=0)
clf.fit(X_train, y_train)

# Прогнозы
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Вывод результатов модели
st.subheader('📊 Результаты модели')
st.metric(label='Точность', value=f"{accuracy:.4f}")
st.metric(label='ROC AUC', value=f"{roc_auc:.4f}")

# Отчет по классификации
st.subheader('📌 Отчет по классификации')
st.write(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

# Визуализация корреляции
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
st.pyplot(fig)

# Форма для ввода данных
with st.sidebar:
    st.header("🔧 Введите признаки: ")
    
    # Длительность обслуживания (tenure)
    tenure = st.slider('Длительность обслуживания', min_value=int(data['tenure'].min()), max_value=int(data['tenure'].max()), value=int(data['tenure'].mean()))
    
    # Ежемесячные платежи (MonthlyCharges)
    MonthlyCharges = st.slider('Ежемесячные платежи', min_value=float(data['MonthlyCharges'].min()), max_value=float(data['MonthlyCharges'].max()), value=float(data['MonthlyCharges'].mean()))
    
    # Тип интернет-услуги (InternetService)
    InternetService = st.selectbox('Тип интернет-услуги', le_dict['InternetService'], index=le_dict['InternetService'].index('DSL'))  # По умолчанию выбрано 'DSL'
    
    # Общая сумма (TotalCharges)
    TotalCharges = st.slider('Общая сумма', min_value=float(data['TotalCharges'].min()), max_value=float(data['TotalCharges'].max()), value=float(data['TotalCharges'].mean()))
    
    # Сервис (PhoneService)
    PhoneService = st.selectbox('Сервис', le_dict['PhoneService'], index=le_dict['PhoneService'].index('Yes'))  # По умолчанию выбрано 'Yes'
    
    # Тип контракта (Contract)
    Contract = st.selectbox('Тип контракта', le_dict['Contract'], index=le_dict['Contract'].index('Month-to-month'))  # По умолчанию выбрано 'Month-to-month'
    
    # Метод оплаты (PaymentMethod)
    PaymentMethod = st.selectbox('Метод оплаты', le_dict['PaymentMethod'], index=le_dict['PaymentMethod'].index('Electronic check'))  # По умолчанию выбрано 'Electronic check'

# Прогнозирование для введенных данных
input_data = {
    'tenure': tenure,
    'MonthlyCharges': MonthlyCharges,
    'InternetService': InternetService,
    'TotalCharges': TotalCharges,
    'PhoneService': PhoneService,
    'Contract': Contract,
    'PaymentMethod': PaymentMethod
}

# Преобразование введенных данных в DataFrame
input_df = pd.DataFrame([input_data])

# Преобразуем категориальные признаки с помощью LabelEncoder
input_df['PhoneService'] = le.transform(input_df['PhoneService'])
input_df['Contract'] = le.transform(input_df['Contract'])
input_df['PaymentMethod'] = le.transform(input_df['PaymentMethod'])
input_df['InternetService'] = le.transform(input_df['InternetService'])

# Масштабирование данных
input_df_scaled = scaler.transform(input_df)

# Прогноз
input_prediction = clf.predict(input_df_scaled)
input_proba = clf.predict_proba(input_df_scaled)[:, 1]

# Результат предсказания
st.subheader("📌 Результат предсказания")
if input_prediction == 1:
    st.error("Этот клиент, вероятно, уйдёт.")
else:
    st.success("Этот клиент, вероятно, останется.")
st.write(f"🔍 Вероятность оттока: {input_proba[0]:.2f}")
