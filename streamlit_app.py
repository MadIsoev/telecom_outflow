import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

st.set_page_config(page_title='Прогноз оттока клиентов', layout='wide')
st.title('📊 Прогнозирование оттока клиентов')
st.write('🔍 Анализ данных и предсказание оттока клиентов телекоммуникационной компании.')

# Загрузка данных
try:
    data = pd.read_csv('telecom_users.csv')
    st.success("✅ Данные успешно загружены!")
except Exception as e:
    st.error(f"⚠️ Ошибка загрузки данных: {e}")

# Обзор данных
with st.expander('📊 Обзор данных'):
    st.write("**Матрица признаков (X)**")
    X_raw = data.drop(columns=["Churn"], errors='ignore')
    st.dataframe(X_raw)

    st.write("**Целевая переменная (y)**")
    y_raw = data["Churn"].astype(int)
    st.dataframe(y_raw)

with st.sidebar:
    st.header("🔧 Введите данные клиента:")
    age = st.slider('Возраст', float(data.age.min()), float(data.age.max()), float(data.age.mean()))
    gender = st.selectbox('Пол', ['Мужской', 'Женский'])

data_input = {
    'age': age,
    'gender': gender
}

input_df = pd.DataFrame(data_input, index=[0])
input_combined = pd.concat([input_df, X_raw], axis=0)

with st.expander('📥 Введенные данные'):
    st.write('**Данные клиента**')
    st.dataframe(input_df)
    st.write('**Объединенные данные (Новые данные + Оригинальные)**')
    st.dataframe(input_combined)

# Обработка данных
imputer = SimpleImputer(strategy="mean")
X = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)
y = y_raw

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Обучение модели
clf = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, verbose=0)
clf.fit(X_train, y_train)

# Предсказание для введенных данных
input_df_aligned = pd.DataFrame(columns=X_raw.columns)
input_df_aligned = pd.concat([input_df_aligned, input_df], ignore_index=True).fillna(0)
df_input_scaled = pd.DataFrame(scaler.transform(input_df_aligned), columns=X_raw.columns)

prediction = clf.predict(df_input_scaled)
prediction_proba = clf.predict_proba(df_input_scaled)

df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Не останется', 'Останется'])

# Вывод предсказаний
st.subheader('🔮 Прогноз выживаемости')
st.dataframe(df_prediction_proba, hide_index=True)

status = np.array(['Не останется', 'Останется'])
st.success(f"Предсказанный статус: **{status[prediction][0]}**")

# Визуализация данных
st.subheader("📊 Визуализация данных")

# График рассеяния
fig1 = px.scatter(data, x='age', y='TotalCharges', color='Churn', title='Возраст vs. Общие расходы')
st.plotly_chart(fig1)

# Гистограмма
fig2 = px.histogram(data, x='age', nbins=30, title='Распределение возраста')
st.plotly_chart(fig2)

# Корреляционная матрица
st.subheader("🔎 Корреляции признаков")
fig, ax = plt.subplots()

data_numeric = data.select_dtypes(include=['float64', 'int64'])
sns.heatmap(data_numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)

st.write("💡 **Совет:** Используйте ползунки на боковой панели для ввода данных клиента и получения прогноза!")

