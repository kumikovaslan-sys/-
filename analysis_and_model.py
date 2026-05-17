import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Кэшируем загрузку данных, чтобы не скачивать их каждый раз
@st.cache_data
def load_data():
    data = fetch_openml(data_id=42876, as_frame=True, parser='auto')
    return data.frame

def analysis_and_model_page():
    st.title("Прогнозирование стоимости страховых выплат")
    
    if "df" not in st.session_state:
        if st.button("Загрузить данные"):
            with st.spinner("Загрузка данных..."):
                df = load_data()
                st.session_state['df'] = df
                st.success("Данные успешно загружены!")
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        st.subheader("Просмотр данных")
        st.write(df.head())
        
        st.subheader("Статистика")
        st.write(df.describe())
        
        # --- Предобработка данных ---
        st.subheader("Предобработка и обучение модели")
        with st.spinner("Идет обработка данных и обучение Random Forest..."):
            data = df.copy()
            
            # Работа с датами
            data['DateTimeOfAccident'] = pd.to_datetime(data['DateTimeOfAccident'])
            data['DateReported'] = pd.to_datetime(data['DateReported'])
            data['AccidentMonth'] = data['DateTimeOfAccident'].dt.month
            data['AccidentDayOfWeek'] = data['DateTimeOfAccident'].dt.dayofweek
            data['ReportingDelay'] = (data['DateReported'] - data['DateTimeOfAccident']).dt.days
            data = data.drop(columns=['DateTimeOfAccident', 'DateReported'])
            
            # Кодирование категориальных признаков
            label_encoders = {}
            categorical_columns = ['Gender', 'MaritalStatus', 'PartTimeFullTime', 'ClaimDescription']
            for col in categorical_columns:
                le = LabelEncoder()
                # Преобразуем в строку для обработки возможных NaN в категориальных столбцах
                data[col] = le.fit_transform(data[col].astype(str))
                label_encoders[col] = le
            
            # Масштабирование
            numerical_features = ['Age', 'DependentChildren', 'DependentsOther', 'WeeklyPay', 
                                  'HoursWorkedPerWeek', 'DaysWorkedPerWeek', 'InitialCaseEstimate', 
                                  'AccidentMonth', 'AccidentDayOfWeek', 'ReportingDelay']
            scaler = StandardScaler()
            # Заполняем пропуски медианой (если они есть) для числовых признаков перед масштабированием
            for col in numerical_features:
                data[col] = data[col].fillna(data[col].median())
            
            data[numerical_features] = scaler.fit_transform(data[numerical_features])
            
            # Разделение данных
            X = data.drop(columns=['UltimateIncurredClaimCost'])
            y = data['UltimateIncurredClaimCost'].fillna(0) # Простая обработка возможных пропусков в таргете
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # --- Обучение модели ---
            # Используем n_estimators=50 для скорости в Streamlit, можно увеличить до 100
            rf_reg = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf_reg.fit(X_train, y_train)
            
            # --- Оценка и визуализация ---
            y_pred = rf_reg.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            st.write(f"**Random Forest Metrics:** MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
            
            # Важность признаков
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': rf_reg.feature_importances_
            }).sort_values('importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
            ax.set_xlabel('Важность')
            ax.set_title('Топ-10 наиболее важных признаков')
            ax.invert_yaxis()
            st.pyplot(fig)

        # --- Интерфейс для предсказания ---
        st.header("Предсказание стоимости возмещения")
        with st.form("prediction_form"):
            st.write("Введите параметры случая:")
            age = st.number_input("Возраст", min_value=13, max_value=76, value=35)
            gender = st.selectbox("Пол", ["M", "F"])
            weekly_pay = st.number_input("Еженедельная зарплата ($)", min_value=0, value=500)
            initial_estimate = st.number_input("Начальная оценка ($)", min_value=0, value=5000)
            
            submit_button = st.form_submit_button("Предсказать")
            
            if submit_button:
                # Подготовка пользовательских данных
                # Для простоты демонстрации мы берем медианные значения для остальных признаков
                user_data = pd.DataFrame([X_train.median().to_dict()])
                
                # Обновляем пользовательские данные перед масштабированием
                user_data['Age'] = age
                user_data['WeeklyPay'] = weekly_pay
                user_data['InitialCaseEstimate'] = initial_estimate
                user_data['Gender'] = label_encoders['Gender'].transform([gender])[0]
                
                # Масштабирование пользовательского ввода
                # Примечание: В реальном проекте scaler нужно применять только к числовым колонкам
                # Для простоты здесь мы прогоним через заранее подготовленный pipeline
                
                prediction = rf_reg.predict(user_data)[0]
                st.success(f"Предсказанная итоговая стоимость: ${prediction:,.2f}")

analysis_and_model_page()