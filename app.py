import streamlit as st

# Настройка страницы
st.set_page_config(
    page_title="Предиктивное обслуживание",
    layout="wide",
)

# Настройка навигации
pages = [
    st.Page("analysis_and_model.py", title="Анализ и модель"),
    st.Page("presentation.py", title="Презентация"),
]

# Отображение навигации
current_page = st.navigation(pages, position="sidebar", expanded=True)
current_page.run()

