import pickle
import streamlit as st

# membaca model
diabetes = pickle.load(open('diabetes_model.sav', 'rb'))

# judul web
# st.title('Machine Learning')
st.markdown(
    """
    <style>
    .title {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with centered class
st.markdown('<h1 class="title">Machine Learning Pendeteksi Diabetes</h1>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input('Masukan Nilai Pregnancies')
with col2:
    Glucose = st.number_input('Masukan Nilai Glucose')
with col1:
    BloodPressure = st.number_input('Masukan Nilai Blood Pressure')
with col2:
    SkinThickness = st.number_input('Masukan Nilai SkinThickness')
with col1:
    Insulin = st.number_input('Masukan Nilai Insulin')
with col2:
    BMI = st.number_input('Masukan Nilai BMI')
with col1:
    DiabetesPedigreeFunction = st.number_input(
        'Masukan Nilai Diabetes Pedigree Function')
with col2:
    Age = st.number_input('Masukan Nilai Umur')
# Tombol
diabetes_diagnosis = ''
# membuat tombol
if st.button('Test Prediksi diabetes'):
    diabetes_prediction = diabetes.predict(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    if (diabetes_prediction[0] == 1):
        diabetes_diagnosis = 'Pasien Terkena Diabetes'
    else:
        diabetes_diagnosis = 'Pasien Tidak Terkena Diabetes'
    st.success(diabetes_diagnosis)
