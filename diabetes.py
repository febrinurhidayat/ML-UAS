import pickle
import streamlit as st

# Membaca model
diabetes = pickle.load(open('diabetes_model.sav', 'rb'))

# CSS tampilan
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        color: #4CAF50;
        font-family: 'Arial', sans-serif;
        font-size: 2.5em;
    }
    .subtitle {
        text-align: center;
        color: #4CAF50;
        font-family: 'Arial', sans-serif;
        font-size: 1.5em;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .input-label {
        font-size: 1.1em;
        color: #333;
    }
    .button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.2em;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Judul dengan kelas yang sudah ditentukan
st.markdown('<h1 class="title">Machine Learning</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subtitle">Pendeteksi Diabetes</h2>', unsafe_allow_html=True)

# Kolom untuk input
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.number_input('Masukan Nilai Pregnancies', min_value=0, step=1, format='%d')
with col2:
    Glucose = st.number_input('Masukan Nilai Glucose', min_value=0, step=1, format='%d')
with col1:
    BloodPressure = st.number_input('Masukan Nilai Blood Pressure', min_value=0, step=1, format='%d')
with col2:
    SkinThickness = st.number_input('Masukan Nilai Skin Thickness', min_value=0, step=1, format='%d')
with col1:
    Insulin = st.number_input('Masukan Nilai Insulin', min_value=0, step=1, format='%d')
with col2:
    BMI = st.number_input('Masukan Nilai BMI', min_value=0.0, format='%.1f')
with col1:
    DiabetesPedigreeFunction = st.number_input('Masukan Nilai Diabetes Pedigree Function', min_value=0.0, format='%.3f')
with col2:
    Age = st.number_input('Masukan Nilai Umur', min_value=0, step=1, format='%d')

# Pusatkan tombol prediksi
st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
if st.button('Test Prediksi Diabetes', key='predict_button'):
    diabetes_prediction = diabetes.predict(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
    )
    if (diabetes_prediction[0] == 1):
        diabetes_diagnosis = 'Pasien Terkena Diabetes'
    else:
        diabetes_diagnosis = 'Pasien Tidak Terkena Diabetes'
    st.success(diabetes_diagnosis)
st.markdown("</div>", unsafe_allow_html=True)
