import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Fungsi untuk memuat dataset
@st.cache
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Fungsi untuk pemrosesan data dan pemodelan
def process_data_model(data):
    # Memisahkan fitur dan label
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisasi fitur
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Membuat model SVM
    svm_model = SVC(kernel='linear', random_state=42)

    # Melatih model
    svm_model.fit(X_train, y_train)

    return svm_model, X_test, y_test

# Fungsi untuk prediksi
def predict(model, X_test, y_test):
    # Memprediksi data uji
    y_pred = model.predict(X_test)

    # Menghitung akurasi
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Akurasi: {accuracy * 100:.2f}%')

    # Menampilkan laporan klasifikasi
    report = classification_report(y_test, y_pred)
    st.write('Laporan Klasifikasi:')
    st.write(report)

# Main program
def main():
    st.title('Prediksi Diabetes Menggunakan SVM')

    # Memuat dataset
    file_path = 'diabetes.sav'  # Ganti dengan nama file dataset yang Anda miliki
    data = load_data(file_path)

    # Menampilkan beberapa baris pertama dari dataset untuk melihat strukturnya
    st.write('Data Awal:')
    st.write(data.head())

    # Memroses data dan membuat model
    model, X_test, y_test = process_data_model(data)

    # Menampilkan form untuk input data pengguna
    st.sidebar.title('Masukkan Data Pasien:')
    pregnancies = st.sidebar.slider('Jumlah Kehamilan', 0, 17, 1)
    glucose = st.sidebar.slider('Glukosa', 0, 200, 100)
    blood_pressure = st.sidebar.slider('Tekanan Darah', 0, 122, 70)
    skin_thickness = st.sidebar.slider('Ketebalan Kulit', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('Indeks Massa Tubuh (BMI)', 0.0, 67.1, 32.0)
    diabetes_pedigree_function = st.sidebar.slider('Fungsi Pedigri Diabetes', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Usia', 21, 81, 29)

    # Membuat prediksi berdasarkan input pengguna
    user_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    })

    if st.sidebar.button('Prediksi'):
        prediction = model.predict(user_data)
        if prediction[0] == 1:
            st.sidebar.error('Pasien kemungkinan memiliki diabetes.')
        else:
            st.sidebar.success('Pasien kemungkinan tidak memiliki diabetes.')

# Jalankan aplikasi
if __name__ == '__main__':
    main()
