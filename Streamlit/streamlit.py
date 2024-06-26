import pickle
import pandas as pd
import numpy as np
import streamlit as st
from scipy.special import boxcox1p
from sklearn.preprocessing import PolynomialFeatures
import requests

# Use the raw URL of the file
url_lambda = 'https://raw.githubusercontent.com/juanvalno/itb/ecc486c3c479818fc61dead72c9751b326bd9ef0/Model/lambda_values.pkl'
url_model = 'https://raw.githubusercontent.com/juanvalno/itb/c74d37ccfe8324d13668d2862bac8f0b81072b47/Model/model_lgbm_tune.pkl'

# Function to load a pickle file from a URL
def load_pickle_from_url(url):
    response = requests.get(url)
    file_content = response.content
    return pickle.loads(file_content)

# Load the lambda values and model
lambda_values = load_pickle_from_url(url_lambda)
model_data = load_pickle_from_url(url_model)

# Extract the model object
model = model_data['best_model']

title = 'MEKAR'
html_string_1 = f"<h1 style='font-size: 50px;'>{title}</h1>"
st.markdown(html_string_1, unsafe_allow_html=True)

title_2 = 'Mengoptimalkan Kesejahteraan Karyawan dengan Prediksi Cerdas'
html_string_2 = f"<h3 style='font-size: 25px;'>{title_2}</h3>"
st.markdown(html_string_2, unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    gender_mapping = {'Male': 1, 'Female': 0}
    Gndr = st.selectbox('Masukan Gender', ['Male', 'Female'])
    Gender = gender_mapping[Gndr]
with col2:
    Usia = st.number_input('Masukkan Usia', format="%.0f")
with col3:
    Tekanan_darah_S = st.number_input('Masukkan Tekanan Sistolik', format="%.2f")

col4, col5, col6 = st.columns(3)
with col4:
    TekananS_D = st.number_input('Masukkan Tekanan Diastolik', format="%.2f")
with col5:
    Tinggi_Badan = st.number_input('Masukkan Tinggi Badan (cm)', format="%.2f")
with col6:
    Berat_Badan = st.number_input('Masukkan Berat Badan (kg)', format="%.2f")

col7, col8, col9 = st.columns(3)
with col7:
    imt = st.number_input('Masukkan IMT (kg/m2)', format="%.2f")
with col8:
    lingkar_perut = st.number_input('Masukkan Lingkar Perut (cm)', format="%.2f")
with col9:
    glukosa_puasa = st.number_input('Masukkan Glukosa (mg/dL)', format="%.2f")

col10, col11, col12 = st.columns(3)
with col10:
    Trigliserida  = st.number_input('Masukkan Trigliserida (mg/dL) ', format="%.2f")
with col11:
    Fat = st.number_input('Masukkan Fat', format="%.2f")
with col12:
    Visceral_Fat = st.number_input('Masukkan Visceral Fat', format="%.2f")

col13 = st.columns(1)
with col13[0]:
    Masa_Kerja  = st.number_input('Masukkan Masa Kerja', format="%.1f")

input_data = pd.DataFrame({
    'Gender': [Gender], 'Usia': [Usia], 'Tekanan_darah_S': [Tekanan_darah_S],
    'Tekanan_darah_D': [TekananS_D], 'Tinggi_badan': [Tinggi_Badan],
    'Berat_badan': [Berat_Badan], 'IMT': [imt], 'Lingkar_perut': [lingkar_perut],
    'Glukosa_Puasa': [glukosa_puasa], 'Trigliserida': [Trigliserida],
    'Fat': [Fat], 'Visceral_Fat': [Visceral_Fat], 'Masa_Kerja': [Masa_Kerja]
})

skewed_feats = ['Berat_badan', 'IMT', 'Glukosa_Puasa', 'Trigliserida', 'Fat', 'Visceral_Fat', 'Masa_Kerja', 'Gender', 'Usia']

for feature in skewed_feats:
    input_data[feature] = boxcox1p(input_data[feature], lambda_values[feature])    

# Apply polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
input_data_poly = poly.fit_transform(input_data)

if st.button('Deteksi Cholesterol'):
    if input_data.isnull().any().any():
        st.write('Data tidak terisi semua. Tolong isi kembali semua data.')
    else:
        prediction = model.predict(input_data_poly)[0]
        prediction_inverse = np.expm1(prediction)
        text = 'Prediksi Cholesterol:'
        html_string = f"<h1 style='font-size: 24px;'>{text} {prediction_inverse}</h1>"
        st.markdown(html_string, unsafe_allow_html=True)



