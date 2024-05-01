import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import PolynomialFeatures
import requests

url = 'https://github.com/juanvalno/itb/raw/a2c2e75ffaa116ffc7a0c08bafbac9bd7f06dfec/Model/model_lgbm_tune.pkl'
response = requests.get(url)
with open('model_lgbm_tune.pkl', 'wb') as file:
    file.write(response.content)

# Load the model and other data
model_data = pickle.load(open('model_lgbm_tune.pkl', 'rb'))

# Extract the model object
model = model_data['best_model']

st.title('Prediksi Tingkat Cholesterol')

col1, col2, col3 = st.columns(3)
with col1:
    Gndr = st.selectbox('Masukan Gender', ['Male', 'Female'])
    gender_encoder = LabelEncoder()
    Gender = gender_encoder.fit_transform([Gndr])[0]
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
with col7:
    Trigliserida  = st.number_input('Masukkan Trigliserida (mg/dL) ', format="%.2f")
with col8:
    Fat = st.number_input('Masukkan Fat', format="%.2f")
with col9:
    Visceral_Fat = st.number_input('Masukkan Visceral Fat', format="%.2f")

col13 = st.columns(1)
with col7:
    Masa_Kerja  = st.number_input('Masukkan Masa Kerja', format="%.1f")

Label = ''

input_data = pd.DataFrame({
    'Gender': [Gender], 'Usia': [Usia], 'Tekanan_darah_S': [Tekanan_darah_S],
    'Tekanan_darah_D': [TekananS_D], 'Tinggi_badan': [Tinggi_Badan],
    'Berat_badan': [Berat_Badan], 'IMT': [imt], 'Lingkar_perut': [lingkar_perut],
    'Glukosa_Puasa': [glukosa_puasa], 'Trigliserida': [Trigliserida],
    'Fat': [Fat], 'Visceral_Fat': [Visceral_Fat], 'Masa_Kerja': [Masa_Kerja]
})

# Checking high skewness and transforming the data
skew_check = input_data.apply(lambda input_data: abs(skew(input_data)))
skewed_feats = skew_check[skew_check > 0.5].index

for feature in skewed_feats:
    optimal_lambda = boxcox_normmax(input_data[feature] + 1)
    input_data[feature] = boxcox1p(input_data[feature], optimal_lambda)

# Apply polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
input_data_poly = poly.fit_transform(input_data)

if st.button('Deteksi Cholesterol'):
    prediction = model.predict(input_data_poly)[0]
    prediction_inverse = np.expm1(prediction)
    st.write('Prediksi Cholesterol:', prediction_inverse)
