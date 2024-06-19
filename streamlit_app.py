import streamlit as st
import joblib
import numpy as np
import pandas as pd
import pickle

# Sayfayı üç sütuna ayırıyoruz
col1, col2, col3 = st.columns([1, 2, 1])

# CSS stilini ekliyoruz
st.markdown(
    """
    <style>
    .small-font {
        font-size:12px !important;
    }
    .red-header {
        text-align: center;
        color: #b30000;
    }
    .background {
        background-color: #FFD6E7;
        padding: 10px;
        border-radius: 10px;
    }
    .compact {
        margin-bottom: 0px;
        margin-top: 0px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sol sütuna başlık ve girdiler ekliyoruz
with col1:
    st.markdown('<div class="background">', unsafe_allow_html=True)
    st.markdown('<h4 class="small-font">Choose Your Wine</h4>', unsafe_allow_html=True)
    
    st.markdown('<p class="small-font compact">Wine Score</p>', unsafe_allow_html=True)
    wine_score = st.slider("", min_value=80, max_value=100, value=90)
    
    st.markdown('<p class="small-font compact">Vintage (Age of Wine)</p>', unsafe_allow_html=True)
    vintage = st.slider("", min_value=0, max_value=100, value=10)
    
    st.markdown('<p class="small-font compact">Quality Index</p>', unsafe_allow_html=True)
    quality_index = st.slider("", min_value=-2.0, max_value=4.0, value=1.0)
    st.markdown('</div>', unsafe_allow_html=True)

# Orta sütuna başlık ve GIF ekliyoruz
with col2:
    st.markdown('<h2 class="red-header">AI Supported Wine Price Estimator</h2>', unsafe_allow_html=True)
    st.image("national-wine-day-wine-day.gif")

# Sağ sütuna "Country of Origin" ve "Special Price" metnini ve altındaki kutuyu ekliyoruz
with col3:
    st.markdown('<div class="background">', unsafe_allow_html=True)
    st.markdown('<p class="small-font compact">Country of Origin</p>', unsafe_allow_html=True)
    country = st.selectbox("", options=["France", "Spain", "England", "New Zealand"])
    
    st.markdown('<h4 class="small-font compact">Special Price:</h4>', unsafe_allow_html=True)
    
    # Modeli ve encoder'ı yükle
    model = joblib.load("final_model.joblib")
    with open('onehot_encoder.pkl', 'rb') as file:
        loaded_encoder = pickle.load(file)
    
    # Verileri hazırlama
    data = pd.DataFrame({
        'wine_score': [np.log(wine_score)],
        'age_of_wine': [np.log(vintage)],
        'quality_index': [quality_index],
        'country': [country]
    })
    
    # One-Hot Encoding (önceden eğitilmiş encoder'ı kullanarak)    
    encoded_country = loaded_encoder.transform(data[['country']])
    encoded_country_df = pd.DataFrame(encoded_country, columns=loaded_encoder.get_feature_names_out(['country'])).astype(int)
    
    data = data.drop('country', axis=1)
    data = pd.concat([data, encoded_country_df], axis=1)
    
    # Gerekli tüm sütunların olduğundan emin olun
    expected_columns = model.feature_names_
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0

    # Tahmin
    predicted_price_log = model.predict(data)[0]
    predicted_price = np.exp(predicted_price_log)  # Inverse logaritma
    
    # Tahmin edilen fiyatı göster
    st.text_input("", value=f"${predicted_price:.2f}", disabled=True)
    st.markdown('</div>', unsafe_allow_html=True)
