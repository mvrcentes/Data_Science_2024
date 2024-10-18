import streamlit as st
import numpy as np
from joblib import load

# Cargar el modelo Gradient Boosting
model = load('model_gb.joblib')

# Características esperadas por el modelo
features = ['area', 'rooms', 'bathroom', 'parking spaces', 'floor', 
            'animal_accepted', 'furniture_furnished', 'hoa (R$)', 
            'property tax (R$)', 'fire insurance (R$)', 
            'city_SaoPaulo', 'city_RioDeJaneiro', 'city_BeloHorizonte', 
            'city_PortoAlegre', 'city_Campinas']

# Función para predecir el precio de alquiler
def predict_price(inputs):
    input_data = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

# Configuración de la página
st.set_page_config(page_title="Estimador de Alquileres en Brasil", layout="wide")

# Estilo CSS personalizado
st.markdown(
    """
    <style>
    .main {background-color: #f7f9fc;}
    .stButton > button {background-color: #007ACC; color: white; font-size: 16px; padding: 10px 20px; border-radius: 8px; transition: 0.3s;}
    .stButton > button:hover {background-color: #005f99;}
    .result {font-size: 28px; font-weight: bold; color: #007ACC; margin-top: 20px;}
    .title {font-size: 36px; color: #007ACC; font-weight: 700;}
    .help {font-size: 16px; color: #555555;}
    </style>
    """,
    unsafe_allow_html=True
)

# Título y descripción de la aplicación
st.markdown("<div class='title'>🔍 Estimador de Precio de Alquiler en Brasil 🇧🇷</div>", unsafe_allow_html=True)
st.markdown("""
**Descubre el precio de alquiler estimado** en ciudades brasileñas ingresando los detalles de la propiedad.
""")

# Formulario de entrada de datos con columnas para mejor organización visual
with st.form(key='property_form'):
    st.markdown("### 🏠 Información de la Propiedad")
    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.number_input("Área en metros cuadrados:", min_value=10, max_value=1000, value=50)
        rooms = st.number_input("Número de habitaciones:", min_value=1, max_value=10, value=2)
        bathroom = st.number_input("Número de baños:", min_value=1, max_value=10, value=1)
        
    with col2:
        parking_spaces = st.number_input("Número de espacios de parqueo:", min_value=0, max_value=5, value=1)
        floor = st.number_input("Piso de la propiedad:", min_value=0, max_value=50, value=1)
        animal = st.selectbox("¿Permite mascotas?", options=["Sí", "No"])
        
    with col3:
        furniture = st.selectbox("¿Está amueblada?", options=["Sí", "No"])
        hoa = st.number_input("Cuota mensual de mantenimiento (HOA) en R$:", min_value=0, max_value=10000, value=500)
        property_tax = st.number_input("Impuesto de propiedad anual (R$):", min_value=0, max_value=10000, value=100)
    
    fire_insurance = st.number_input("Costo del seguro contra incendios (R$):", min_value=0, max_value=10000, value=50)
    city = st.selectbox("Ubicación de la propiedad:", options=["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Porto Alegre", "Campinas"])
    
    # Validación del formulario
    submitted = st.form_submit_button(label='📊 Calcular Precio de Alquiler')

    if submitted:
        # Codificación de variables categóricas
        animal_value = 1 if animal == "Sí" else 0
        furniture_value = 1 if furniture == "Sí" else 0
        city_encoded = [
            1 if city == "São Paulo" else 0, 
            1 if city == "Rio de Janeiro" else 0, 
            1 if city == "Belo Horizonte" else 0, 
            1 if city == "Porto Alegre" else 0, 
            1 if city == "Campinas" else 0
        ]
        
        # Crear la lista de características en el orden correcto
        inputs = [area, rooms, bathroom, parking_spaces, floor, animal_value, furniture_value, hoa, property_tax, fire_insurance] + city_encoded
        
        # Realizar la predicción
        predicted_price = predict_price(inputs)
        
        # Mostrar la predicción de una manera más visual
        st.markdown("<div class='result'>✨ Precio estimado de alquiler: **R$ {:,.2f}** / mes ✨</div>".format(predicted_price), unsafe_allow_html=True)

# Barra lateral de ayuda
st.sidebar.title("📌 Ayuda y Soporte")
st.sidebar.info("""
- Completa el formulario con los datos de la propiedad.
- Haz clic en **Calcular Precio de Alquiler** para obtener una estimación.
- Si tienes dudas, consulta esta sección o contacta al soporte.
""")
