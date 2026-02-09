import joblib
import streamlit as st
import numpy as np
import pandas as pd
import random

## Load trained model
model = joblib.load("energy_gbt_model.pkl")

## Streamlit app
st.title("Household Appliances Energy Usage Prediction")

st.caption("Wh stands for watt-hour, which is power (watts) multiplied by time (hours) and this is to measure the total amount of electricity an appliance consumes over an hour.")
st.caption("mmHg means millimetres of mercury, which is a manometric unit of pressure defined as the pressure exerted by a column of mercury 1 millimeter high.")

col1, col2 = st.columns(2, border=True)
col3, col4 = st.columns(2, border=True)
col5, col6 = st.columns(2, border=True)
col7, col8 = st.columns(2, border=True)

## User inputs
with col1:
    st.header("Kitchen")
    kitchen_temp = st.slider("Select Kitchen Temperature (Celsius)", min_value=15, max_value=35, value=24)
    kitchen_humid = st.slider("Select Kitchen Humidity (%)", min_value=20, max_value=70, value=40)

with col2:
    st.header("Living Room")
    living_rm_temp = st.slider("Select Living Room Temperature (Celsius)", min_value=15, max_value=35, value=24)
    living_rm_humid = st.slider("Select Living Room Humidity (%)", min_value=20, max_value=60, value=40)

with col3:
    st.header("Laundry Room")
    laundry_temp = st.slider("Select Laundry Room Temperature (Celsius)", min_value=15, max_value=35, value=24)
    laundry_humid = st.slider("Select Laundry Room Humidity (%)", min_value=20, max_value=60, value=40)

with col4:
    st.header("Office Room")
    office_temp = st.slider("Select Office Room Temperature (Celsius)", min_value=10, max_value=35, value=24)
    office_humid = st.slider("Select Office Room Humidity (%)", min_value=20, max_value=60, value=40)

with col5:
    st.header("Bathroom")
    bathroom_temp = st.slider("Select Bathroom Temperature (Celsius)", min_value=10, max_value=30, value=24)
    bathroom_humid = st.slider("Select Bathroom Humidity (%)", min_value=20, max_value=100, value=40)

with col6:
    st.header("Outdoor Area")
    outdoor_temp = st.slider("Select Outdoor Area Temperature (Celsius)", min_value=-10, max_value=35, value=20)
    outdoor_humid = st.slider("Select Outdoor Area Humidity (%)", min_value=0, max_value=100, value=40)

with col7:
    st.header("Ironing Room")
    ironing_temp = st.slider("Select Ironing Room Temperature (Celsius)", min_value=10, max_value=30, value=24)
    ironing_humid = st.slider("Select Ironing Room Humidity (%)", min_value=20, max_value=60, value=40)

with col8:
    st.header("Teenage Bedroom")
    teen_bedrm_temp = st.slider("Select Teenage Bedroom Temperature (Celsius)", min_value=10, max_value=30, value=24)
    teen_bedrm_humid = st.slider("Select Teenage Bedroom Humidity (%)", min_value=20, max_value=65, value=40)

with st.container(border=True):
    st.header("Parents Bedroom")
    parents_bedrm_temp = st.slider("Select Parents Bedroom Temperature (Celsius)", min_value=10, max_value=30, value=20)
    parents_bedrm_humid = st.slider("Select Parents Bedroom Humidity (%)", min_value=20, max_value=65, value=40)

with st.container(border=True):
    st.header("Lights")
    lights_usage = st.slider("Select Energy Usage of Lights (Wh)", min_value=0, max_value=100, value=50)

with st.container(border=True):
    st.header("Climate Data from Closest Airport")
    airport_temp = st.slider("Select Temperature from the Closest Airport (Celsius)", min_value=-10, max_value=30, value=10)
    airport_pressure = st.slider("Select Pressure from the Closest Airport (mmHg)", min_value=700, max_value=800, value=750)
    airport_humid = st.slider("Select Humidity from the Closest Airport (%)", min_value=20, max_value=100, value=50)
    airport_windspeed = st.slider("Select Wind Speed from the Closest Airport (km/h)", min_value=0, max_value=20, value=10)
    airport_visibility = st.slider("Select Visibility from the Closest Airport (km)", min_value=0, max_value=70, value=30)

## Randomly generate data for 3 unknown variables (1 unexplained, 2 random variables) so user doesn't have to input them
tdewpoint = random.uniform(-10, 15)
rv1 = random.uniform(0, 50)
rv2 = random.uniform(0, 50)

## Predict button
if st.button("Predict appliances energy usage"):

    ## Create dict for input features
    input_data = {
        'lights': lights_usage,
        'T1': kitchen_temp,
        'RH_1': kitchen_humid,
        'T2': living_rm_temp,
        'RH_2': living_rm_humid,
        'T3': laundry_temp,
        'RH_3': laundry_humid,
        'T4': office_temp,
        'RH_4': office_humid,
        'T5': bathroom_temp,
        'RH_5': bathroom_humid,
        'T6': outdoor_temp,
        'RH_6': outdoor_humid,
        'T7': ironing_temp,
        'RH_7': ironing_humid,
        'T8': teen_bedrm_temp,
        'RH_8': teen_bedrm_humid,
        'T9': parents_bedrm_temp,
        'RH_9': parents_bedrm_humid,
        'T_out': airport_temp,
        'Press_mm_hg': airport_pressure,
        'RH_out': airport_humid,
        'Windspeed': airport_windspeed,
        'Visibility': airport_visibility,
        'Tdewpoint': tdewpoint,
        'rv1': rv1,
        'rv2': rv2
    }

    ## Convert input data to a DataFrame
    df_input = pd.DataFrame({
        'lights': [lights_usage],
        'T1': [kitchen_temp],
        'RH_1': [kitchen_humid],
        'T2': [living_rm_temp],
        'RH_2': [living_rm_humid],
        'T3': [laundry_temp],
        'RH_3': [laundry_humid],
        'T4': [office_temp],
        'RH_4': [office_humid],
        'T5': [bathroom_temp],
        'RH_5': [bathroom_humid],
        'T6': [outdoor_temp],
        'RH_6': [outdoor_humid],
        'T7': [ironing_temp],
        'RH_7': [ironing_humid],
        'T8': [teen_bedrm_temp],
        'RH_8': [teen_bedrm_humid],
        'T9': [parents_bedrm_temp],
        'RH_9': [parents_bedrm_humid],
        'T_out': [airport_temp],
        'Press_mm_hg': [airport_pressure],
        'RH_out': [airport_humid],
        'Windspeed': [airport_windspeed],
        'Visibility': [airport_visibility],
        'Tdewpoint': [float(tdewpoint)],
        'rv1': [float(rv1)],
        'rv2': [float(rv2)]
    })
    
    # df_input = df_input.to_numpy()

    # df_input = df_input.reindex(columns = model.feature_names_in_,
    #                             fill_value=0)

    ## Predict
    y_unseen_pred = model.predict(df_input)[0]
    st.success(f"Predicted Energy Usage: {y_unseen_pred:,.2f} Wh")

# Page design
# st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background: url("");
#         background-size: cover
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
# )