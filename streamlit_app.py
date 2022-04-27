import pandas as pd
import numpy as np
import streamlit as st
from streamlit_folium import folium_static
from PIL import Image
import xgboost as xgb
import matplotlib.cm as cm
import folium
import matplotlib.colors as colors

xgb1 = xgb.Booster()
xgb1.load_model("xgb.model")

def main():
    
    st.sidebar.title('Select what you would like to do')
    app_mode = st.sidebar.selectbox('Select', ['Map', 'Prediction'])
    
    if app_mode == 'Map':
        show_map()
    elif app_mode == 'Prediction':
        prediction_page()
        
        
def prediction(gps_height, installer_cat, longitude, latitude,
       basin, region, district_code, lga, ward, subvillage,
       permit,  extraction_type_group, management,
       water_quality, quantity, source, waterpoint_type, funder_cat,
       population_log, age, decade):
    
    test = [[gps_height, installer_cat, longitude, latitude,
       basin, region, district_code, lga, ward, subvillage,
       permit,  extraction_type_group, management,
       water_quality, quantity, source, waterpoint_type, funder_cat,
       population_log, age, decade]]
    
    prediction = xgb1.predict(xgb.DMatrix(test))
    print(prediction)
    return prediction

def prediction_page():
    "# Pump Functionality Prediction"
    
    gps_height =  st.number_input('GPS Height',format="%.2f")
    installer_cat = st.number_input('Installer Category')
    longitude = st.number_input('Longitude',format="%.2f")
    latitude = st.number_input('Latitude',format="%.2f")
    basin = st.number_input('Basin Category')
    region = st.number_input('Region Category')
    district_code = st.number_input('District Code')
    lga = st.number_input('LGA Category')
    ward = st.number_input('Ward Category')
    subvillage = st.number_input('Subvillage Category')
    permit = st.number_input('Permit - 1 or 0')
    decade = st.number_input('Decade Category')
    extraction_type_group = st.number_input('Extraction Type')
    management = st.number_input('Management Category')
    water_quality = st.number_input('Water Quality')
    quantity = st.number_input('Water Quantity Category')
    source = st.number_input('Source Type')
    waterpoint_type = st.number_input('Waterpoint Type')
    funder_cat = st.number_input('Funder Category')
    population_log = st.number_input('Population log10')
    age = st.number_input('Age',format="%.2f")
    result = ""
    
    if st.button("Predict"):
        result = prediction(gps_height, installer_cat, longitude, latitude,
       basin, region, district_code, lga, ward, subvillage,
       permit,  extraction_type_group, management,
       water_quality, quantity, source, waterpoint_type, funder_cat,
       population_log, age, decade)
        st.success("The output is {}, where 0 = functional, 1 = non functional, 2 = functional needs repair".format(result))

        
latitude = -6.3728253
longitude = 34.8924826

xgb_m = pd.read_csv('XGB(2).csv')
etc = pd.read_csv('etc(1).csv')
knn = pd.read_csv('submission (13).csv')

def show_map():
    st.title = "Pump Functionality Prediction Mapping"
    "# Pump Functionality Mapping"
    "This is the interactive interface for the COMP6940 project. \
    Select which model you would like to use to for predictions. \
    The results shown are the predicted values for each waterpoint pump \
    in the test value dataset."
    
    modelselect = st.selectbox('Select Model', 
                               ('kNN','XGBoost', 'ExtraTrees'))
    
    if modelselect == 'XGBoost':
        m = folium.Map(location=[latitude, longitude], zoom_start=8)
            
        x = np.arange(3)
        ys = [i + x + (i*x)**2 for i in range(3)]
        colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
        rainbow = [colors.rgb2hex(i) for i in colors_array]

        # add markers to the map
        markers_colors = []
        for lat, lon, poi, id, cluster in zip(xgb_m.latitude, xgb_m.longitude, xgb_m.status_group, xgb_m.id, xgb_m.label):
            label = folium.Popup(str(poi) + ' ID ' + str(id), parse_html=True)
            folium.CircleMarker(
                [lat, lon],
                radius=5,
                popup=label,
                color=rainbow[cluster-1],
                fill=True,
                fill_color=rainbow[cluster-1],
                fill_opacity=0.7).add_to(m)

        folium_static(m)
    
    elif modelselect == 'ExtraTrees':
        m = folium.Map(location=[latitude, longitude], zoom_start=8)
        
        x = np.arange(3)
        ys = [i + x + (i*x)**2 for i in range(3)]
        colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
        rainbow = [colors.rgb2hex(i) for i in colors_array]

        # add markers to the map
        markers_colors = []
        for lat, lon, poi, id,  cluster in zip(etc.latitude, etc.longitude, etc.status_group, etc.id, etc.label):
            label = folium.Popup(str(poi) + ' ID ' + str(id), parse_html=True)
            folium.CircleMarker(
                [lat, lon],
                radius=5,
                popup=label,
                color=rainbow[cluster-1],
                fill=True,
                fill_color=rainbow[cluster-1],
                fill_opacity=0.7).add_to(m)

        folium_static(m)
        
    elif modelselect == 'kNN':
        m = folium.Map(location=[latitude, longitude], zoom_start=8)
        
        x = np.arange(3)
        ys = [i + x + (i*x)**2 for i in range(3)]
        colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
        rainbow = [colors.rgb2hex(i) for i in colors_array]

        # add markers to the map
        markers_colors = []
        for lat, lon, poi, id, cluster in zip(knn.latitude, knn.longitude, knn.status_group, knn.id, knn.label):
            label = folium.Popup(str(poi) + ' ID ' + str(id), parse_html=True)
            folium.CircleMarker(
                [lat, lon],
                radius=5,
                popup=label,
                color=rainbow[cluster-1],
                fill=True,
                fill_color=rainbow[cluster-1],
                fill_opacity=0.7).add_to(m)

        folium_static(m)
        
    
    
if __name__=='__main__':
    main()