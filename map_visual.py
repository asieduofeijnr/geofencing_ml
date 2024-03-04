import streamlit as st
import pandas as pd
import os
import folium
from streamlit_folium import st_folium
import pandas as pd
from truck import *
import folium
from itertools import cycle

# Streamlit Page
st.set_page_config(
    page_title="Geofence Recommender", page_icon="truck:", layout="wide"
)
st.title(
    "TruckX - Geofence Recommendation"
)


st.subheader("Visualize Truck Routes")

# Instructions and Overview on separate lines
st.write("1. Select Truck(s) to visualize their geofenced stops on the map.")
st.write("2. After selection, the map below will display each truck's stops with distinct colors for clarity.")
st.write("3. Use the 'Select Truck(s)' dropdown to choose and add trucks to the map.")
st.write("4. You can select multiple trucks to compare their stops.")

# Read Pickle
dfs = os.path.expanduser("~/data/geofence_data/merged_csv.pickle")
df_fleet = pd.read_pickle(dfs)
device_ids = df_fleet.device_id.unique()
truck_ids = {"Truck_" + str(count + 1): id for count,
             id in enumerate(device_ids)}

options = st.multiselect(
    ' ',
    list(truck_ids.keys()), label_visibility='hidden', placeholder="Select Truck(s)", key="truck_stops"
)

selected_truck_ids = []

# Check if any truck is selected and display the corresponding IDs in a list
if options:
    selected_truck_ids = [int(truck_ids[option]) for option in options]


if selected_truck_ids:
    colors = cycle(['blue', 'green', 'red', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
                    'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray'])  # Cycle through this list for different trucks

    first_truck = True

    for truck_id in selected_truck_ids:
        truck_data = FLEET([truck_id], df_fleet).get_data_frame()
        df = truck_data.get_stops().fleetstops_dataframe

        if first_truck:  # Initialize the map with the first truck's location
            start_latitude = df.iloc[0]['latitude']
            start_longitude = df.iloc[0]['longitude']
            m = folium.Map(
                location=[start_latitude, start_longitude], zoom_start=4)
            first_truck = False

        truck_color = next(colors)  # Get the next color from the cycle

        # Adding points for the truck's route
        for _, row in df.iterrows():
            folium.CircleMarker(location=[row['latitude'], row['longitude']],
                                radius=5,
                                color=truck_color,
                                fill=False,
                                fill_color=truck_color,
                                fill_opacity=0.6).add_to(m)

        # Adding lines to connect the points and show the route
        # folium.PolyLine(df[['latitude', 'longitude']].values,
            # color=truck_color).add_to(m)

    # Call to render Folium map in Streamlit
    st_data = st_folium(m, height=500, width=1200)

st.subheader("Visualize Clusters of Stops for Selected Truck(s)")

# Detailed instructions on separate lines
st.write("1. Use this section to select one or more trucks and define parameters to visualize how frequently trucks stop at various locations.")
st.write("2. Clusters are determined based on your specified 'Stop Threshold' and 'Hours' parameters, representing the minimum number of stops within a certain time frame to form a cluster.")
st.write("3. Adjust these parameters and select trucks to see their stop clusters on the map.")

options_clusters = st.multiselect(
    ' ',
    list(truck_ids.keys()), label_visibility='hidden', placeholder="Select Truck(s)", key="clusters"
)

selected_truck_ids_cluster = []


col1, col2 = st.columns(2)

with col1:
    stop_threshold = st.number_input(
        'Insert a number for stop threshold', min_value=0.2, value=0.2, key="stop_threshold")
with col2:
    hours = st.number_input(
        'Insert a number for hours', min_value=1, value=1, key='hours')


# Check if any truck is selected and display the corresponding IDs in a list
if options_clusters:
    selected_truck_ids_cluster = [
        int(truck_ids[options_cluster]) for options_cluster in options_clusters]


if selected_truck_ids_cluster:
    # Fixed color for the single truck

    # Get data for the selected truck
    truck_data = FLEET(selected_truck_ids_cluster, df_fleet).get_data_frame().get_stops().getClustersFrequency(
        stop_threshold=stop_threshold, time_threshold=pd.Timedelta(hours=hours))
    df = truck_data.clusters

    # Initialize the map with the truck's first location
    start_latitude = df.iloc[0]['cluster_centroid'][0]
    start_longitude = df.iloc[0]['cluster_centroid'][1]
    m = folium.Map(
        location=[start_latitude, start_longitude], zoom_start=5)

    # Adding points for the truck's route
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['cluster_centroid']
                      [0], row['cluster_centroid'][1]],
            radius=row['Frequency_of_stops'],
            color="red",
            fill=True,
            fill_color=truck_color,
            fill_opacity=0.6
        ).add_to(m)

    # Optionally, add lines to connect the points and show the route
    # if df.shape[0] > 1:  # Check if there are at least two points to connect
    #     folium.PolyLine(df['cluster_centroid'].tolist(), color=truck_color).add_to(m)

    # Call to render Folium map in Streamlit
    st_data = st_folium(m, height=500, width=1200)
