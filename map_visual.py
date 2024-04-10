import streamlit as st
import pandas as pd
from data_preprocessing import recommendation_algo
import os
import folium
from streamlit_folium import st_folium
import pandas as pd
from truck import *
import folium
from shapely.geometry import Polygon
from itertools import cycle

######################################
# Function to load data, if not already in session state


def load_data():
    if 'df_fleet' not in st.session_state:
        dfs = os.path.expanduser("~/data/geofence_data/merged_csv.pickle")
        st.session_state.df_fleet = pd.read_pickle(dfs)


def load_data_cluster():
    if 'df_fleet_cluster' not in st.session_state:
        dfs_cluster = os.path.expanduser(
            "~/data/geofence_data/fleet_dataframe.pickle")
        st.session_state.df_fleet_cluster = pd.read_pickle(dfs_cluster)

######################################


st.set_page_config(
    page_title="Geofence Recommender",
    page_icon=":truck:",
    layout="wide"
)


# Using columns and markdown for a cleaner, more structured layout
col1, col2 = st.columns([1, 3])

with col1:
    # Using markdown with HTML to center the image within the column
    st.image("images/truckx.png", width=200)

with col2:
    # Using markdown with HTML to center the title within the column
    st.markdown(
        f"""
        <div style="text-align: center;">
            <h1>TruckX - Geofence Recommendation</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("***")

# Improving the instructions and overview for better readability and aesthetics
st.markdown("""
    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 10px;">
        <h4>Visualize Truck Routes:</h4>
        <ol>
            <li>Select Truck(s) to visualize their geofenced stops on the map.</li>
            <li>After selection, the map below will display each truck's stops with distinct colors for clarity.</li>
            <li>Use the 'Select Truck(s)' dropdown to choose and add trucks to the map.</li>
            <li>You can select multiple trucks to compare their stops.</li>
        </ol>
    </div>
""", unsafe_allow_html=True)

# Read Pickle for truck data
load_data()
load_data_cluster()

# Extracting unique device IDs and mapping them to truck IDs
device_ids = st.session_state.df_fleet.device_id.unique()
truck_ids = {f"Truck_{count + 1}": id for count,
             id in enumerate(device_ids)}

options = st.multiselect(
    label="Select Truck(s) to Visualize:",
    options=truck_ids.keys(),
    placeholder="Select Truck(s)",
    key="truck_stops",
    help="Choose one or more trucks to display their routes."
)

selected_truck_ids = []

if options:
    selected_truck_ids = [int(truck_ids[option]) for option in options]

if selected_truck_ids:
    colors = cycle(['blue', 'green', 'red', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
                    'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray'])  # Cycle through this list for different trucks

    first_truck = True

    for truck_id in selected_truck_ids:
        truck_data = FLEET(
            [truck_id],  st.session_state.df_fleet).get_data_frame()
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


# Improving the instructions and overview for better readability and aesthetics
st.markdown("""
    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 10px;">
        <h4>Visualize Truck Clusters:</h4>
        <ol>
            <li>Select Truck(s) stop threshold (the minimum distance from another truck).</li>
            <li>Select Truck(s) hours stopped (the minimum hours a truck has stopped for)</li>
        </ol>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    stop_threshold = st.number_input(
        'Insert a number for stop threshold', min_value=0.2, value=0.2, key="stop_threshold")
with col2:
    hours = st.number_input(
        'Insert a number for hours', min_value=1, value=1, key='hours')


if selected_truck_ids:
    # Get data for the selected truck
    # truck_data = FLEET(selected_truck_ids, st.session_state.df_fleet).get_data_frame().get_stops().getClustersFrequency(
    #     stop_threshold=stop_threshold, time_threshold=pd.Timedelta(hours=hours))
    # df = truck_data.clusters

    df = st.session_state.df_fleet_cluster

    # Initialize the map with the truck's first location
    start_latitude = float(df.iloc[0]['cluster_centroid'][0])
    start_longitude = float(df.iloc[0]['cluster_centroid'][1])
    m = folium.Map(
        location=[start_latitude, start_longitude], zoom_start=5)

    # Adding points for the truck's route
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['cluster_centroid']
                      [0], row['cluster_centroid'][1]],
            radius=(row['Frequency_of_stops']/10),
            color="red",
            tooltip=row['clusters'],
            fill=True,
            fill_color='red',
            fill_opacity=0.6
        ).add_to(m)

    # Optionally, add lines to connect the points and show the route
    # if df.shape[0] > 1:  # Check if there are at least two points to connect
    #     folium.PolyLine(df['cluster_centroid'].tolist(), color=truck_color).add_to(m)

    # Call to render Folium map in Streamlit
    st_data = st_folium(m, height=500, width=1200)

######################################

# row1, row2, row3 = st.columns([1, 4, 1])

# with row2:

st.markdown("""
    <div style="background-color: #f0f2f6; padding: 10px; border-radius: 10px;">
        <h4>Visualize Interesting Truck Clusters based of parameters:</h4>
        <ol>
            <li>Select Truck(s) homogenous state (the minimum distance from another truck).</li>
            <li>Select Truck(s) hours stopped (the minimum hours a truck has stopped for)</li>
        </ol>
    </div>
""", unsafe_allow_html=True)


# homogenous = st.slider("Homogenous", min_value=min(
#     df.homogeneous), max_value=max(df.homogeneous), key='homogenous')

homogenous = st.slider("Homogenous", min_value=0.0,
                       max_value=0.9, key='homogenous')

num_stops = st.number_input(
    f'Insert a number for truck stops in Cluster, MAX_stops = {max(df.Frequency_of_stops)}', min_value=1, value=50, key='num_stops')

percentage_of_trucks = st.slider(
    "Percentage of trucks in cluster", min_value=0.0, max_value=0.9, key='percentage_of_trucks')

# avg_wait_time = st.number_input(
#     f'Insert a number for Average wait time in hours, MAX_hours = {max(df.avg_wait_time_hours)} ', min_value=1, value=1, key='avg_wait_time')

avg_wait_time = st.number_input(
    f'Insert a number for Average wait time in hours, MAX_hours = 100 ', min_value=1, value=1, key='avg_wait_time')


unique_ids = st.number_input(
    'Insert a number for unique_ids', min_value=1, value=31, key="unique_ids")


st.session_state.final_df = recommendation_algo(
    df, homogenous, num_stops, percentage_of_trucks, avg_wait_time, unique_ids)

start_latitude = float(
    st.session_state.final_df.iloc[0]['cluster_centroid'][0])
start_longitude = float(
    st.session_state.final_df.iloc[0]['cluster_centroid'][1])
m_cluster = folium.Map(
    location=[start_latitude, start_longitude], zoom_start=5)

# Adding points for the truck's route
for _, row in st.session_state.final_df.iterrows():
    folium.CircleMarker(
        location=[row['cluster_centroid']
                  [0], row['cluster_centroid'][1]],
        radius=10,
        color="purple",
        popup=f'''Cluster: {row['clusters']} <br> 
                  No Trucks: {len(row['num_trucks_stops'])} <br>
                  Total wait time : {row['total_wait_time']} <br>
                  Average wait time : {row['avg_wait_time']} <br>
                  Frequency of Stops: {row['Frequency_of_stops']} <br>
                  Truck Stops: {row['num_trucks_stops']}''',
        fill=True,
        fill_color='purple',
        fill_opacity=0.6
    ).add_to(m_cluster)

for i in st.session_state.final_df.Points:
    # Find the minimum and maximum latitude and longitude values
    min_lat = min(coord[0] for coord in i)
    max_lat = max(coord[0] for coord in i)
    min_lon = min(coord[1] for coord in i)
    max_lon = max(coord[1] for coord in i)

    boundary_coordinates = [
        (min_lat, min_lon),
        (min_lat, max_lon),
        (max_lat, max_lon),
        (max_lat, min_lon),
        (min_lat, min_lon)
    ]
    # Create a polygon to highlight the cluster
    area = Polygon(boundary_coordinates).area * 10**6

    folium.Polygon(locations=boundary_coordinates, color='red', fill=True,
                   fill_color='red', fill_opacity=0.5, weight=0.2,
                   popup=folium.Popup(f'Area: {area:2f} sq.km', parse_html=True)).add_to(m_cluster)

    for coord in i:
        folium.CircleMarker(location=coord, radius=2, color='blue',
                            fill=True, fill_color='blue').add_to(m_cluster)


# Optionally, add lines to connect the points and show the route
# if df.shape[0] > 1:  # Check if there are at least two points to connect
#     folium.PolyLine(df['cluster_centroid'].tolist(), color=truck_color).add_to(m)

# Call to render Folium map in Streamlit
st_data_cluster = st_folium(m_cluster, height=500, width=1200, key="m_cluster")
