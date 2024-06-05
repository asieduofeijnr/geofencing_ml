import streamlit as st
import pandas as pd
import folium
import os


from data_preprocessing import recommendation_algo
from data_preprocessing import create_cluster_map
from streamlit_folium import st_folium

from itertools import cycle
from truck import *


######################################
# Function to load data, if not already in session state


def load_data():
    if 'df_fleet' not in st.session_state:
        dfs = os.path.expanduser(
            "original.csv")
        st.session_state.df_fleet = pd.read_csv(dfs)


# def load_data_cluster():
#     if 'df_fleet_cluster' not in st.session_state:
#         dfs_cluster = os.path.expanduser(
#             "~/data/geofence_data/fleet_dataframe.pickle")
#         st.session_state.df_fleet_cluster = pd.read_pickle(dfs_cluster)

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
    st.image("truckx.png", width=200)

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
# load_data_cluster()

# Extracting unique device IDs and mapping them to truck IDs

device_ids = st.session_state.df_fleet.device_id.unique()
truck_ids = {f"Truck_{count + 1}": id for count, id in enumerate(device_ids)}
truck_ids["All Trucks"] = "all"

options = st.multiselect(
    label="Select Truck(s) to Visualize:",
    options=list(truck_ids.keys()),  # Convert keys to a list
    placeholder="Select Truck(s)",
    key="truck_stops",
    help="Choose one or more trucks to display their routes."
)

selected_truck_ids = []

if "All Trucks" in options:
    # Select all device IDs if "All Trucks" is selected
    selected_truck_ids = device_ids.tolist()
else:
    selected_truck_ids = [int(truck_ids[option])
                          for option in options if option != "All Trucks"]


colh, cold = st.columns(2)

with colh:
    hours = st.number_input(
        'Hours (Time Parked)', min_value=1, value=1, key='hours')
with cold:
    days = st.number_input(
        'Days (for dataset)', min_value=1, value=90, key='days')


if selected_truck_ids:
    colors = cycle(['blue', 'green', 'red', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
                    'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray'])  # Cycle through this list for different trucks

    first_truck = True

    for truck_id in selected_truck_ids:
        truck_data = FLEET(
            [truck_id],  st.session_state.df_fleet).get_data_frame(n_days=days)

        df = truck_data.get_stops(
            time_threshold=pd.Timedelta(hours=hours)).fleetstops_dataframe

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


stop_threshold = st.number_input(
    'Insert a number for stop threshold', min_value=0.2, value=0.2, key="stop_threshold")


if selected_truck_ids:
    # Get data for the selected truck
    cluster_truck_data = FLEET(selected_truck_ids, st.session_state.df_fleet).get_data_frame(n_days=days).get_stops(time_threshold=pd.Timedelta(hours=hours)).getClustersFrequency(
        stop_threshold=stop_threshold).get_proximity_df()

    df = cluster_truck_data.clusters

# ------>DO NOT NEED<-------
    # df = st.session_state.df_fleet_cluster
# ------>DO NOT NEED<-------

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

st.write('----')

# Streamlit app layout
st.sidebar.title('Map Use Case Selector')
# Setting 'reset' as the default value directly in the radio widget
use_case = st.sidebar.radio(
    "Select Use Case:", ['Optimization', 'Tracking', 'Safety', 'Reset'], index=3)

# Mapping use cases to colors
use_case_to_color = {
    'Optimization': 'red',
    'Tracking': 'blue',
    'Safety': 'green',
    'Reset': 'purple'
}

case_color = use_case_to_color[use_case]
unique_ids = 32

proximity_df = cluster_truck_data.proximity_df


st.session_state.final_df = recommendation_algo(
    proximity_df, unique_ids, size='small', use_case=use_case)

map_object = create_cluster_map(st.session_state.final_df, case_color)
st.components.v1.html(folium.Map._repr_html_(
    map_object), height=500, width=1200)
