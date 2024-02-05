import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).

    Parameters:
    - lon1, lat1: Longitude and latitude of the first point.
    - lon2, lat2: Longitude and latitude of the second point.

    Returns:
    - The distance between the two points in kilometers.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r


def get_stop_groups(data_frame):
    """
    Identifies groups of stops in the given DataFrame based on location data and stop duration,
    then calculates the percentage of total stop time for each stop.

    Parameters:
    - data_frame (DataFrame): A pandas DataFrame containing the columns 'lon', 'lat', and 'timestamp'.

    Returns:
    - DataFrame: A pandas DataFrame with stop groups, their duration, coordinates, percentage of total stop time,
        and the corresponding timestamp of when the stop started.
    """

    # Threshold for considering the car stopped, in kilometers (e.g., 0.05 km)
    stop_threshold = 0.05
    # Initialize lists for latitude, longitude, percentage, and timestamp
    percent = []
    timestamp = []
    latvalues = []
    longvalues = []

    # Calculate previous longitude and latitude
    data_frame['prev_lon'] = data_frame['lon'].shift()
    data_frame['prev_lat'] = data_frame['lat'].shift()

    # Calculate the distance using the Haversine formula
    data_frame['distance'] = data_frame.apply(lambda row:
                                              haversine(
                                                  row['lon'], row['lat'], row['prev_lon'], row['prev_lat'])
                                              if pd.notnull(row['prev_lon']) and pd.notnull(row['prev_lat']) else 0, axis=1)

    # Identify rows where the car is stopped
    data_frame['stopped'] = data_frame['distance'] <= stop_threshold

    # Group continuous stopped periods
    data_frame['stopped_group'] = (
        data_frame['stopped'] != data_frame['stopped'].shift()).cumsum()
    data_frame_stopped_group = data_frame[data_frame['stopped']].groupby('stopped_group')\
        .agg(start_time=('timestamp', 'min'), end_time=('timestamp', 'max'))

    # Convert timestamps to datetime objects
    stopped_groups_df = pd.DataFrame(data_frame_stopped_group)
    stopped_groups_df['start_time'] = pd.to_datetime(
        stopped_groups_df['start_time'])
    stopped_groups_df['end_time'] = pd.to_datetime(
        stopped_groups_df['end_time'])
    stopped_groups_df['time_diff'] = stopped_groups_df['end_time'] - \
        stopped_groups_df['start_time']
    sorted_df_stopped_group = stopped_groups_df.sort_values(
        by='time_diff', ascending=False)

    # Create a dictionary to store stopped groups with their duration
    list_stopped = {}
    for index, row in sorted_df_stopped_group.iterrows():
        time_diff = row['time_diff']
        list_stopped[index] = [time_diff]

    # Associate each stop group with its details
    dict_stopps = {}
    for _, row in data_frame.iterrows():
        if row['stopped_group'] not in dict_stopps:
            dict_stopps[row['stopped_group']] = row

    # Update list_stopped with stop details
    for key in dict_stopps.keys():
        if key in list_stopped.keys():
            list_stopped[key].append(dict_stopps[key])

    large_master_df = pd.DataFrame(list_stopped).T
    df_reset = large_master_df.reset_index()

    total_stopped = df_reset[0].sum()

    # Populate lists with data
    for i in range(len(df_reset)):
        timestamp.append(df_reset[1][i]['timestamp'])
        latvalues.append(df_reset[1][i]['lat'])
        longvalues.append(df_reset[1][i]['lon'])
        percent.append(df_reset[0][i]/total_stopped)

    # Update DataFrame with new columns
    df_reset['lat'] = latvalues
    df_reset['long'] = longvalues
    df_reset['percent'] = percent
    df_reset['timestamp'] = timestamp

    # Drop the now unnecessary column
    df_with_stops = df_reset.drop(df_reset.columns[2], axis=1)

    return df_with_stops


def get_location(lon, lat):
    """
    Retrieves the address corresponding to the given longitude and latitude.

    Parameters:
    - lon (str): The longitude of the location as a string.
    - lat (str): The latitude of the location as a string.

    Returns:
    dict: A dictionary containing the address components of the location.
    """
    geolocator = Nominatim(user_agent='geoapiExcises')
    location = geolocator.reverse(f'{lat},{lon}')
    address = location.raw['address']
    return address
