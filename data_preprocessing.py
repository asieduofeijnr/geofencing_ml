import pandas as pd
import numpy as np
import time
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
    - data_frame (DataFrame): A pandas DataFrame containing the columns 'lon', 'latitude', and 'ts'.

    Returns:
    - DataFrame: A pandas DataFrame with stop groups, their duration, coordinates, percentage of total stop time,
        and the corresponding ts of when the stop started.
    """

    # Threshold for considering the car stopped, in kilometers (e.g., 0.05 km)
    stop_threshold = 0.05
    # Initialize lists for latitude, longitude, percentage, and ts
    percent = []
    ts = []
    latvalues = []
    longvalues = []

    # Calculate previous longitude and latitude
    data_frame['prev_lon'] = data_frame['longitude'].shift()
    data_frame['prev_lat'] = data_frame['latitude'].shift()

    # Calculate the distance using the Haversine formula
    data_frame['distance'] = data_frame.apply(lambda row:
                                              haversine(
                                                  row['longitude'], row['latitude'], row['prev_lon'], row['prev_lat'])
                                              if pd.notnull(row['prev_lon']) and pd.notnull(row['prev_lat']) else 0, axis=1)

    # Identify rows where the car is stopped
    data_frame['stopped'] = data_frame['distance'] <= stop_threshold

    # Group continuous stopped periods
    data_frame['stopped_group'] = (
        data_frame['stopped'] != data_frame['stopped'].shift()).cumsum()
    data_frame_stopped_group = data_frame[data_frame['stopped']].groupby('stopped_group')\
        .agg(start_time=('ts', 'min'), end_time=('ts', 'max'))

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
        ts.append(df_reset[1][i]['ts'])
        latvalues.append(df_reset[1][i]['latitude'])
        longvalues.append(df_reset[1][i]['longitude'])
        percent.append(df_reset[0][i]/total_stopped)

    # Update DataFrame with new columns
    df_reset['latitude'] = latvalues
    df_reset['longitude'] = longvalues
    df_reset['percent'] = percent
    df_reset['timestamp'] = ts
    df_reset['id'] = data_frame['device_id']

    # Drop the now unnecessary column
    df_with_stops = df_reset.drop(df_reset.columns[2], axis=1)

    return df_with_stops


def get_location(lon, lat):
    """
    Retrieves the address corresponding to the given longitude and latitude.

    Parameters:
    - lon (str): The longitude of the location as a string.
    -'latitude (str): The latitude of the location as a string.

    Returns:
    dict: A dictionary containing the address components of the location.
    """
    time.sleep(1)  # to reduce the request made each time
    geolocator = Nominatim(user_agent='geoapiExcises')
    location = geolocator.reverse(f'{lat},{lon}')
    address = location.raw['address']
    return address


def get_clusters_and_frequency(dataframe, stop_threshold=0.2):
    """
    Identify clusters of stops and calculate their frequency and total waiting time.

    Parameters:
    - dataframe (DataFrame): DataFrame containing stop data
    - stop_threshold (float): Distance threshold to consider stops as part of the same cluster

    Returns:
    - DataFrame: DataFrame containing clusters of stops, their frequencies, and total waiting time
    """
    # Use a dictionary to store unique cluster centroids and their corresponding waiting times
    cluster_details = {}
    # List to store mean coordinates of each cluster centroid
    cluster_centroid_cords = []

    # Iterate over each row in the DataFrame
    for index, row_0 in dataframe.iterrows():
        distances = []
        # Calculate distances to other cluster centroids in the DataFrame
        for row in range(len(dataframe)):
            distance = haversine(
                row_0['longitude'], 
                row_0['latitude'], 
                dataframe.iloc[row]['longitude'], 
                dataframe.iloc[row]['latitude'])
            # Check if the distance is below the threshold
            if distance < stop_threshold:
                lat = dataframe.iloc[row]['latitude']
                lon = dataframe.iloc[row]['longitude']
                time_stopped = dataframe.iloc[row]['waiting_time']
                       
                timestamp = dataframe.iloc[row]['timestamp']
                row_tuple = (lat, lon)
                # Add the row to the dictionary if it's unique      
                if row_tuple not in cluster_details:
                    cluster_details[row_tuple] = {'waiting_time': time_stopped, 'timestamp': timestamp}
                    distances.append((lat, lon))
                else:
                    # Update the waiting time for existing cluster centroid
                    cluster_details[row_tuple]['waiting_time'] += time_stopped

        # Calculate the mean latitude and longitude for the cluster centroid
        if distances:
            mean_lat = np.mean([lat for lat, lon in distances])
            mean_lon = np.mean([lon for lat, lon in distances])
            # Extend the list with the mean coordinates for each existing coordinate in the group
            cluster_centroid_cords.extend([(mean_lat, mean_lon)] * len(distances))

    # Create a DataFrame from the unique cluster centroid details
    df = pd.DataFrame(cluster_details.values())

    # Add mean latitude and longitude columns to the DataFrame
    df['mean_latitude'] = [lat for lat, _ in cluster_centroid_cords[:len(df)]]
    df['mean_longitude'] = [lon for _, lon in cluster_centroid_cords[:len(df)]]

    # Create a tuple of mean latitude and longitude as a cluster centroid identifier
    df['cluster_centroid'] = list(zip(round(df['mean_latitude'], 6), round(df['mean_longitude'], 6)))

    # Assign cluster labels to each cluster centroid
    cluster_labels = {tuple(row): f'Cluster {i + 1}' for i, row in enumerate(
        df[['mean_latitude', 'mean_longitude']].drop_duplicates().itertuples(index=False))}
    df['clusters'] = df[['mean_latitude', 'mean_longitude']].apply(tuple, axis=1).map(cluster_labels)

    # Calculate the frequency of each cluster
    df['Frequency_of_stops'] = df.groupby('cluster_centroid')['cluster_centroid'].transform('count')

    # Drop duplicate cluster_centroids
    df = df.drop_duplicates(subset='cluster_centroid')
    df = df.drop(columns=['mean_latitude', 'mean_longitude'])

    return df


def estimate_proximity_and_closest_cluster(dataframe):
    """
    Calculate proximity distances between clusters and determine \
        the index of the closest cluster for each cluster.

    Parameters:
    - dataframe (DataFrame): DataFrame containing cluster data

    Returns:
    - DataFrame: DataFrame with new columns \
        'proximity_of_clusters' and 'closest_cluster_index'
    """
    # Initialize empty lists to store proximity distances and corresponding cluster indices
    proximity_of_clusters = []
    closest_clusters = []

    # Calculate distance between each pair of clusters
    for i in range(len(dataframe)):
        distances = []
        for j in range(len(dataframe)):
            if i != j:
                # Convert latitude and longitude to radians
                lat1, lon1 = dataframe.cluster_centroid.apply(
                    lambda x: x[0]).iloc[i], dataframe.cluster_centroid.apply(lambda x: x[1]).iloc[i]
                lat2, lon2 = dataframe.cluster_centroid.apply(
                    lambda x: x[0]).iloc[j], dataframe.cluster_centroid.apply(lambda x: x[1]).iloc[j]

                # Calculate haversine distance
                distance = haversine(lat1, lon1, lat2, lon2)
                distances.append(distance)
            else:
                distances.append(np.inf)  # Set distance to infinity for the same cluster (to be ignored)

        min_distance = min(distances)
        proximity_of_clusters.append(min_distance)  
        closest_cluster_index = distances.index(min_distance)  
        closest_clusters.append(closest_cluster_index)  

    # Add new columns 'proximity_of_clusters' and 'closest_cluster_index' to the DataFrame
    dataframe['proximity_of_clusters'] = proximity_of_clusters
    dataframe['closest_cluster_index'] = closest_clusters

    return dataframe


def create_geofence_target_label(dataframe, 
                                    waiting_time_threshold, 
                                    frequency_threshold, 
                                    proximity_threshold):
    """
    Assign labels to rows based on specified thresholds \
        for waiting time, frequency, and proximity.

    Parameters:
    - dataframe (DataFrame): DataFrame containing cluster data
    - waiting_time_threshold (pd.Timedelta): Threshold for waiting time
    - frequency_threshold (int): Threshold for frequency of stops
    - proximity_threshold (float): \
        Threshold for proximity distance (in kilometers)

    Returns:
    - DataFrame: DataFrame with a new column 'is_geofence' \
        indicating whether each row corresponds to a geofence
    """
    # Find the maximum Frequency_of_stops for each closest_cluster_index
    max_freq_per_index = dataframe.groupby(
        'closest_cluster_index')['Frequency_of_stops'].max()

    # Apply rules to create target variable
    def assign_label(row):
        if (row['waiting_time'] > waiting_time_threshold) and \
           (row['Frequency_of_stops'] == \
            max_freq_per_index[row['closest_cluster_index']]) and \
           (row['proximity_of_clusters'] < proximity_threshold) and \
           (row['Frequency_of_stops'] >= frequency_threshold):
            return 1
        else:
            return 0

    # Apply the function to create the target variable
    dataframe['is_geofence'] = dataframe.apply(assign_label, axis=1)

    return dataframe