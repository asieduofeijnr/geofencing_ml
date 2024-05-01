import pandas as pd
import numpy as np
import folium
import time

from geopy.geocoders import Nominatim
from shapely.geometry import Polygon
from scipy.stats import entropy
from collections import Counter

geolocator = Nominatim(user_agent='geoapiExcises')


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


def get_location(coordinates):
    """
    Retrieves the address corresponding to the given longitude and latitude.

    Parameters:
    - lon (str): The longitude of the location as a string.
    -'latitude (str): The latitude of the location as a string.

    Returns:
    dict: A dictionary containing the address components of the location.
    """
    time.sleep(1)  # to reduce the request made each time

    location = geolocator.reverse(f'{coordinates[0]},{coordinates[1]}')
    address = location.raw['address']
    return address


def count_truck_occurrences(coordinates, waiting_time):
    # Extract truck IDs from each tuple using list comprehension
    truck_wait_times_sum = {}
    truck_ids = [coord[-1] for coord in coordinates]
    wait_times = list(zip(truck_ids, waiting_time))
    truck_id_counts = Counter(truck_ids)
    for truck_id, wait_time in wait_times:
        if truck_id in truck_wait_times_sum:
            truck_wait_times_sum[truck_id] += wait_time
        else:
            truck_wait_times_sum[truck_id] = wait_time
    return truck_id_counts, truck_wait_times_sum


def get_clusters_and_frequency(dataframe, stop_threshold=0.5):
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
    visited = []

    # Iterate over each row in the DataFrame
    for index, row_0 in dataframe.iterrows():
        for row in range(len(dataframe)):
            id = (dataframe.iloc[row]['latitude'],
                  dataframe.iloc[row]['longitude'])
            if id not in visited:
                distance = haversine(
                    row_0['longitude'],
                    row_0['latitude'],
                    dataframe.iloc[row]['longitude'],
                    dataframe.iloc[row]['latitude'])
                # Check if the distance is below the threshold
                if distance < stop_threshold:
                    lat = dataframe.iloc[row]['latitude']
                    lon = dataframe.iloc[row]['longitude']
                    # Assuming 'id' column has truck IDs
                    truck_id = dataframe.iloc[row]['id']
                    time_stopped = dataframe.iloc[row]['waiting_time']
                    timestamp = dataframe.iloc[row]['timestamp']

                    # Key for the cluster based on the first point that forms the cluster
                    cluster_key = (
                        index, row_0['latitude'], row_0['longitude'])

                    if cluster_key not in cluster_details:
                        cluster_details[cluster_key] = {
                            'coords': [(lat, lon, truck_id)],
                            'waiting_time': [time_stopped],
                            'arrival_time': [timestamp],
                            'departure_time': [timestamp+time_stopped],
                            'total_wait_time': time_stopped
                        }
                    else:
                        cluster_details[cluster_key]['coords'].append(
                            (lat, lon, truck_id))
                        cluster_details[cluster_key]['waiting_time'].append(
                            time_stopped)
                        cluster_details[cluster_key]['total_wait_time'] += time_stopped
                        cluster_details[cluster_key]['arrival_time'].append(
                            timestamp)
                        cluster_details[cluster_key]['departure_time'].append(
                            timestamp+time_stopped)

                    visited.append(id)

    # Process the cluster details
    cluster_data = []
    for key, details in cluster_details.items():
        mean_lat = np.mean([lat for lat, _, _ in details['coords']])
        mean_lon = np.mean([lon for _, lon, _ in details['coords']])
        num_trucks = count_truck_occurrences(
            details['coords'], details['waiting_time'])
        total_stops = sum(num_trucks[0].values())
        cluster_data.append({
            'arrival_time': details['arrival_time'],
            'departure_time': details['departure_time'],
            'waiting_time': details['waiting_time'],
            'total_wait_time': details['total_wait_time'],
            'mean_latitude': mean_lat,
            'mean_longitude': mean_lon,
            'coordinates': details['coords'],
            'num_trucks_stops': num_trucks[0],
            'num_trucks_waittime': num_trucks[1],
            'avg_wait_time_per_truck':  dict(zip(num_trucks[0], [num_trucks[1][i]/num_trucks[0][i] for i in num_trucks[0]])),
            'avg_wait_time': np.mean([num_trucks[1][i]/num_trucks[0][i] for i in num_trucks[0]]),
            'Frequency_of_stops': total_stops
        })

    # Create a DataFrame from the cluster data
    df = pd.DataFrame(cluster_data)

    # Create a tuple of mean latitude and longitude as a cluster centroid identifier
    df['cluster_centroid'] = list(
        zip(df['mean_latitude'], df['mean_longitude']))

    # Assign cluster labels to each cluster centroid
    cluster_labels = {tuple(row): f'Cluster {i + 1}' for i, row in enumerate(
        df[['mean_latitude', 'mean_longitude']].drop_duplicates().itertuples(index=False))}
    df['clusters'] = df[['mean_latitude', 'mean_longitude']].apply(
        tuple, axis=1).map(cluster_labels)

    # Get location of coordinates
    # df['location_address'] = df['cluster_centroid'].apply(get_location)
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
        distances = {}
        for j in range(len(dataframe)):
            cluster_j = dataframe.clusters.iloc[j]
            if i != j:
                # Convert latitude and longitude to radians
                lat1, lon1 = dataframe.cluster_centroid.apply(
                    lambda x: x[0]).iloc[i], dataframe.cluster_centroid.apply(lambda x: x[1]).iloc[i]
                lat2, lon2 = dataframe.cluster_centroid.apply(
                    lambda x: x[0]).iloc[j], dataframe.cluster_centroid.apply(lambda x: x[1]).iloc[j]

                # Calculate haversine distance but make sure each value is float
                distance = haversine(lat1, lon1, lat2, lon2)
                if cluster_j not in distances:
                    distances[cluster_j] = distance

            else:
                # Set distance to infinity for the same cluster (to be ignored)
                distances[cluster_j] = np.inf

        min_distance = min(distances.values())

        # Append the cluster aand its distance to the list
        proximity_of_clusters.append(min_distance)

        # Apeend the closest cluster name to the list
        closest_clusters.append(
            list(distances.keys())[list(distances.values()).index(min_distance)])

    # Add new columns 'proximity_of_clusters' and 'closest_cluster_index' to the DataFrame
    dataframe['proximity_of_clusters'] = proximity_of_clusters
    dataframe['closest_cluster'] = closest_clusters

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
        'closest_cluster')['Frequency_of_stops'].max()

    # Apply rules to create target variable
    def assign_label(row):
        if (row['total_wait_time'] > waiting_time_threshold) and \
           (row['Frequency_of_stops'] ==
            max_freq_per_index[row['closest_cluster']]) and \
           (row['proximity_of_clusters'] < proximity_threshold) and \
           (row['Frequency_of_stops'] >= frequency_threshold):
            return 1
        else:
            return 0

    # Apply the function to create the target variable
    dataframe['is_geofence'] = dataframe.apply(assign_label, axis=1)

    return dataframe


def recommendation_algo(df, unique_ids, use_case, size):
    bounds = {
        'Optimization': {
            'lower_bounds': {
                'Frequency_of_stops': 2,
                'homogeneous': 0,
                'num_ids': 0,
                'avg_wait_time_hours': 3
            },
            'upper_bounds': {
                'Frequency_of_stops': float('inf'),
                'homogeneous': float('inf'),
                'num_ids': 1,
                'avg_wait_time_hours': 5
            }
        },
        'Tracking': {
            'lower_bounds': {
                'Frequency_of_stops': 20,
                'homogeneous': 0.2,
                'num_ids': 0.2,
                'avg_wait_time_hours': 1
            },
            'upper_bounds': {
                'Frequency_of_stops': float('inf'),
                'homogeneous': float('inf'),
                'num_ids': float('inf'),
                'avg_wait_time_hours': float('inf')
            }
        },
        'Safety': {
            'lower_bounds': {
                'Frequency_of_stops': 1,
                'homogeneous': 0,
                'num_ids': 0,
                'avg_wait_time_hours': 10
            },
            'upper_bounds': {
                'Frequency_of_stops': float('inf'),
                'homogeneous': float('inf'),
                'num_ids': 0.1,
                'avg_wait_time_hours': float('inf')
            }
        },
        'Reset': {
            'lower_bounds': {
                'Frequency_of_stops': 0,
                'homogeneous': 0,
                'num_ids': 0,
                'avg_wait_time_hours': 0
            },
            'upper_bounds': {
                'Frequency_of_stops': float('inf'),
                'homogeneous': float('inf'),
                'num_ids': 0,
                'avg_wait_time_hours': float('inf')
            }
        }
    }

    def filter_threshold(df, bounds, use_case, size):
        # Retrieve the specific lower and upper bounds for the use case
        lower_bounds = bounds[use_case]['lower_bounds']
        upper_bounds = bounds[use_case]['upper_bounds']

        # Adjust bounds based on the size parameter
        if size == 'small':
            lower_bounds['num_ids'] = 0

        # Filter the dataframe based on the bounds
        filtered_df = df[
            (df['Frequency_of_stops'] >= lower_bounds['Frequency_of_stops']) &
            (df['Frequency_of_stops'] <= upper_bounds['Frequency_of_stops']) &
            (df['homogeneous'] >= lower_bounds['homogeneous']) &
            (df['homogeneous'] <= upper_bounds['homogeneous']) &
            (df['num_ids'] >= lower_bounds['num_ids']) &
            (df['num_ids'] <= upper_bounds['num_ids']) &
            (df['avg_wait_time_hours'] >= lower_bounds['avg_wait_time_hours']) &
            (df['avg_wait_time_hours'] <= upper_bounds['avg_wait_time_hours'])
        ]

        return filtered_df

    def calculate_homogeneous_score(value_dict, smoothing=1):
        values = np.array(list(value_dict.values()), dtype=np.float64)
        values += smoothing  # Applying Laplace smoothing
        total = values.sum()
        if total > 0:
            proportions = values / total

            entropy_value = entropy(proportions, base=2)
            if entropy_value == 0:
                if len(proportions) == 1:
                    return 1
            elif entropy_value > 0:
                score = 1 / entropy_value
                return score
        return 0

    def count_ids(value_dict):
        return len(value_dict)/unique_ids

    def get_polygon_coords(data):
        polygon_coords = []
        for i in data:
            polygon_coords.append((i[0], i[1]))
        return polygon_coords

    # Example usage with a DataFrame column
    df['homogeneous'] = df['num_trucks_stops'].apply(
        calculate_homogeneous_score)
    df['num_ids'] = df['num_trucks_stops'].apply(count_ids)
    df['avg_wait_time_hours'] = df['avg_wait_time'].dt.total_seconds() / 3600
    df['Points'] = df['coordinates'].apply(get_polygon_coords)

    recommended_stops = filter_threshold(df, bounds, use_case, size)

    return recommended_stops


def create_cluster_map(final_df, fill_colour):
    # Initialize the map at the centroid of the first cluster
    start_latitude = float(final_df.iloc[0]['cluster_centroid'][0])
    start_longitude = float(final_df.iloc[0]['cluster_centroid'][1])
    m_cluster = folium.Map(
        location=[start_latitude, start_longitude], zoom_start=4)

    # Loop through each cluster's points to create polygons
    for points in final_df.Points:
        # Calculate the boundary coordinates for the polygon
        min_lat = min(coord[0] for coord in points)
        max_lat = max(coord[0] for coord in points)
        min_lon = min(coord[1] for coord in points)
        max_lon = max(coord[1] for coord in points)

        boundary_coordinates = [
            (min_lat, min_lon),
            (min_lat, max_lon),
            (max_lat, max_lon),
            (max_lat, min_lon),
            (min_lat, min_lon)
        ]

        # Calculate the area and create a polygon on the map
        area = Polygon(boundary_coordinates).area * 10**6
        folium.Polygon(
            locations=boundary_coordinates,
            color='red',
            fill=True,
            fill_color=fill_colour,
            fill_opacity=0.5,
            weight=0.2,
            popup=folium.Popup(f'Area: {area:.2f} sq.km', parse_html=True)
        ).add_to(m_cluster)

        # Add CircleMarkers for each point within the cluster
        for coord in points:
            folium.CircleMarker(
                location=coord,
                radius=2,
                color='blue',
                fill=True,
                fill_color='blue'
            ).add_to(m_cluster)

    # Adding points for each cluster's centroid with additional data
    for _, row in final_df.iterrows():
        folium.CircleMarker(
            location=[row['cluster_centroid'][0], row['cluster_centroid'][1]],
            radius=10,
            color="purple",
            popup=f'''Cluster: {row['clusters']} <br>
                      No Trucks: {len(row['num_trucks_stops'])} <br>
                      Total wait time: {row['total_wait_time']} <br>
                      Homogenous Score: {row['homogeneous']} <br>
                      Average wait time: {row['avg_wait_time']} <br>
                      Frequency of Stops: {row['Frequency_of_stops']} <br>
                      Truck Stops: {row['num_trucks_stops']}''',
            fill=True,
            fill_color=fill_colour,
            fill_opacity=0.6
        ).add_to(m_cluster)

    # Optionally, add lines to connect the points and show the route
    # if final_df.shape[0] > 1:
    #     folium.PolyLine(final_df['cluster_centroid'].tolist(), color='truck_color').add_to(m_cluster)

    # Return the map object
    return m_cluster

# Example usage:
# st_data_cluster = st_folium(create_cluster_map(final_df, 'red'), height=500, width=1200, key="m_cluster")
