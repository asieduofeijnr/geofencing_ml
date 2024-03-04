import data_preprocessing
import pandas as pd


class FLEET():

    def __init__(self, truck_id, df):
        self.fleetstops_dataframe = pd.DataFrame()
        self.truck_id = truck_id
        self.clusters = pd.DataFrame()
        self.df = df
        self.fleet_dataframe = pd.DataFrame()
        self.proximity_df = pd.DataFrame()
        self.geofence_df = pd.DataFrame()

    def get_data_frame(self):
        self.fleet_dataframe = self.df[
            self.df['device_id'].isin(self.truck_id)
        ]
        return self

    def get_stops(self, time_threshold=None):
        if self.fleet_dataframe.empty:
            self.get_data_frame()
        for i in self.truck_id:
            new_stopped_df = data_preprocessing.get_stop_groups(
                self.fleet_dataframe[
                    self.fleet_dataframe['device_id'] == i
                ]).rename(columns={0: "waiting_time"})
            if self.fleetstops_dataframe.empty:
                self.fleetstops_dataframe = new_stopped_df
            else:
                self.fleetstops_dataframe = pd.concat(
                    [new_stopped_df, self.fleetstops_dataframe]
                )
        if time_threshold:
            self.fleetstops_dataframe = self.fleetstops_dataframe[
                self.fleetstops_dataframe['waiting_time'] > time_threshold
            ]
        return self

    def get_address(self):
        self.fleetstops_dataframe['address'] = self.fleetstops_dataframe.apply(
            lambda row: data_preprocessing.get_location(str(row['longitude']),
                                                        str(row['latitude'])),
            axis=1
        )
        return self

    def getClustersFrequency(self, stop_threshold=0.000001, time_threshold=None):
        if self.fleetstops_dataframe.empty:
            self.get_stops(time_threshold=time_threshold)
        clusters_df = data_preprocessing.get_clusters_and_frequency(
            self.fleetstops_dataframe, stop_threshold)
        self.clusters = clusters_df
        return self

    def get_proximity_df(self):
        if self.clusters.empty:
            self.getClustersFrequency()
        self.proximity_df = data_preprocessing.estimate_proximity_and_closest_cluster(
            self.clusters)
        return self

    def get_geofence_df(self, waiting_time_threshold=None,
                        frequency_threshold=None,
                        proximity_threshold=None):
        if self.proximity_df.empty:
            self.get_proximity_df()
        self.geofence_df = \
            data_preprocessing.create_geofence_target_label(
                self.proximity_df,
                waiting_time_threshold=waiting_time_threshold,
                frequency_threshold=frequency_threshold,
                proximity_threshold=proximity_threshold)
        return self

    def stop_duration(self, df, centroid):
        pass

    def stop_freq(sefl, df, centroid):
        pass

    def get_centriod(self, df):
        pass

    def create_clusters(self):
        pass
