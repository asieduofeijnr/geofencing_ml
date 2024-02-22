import data_preprocessing
import pandas as pd


class FLEET():

    def __init__(self, truck_id, df):
        self.fleetstops_dataframe = pd.DataFrame()
        self.truck_id = truck_id
        self.cluster = None
        self.df = df
        self.fleet_dataframe = pd.DataFrame()

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
                    ]).rename(columns={0:"waiting_time"})
            if self.fleetstops_dataframe.empty:
                self.fleetstops_dataframe = new_stopped_df
            else:
                self.fleetstops_dataframe = pd.concat(
                    [new_stopped_df, self.fleetstops_dataframe]
                    )
        if time_threshold:
            self.fleetstops_dataframe = self.fleetstops_dataframe[
                self.fleetstops_dataframe['waiting_time']>time_threshold
                ]
        return self
        
    def get_address(self):
        self.fleetstops_dataframe['address'] = self.fleetstops_dataframe.apply(
            lambda row: data_preprocessing.get_location(str(row['longitude']),
                                                        str(row['latitude'])),
                                                        axis=1
                                                        )
        return self

    def stop_duration(self, df, centroid):
        pass

    def stop_freq(sefl, df, centroid):
        pass

    def get_centriod(self, df):
        pass

    def create_clusters(self):
        pass