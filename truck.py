import data_preprocessing
import pandas as pd


class FLEET():

    def __init__(self, truck_id, df):
        self.fleetstops_dataframe = None
        self.truck_id = truck_id
        self.cluster = None
        self.df = df
        self.fleet_dataframe = None

    def get_data_frame(self):
        self.fleet_dataframe = self.df[self.df['device_id'].isin(
            self.truck_id)]

    def get_stops(self):
        if not self.fleetstops_dataframe:
            self.get_data_frame()
        for i in self.truck_id:
            new_stopped_df = data_preprocessing.get_stop_groups(
                self.fleet_dataframe[self.fleet_dataframe['device_id'] == i])
            if not self.fleetstops_dataframe:
                self.fleetstops_dataframe = new_stopped_df
            else:
                self.fleetstops_dataframe = pd.concat(
                    [new_stopped_df, self.fleetstops_dataframe])

    def get_centriod(self, df):
        pass

    def stop_duration(self, df, centroid):
        pass

    def stop_freq(sefl, df, centroid):
        pass