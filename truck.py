

class FLEET():
    def __init__(self,truck_id,df):
        self.truck_id = truck_id
        self.cluster = None
        self.df = df
        self.data_unique = None
        
    def get_data_frame(self):
    
        self.data_unique = self.df[self.df['device_id'].isin(self.truck_id)]
        return self.data_unique

    def get_stops(self,df):
        pass
    def get_centriod(self,df):
        pass
    def stop_duration(self,df,centroid):
        pass
    def stop_freq(sefl,df,centroid):
        pass

