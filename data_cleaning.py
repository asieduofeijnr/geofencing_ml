import pandas as pd

def cleaning(file_location):
    df = pd.read_csv(file_location)
    ## Everything related to cleaning
    return df