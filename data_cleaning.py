import pandas as pd


def cleaning(Path):
    """
    This Function takes in a path of the data
    returns a dataframe with id, lat , lon, and timestamp
    """
    data = pd.read_csv(Path,header=None)
    selected_columns = data.iloc[:, [0, 2, 3, 4]]
    selected_columns.columns = ['id','lat','lon','timestamp']
    return selected_columns