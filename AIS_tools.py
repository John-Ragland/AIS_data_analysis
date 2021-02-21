import pandas as pd
import os

def get_AIS_data(AIS_data_dir=None):
    '''
    get_AIS_data - loads AIS from csvs into pandas data frame

    Parameters
    ----------
    AIS_data_dir : str
        absolute path to AIS data. Should be a folder contain CSVs

    Returns
    -------
    df : pandas.DataFrame
        pandas data frame containing AIS data with time column (datetime64)
        added
    '''
    if AIS_data_dir == None:
        AIS_data_dir = '/Volumes/Ocean_Acoustics/AIS_Data/Axial_Seamount/'
    
    files = os.listdir(AIS_data_dir)

    for file in files:
        try:
            df
        except NameError:
            df = pd.read_csv(AIS_data_dir + file)
        else:
            df = pd.concat([pd.read_csv(AIS_data_dir + file), df])
        
    df = df.reset_index()
    df["time"] = df["TIMESTAMP UTC"].astype("datetime64")

    return df


def single_ship_pass(df):
    '''
    single_ship_pass - seperate AIS dataframe into list of dataframes for each
    ship pass instance

    Parameter
    ---------
    df : pandas df
        df containing AIS shipping data
    
    Returns:
    --------
    df_ls : list
        list of pandas dataframes where each list entry contains single ship
        passing instance
    '''
    group = df.groupby('MMSI')
    MMSIs = df['MMSI'].unique()

    dataframe_by_ship = []
    for k, MMSI in enumerate(MMSIs):
        single_ship = df[df['MMSI']==MMSI])

        single_ship.reset_index()