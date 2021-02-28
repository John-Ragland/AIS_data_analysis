import pandas as pd
import os
import numpy as np
import scipy
import math as m
import pickle

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
    

    df["time"] = df["TIMESTAMP UTC"].astype("datetime64")
    df = df.sort_values(by=['time'])
    df = df.reset_index()

    # add timestamp value
    time_val = []
    print('Converting Time to Values...')
    for k in range(len(df)):
        time_val.append(df['time'][k].value)
    df['time_val'] = time_val
    
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
    MMSIs = df['MMSI'].unique()

    ship_passes = []
    for n, MMSI in enumerate(MMSIs):
        single_ship = df[df['MMSI']==MMSI]

        single_ship = single_ship.reset_index()

        # Split into different dataframes for each pass
        for k in range(len(single_ship)):
            if k == 0:
                pass_start_idx = 0
            else:
                if (single_ship['time'][k] - single_ship['time'][k-1]).seconds > 3600:
                    # save last pass to list
                    single_pass_df = single_ship[pass_start_idx:k-1]
                    single_pass_df = single_pass_df.reset_index(drop=True)
                    ship_passes.append(single_pass_df)
                    # start new pass
                    pass_start_idx = k
                
                else:
                    pass
        print((n+1)/len(MMSIs)*100, end='\r')
    print('Complete                     ')

    # remove empty list entries
    for k in range(len(ship_passes)-1,-1, -1):
        if len(ship_passes[k]) == 0:
            _ = ship_passes.pop(k)
    return ship_passes


def grid_coord(df):
    '''
    Parameters
    ----------
    df

    Returns
    -------
    gridded_data : pd Dataframe
        data frame containing lats, lons, and times. which are the gridded data
        from the ship pass
    '''
    lat_grid = np.linspace(44.9, 46.9, 500)
    lon_grid = np.linspace(-131.3, -128.4, 500)

    if (len(df) == 1):
        lat_sampled = df.LAT[0]
        lon_sampled = df.LON[0]

        lats = [np.array(lat_grid)[np.searchsorted(lat_grid, lat_sampled)-1]]
        lons = [np.array(lon_grid)[np.searchsorted(lon_grid, lon_sampled)-1]]
        times = [df.time_val[0]]

    else:
        # define interpolation
        f_lat = scipy.interpolate.interp1d(df.time_val, df.LAT, bounds_error=False)
        f_lon = scipy.interpolate.interp1d(df.time_val, df.LON, bounds_error=False)

        # create time array with \Delta t = 1S
        time_sampled = np.arange(df['time_val'].min(), df['time_val'].max(), 1e+9)
        # linearly interpolate lat/lon with time array
        lat_sampled = f_lat(time_sampled)
        lon_sampled = f_lon(time_sampled)

        # round sampled lat/lon to lat/lon grid (and removed NAN)
        rounded_lat = np.array(lat_grid)[np.searchsorted(lat_grid, lat_sampled)-1]
        rounded_lat = rounded_lat[~np.isnan(lat_sampled)]
        rounded_lon = np.array(lon_grid)[np.searchsorted(lon_grid, lon_sampled)-1]
        rounded_lon = rounded_lon[~np.isnan(lon_sampled)]

        diff_lat = np.insert(np.diff(rounded_lat),0,0)
        diff_lon = np.insert(np.diff(rounded_lon),0,0,)

        mask = np.array((diff_lat != 0) | (diff_lon != 0))
        lats = rounded_lat[mask]
        lons = rounded_lon[mask]

        times = time_sampled[mask]

    # Create Lat/Lon indexes
    lat_idx = np.zeros(np.array(lats).shape[0])
    lon_idx = np.zeros(np.array(lons).shape[0])
    for k in range(np.array(lats).shape[0]):
        lat_idx[k] = np.argwhere(lats[k] == lat_grid)[0][0]
        lon_idx[k] = np.argwhere(lons[k] == lon_grid)[0][0]
    
    d = {'lats':lats, 'lons':lons, 'times':times, 'lat_idx':lat_idx, 'lon_idx':lon_idx}
    gridded_data = pd.DataFrame(data=d)
    return gridded_data

def grid_all_passes(ship_passes):
    ship_passes_grid = []
    for k, ship_pass in enumerate(ship_passes):
        try:
            ship_pass_grid = grid_coord(ship_pass)
        except:
            print(f'Ship Pass {k} skipped: error')
        else:
            ship_passes_grid.append(ship_pass_grid)
        print((k+1)/len(ship_passes)*100, end='\r')
    return ship_passes_grid

def create_spacetime_distribution():
    '''
    create space-time distribution from raw AIS Data
    '''
    # load ship_passes_grid.pkl
    print('loading ship_passes_grid.pkl...')
    ship_passes_grid = pickle.load( open( "ship_passes_grid.pkl", "rb" ) )
    
    # Every hour betwee 2015-2020
    time_dim = 52608    #hours

    print('combining data into single dataframe....')
    full_data = pd.concat(ship_passes_grid)
    full_data['timestamp'] = pd.to_datetime(full_data['times'])
    full_data = full_data.reset_index(drop=True)

    td = full_data['timestamp'] - pd.Timestamp('2015-01-01')
    hours = td / np.timedelta64(1, 'h')

    lat_grid = np.linspace(44.9, 46.9, 500)
    lon_grid = np.linspace(-131.3, -128.4, 500)

    st_dist = np.zeros((lat_grid.shape[0], lon_grid.shape[0], time_dim))
    print('populating space time distribution...')
    # Populate Space Time Distribution
    for k in range(len(full_data)):
        time_idx = m.trunc(hours[k])
        st_dist[int(full_data.lat_idx[k]), int(full_data.lon_idx[k]), time_idx] += 1
        if k % 1000 == 0:
            print(f'Percent Complete: {k/len(full_data)*100}', end='\r')

    return st_dist
        

def resample_time(df):
    '''
    Parameters
    ----------
    df

    Returns
    -------
    gridded_data : pd Dataframe
        data frame containing lats, lons, and times. which are the gridded data
        from the ship pass
    '''
    lat_grid = np.linspace(44.9, 46.9, 500)
    lon_grid = np.linspace(-131.3, -128.4, 500)

    if (len(df) == 1):
        lat_sampled = df.LAT[0]
        lon_sampled = df.LON[0]

        lats = [np.array(lat_grid)[np.searchsorted(lat_grid, lat_sampled)-1]]
        lons = [np.array(lon_grid)[np.searchsorted(lon_grid, lon_sampled)-1]]
        times = [df.time_val[0]]

    else:
        # define interpolation
        f_lat = scipy.interpolate.interp1d(df.time_val, df.LAT, bounds_error=False)
        f_lon = scipy.interpolate.interp1d(df.time_val, df.LON, bounds_error=False)

        # create time array with \Delta t = 5 min
        time_sampled = np.arange(df['time_val'].min(), df['time_val'].max(), 5*60e+9) # 5 min
        # linearly interpolate lat/lon with time array
        lats = f_lat(time_sampled)
        lons = f_lon(time_sampled)

        times = time_sampled
    
    d = {'lats':lats, 'lons':lons, 'times':times}
    resampled_data = pd.DataFrame(data=d)
    return resampled_data

def resample_time_all(df_ls):
    resampled_ls = []
    for k, df in enumerate(df_ls):
        try:
            resampled = resample_time(df)
        except:
            print(f'Ship Pass {k} skipped: error')
        else:
            resampled_ls.append(resampled)
        print(k/len(df_ls), end='\r')

    return pd.concat(resampled_ls)