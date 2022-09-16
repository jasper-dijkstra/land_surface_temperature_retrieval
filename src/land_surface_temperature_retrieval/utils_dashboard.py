import pandas as pd
import geopandas as gpd
import numpy as np

from itertools import combinations
from copy import deepcopy
from scipy.stats import pearsonr


def evaluate(x, x_roll, y, y_roll):
    """
    Determine the Pearson r and bias based on rolling window of timeseries (x_roll, y_roll)
    Determine the unbiased RMSE (ubRMSE) based on the raw measurements (x, y)

    NOTE: the input pd.Series or np.arrays should not contain NaN!
    """
    try:
        r, p = pearsonr(x, y)  # Compute P. corr. coeff and ignore the p-vals.
        bias = np.mean(x - y)  # Compute bias, via mean (x - y)
        mae = np.mean(np.abs(x - y)) # Compute mae
        ub_rmse = (((x - y) - np.mean(x - y)) ** 2).mean() ** .5  # Compute unbiased RMSE
        n = len(x) # Number of observations metric is based on
        return {"r": r, "bias": bias, "mae":mae, "ub_rmse": ub_rmse, "n":n}
    except ValueError:
        return {"r": np.nan, "bias": np.nan, "mae":np.nan, "ub_rmse": np.nan, "n" : 0}


def interpolate_linear(vmin, vmax, center=None):
    """
    Generate list with linearly interpolated values between 'vmin' and 'vmax'
        Optionally, specify a center around which the interpolation takes place
    """
    classes = 5
    if center != None:
        index = [0, int(classes / 2), classes - 1]
        value = [vmin, center, vmax]
        values_left = [np.round(np.interp(i, index, value), 2) for i in range(index[0], index[1])]
        values_right = [np.round(np.interp(i, index, value), 2) for i in range(index[1], index[2])]
        values = values_left + values_right
        values.append(np.round(vmax, 2))
    else:
        index = [0, classes - 1]
        value = [vmin, vmax]
        values = [np.round(np.interp(i, index, value), 2) for i in range(index[0], index[-1])]
        values.append(np.round(vmax, 2))

    return values


def interpolate_log(vmin=0, vmax=1, classes=5):
    """
    Generate list with logartihmic interpolated values between 'vmin' and 'vmax'
        !!! Improve this?
    """
    if vmin == 0:
        vmin == 1e-10
    output = list(np.round(np.geomspace(vmin*10, vmax*10, classes) / 10, 2))
    return output


def StationTable(df, station_name, products):
    """
    Determine how 'products' compare to each based on Pearson r, bias, mae and ubRMSE
    over a period of time for a defined station ('station_name').

    Returns:
        evaluation_df: pd.DataFrame with evaluation metrics per product combination

    """
    # Filter to required station
    station_filtered = pd.DataFrame(df[df["name"] == station_name][products + ["roll_"+i for i in products]])

    # Initiate dataframe to which the evaluation metrics can be appended later
    evaluation_df = pd.DataFrame(index=["r", "bias", "mae", "ub_rmse", "n"])

    # Loop over the different product combinations (c) and compare
    for c in list(combinations(sorted(products), 2)):
        c_roll = ["roll_"+i for i in c] # Also define the "rolled" version of the products (c_roll)

        # First make sure there are no nan's
        xy_df = pd.DataFrame({"x" : station_filtered[c[0]].iloc[0],
                              "y" : station_filtered[c[1]].iloc[0]}).dropna()
        xy_roll_df = pd.DataFrame({"x" : station_filtered[c_roll[0]].iloc[0],
                                   "y" : station_filtered[c_roll[1]].iloc[0]}).dropna()

        # Determine evaluation metrics
        evaluation_dict = evaluate(x = xy_df["x"], x_roll = xy_roll_df["x"],
                                   y = xy_df["y"], y_roll = xy_roll_df["y"])

        # Append to evaluation dataframe
        evaluation_df[f"{c[0]} x {c[1]}"] = pd.DataFrame.from_dict(evaluation_dict, orient="index")

    return evaluation_df


def EvaluationMetrics(df, products):
    """
    Like the Dynamic Table function: determine some evaluation metrics over the timeseries of stations
    The StaticTable() function takes the spatial average of all individual station metrics.

    Args:
        df: pd.DataFrame that is output of DataPerStation() function
        products: list() with the products that will be compared to each other

    Returns:
        evaluation_metrics: multi-indexed pd.DataFrame with the evaluation metrics aggregated over the stations
    """
    # Create a multi-index dataframe with all evaluation metrics
    evaluation_dict = {}
    for station in df["name"]:
        station_evaluation = StationTable(df, station, products).to_dict()
        evaluation_dict[station] = station_evaluation
    evaluation_df = pd.DataFrame.from_dict(evaluation_dict, orient="index").stack().to_frame()
    evaluation_df = pd.DataFrame(evaluation_df[0].values.tolist(), index=evaluation_df.index)

    return evaluation_df


def AddMetricColumn(evaluation_df, product_combination, metric="mae"):
    """
    Create a df with a station-wise metric column of a 'product_combination'

    This is required to color choropleth map in the dashboard
    """
    metric_col = {}
    for station, station_metrics in evaluation_df.groupby(level=0):
        station_metrics = station_metrics.loc[station]
        metric_col[station] = station_metrics.loc[product_combination, metric]
    metric_col = pd.DataFrame.from_dict(metric_col, orient="index").reset_index()
    metric_col = metric_col.rename(columns={"index": "name", 0: metric + "_" + product_combination})

    return metric_col


def AppendAllMetricCombinations(df, evaluation_df, products, metrics=["r", "bias", "mae", "ub_rmse", "n"]):
    """
    Args:
        df: df with at least station names ("name") and geometries,
        evaluation_df: df returned from EvaluationMetrics()
        products: list() with product combinations to consider
        metrics: list() with matrices to consider

    Returns: input 'df' with columns with evaluation metrics appended

    """
    for combi in list(combinations(sorted(products), 2)):
        for metric in metrics:
            evaluation_column = AddMetricColumn(evaluation_df, f"{combi[0]} x {combi[1]}", metric)
            df = pd.merge(df, evaluation_column, on="name")

    return df


def TimeFilter(df, start_date, end_date):
    """ Filter all arrays in data, based on defined time window """

    time_filtered_df = df.copy()
    for i, data_slice in time_filtered_df.iterrows():

        # Get indices where t=valid
        idx = np.where((data_slice["date_time_local"][:] >= np.datetime64(start_date)) & (
                    data_slice["date_time_local"][:] <= np.datetime64(end_date)))

        # If a column contains an array, filter it with the valid indices
        row_list = []
        for c in time_filtered_df.columns:
            if type(data_slice[c]) == np.ndarray:
                row_list.append(np.array(data_slice[c][idx]))
            else:
                row_list.append(data_slice[c])

        # Correct the df, for the station involved
        time_filtered_df.loc[i, time_filtered_df.columns] = row_list

    return time_filtered_df


def clip_to_col(in_df, products, clipping_col):
    """ Clip the input df to limited data """
    out_df = gpd.GeoDataFrame(columns=in_df.columns, data=deepcopy(in_df.values))
    selected_products = products + ["roll_" + p for p in products]
    for row, dff in out_df.iterrows():
        for p in products:
            tt = dff[p] # Cut to timeseries
            tt[np.isnan(dff[clipping_col])]=np.nan # Clip to nan
            out_df.loc[out_df.name == dff['name'], p].iloc[0] = tt # append to df

    return out_df


def TemperatureFilter(df, products, lower_lim=-1e10, upper_lim=1e10):
    """ Filter temperature from """

    def Filter(row):
        row[col][row[col] <= lower_lim] = np.nan
        row[col][row[col] >= upper_lim] = np.nan
        return row[col]

    products = products + ["roll_" + p for p in products]
    for col in products:
        df[col] = df.apply(lambda row: Filter(row), axis=1)
        #df[df[products] <= lower_lim] = np.nan
        #df[df[products] >= upper_lim] = np.nan
    return df


def filter_df(in_df, col_filter, col_to_filter):
    """ Filter a df column """
    return in_df.groupby(col_filter)[col_to_filter].apply(np.array)


def open_csv(in_csv, date_time_col):
    # Read the data
    data = pd.read_csv(in_csv, index_col=[0])

    # Set the dtypes of the time columns to datetime instead of str()
    try:
        data[date_time_col] = pd.to_datetime(data[date_time_col], format="%Y-%m-%d %H:%M:%S")
    except KeyError:
        print("date_time column is not the csv .. was not converted")
    return data


def kelvin_to_celsius(df, columns):
    return df[columns] - 273.15


def df_stations(data, station_info_cols):
    stations = data.drop_duplicates(subset=["name"])
    stations = gpd.GeoDataFrame(data=stations,
                                geometry=gpd.points_from_xy(stations.longitude, stations.latitude),
                                crs="epsg:4326")
    return stations[station_info_cols].reset_index(drop=True)


def add_rolling_ts(stations, ts_col, window=30, min_periods=1):
    # add rolling for each station
    rolled = pd.DataFrame(columns=["name", "roll_" + ts_col])
    for station in stations["name"]:
        station_timeseries = pd.Series(stations[stations["name"] == station][ts_col].iloc[0])
        station_timeseries = np.array(station_timeseries.rolling(window=window, min_periods=min_periods, center=True).mean())

        # Remove all values below 0:
        # station_timeseries[station_timeseries < 0] = np.nan

        # Append all to stations df
        rolled = pd.concat(
            [rolled, pd.DataFrame.from_records([{
                "name": station,
                "roll_" + ts_col: station_timeseries  # col to label!
            }])])

    return pd.merge(left=stations, right=rolled, on="name")


def reformat_per_station(data, kelvin_cols, celsius_cols, station_info_cols, date_time_col, hour,
                         T_min=-40, T_max=50, add_rolling=True, window=20, min_periods=3):
    # Convert the columns with temperature in K, to °C:
    data[kelvin_cols] = kelvin_to_celsius(data, kelvin_cols)

    # Create a GeoDataFrame with all stations, their geolocation and some station specific variables
    stations = df_stations(data, station_info_cols)

    # Create timeseries for each station for selected time, for several measurement sources, when all have values
    time_filtered = data[data[date_time_col].dt.hour == hour]

    # Make sure all dates have been filtered
    time_filtered = time_filtered.sort_values(by=[date_time_col])

    # add list with dates
    dt = filter_df(time_filtered, "name", date_time_col)
    stations = pd.merge(left=stations, right=dt, on="name")

    # Loop over each column containing temperature
    for col in celsius_cols+kelvin_cols:
        # Remove all measurements with temperatures below -40 and above 50°C and those equal to 0°C
        time_filtered.loc[time_filtered[col] <= T_min, col] = np.nan
        time_filtered.loc[time_filtered[col] >= T_max, col] = np.nan
        time_filtered.loc[time_filtered[col] == 0, col] = np.nan  # Some 0 values are errors

        # Group them by station, into list
        r = filter_df(time_filtered, "name", col)
        stations = pd.merge(left=stations, right=r, on="name")

    if add_rolling:
        for col in celsius_cols+kelvin_cols:
            stations = add_rolling_ts(stations, ts_col=col, window=window, min_periods=min_periods)

    return stations


def inverted_ubrmse(data, left, right):
    """
    The inverse ubRMSE (switched around left and right products) needs to be calculated separately
    """
    new_col_name = f"ub_rmse_{left} x {right}"
    filtered = data[["name", left, right]].copy()
    filtered[new_col_name] = np.nan
    for i, station in enumerate(filtered["name"]):
        # First make sure there are no nan's
        xy_df = pd.DataFrame({"x": filtered[left].iloc[i],
                              "y": filtered[right].iloc[i]}).dropna()

        # Evaluate the data
        evaluation_dict = evaluate(xy_df["x"], None, xy_df["y"], None)

        # Append to output df
        filtered.loc[i, new_col_name] = evaluation_dict["ub_rmse"]

    data = pd.merge(left=data, right=filtered[["name", new_col_name]], on="name")
    return data
