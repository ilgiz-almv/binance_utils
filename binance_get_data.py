import glob
import logging
import math
import os
import time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
from tabulate import tabulate


def get_data_last_raw(
        start_time, client, symbol, interval_str,
        prev_sumOpenInterest=None, limit=2
):
    """
    Retrieves the latest spot and futures candlestick data, along with open
    interest data, from a given trading client.

    This function fetches the latest spot and futures market data, ensuring that
    a complete set of candles is obtained. It also attempts to retrieve open
    interest data and handles missing data scenarios with retries and fallback
    mechanisms.

    Args:
        start_time (int): The starting timestamp (in milliseconds) for fetching data.
        client (object): The trading client instance to query market data.
        symbol (str): The trading pair symbol (e.g., "BTCUSDT").
        interval_str (str): The time interval for candlestick data (e.g., "5m", "1h").
        prev_sumOpenInterest (float, optional): Previous open interest value for
            fallback use. Defaults to None.
        limit (int, optional): The number of candles to fetch. Defaults to 2.

    Returns:
        pd.DataFrame: A DataFrame containing spot and futures market data with timestamps.
    """
    max_attempts = 10

    # Fetching spot market data
    spot_data = []
    attempts = 0
    while len(spot_data) < 2 and attempts < max_attempts:
        try:
            spot_data = client.get_klines(
                symbol=symbol, interval=interval_str, limit=limit,
                startTime=start_time
            )
        except Exception as e:
            attempts += 1
            logging.error(
                f"Exception occurred while fetching spot data (Attempt {attempts}): {e}"
            )
        else:
            if len(spot_data) == 2:
                break
            attempts += 1
            logging.info(f"Spot data is incomplete, retrying... (Attempt {attempts})")
            time.sleep(1)

    # If no data was retrieved after max attempts
    if not spot_data:
        logging.info("Max attempts reached. No spot data received.")
        return pd.DataFrame()

    # Convert spot data to DataFrame
    spot_df = pd.DataFrame(
        [spot_data[0]], columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_vol", "taker_buy_quote_asset_volume", "ignore"
        ]
    )[["time", "open", "high", "low", "close", "volume",
       "number_of_trades", "taker_buy_vol"]]

    # Fetching futures market data
    futures_data = []
    attempts = 0
    while len(futures_data) < 2 and attempts < max_attempts:
        try:
            futures_data = client.futures_klines(
                symbol=symbol, interval=interval_str, limit=limit,
                startTime=start_time
            )
        except Exception as e:
            attempts += 1
            logging.error(
                f"Exception occurred while fetching futures data (Attempt {attempts}): {e}"
            )
        else:
            if len(futures_data) == 2:
                break
            attempts += 1
            logging.info(f"Futures data is incomplete, retrying... (Attempt {attempts})")
            time.sleep(1)

    # If no futures data was retrieved
    if not futures_data:
        logging.info("Max attempts reached. No futures data received.")
        return pd.DataFrame()

    # Convert futures data to DataFrame
    futures_df = pd.DataFrame(
        [futures_data[0]], columns=[
            "time_d", "open_d", "high_d", "low_d", "close_d", "volume_d",
            "close_time_d", "quote_asset_volume_d", "number_of_trades_d",
            "taker_buy_vol_d", "taker_buy_quote_asset_volume_d", "ignore_d"
        ]
    )[["open_d", "high_d", "low_d", "close_d", "volume_d",
       "number_of_trades_d", "taker_buy_vol_d"]]

    # Merge spot and futures data
    merged_table = pd.concat([spot_df, futures_df], axis=1).astype(float)

    # Adjust time to align with the close time of the interval
    interval_int_ms = convert_to_msec(interval_str)
    merged_table["time"] = np.int64(merged_table["time"] + interval_int_ms)

    # Fetch open interest data if previous value is provided
    if prev_sumOpenInterest is not None:
        open_interest_data = []
        attempts = 0
        sumOpenInterest = prev_sumOpenInterest
        timestamp_OI_reserve = start_time

        while len(open_interest_data) < 2 and attempts < max_attempts:
            try:
                open_interest_data = client.futures_open_interest_hist(
                    symbol=symbol, period=interval_str, limit=limit,
                    startTime=start_time
                )
            except Exception as e:
                logging.error(
                    f"Exception while fetching open interest history (Attempt {attempts+1}): {e}"
                )
            else:
                if len(open_interest_data) == 2:
                    sumOpenInterest = float(open_interest_data[-1]['sumOpenInterest'])
                    break
                logging.info(f"Open interest data incomplete, retrying... (Attempt {attempts+1})")
            finally:
                attempts += 1

                if len(open_interest_data) < 2:
                    try:
                        open_interest_reserve = client.futures_open_interest(symbol=symbol)
                        if abs(np.int64(open_interest_reserve['time']) - (start_time + 300000)) < \
                           abs(timestamp_OI_reserve - (start_time + 300000)):
                            timestamp_OI_reserve = np.int64(open_interest_reserve['time'])
                            sumOpenInterest = float(open_interest_reserve['openInterest'])
                    except Exception as e:
                        logging.error(
                            f"Exception while fetching open interest reserve: {e}"
                        )

        if sumOpenInterest == prev_sumOpenInterest and attempts == max_attempts:
            logging.info(
                f"Open interest retrieval failed, using previous value: {sumOpenInterest}"
            )

        merged_table["sumOpenInterest"] = sumOpenInterest

    # Convert time to UTC datetime and set as index
    merged_table["time_UTC"] = pd.to_datetime(merged_table["time"], unit='ms', utc=True)
    merged_table.set_index('time_UTC', inplace=True)
    merged_table.drop(columns='time', inplace=True)

    return merged_table


def convert_to_msec(time_str):
    """
    Converts a time interval string (e.g., "5m", "1h", "3d", "1w") into milliseconds.

    This function parses a time string where the last character represents the time unit
    (seconds, minutes, hours, days, weeks), and the preceding characters represent the numerical value.
    It then converts this into an integer value representing milliseconds.

    Args:
        time_str (str): Time interval string (e.g., "5m", "1h", "3d", "1w").

    Returns:
        int: The equivalent duration in milliseconds.

    Raises:
        ValueError: If the provided time format is unsupported.
    """
    unit = time_str[-1]  # Extract the unit character (e.g., 'm', 'h', 'd', etc.)
    try:
        value = int(time_str[:-1])  # Extract the numeric portion and convert to integer
    except ValueError:
        raise ValueError("Invalid time format: numerical value expected before unit.")

    # Mapping of time units to their equivalent in milliseconds
    time_units = {
        's': 1000,  # Seconds to milliseconds
        'm': 60 * 1000,  # Minutes to milliseconds
        'h': 3600 * 1000,  # Hours to milliseconds
        'd': 86400 * 1000,  # Days to milliseconds
        'w': 604800 * 1000  # Weeks to milliseconds
    }

    if unit not in time_units:
        raise ValueError(f"Unsupported time format: '{unit}' is not a valid unit.")

    return value * time_units[unit]


def check_and_fill_time_steps(df, start_time, end_time, step, threshold_pct):
    """
    Checks for missing time steps in a DataFrame index and fills gaps if within a given threshold.

    This function verifies that the DataFrame's index covers the expected time range at the specified
    step interval. If missing timestamps are found, it logs their percentage and optionally fills them
    if the missing rate is below a given threshold.

    Args:
        df (pd.DataFrame): DataFrame with a datetime index.
        start_time (int): Start timestamp in milliseconds.
        end_time (int): End timestamp in milliseconds.
        step (int): Time step interval in milliseconds.
        threshold_pct (float): Percentage threshold for missing data before logging an error.

    Returns:
        pd.DataFrame: Updated DataFrame with missing timestamps reindexed if necessary.
    """
    # Generate expected time range based on start, end, and step size
    expected_times = pd.date_range(
        start=pd.to_datetime(start_time, unit='ms', utc=True),
        end=pd.to_datetime(end_time, unit='ms', utc=True),
        freq=f'{step}ms'
    )

    # Identify missing timestamps
    missing_times = expected_times.difference(df.index)
    missing_times_pct = (len(missing_times) / len(expected_times)) * 100

    logging.info(f"Missing Times Percentage: {missing_times_pct:.1f}%")

    if 0 < missing_times_pct < threshold_pct:
        missing_times_df = pd.DataFrame({'Missing Timestamps': missing_times})
        missing_times_df['Missing Timestamps'] = missing_times_df['Missing Timestamps'].astype(str)

        logging.info(
            f"\nMissing Times:\n"
            f"{tabulate(missing_times_df, headers='keys', tablefmt='grid')}\n"
        )

        # Reindex DataFrame to include missing timestamps
        df = df.reindex(df.index.union(expected_times))
    elif missing_times_pct >= threshold_pct:
        logging.error(f"More than {threshold_pct}% of timestamps are missing.")
        df = df.reindex(df.index.union(expected_times))

    return df


def time_info_to_log(start_time, end_time):
    """
    Logs time information, including both raw timestamps and their UTC datetime representations.

    Args:
        start_time (int): The start timestamp in milliseconds.
        end_time (int): The end timestamp in milliseconds.
    """
    info_df = pd.DataFrame({
        'Parameter': ['start_time', 'end_time', 'start_time (datetime)', 'end_time (datetime)'],
        'Value': [
            start_time,
            end_time,
            pd.to_datetime(start_time, unit='ms', utc=True),
            pd.to_datetime(end_time, unit='ms', utc=True)
        ]
    })

    logging.info(
        f"\nTime Information:\n"
        f"{tabulate(info_df, headers='keys', tablefmt='grid', showindex=False)}\n"
    )


def data_preparation(df, time_attr, start_time, end_time, step, attributes,
                     threshold_pct, conv_to_close_time=True):
    """
    Prepares and validates a DataFrame by checking for missing or problematic values,
    synchronizing timestamps, and filling missing time steps.

    Args:
        df (pd.DataFrame): DataFrame containing time-series data.
        time_attr (str): Column name containing timestamps in milliseconds.
        start_time (int): Start timestamp in milliseconds.
        end_time (int): End timestamp in milliseconds.
        step (int): Time step interval in milliseconds.
        attributes (list): List of attribute names to check for NaN or zero values.
        threshold_pct (float): Percentage threshold for missing/zero values before logging an error.
        conv_to_close_time (bool, optional): Whether to shift timestamps to closing times. Defaults to True.

    Returns:
        pd.DataFrame: Processed DataFrame with missing values replaced and time steps aligned.
    """
    df = df.astype(float)  # Ensure numerical consistency

    logging.info(f"\nconv_to_close_time = {conv_to_close_time}:\n")
    time_info_to_log(start_time, end_time)

    summary_data = {
        'Attribute': [], 'NaN Count': [], 'Zero Count': [],
        'NaN Percentage': [], 'Zero Percentage': []
    }

    # Analyze attributes for NaN and zero values
    for attr in attributes:
        nan_count = df[attr].isna().sum()
        zero_count = df[attr].eq(0).sum()
        total_count = len(df)

        nan_pct = (nan_count / total_count) * 100
        zero_pct = (zero_count / total_count) * 100

        summary_data['Attribute'].append(attr)
        summary_data['NaN Count'].append(nan_count)
        summary_data['Zero Count'].append(zero_count)
        summary_data['NaN Percentage'].append(nan_pct)
        summary_data['Zero Percentage'].append(zero_pct)

        if nan_pct > threshold_pct or zero_pct > threshold_pct:
            logging.error(
                f"\nThe attribute {attr} contains more than {threshold_pct}% NaN or zero values.\n"
            )

    # Convert summary data to DataFrame
    summary_df = pd.DataFrame(summary_data)
    logging.info(
        f"\nAttribute Validation Summary:\n"
        f"{tabulate(summary_df, headers='keys', tablefmt='grid', showindex=False)}\n"
    )

    if conv_to_close_time:
        # Adjust time column to reflect close time
        df[time_attr] = np.int64(df[time_attr] + step)
        start_time += step
        end_time += step

        logging.info("\nTime after shifting to close time:\n")
        time_info_to_log(start_time, end_time)
    else:
        df[time_attr] = np.int64(df[time_attr])

    # Convert time column to UTC and set as index
    df["time_UTC"] = pd.to_datetime(df[time_attr], unit='ms', utc=True)
    df.set_index('time_UTC', inplace=True)
    df.sort_index(inplace=True)

    # Fill missing time steps
    df = check_and_fill_time_steps(df, start_time, end_time, step, threshold_pct)

    # Identify rows containing NaN or zero values
    mask = df.isna().any(axis=1) | (df == 0).any(axis=1)
    filtered_df = df[mask]

    if not filtered_df.empty:
        indexes_df = pd.DataFrame({'Problematic Indexes': filtered_df.index})

        if isinstance(filtered_df.index, pd.DatetimeIndex):
            indexes_df['Problematic Indexes'] = indexes_df['Problematic Indexes'].apply(
                lambda x: x.strftime('%Y-%m-%d %H:%M:%S')
            )

        logging.info(
            f"\nRows with NaN or 0 in any attribute:\n"
            f"{tabulate(indexes_df, headers='keys', tablefmt='grid')}\n"
        )

        # Replace identified problematic values with NaN
        df.loc[mask] = np.nan
    else:
        logging.info("\nNo rows with NaN or 0 in any attribute found.\n")

    # Drop the original time column
    df.drop(time_attr, axis=1, inplace=True)

    return df


def convert_time_into_ms(time_input, interval_int_ms):
    """
    Converts a given time input into UTC milliseconds (np.int64), aligning it to a specified interval.

    The function supports input as a string (formatted as 'dd.mm.yyyy' or 'dd.mm.yyyy HH:MM:SS'),
    a datetime object (must be in UTC), or an integer representing milliseconds.

    Args:
        time_input (str | datetime | int | np.int64): The time input to convert.
        interval_int_ms (int): The interval in milliseconds to which the result should be aligned.

    Returns:
        np.int64: The converted time in UTC milliseconds, rounded down to the nearest interval.

    Raises:
        ValueError: If the input format is not recognized or not in UTC (for datetime objects).
    """
    logging.info(f"\nConverting time = {time_input} into msec UTC (np.int64)..\n")

    if isinstance(time_input, str):
        try:
            # Convert string with time included
            time_dt = datetime.strptime(time_input, "%d.%m.%Y %H:%M:%S").replace(tzinfo=timezone.utc)
        except ValueError:
            try:
                # Convert string without time (defaults to midnight UTC)
                time_dt = datetime.strptime(time_input, "%d.%m.%Y").replace(tzinfo=timezone.utc)
            except ValueError:
                raise ValueError(
                    f"{time_input} does not match expected date/time formats ('dd.mm.yyyy' or 'dd.mm.yyyy HH:MM:SS')."
                )
    elif isinstance(time_input, datetime):
        if (
            time_input.tzinfo is not None
            and time_input.tzinfo.utcoffset(time_input) == timezone.utc.utcoffset(time_input)
        ):
            time_dt = time_input
        else:
            raise ValueError(f"Datetime input {time_input} must be in UTC.")
    elif isinstance(time_input, (int, np.int64)):
        # Convert milliseconds to datetime UTC
        time_dt = datetime.fromtimestamp(time_input / 1000, tz=timezone.utc)
    else:
        raise ValueError("Unsupported time input type.")

    # Convert datetime to UTC milliseconds, aligned to the given interval
    time_ms = np.int64(math.floor(time_dt.timestamp() * 1000 / interval_int_ms) * interval_int_ms)

    logging.info(f"\nNew time (msec) UTC = {time_ms}\n")

    return time_ms


def get_precision(value):
    """
    Determines the number of decimal places in a given numeric string.

    If the value contains a decimal point, trailing zeros are ignored when calculating precision.

    Args:
        value (str): Numeric value in string format.

    Returns:
        int: Number of decimal places.
    """
    if '.' in value:
        return len(value.split('.')[1].rstrip('0'))
    return 0


def determine_max_precision(df):
    """
    Determines the maximum decimal precision for each column in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing numeric values.

    Returns:
        dict: A dictionary where keys are column names and values are the maximum decimal precision.
    """
    precision_dict = {}

    for column in df.columns:
        max_precision = df[column].astype(str).apply(lambda x: get_precision(x)).max()
        precision_dict[column] = max_precision

    return precision_dict


def get_max_decimals_dict(precisions_list, time_attr_to_del):
    """
    Computes the maximum decimal precision for each column from a list of precision dictionaries.

    Args:
        precisions_list (list of dict): List of dictionaries containing precision values.
        time_attr_to_del (str): Column name to exclude from the final dictionary.

    Returns:
        dict: Dictionary with column names as keys and maximum precision as values, excluding the specified column.
    """
    decimals_df = pd.DataFrame(precisions_list)
    max_decimals_dict = decimals_df.max().to_dict()

    if time_attr_to_del in max_decimals_dict:
        del max_decimals_dict[time_attr_to_del]

    return max_decimals_dict


def get_klines_data(
    client, symbol, interval_str,
    start_time_str, end_time_str=None,
    limit=1000, threshold_pct=5,
    is_futures_included=True
):
    """
    Retrieves historical candlestick (klines) data for a given symbol from a trading client.

    Supports both spot and futures data, with options to process missing data and apply precision rounding.

    Args:
        client (object): Trading client instance.
        symbol (str): Trading pair symbol (e.g., "BTCUSDT").
        interval_str (str): Time interval for candlesticks (e.g., "5m", "1h").
        start_time_str (str | datetime | int): Start time in format "%d.%m.%Y %H:%M:%S", "%d.%m.%Y",
            as a datetime (UTC), or as an integer in milliseconds.
        end_time_str (str | datetime | int, optional): End time in the same formats. Defaults to current time if None.
        limit (int, optional): Maximum number of candles per API request. Defaults to 1000.
        threshold_pct (int, optional): Threshold for missing data before logging an error. Defaults to 5.
        is_futures_included (bool, optional): Whether to include futures data. Defaults to True.

    Returns:
        pd.DataFrame: Processed DataFrame with spot and optionally futures data, indexed by UTC time.
    """
    logging.info("\nget_klines_data() run..\n")

    final_df = pd.DataFrame()
    spot_precisions_list = []
    fut_precisions_list = []

    interval_int = convert_to_msec(interval_str)
    logging.info(f"\nConverted interval: {interval_str} -> {interval_int} msec\n")

    start_time = convert_time_into_ms(start_time_str, interval_int) - interval_int

    if end_time_str is None:
        end_time_close = np.int64(
            math.floor(datetime.now(timezone.utc).timestamp() * 1000 / interval_int) * interval_int
        )
        end_time_open = end_time_close - interval_int
    else:
        end_time_open = convert_time_into_ms(end_time_str, interval_int) - interval_int

    if start_time > end_time_open:
        logging.error(f"\nstart_time ({start_time}) must be <= end_time_open ({end_time_open})")
        raise ValueError(f"start_time ({start_time}) must be <= end_time_open ({end_time_open})")

    max_attempts = 5

    while end_time_open >= start_time:
        start_time_open = max(start_time, end_time_open - (limit - 1) * interval_int)

        spot_data = []
        attempts = 0
        while attempts < max_attempts:
            try:
                spot_data = client.get_klines(
                    symbol=symbol, interval=interval_str,
                    limit=limit, startTime=start_time_open,
                    endTime=end_time_open if end_time_open != start_time else None
                )
                break
            except Exception as e:
                attempts += 1
                logging.error(f"\nException in client.get_klines (Attempt {attempts}): {e}\n")

        if not spot_data:
            logging.error("\nMax attempts reached. No spot data received.\n")
            return pd.DataFrame()

        if is_futures_included:
            futures_data = []
            attempts = 0
            while attempts < max_attempts:
                try:
                    futures_data = client.futures_klines(
                        symbol=symbol, interval=interval_str,
                        limit=limit, startTime=start_time_open,
                        endTime=end_time_open if end_time_open != start_time else None
                    )
                    break
                except Exception as e:
                    attempts += 1
                    logging.error(f"\nException in client.futures_klines (Attempt {attempts}): {e}\n")

            if not futures_data:
                logging.error("\nMax attempts reached. No futures data received.\n")
                return pd.DataFrame()

        attributes = ["time", "open", "high", "low", "close", "volume", "number_of_trades", "taker_buy_vol"]
        spot_df = pd.DataFrame(spot_data, columns=[
            "time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_vol", "taker_buy_quote_asset_volume", "ignore"
        ])[attributes]

        logging.info("\nProcessing spot_df with data_preparation()..\n")
        spot_decimals_dict = determine_max_precision(spot_df)
        spot_precisions_list.append(spot_decimals_dict)

        spot_df = data_preparation(
            spot_df, 'time', start_time_open, end_time_open,
            interval_int, attributes, threshold_pct, conv_to_close_time=True
        )

        if is_futures_included:
            attributes = [
                "time_d", "open_d", "high_d", "low_d", "close_d", "volume_d", "number_of_trades_d", "taker_buy_vol_d"
            ]
            futures_df = pd.DataFrame(futures_data, columns=[
                "time_d", "open_d", "high_d", "low_d", "close_d", "volume_d",
                "close_time_d", "quote_asset_volume_d", "number_of_trades_d",
                "taker_buy_vol_d", "taker_buy_quote_asset_volume_d", "ignore_d"
            ])[attributes]

            logging.info("\nProcessing futures_df with data_preparation()..\n")
            fut_decimals_dict = determine_max_precision(futures_df)
            fut_precisions_list.append(fut_decimals_dict)

            futures_df = data_preparation(
                futures_df, 'time_d', start_time_open,
                end_time_open, interval_int, attributes,
                threshold_pct, conv_to_close_time=True
            )

            spot_df = spot_df.join(futures_df, how='outer')

        final_df = pd.concat([spot_df, final_df])
        end_time_open = start_time_open - interval_int

    max_spot_decimals_dict = get_max_decimals_dict(spot_precisions_list, 'time')
    logging.info(
        f"\nMax precisions for spot attributes:\n"
        f"{tabulate([max_spot_decimals_dict], headers='keys', tablefmt='pipe')}\n"
    )

    if is_futures_included:
        max_fut_decimals_dict = get_max_decimals_dict(fut_precisions_list, 'time_d')
        logging.info(
            f"\nMax precisions for futures attributes:\n"
            f"{tabulate([max_fut_decimals_dict], headers='keys', tablefmt='pipe')}\n"
        )

    logging.info("\nApplying linear interpolation where needed..\n")
    final_df.interpolate(method='linear', inplace=True)

    logging.info("\nRounding data with max precisions..\n")
    final_df = final_df.round(max_spot_decimals_dict)
    if is_futures_included:
        final_df = final_df.round(max_fut_decimals_dict)

    final_df.index.name = final_df.index.name or 'time_UTC'
    return final_df


def get_sumOpenInterest(
    client, symbol, interval_str,
    start_time_str=None, end_time_str=None,
    limit=500, threshold_pct=5
):
    """
    Retrieves historical sumOpenInterest data for a given symbol over a specified interval.

    This function queries the futures open interest history and processes the data
    with appropriate time alignment, precision checks, and interpolation.

    Args:
        client (object): Trading client instance.
        symbol (str): Trading pair symbol (e.g., "BTCUSDT").
        interval_str (str): Time interval for sumOpenInterest (e.g., "5m", "1h").
        start_time_str (str | datetime | int, optional): Start time in format "%d.%m.%Y %H:%M:%S",
            "%d.%m.%Y", as a datetime (UTC), or as an integer in milliseconds.
        end_time_str (str | datetime | int, optional): End time in the same formats. Defaults to current time if None.
        limit (int, optional): Maximum number of data points per API request. Defaults to 500.
        threshold_pct (int, optional): Threshold for missing data before logging an error. Defaults to 5.

    Returns:
        pd.DataFrame: Processed DataFrame containing sumOpenInterest data, indexed by UTC time.
    """
    logging.info("\nget_sumOpenInterest() run..\n")
    MAX_DAYS_TO_LOAD_MSEC = np.int64(28 * 24 * 60 * 60 * 1000)  # 28 days in milliseconds
    max_attempts = 5
    final_df = pd.DataFrame()
    OI_precisions_list = []
    interval_int = convert_to_msec(interval_str)

    logging.info(f"\nConverted interval: {interval_str} -> {interval_int} msec\n")

    current_time_ms = np.int64(
        math.floor(datetime.now(timezone.utc).timestamp() * 1000 / interval_int) * interval_int
    )

    # Determine the start time
    if start_time_str is None:
        start_time_close_initial = current_time_ms - MAX_DAYS_TO_LOAD_MSEC
        logging.info(
            f"\nUsing default start time (28 days before {current_time_ms}): {start_time_close_initial}\n"
        )
    else:
        start_time_close_initial = convert_time_into_ms(start_time_str, interval_int)
        if current_time_ms - start_time_close_initial > MAX_DAYS_TO_LOAD_MSEC:
            start_time_close_initial = current_time_ms - MAX_DAYS_TO_LOAD_MSEC
            logging.info(f"\nStart time exceeds 28-day limit. Adjusted to: {start_time_close_initial}\n")

    # Determine the end time
    end_time_close = current_time_ms if end_time_str is None else convert_time_into_ms(end_time_str, interval_int)

    if start_time_close_initial > end_time_close:
        logging.error(f"\nstart_time ({start_time_close_initial}) must be <= end_time ({end_time_close})")
        raise ValueError(f"start_time ({start_time_close_initial}) must be <= end_time ({end_time_close})")

    while end_time_close >= start_time_close_initial:
        start_time_close = max(start_time_close_initial, end_time_close - (limit - 1) * interval_int)

        open_interest_data = []
        attempts = 0
        while attempts < max_attempts:
            try:
                open_interest_data = client.futures_open_interest_hist(
                    symbol=symbol, period=interval_str, limit=limit,
                    startTime=start_time_close, endTime=end_time_close
                )
                break
            except Exception as e:
                attempts += 1
                logging.error(
                    f"\nException in client.futures_open_interest_hist (Attempt {attempts}): {e}\n"
                )

        if not open_interest_data:
            logging.error("\nMax attempts reached. No open interest data received.\n")
            return pd.DataFrame()

        open_interest_df = pd.DataFrame.from_records(
            open_interest_data, columns=["timestamp", "sumOpenInterest"]
        )
        attributes = ["sumOpenInterest"]

        logging.info("\nProcessing open_interest_df with data_preparation()..\n")
        OI_decimals_dict = determine_max_precision(open_interest_df)
        OI_precisions_list.append(OI_decimals_dict)

        open_interest_df = data_preparation(
            open_interest_df, 'timestamp', start_time_close, end_time_close,
            interval_int, attributes, threshold_pct, conv_to_close_time=False
        )

        final_df = pd.concat([open_interest_df, final_df])
        end_time_close = start_time_close - interval_int

    max_OI_decimals_dict = get_max_decimals_dict(OI_precisions_list, 'timestamp')
    logging.info(
        f"\nMax precisions for attributes:\n"
        f"{tabulate([max_OI_decimals_dict], headers='keys', tablefmt='pipe')}\n"
    )

    logging.info("\nApplying linear interpolation where needed..\n")
    final_df.interpolate(method='linear', inplace=True)

    logging.info("\nRounding data with max precisions..\n")
    final_df = final_df.round(max_OI_decimals_dict)

    final_df.index.name = final_df.index.name or 'time_UTC'
    return final_df


def check_datetime_utc(t):
    """
    Validates if the provided datetime object is in UTC.

    Args:
        t (datetime): The datetime object to check.

    Raises:
        ValueError: If `t` is not a datetime object or is not in UTC.
    """
    if not isinstance(t, datetime):
        raise ValueError(f"Time {t} must be of type datetime.")

    if t.tzinfo is None or t.tzinfo.utcoffset(t) != timedelta(0):
        raise ValueError(f"Time {t} must be in UTC timezone!")


def get_path(symbol):
    """
    Generates a file path for storing historical data based on the given symbol.

    Args:
        symbol (str): Trading pair symbol.

    Returns:
        str: The constructed file path.
    """
    return f'hist_data/{symbol}'


def log_initialize(symbol, path=None, log_filename=None):
    """
    Initializes logging for a given trading symbol, ensuring logs are stored in the correct location.

    This function sets up logging by creating necessary directories and configuring log file handlers.

    Args:
        symbol (str): Trading pair symbol.
        path (str, optional): Directory path for log storage. Defaults to None (generated dynamically).
        log_filename (str, optional): Custom log filename. Defaults to None (auto-generated).
    """
    if path is None:
        path = get_path(symbol)

    if not os.path.exists(path):
        os.makedirs(path)

    # Remove existing handlers to prevent duplicate logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_filename = f"{path}/{log_filename}" if log_filename else f"{path}/logfile_{symbol}.log"

    logging.basicConfig(
        filename=log_filename, level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )


def update_klines_and_OI(
    client, symbol, interval_str, start_time_reserve=None,
    is_OI_included=False, is_futures_included=True, num_days_to_get=None
):
    """
    Updates and merges historical kline (candlestick) and Open Interest (OI) data.

    This function loads existing data from CSV files, fetches new data from the trading client,
    and updates the dataset while ensuring no duplicates exist.

    Args:
        client (object): Trading client instance.
        symbol (str): Trading pair symbol (e.g., "BTCUSDT").
        interval_str (str): Time interval for klines (e.g., "5m", "1h").
        start_time_reserve (str, optional): Start time in "%Y-%m-%d %H:%M:%S" format if no prior data exists.
        is_OI_included (bool, optional): Whether to include Open Interest data. Defaults to False.
        is_futures_included (bool, optional): Whether to include futures market data. Defaults to True.
        num_days_to_get (int, optional): Limits the returned dataset to the last `num_days_to_get` days.

    Returns:
        pd.DataFrame: The updated kline and Open Interest data.
    """
    days_to_update_klines_csv = 10
    days_to_update_OI_csv = 5
    path = get_path(symbol)

    if not os.path.exists(f"{path}/{interval_str}"):
        os.makedirs(f"{path}/{interval_str}")

    df = pd.DataFrame()
    csv_files = glob.glob(f"{path}/{interval_str}/klines*.csv")
    interval_int_ms = convert_to_msec(interval_str)
    current_time = datetime.now(timezone.utc)

    # Load and merge CSV files
    for file in csv_files:
        data = pd.read_csv(file, index_col=0, parse_dates=[0], sep=';')
        data.index = pd.to_datetime(data.index, utc=True)
        df = pd.concat([df, data])

    if not df.empty:
        df = df[~df.index.duplicated(keep='first')].sort_index()
        start_time = df.index[-1] + timedelta(milliseconds=interval_int_ms)
    else:
        if start_time_reserve is None:
            raise ValueError(f"start_time_reserve must be specified for new coin {symbol}")
        start_time = datetime.strptime(start_time_reserve, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    if current_time > start_time:
        df_upd = get_klines_data(client, symbol, interval_str, start_time, current_time,
                                 is_futures_included=is_futures_included)
        if df_upd.empty:
            return pd.DataFrame()

        if df_upd.index[-1] - df_upd.index[0] >= timedelta(days=days_to_update_klines_csv):
            start_date = df_upd.index[0].strftime("%d.%m.%Y")
            end_date = df_upd.index[-1].strftime("%d.%m.%Y")
            filename = f"{path}/{interval_str}/klines_{symbol}_{interval_str}_{start_date}-{end_date}.csv"
            df_upd.to_csv(filename, sep=';', index=True)

        df = pd.concat([df, df_upd]).drop_duplicates().sort_index()

    if is_OI_included:
        min_interval_str_OI = '5m'
        max_interval_to_load_OI_d = 30
        min_interval_str_OI_ms = convert_to_msec(min_interval_str_OI)
        path_OI = f"{path}/sumOpenInterest"

        if not os.path.exists(path_OI):
            os.makedirs(path_OI)

        df_OI = pd.DataFrame()
        csv_files = glob.glob(f"{path_OI}/sumOpenInterest*.csv")

        for file in csv_files:
            data = pd.read_csv(file, index_col=0, parse_dates=[0], sep=';')
            data.index = pd.to_datetime(data.index, utc=True)
            df_OI = pd.concat([df_OI, data])

        if not df_OI.empty:
            df_OI = df_OI[~df_OI.index.duplicated(keep='first')].sort_index()

            if (df_OI.index[-1] + timedelta(milliseconds=min_interval_str_OI_ms)) >= \
               (current_time - timedelta(days=max_interval_to_load_OI_d)):
                start_time = df_OI.index[-1] + timedelta(milliseconds=min_interval_str_OI_ms)
            else:
                start_time = current_time - timedelta(days=max_interval_to_load_OI_d)
        else:
            start_time = current_time - timedelta(days=max_interval_to_load_OI_d)

        if current_time > start_time:
            df_upd = get_sumOpenInterest(client, symbol, min_interval_str_OI, start_time, current_time)

            if df_upd.index[-1] - df_upd.index[0] >= timedelta(days=days_to_update_OI_csv):
                start_date = df_upd.index[0].strftime("%d.%m.%Y")
                end_date = df_upd.index[-1].strftime("%d.%m.%Y")
                filename = f"{path_OI}/sumOpenInterest_{symbol}_{min_interval_str_OI}_{start_date}-{end_date}.csv"
                df_upd.to_csv(filename, sep=';', index=True)

            df_OI = pd.concat([df_OI, df_upd]).drop_duplicates().sort_index()

        df = df.join(df_OI['sumOpenInterest'], how='left')

    if num_days_to_get is not None:
        end_date_cut = df.index[-1]
        start_date_cut = max(df.index[0], end_date_cut - pd.Timedelta(days=num_days_to_get))
        df = df[start_date_cut:end_date_cut]

    return df


def save_to_csv(df, filename, mode='w'):
    """
    Saves a DataFrame to a CSV file.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The file path.
        mode (str): The writing mode ('w' for overwrite, 'a' for append). Default is 'w'.
    """
    df.to_csv(filename, mode=mode, header=mode == 'w', index=False, sep=';')


def save_orderbook_to_csv(client, symbol, time_UTC, path, limit_spot=5000, limit_perp=1000, aggr_step=25.0):
    """
    Fetches and saves order book data to CSV files, applying aggregation and trimming as needed.

    Parameters:
        client: Trading API client.
        symbol (str): Trading pair symbol (e.g., "BTCUSDT").
        time_UTC (datetime): Timestamp of the order book snapshot.
        path (str): Directory path to save CSV files.
        limit_spot (int): Number of levels to retrieve for spot market.
        limit_perp (int): Number of levels to retrieve for perpetual market.
        aggr_step (float): Step size for price aggregation.
    """
    MAX_ORDERBOOK_DEPTH_PCT = 5
    STEP_PCT_SPOT = 0.1
    STEP_PCT_PERP = 0.05

    if symbol == 'BTCUSDT':
        SPOT_PRICE_PRECISION = 2
        PERP_PRICE_PRECISION = 1
    else:
        SPOT_PRICE_PRECISION = PERP_PRICE_PRECISION = 0  # default precision

    def aggregate(df: pd.DataFrame, step: float, is_bid: bool) -> pd.DataFrame:
        """
        Aggregates order book data by price steps.

        Parameters:
            df (pd.DataFrame): Order book data containing 'price' and 'quantity'.
            step (float): Step size for price aggregation.
            is_bid (bool): If True, rounding is done differently for bids and asks.

        Returns:
            pd.DataFrame: Aggregated order book data grouped by price levels.
        """
        df['price'] = (df['price'] // step) * step if is_bid else np.where(
            df['price'] % step == 0,
            (df['price'] // step) * step,
            (df['price'] // step + 1) * step
        )
        return df.groupby('price').sum(numeric_only=True).reset_index()

    def trim_orderbook(
        df: pd.DataFrame, is_bid: bool, is_spot: bool, trim_pct: float = MAX_ORDERBOOK_DEPTH_PCT
    ) -> pd.DataFrame:
        """
        Trims the order book by removing orders outside a given percentage range from the best price.

        Parameters:
            df (pd.DataFrame): Order book data containing 'price'.
            is_bid (bool): If True, trims bids; otherwise, trims asks.
            is_spot (bool): If True, uses spot price precision; otherwise, uses perpetual precision.
            trim_pct (float): Percentage depth at which to trim the order book.

        Returns:
            pd.DataFrame: Trimmed order book data.
        """
        price_prec = SPOT_PRICE_PRECISION if is_spot else PERP_PRICE_PRECISION
        max_price = df['price'].max()
        min_price = df['price'].min()

        if is_bid:
            min_price_trim = round(max_price * (100 - trim_pct) / 100, price_prec)
            if min_price_trim > min_price:
                logging.info(f"\nBids trimmed by price {min_price_trim}\n")
                mask = df['price'] >= min_price_trim
                return df[mask]
        else:
            max_price_trim = round(min_price * (100 + trim_pct) / 100, price_prec)
            if max_price_trim < max_price:
                logging.info(f"\nAsks trimmed by price {max_price_trim}\n")
                mask = df['price'] <= max_price_trim
                return df[mask]
        return df

    def calculate_step_sums(df: pd.DataFrame, is_bid: bool, is_spot: bool) -> pd.DataFrame:
        """
        Calculates cumulative sums of order quantities within price step ranges.

        Parameters:
            df (pd.DataFrame): Order book data containing 'price' and 'quantity'.
            is_bid (bool): If True, processes bids; otherwise, processes asks.
            is_spot (bool): If True, uses spot price settings; otherwise, uses perpetual settings.

        Returns:
            pd.DataFrame: Dataframe containing step-wise aggregated quantities.
        """
        price_prec = SPOT_PRICE_PRECISION if is_spot else PERP_PRICE_PRECISION
        step_pct = STEP_PCT_SPOT if is_spot else STEP_PCT_PERP

        min_price = df['price'].min()
        max_price = df['price'].max()
        results = []
        i = 1

        step_value = round((step_pct / 100.0) * (max_price if is_bid else min_price), price_prec)

        if is_bid:
            bid0 = max_price
            bid1 = max_price - step_value
            while bid1 >= min_price:
                mask = (df['price'] >= bid1) & (df['price'] <= bid0)
                sum_quantity = df.loc[mask, 'quantity'].sum()
                results.append(
                    {'step_pct_from_bid0': step_pct * i, 'bid0': bid0, 'bid1': bid1, 'sum_qty': sum_quantity}
                )
                i += 1
                bid0 = bid1 - 10 ** (-price_prec)
                bid1 -= step_value
        else:
            ask0 = min_price
            ask1 = min_price + step_value
            while ask1 <= max_price:
                mask = (df['price'] >= ask0) & (df['price'] <= ask1)
                sum_quantity = df.loc[mask, 'quantity'].sum()
                results.append(
                    {'step_pct_from_ask0': step_pct * i, 'ask0': ask0, 'ask1': ask1, 'sum_qty': sum_quantity}
                )
                i += 1
                ask0 = ask1 + 10 ** (-price_prec)
                ask1 += step_value

        return pd.DataFrame(results)

    def get_order_book_with_retries(client, symbol: str, limit_spot: int, limit_perp: int):
        """
        Fetches the order book from a trading client with retry logic in case of failures.

        Parameters:
            client: Trading API client.
            symbol (str): Trading pair symbol (e.g., "BTC/USDT").
            limit_spot (int): Number of levels to retrieve for spot market.
            limit_perp (int): Number of levels to retrieve for perpetual market.

        Returns:
            tuple: (order_book_spot, order_book_perp, client_t_before_req, client_t_after_req, server_time)
        """
        MAX_RETRIES = 10

        for attempt in range(MAX_RETRIES):
            try:
                client_t_before_req = datetime.now(timezone.utc)
                order_book_spot = client.get_order_book(symbol=symbol, limit=limit_spot)
                order_book_perp = client.futures_order_book(symbol=symbol, limit=limit_perp)
                server_time = datetime.fromtimestamp(client.get_server_time()['serverTime'] / 1000, tz=timezone.utc)
                client_t_after_req = datetime.now(timezone.utc)
                break
            except Exception as e:
                logging.error(f"\nError fetching order book (attempt {attempt + 1}): {e}\n")
                time.sleep(0.5)
        else:
            logging.error("\nFailed to fetch order book after multiple retries!\n")
            return None, None, None, None, None

        return order_book_spot, order_book_perp, client_t_before_req, client_t_after_req, server_time

    def save_orderbook(order_book, time_UTC, path, timestamp, is_spot, aggr_step=None):
        """
        Processes and saves order book data (bids and asks) with optional aggregation.
        """
        suffix = 'spot' if is_spot else 'perp'
        bids = pd.DataFrame(order_book['bids'], columns=['price', 'quantity'], dtype=float)
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'quantity'], dtype=float)

        bids = trim_orderbook(bids, True, is_spot)
        asks = trim_orderbook(asks, False, is_spot)

        bids_step_sums = calculate_step_sums(bids, True, is_spot)
        asks_step_sums = calculate_step_sums(asks, False, is_spot)

        filename_bids_step_sums = f"{path}/step_sums_orderbook_{suffix}_bids_{timestamp}.csv"
        filename_asks_step_sums = f"{path}/step_sums_orderbook_{suffix}_asks_{timestamp}.csv"

        if aggr_step:
            bids = aggregate(bids, aggr_step, True)
            asks = aggregate(asks, aggr_step, False)
            filename_bids = f"{path}/aggr_orderbook_{suffix}_bids_{timestamp}.csv"
            filename_asks = f"{path}/aggr_orderbook_{suffix}_asks_{timestamp}.csv"
        else:
            filename_bids = f"{path}/orderbook_{suffix}_bids_{timestamp}.csv"
            filename_asks = f"{path}/orderbook_{suffix}_asks_{timestamp}.csv"

        bids['time_UTC'] = asks['time_UTC'] = time_UTC
        bids_step_sums['time_UTC'] = asks_step_sums['time_UTC'] = time_UTC

        save_to_csv(bids, filename_bids, mode='a' if os.path.isfile(filename_bids) else 'w')
        save_to_csv(asks, filename_asks, mode='a' if os.path.isfile(filename_asks) else 'w')
        save_to_csv(
            bids_step_sums, filename_bids_step_sums, mode='a' if os.path.isfile(
                filename_bids_step_sums
            ) else 'w'
        )
        save_to_csv(
            asks_step_sums, filename_asks_step_sums, mode='a' if os.path.isfile(
                filename_asks_step_sums
            ) else 'w'
        )

        logging.info(f"\nOrderbook {suffix} saved to {filename_bids} and {filename_asks}\n")

    logging.info("\nsave_orderbook_info_to_csv is running..\n")

    if not os.path.exists(path):
        os.makedirs(path)

    if not isinstance(time_UTC, datetime):
        raise ValueError(f"time_UTC {time_UTC} must be a datetime object")

    timestamp = time_UTC.strftime("%d.%m.%Y")

    order_book_spot, order_book_perp, client_t_before_req, client_t_after_req, server_time = (
        get_order_book_with_retries(client, symbol, limit_spot, limit_perp)
    )

    if order_book_spot and order_book_perp:
        timeinfo_data = {
            'time_UTC': time_UTC,
            'client_time_before_request': client_t_before_req,
            'client_time_after_request': client_t_after_req,
            'server_time_after_request': server_time,
            'last_update_id_spot': order_book_spot['lastUpdateId'],
            'last_update_id_perp': order_book_perp['lastUpdateId'],
            'E_perp': order_book_perp['E'],
            'T_perp': order_book_perp['T']
        }
        save_to_csv(
            pd.DataFrame(timeinfo_data, index=[0]), f"{path}/orderbook_timeinfo_{timestamp}.csv",
            mode='a' if os.path.isfile(f"{path}/orderbook_timeinfo_{timestamp}.csv") else 'w'
        )

        if aggr_step:
            save_orderbook(order_book_spot, time_UTC, path, timestamp, True, aggr_step)
            save_orderbook(order_book_perp, time_UTC, path, timestamp, False, aggr_step)
        else:
            save_orderbook(order_book_spot, time_UTC, path, timestamp, True)
            save_orderbook(order_book_perp, time_UTC, path, timestamp, False)

    else:
        logging.error("\nNo data to save. The orderbook data is empty.\n")
