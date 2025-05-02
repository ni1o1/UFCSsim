import pandas as pd
import numpy as np
import os
import warnings
from ortools.linear_solver import pywraplp
from datetime import time, timedelta, datetime, date
from tqdm import tqdm
import copy
from math import ceil
from collections import defaultdict
import pandas as pd
import io

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from math import ceil
from tqdm import tqdm
import numba

def simulate_ufcs_upgrade(city, target_ufcs_stations, ufcs_power=480, ufcs_guns_per_station=2, station_types=None):
    """
    Simulates the impact on charging records after upgrading some charging guns to UFCS.
    Guns are selected based on the power increase they provide.

    Args:
        city (str): City identifier ('bj', 'gz', or 'sh').
        target_ufcs_stations (int): The target number of stations to conceptually upgrade.
                                     Used to calculate the total *nominal* power increase target.
        ufcs_power (int, optional): The target power for upgraded UFCS guns. Defaults to 480 kW.
        ufcs_guns_per_station (int, optional): The number of guns per station assumed for the target calculation. Defaults to 2.
        station_types (list or str, optional): A list of station types (from stationtype.csv 'type' column)
                                               to prioritize for upgrades. If None, all types are considered equally.

    Returns:
        tuple:
            - pd.DataFrame: A new DataFrame with simulated charging records after upgrades.
                            For upgraded guns, the 'power' is set to `ufcs_power`, and 'time_end' is adjusted
                            based on the assumption that energy remains constant.
            - set: A set of gunId values that were selected for the upgrade simulation.
    """
    data_path = 'data'
    gunrecords_file = os.path.join(data_path, f'{city}gunrecords.csv')
    stationtype_file = os.path.join(data_path, f'{city}stationtype.csv')

    if not os.path.exists(gunrecords_file):
        raise FileNotFoundError(f"Charge record file not found: {gunrecords_file}")
    if not os.path.exists(stationtype_file):
        raise FileNotFoundError(f"Station type file not found: {stationtype_file}")

    gunrecords = pd.read_csv(gunrecords_file)
    stationtype = pd.read_csv(stationtype_file)

    if city == 'gz':
        expansion_factor = 3.99
    elif city == 'bj':
        expansion_factor = 4.49
    elif city == 'sh':
        expansion_factor = 5.25
    else:
        raise ValueError("Unsupported city name, please use 'bj', 'gz', or 'sh'")

    stationtype = stationtype.rename(columns={'id': 'stationId'})
    merged_data = pd.merge(gunrecords, stationtype[['stationId', 'type']], on='stationId', how='left')
    merged_data['time_start'] = pd.to_datetime(merged_data['time_start'])
    merged_data['time_end'] = pd.to_datetime(merged_data['time_end'])

    invalid_time_mask = merged_data['time_end'] <= merged_data['time_start']
    if invalid_time_mask.any():
        pass

    target_nominal_power_increase = target_ufcs_stations * ufcs_guns_per_station * ufcs_power
    target_upgrade_power_increase_sampled = target_nominal_power_increase / expansion_factor

    unique_guns = merged_data[['gunId', 'power', 'type']].drop_duplicates(subset=['gunId']).copy()

    original_gun_count = len(unique_guns)
    unique_guns = unique_guns[unique_guns['power'] < ufcs_power].copy()
    filtered_gun_count = len(unique_guns)

    unique_guns['power_increase'] = ufcs_power - unique_guns['power']

    selected_gun_ids = set()
    accumulated_increase = 0.0
    remaining_guns = unique_guns.copy()

    if station_types:
        if isinstance(station_types, str):
            station_types = [station_types]

        priority_mask = remaining_guns['type'].isin(station_types)
        priority_guns = remaining_guns[priority_mask]
        total_priority_increase = priority_guns['power_increase'].sum()

        if total_priority_increase >= target_upgrade_power_increase_sampled:
            priority_guns_shuffled = priority_guns.sample(frac=1).reset_index(drop=True)
            for _, gun in priority_guns_shuffled.iterrows():
                if accumulated_increase < target_upgrade_power_increase_sampled:
                    selected_gun_ids.add(gun['gunId'])
                    accumulated_increase += gun['power_increase']
                else:
                    break
            remaining_guns = remaining_guns[~remaining_guns['gunId'].isin(selected_gun_ids)]
        else:
            selected_gun_ids.update(priority_guns['gunId'])
            accumulated_increase += total_priority_increase
            remaining_guns = remaining_guns[~priority_mask]
            remaining_target_increase = target_upgrade_power_increase_sampled - accumulated_increase

            if remaining_target_increase > 0 and not remaining_guns.empty:
                remaining_guns_shuffled = remaining_guns.sample(frac=1).reset_index(drop=True)
                for _, gun in remaining_guns_shuffled.iterrows():
                    if accumulated_increase < target_upgrade_power_increase_sampled:
                        selected_gun_ids.add(gun['gunId'])
                        accumulated_increase += gun['power_increase']
                    else:
                        break
                remaining_guns = remaining_guns[~remaining_guns['gunId'].isin(selected_gun_ids)]
            elif remaining_target_increase > 0:
                warnings.warn(f"Total power increase from specified types is insufficient, and remaining gun pool is empty or also insufficient to reach target {target_upgrade_power_increase_sampled:.2f} kW. Final accumulated power increase is {accumulated_increase:.2f} kW.")

    if not station_types or (accumulated_increase < target_upgrade_power_increase_sampled and not remaining_guns.empty):
        if not station_types:
             pass
        remaining_guns_shuffled = remaining_guns.sample(frac=1).reset_index(drop=True)
        for _, gun in remaining_guns_shuffled.iterrows():
             if accumulated_increase < target_upgrade_power_increase_sampled:
                 selected_gun_ids.add(gun['gunId'])
                 accumulated_increase += gun['power_increase']
             else:
                 break

    if accumulated_increase < target_upgrade_power_increase_sampled:
         warnings.warn(f"Total power increase from all upgradable guns in the dataset is insufficient to reach target {target_upgrade_power_increase_sampled:.2f} kW. Final accumulated power increase is {accumulated_increase:.2f} kW.")

    simulated_data = merged_data.copy()
    upgrade_mask = simulated_data['gunId'].isin(selected_gun_ids)

    records_to_modify = simulated_data.loc[upgrade_mask].copy()
    original_duration_hours = (records_to_modify['time_end'] - records_to_modify['time_start']) / pd.Timedelta(hours=1)
    original_duration_hours = original_duration_hours.clip(lower=0)
    original_energy = records_to_modify['power'] * original_duration_hours

    if ufcs_power <= 0:
        raise ValueError("ufcs_power must be greater than 0")
    new_duration_hours = original_energy / ufcs_power
    new_duration_hours = new_duration_hours.clip(lower=0)

    new_time_end = records_to_modify['time_start'] + pd.to_timedelta(new_duration_hours, unit='h')

    simulated_data.loc[upgrade_mask, 'power'] = ufcs_power
    simulated_data.loc[upgrade_mask, 'time_end'] = new_time_end

    return simulated_data, selected_gun_ids

def get_load_profile(minute_load_data: pd.Series, profile_type) -> pd.Series:
    """
    Calculates or extracts a daily load profile based on the specified type.

    Args:
        minute_load_data (pd.Series): A time series of minute-level load data
                                     with a DatetimeIndex.
        profile_type (str or datetime.date or datetime.datetime or pd.Timestamp):
            - If a string, must be one of pandas aggregation methods applicable to groupby ('mean', 'std', etc.)
              to compute an aggregated daily profile across all available days.
            - If a date/datetime object or string parseable as date, extracts the load profile for that specific date.

    Returns:
        pd.Series: A daily load profile (1440 minutes) with time-of-day index.
                   Returns a zero profile if no data or invalid type.
    """
    if not isinstance(minute_load_data, pd.Series):
        raise TypeError("Input 'minute_load_data' must be a pandas Series.")
    if not isinstance(minute_load_data.index, pd.DatetimeIndex):
        raise TypeError("Input Series index must be a DatetimeIndex.")

    if minute_load_data.empty:
        time_index = pd.timedelta_range(start=0, periods=1440, freq='T') + pd.Timestamp('1900-01-01')
        return pd.Series(0.0, index=time_index.time, name="empty_profile")


    valid_aggregations = {'mean', 'std', 'median', 'max', 'min', 'sum'}

    base_time_index_dt = pd.date_range("2000-01-01", periods=1440, freq='T')
    full_day_minutes_time = base_time_index_dt.time

    target_date = None
    is_aggregation = False

    if isinstance(profile_type, str):
        if profile_type in valid_aggregations:
            is_aggregation = True
        else:
            try:
                target_date = pd.to_datetime(profile_type).date()
            except ValueError:
                raise ValueError(f"Invalid profile_type: '{profile_type}'. Must be a valid date string or one of {valid_aggregations}.")
    elif isinstance(profile_type, (datetime.date, datetime.datetime, pd.Timestamp)):
        if isinstance(profile_type, (datetime.datetime, pd.Timestamp)):
             target_date = profile_type.date()
        else:
             target_date = profile_type
    else:
         raise ValueError(f"Invalid profile_type: '{profile_type}'. Must be a date/datetime object, date string, or one of {valid_aggregations}.")

    profile_result = None

    if is_aggregation:
        try:
            grouped_by_time = minute_load_data.groupby(minute_load_data.index.time)
            aggregated_data = getattr(grouped_by_time, profile_type)()
            profile_result = aggregated_data.reindex(full_day_minutes_time).fillna(0)
            profile_result.name = f"{profile_type}_daily_profile"
        except Exception as e:
            profile_result = pd.Series(0.0, index=full_day_minutes_time, name=f"error_{profile_type}_profile")
    elif target_date:
        daily_data = minute_load_data[minute_load_data.index.date == target_date]
        if daily_data.empty:
            profile_result = pd.Series(0.0, index=full_day_minutes_time, name=f"profile_{target_date}_nodata")
        else:
            daily_data.index = daily_data.index.time
            profile_result = daily_data.reindex(full_day_minutes_time).fillna(0)
            profile_result.name = f"profile_{target_date}"
    else:
         raise RuntimeError("Internal logic error: Should be either aggregation or specific date.")

    return profile_result

@numba.njit
def _estimate_non_tou_numba_core(
    start_offsets,
    stay_durations_min,
    energies_needed,
    total_power_array,
    charger_rated_kw,
    alpha, beta, avg_ratio, min_tail_ratio
):
    """
    Numba JIT function for the core calculation logic: iterate stays, calculate charge power, accumulate.

    Args:
        start_offsets (np.ndarray): Minute offset from the global start time for each stay (int).
        stay_durations_min (np.ndarray): Maximum duration of each stay (minutes, int).
        energies_needed (np.ndarray): Energy needed for each stay (kWh, float).
        total_power_array (np.ndarray): NumPy array to accumulate total power (float, modified in-place).
        charger_rated_kw (float): Charger rated power (float, scalar).
        alpha, beta, avg_ratio, min_tail_ratio (float): Charging curve parameters (float, scalars).
    """
    n_stays = len(start_offsets)
    total_minutes = len(total_power_array)
    processed_count = 0

    for i in range(n_stays):
        stay_duration_min_int = stay_durations_min[i]
        energy_needed = energies_needed[i]
        start_offset = start_offsets[i]

        if stay_duration_min_int <= 0 or energy_needed <= 1e-6 or start_offset < 0:
            continue

        potential_power_kw = piecewise_charging_profile_numba(
            rated_kw=charger_rated_kw,
            duration_min=stay_duration_min_int,
            alpha=alpha, beta=beta, avg_ratio=avg_ratio, min_tail_ratio=min_tail_ratio
        )

        if potential_power_kw.size == 0:
            continue

        energy_per_minute_kwh = potential_power_kw / 60.0
        cumulative_energy_kwh = np.cumsum(energy_per_minute_kwh)

        minutes_to_charge_idx = np.searchsorted(cumulative_energy_kwh, energy_needed, side='left')

        actual_charge_duration_min = min(minutes_to_charge_idx + 1, stay_duration_min_int)

        if actual_charge_duration_min <= 0:
             continue

        power_to_add = np.zeros(actual_charge_duration_min, dtype=np.float64)
        power_to_add[:] = potential_power_kw[:actual_charge_duration_min]

        if minutes_to_charge_idx < stay_duration_min_int:
            energy_charged_before_last = 0.0
            if minutes_to_charge_idx > 0:
                energy_charged_before_last = cumulative_energy_kwh[minutes_to_charge_idx - 1]

            energy_needed_in_last_minute = energy_needed - energy_charged_before_last
            power_for_last_minute = max(0.0, energy_needed_in_last_minute * 60.0)

            original_last_minute_power = power_to_add[-1]
            adjusted_last_minute_power = max(0.0, min(power_for_last_minute, min(original_last_minute_power, charger_rated_kw)))
            power_to_add[-1] = adjusted_last_minute_power

        end_offset = start_offset + actual_charge_duration_min

        actual_end_offset = min(end_offset, total_minutes)
        actual_len_to_add = actual_end_offset - start_offset

        if actual_len_to_add > 0:
            total_power_array[start_offset:actual_end_offset] += power_to_add[:actual_len_to_add]
            processed_count += 1

    return processed_count

def estimate_non_tou_charging_demand(stay_df: pd.DataFrame,
                                           battery_capacity_kwh: float,
                                           charger_rated_kw: float,
                                           target_soc: float = 100.0,
                                           profile_params: dict = None
                                           ) -> pd.Series:
    """
    (Numba accelerated version) Estimates the total EV charging power demand time series (1-minute resolution)
    assuming no TOU (Time-of-Use) charging behavior.

    Args:
        stay_df (pd.DataFrame): DataFrame containing stay records with 'stime', 'etime', 'ssoc'.
                                'duration' column is also supported if present.
        battery_capacity_kwh (float): The battery capacity of the EVs (assumed constant).
        charger_rated_kw (float): The rated power of the charger (assumed constant).
        target_soc (float, optional): The target SOC (State of Charge) to reach during the stay. Defaults to 100.0.
        profile_params (dict, optional): Dictionary containing parameters for the piecewise charging profile.
                                         Keys can include 'alpha', 'beta', 'avg_ratio', 'min_tail_ratio'.
                                         Defaults are used if not provided.

    Returns:
        pd.Series: A time series (1-minute resolution) of estimated total charging power demand (kW).
                   Index is DatetimeIndex. Returns an empty Series if no valid records or time range.
    """
    import time
    start_time_total = time.time()

    if profile_params is None:
        profile_params = {}
    alpha = profile_params.get('alpha', 0.05)
    beta = profile_params.get('beta', 0.10)
    avg_ratio = profile_params.get('avg_ratio', 0.55)
    min_tail_ratio = profile_params.get('min_tail_ratio', 0.10)

    df = stay_df.copy()
    df['stime'] = pd.to_datetime(df['stime'])
    df['etime'] = pd.to_datetime(df['etime'])

    if 'duration' not in df.columns or df['duration'].isnull().any():
        df['duration_sec'] = (df['etime'] - df['stime']).dt.total_seconds()
    else:
        df['duration_sec'] = df['duration']

    df['duration_min'] = np.maximum(1, np.ceil(df['duration_sec'] / 60.0)).astype(int)

    df['energy_needed_kwh'] = (target_soc - df['ssoc'].clip(upper=target_soc)) / 100.0 * battery_capacity_kwh

    df_filtered = df[(df['duration_min'] > 0) & (df['energy_needed_kwh'] > 1e-6)].copy()

    if df_filtered.empty:
        return pd.Series(dtype=float)

    min_time = df_filtered['stime'].min().floor('min')
    max_time = df_filtered['etime'].max().ceil('min')

    if pd.isna(min_time) or pd.isna(max_time) or max_time < min_time:
         return pd.Series(dtype=float)

    time_index = pd.date_range(start=min_time, end=max_time, freq='min')
    total_minutes = len(time_index)
    total_power_array = np.zeros(total_minutes, dtype=np.float64)

    global_start_ts_sec = min_time.timestamp()
    start_offsets_np = ((pd.to_datetime(df_filtered['stime']).values.astype(np.int64) // 10**9 - global_start_ts_sec) / 60).astype(int)

    stay_durations_min_np = df_filtered['duration_min'].values
    energies_needed_np = df_filtered['energy_needed_kwh'].values

    valid_indices = start_offsets_np >= 0
    if not np.all(valid_indices):
        warnings.warn(f"Warning: Filtering out {np.sum(~valid_indices)} records starting before the global minimum time.")
        start_offsets_np = start_offsets_np[valid_indices]
        stay_durations_min_np = stay_durations_min_np[valid_indices]
        energies_needed_np = energies_needed_np[valid_indices]

    if len(start_offsets_np) == 0:
        return pd.Series(0.0, index=time_index, name='Total Power (kW)')

    start_numba_time = time.time()
    processed_count = _estimate_non_tou_numba_core(
        start_offsets_np,
        stay_durations_min_np,
        energies_needed_np,
        total_power_array,
        charger_rated_kw,
        alpha, beta, avg_ratio, min_tail_ratio
    )
    end_numba_time = time.time()

    total_power_profile = pd.Series(total_power_array, index=time_index, name='Total Power (kW)')

    end_time_total = time.time()

    return total_power_profile

@numba.njit
def piecewise_charging_profile_numba(rated_kw: float,
                                     duration_min: int,
                                     alpha: float = 0.05,
                                     beta: float = 0.10,
                                     avg_ratio: float = 0.55,
                                     min_tail_ratio: float = 0.10
                                     ) -> np.ndarray:
    """
    (Numba accelerated version) Generates a 1-minute resolution EV fast charging power sequence (kW) as a NumPy array.

    Args:
        rated_kw (float): Rated power of the charger (kW).
        duration_min (int): Total duration of the charging session (minutes).
        alpha (float, optional): Ratio of ramp-up phase duration to total duration (0-1). Defaults to 0.05.
        beta (float, optional): Ratio of peak phase duration to total duration (0-1). Defaults to 0.10.
        avg_ratio (float, optional): Ratio of average power to rated power (0-1). Defaults to 0.55.
                                     Used to infer the ending power of the tail phase.
        min_tail_ratio (float, optional): Minimum power during the tail phase as a ratio of rated power (0-1). Defaults to 0.10.

    Returns:
        np.ndarray: A NumPy array of power values (kW) for each minute of the duration.
                    Returns an empty array if duration_min is <= 0.
    """
    if duration_min <= 0:
        return np.zeros(0, dtype=np.float64)

    T = duration_min
    tau_ramp = alpha
    tau_peak = beta
    tau_tail = 1.0 - tau_ramp - tau_peak

    if tau_tail < 0:
        tau_peak = 1.0 - tau_ramp
        tau_tail = 0.0
        if tau_peak < 0:
             tau_ramp = 1.0
             tau_peak = 0.0
             tau_tail = 0.0

    P_end = rated_kw
    if tau_tail > 1e-6:
        P_end_calc = (2.0 * (avg_ratio - (tau_ramp / 2.0 + tau_peak)) / tau_tail) * rated_kw - rated_kw
        P_end = max(min_tail_ratio * rated_kw, min(P_end_calc, rated_kw))
    else:
         P_end = rated_kw
         if tau_ramp < 1.0:
             tau_peak = 1.0 - tau_ramp
         else:
             tau_peak = 0.0

    m = np.arange(duration_min)

    m_ramp_end = int(np.ceil(T * tau_ramp))
    m_peak_end = m_ramp_end + int(np.ceil(T * tau_peak))

    m_ramp_end = min(m_ramp_end, T)
    m_peak_end = min(m_peak_end, T)

    m_tail_start = m_peak_end

    power = np.zeros(T, dtype=np.float64)

    if m_ramp_end > 0:
        for i in range(m_ramp_end):
             power[i] = rated_kw * (i + 1.0) / m_ramp_end

    if m_peak_end > m_ramp_end:
        for i in range(m_ramp_end, m_peak_end):
            power[i] = rated_kw

    if T > m_peak_end and tau_tail > 1e-6:
        tail_duration_m = T - m_tail_start
        if tail_duration_m > 0:
             if m_tail_start < T:
                  power[m_tail_start] = rated_kw
                  if tail_duration_m > 1:
                      for i in range(1, tail_duration_m):
                           current_m_in_tail = i
                           idx = m_tail_start + i
                           if idx < T:
                               power[idx] = rated_kw - (rated_kw - P_end) * (current_m_in_tail / (tail_duration_m - 1.0))

    elif T > m_peak_end:
        for i in range(m_peak_end, T):
             power[i] = P_end

    for i in range(T):
        power[i] = max(0.0, min(power[i], rated_kw))

    return power

@numba.njit
def _accumulate_load_numba(start_offsets, durations, powers,
                           total_load_np, global_start_ts_minutes,
                           alpha, beta, avg_ratio, min_tail_ratio):
    """
    Numba JIT function to calculate and accumulate power profile of each charging session
    to the total load array.

    Args:
        start_offsets (np.ndarray): Minute offset from the global start time for each session (int).
        durations (np.ndarray): Duration of each session (minutes, int).
        powers (np.ndarray): Rated power of each session (float).
        total_load_np (np.ndarray): NumPy array for accumulating total load (modified in-place).
        global_start_ts_minutes: Unix timestamp of the global start time (in minutes).
        alpha, beta, avg_ratio, min_tail_ratio: Charging curve parameters.
    """
    n_records = len(start_offsets)
    total_minutes = len(total_load_np)

    for i in range(n_records):
        start_offset = start_offsets[i]
        duration_min = durations[i]
        rated_power = powers[i]

        if duration_min <= 0 or rated_power <= 0:
            continue

        power_profile_np = piecewise_charging_profile_numba(
            rated_kw=rated_power, duration_min=duration_min,
            alpha=alpha, beta=beta, avg_ratio=avg_ratio, min_tail_ratio=min_tail_ratio
        )

        start_idx = start_offset
        end_idx = min(start_idx + duration_min, total_minutes)
        profile_len_to_add = end_idx - start_idx

        if profile_len_to_add <= 0:
            continue

        total_load_np[start_idx : end_idx] += power_profile_np[:profile_len_to_add]

def calculate_minute_load(gunrecords: pd.DataFrame,
                                city: str,
                                freq: str = 'T',
                                alpha: float = 0.05,
                                beta: float = 0.10,
                                avg_ratio: float = 0.55,
                                min_tail_ratio: float = 0.10) -> pd.Series:
    """
    (Numba accelerated version) Calculates the scaled total charging load at the specified frequency.

    Args:
        gunrecords (pd.DataFrame): DataFrame containing gun records with 'time_start', 'time_end', 'power'.
        city (str): City identifier ('bj', 'gz', or 'sh') used for scaling.
        freq (str, optional): The frequency for the output time series (e.g., 'T' for minute, 'H' for hour). Defaults to 'T'.
        alpha (float, optional): Charging profile parameter. Defaults to 0.05.
        beta (float, optional): Charging profile parameter. Defaults to 0.10.
        avg_ratio (float, optional): Charging profile parameter. Defaults to 0.55.
        min_tail_ratio (float, optional): Charging profile parameter. Defaults to 0.10.

    Returns:
        pd.Series: A time series of scaled total load (kW) at the specified frequency.
                   Index is DatetimeIndex. Returns an empty Series if no valid records.
    """
    required_cols = ['time_start', 'time_end', 'power']
    if not all(col in gunrecords.columns for col in required_cols):
        raise KeyError(f"Input DataFrame is missing required columns: {required_cols}")

    scaling_factors = {'gz': 3.99, 'bj': 4.49, 'sh': 5.25}
    if city not in scaling_factors:
        raise ValueError(f"Invalid city identifier '{city}'. Must be one of {list(scaling_factors.keys())}")
    r = scaling_factors[city]

    if freq not in ['T', 'min', 'Min', 'T']:
         freq = 'T'

    df = gunrecords.copy()
    df['time_start'] = pd.to_datetime(df['time_start'])
    df['time_end'] = pd.to_datetime(df['time_end'])
    df['power'] = pd.to_numeric(df['power'])

    df = df[(df['time_start'] < df['time_end']) & (df['power'] > 0)].copy()

    if df.empty:
        try:
            min_time_orig = pd.to_datetime(gunrecords['time_start']).min()
            max_time_orig = pd.to_datetime(gunrecords['time_end']).max()
            if pd.isna(min_time_orig) or pd.isna(max_time_orig): return pd.Series(dtype=float, name='scaled_load')
            start = min_time_orig.floor(freq)
            end = max_time_orig.ceil(freq)
            if start <= end:
                time_index = pd.date_range(start=start, end=end, freq=freq, name='timestamp')
                return pd.Series(0.0, index=time_index, name='scaled_load')
            else: return pd.Series(dtype=float, name='scaled_load')
        except Exception: return pd.Series(dtype=float, name='scaled_load')

    min_time = df['time_start'].min().floor(freq)
    max_time = df['time_end'].max().ceil(freq)

    if pd.isna(min_time) or pd.isna(max_time) or max_time < min_time:
        return pd.Series(dtype=float, name='scaled_load')

    full_time_index = pd.date_range(start=min_time, end=max_time, freq=freq, name='timestamp')
    total_minutes = len(full_time_index)
    total_load_np = np.zeros(total_minutes, dtype=np.float64)

    global_start_ts_sec = min_time.timestamp()
    df['start_offset_min'] = ((pd.to_datetime(df['time_start']).values.astype(np.int64) // 10**9 - global_start_ts_sec) / 60).astype(int)

    df['duration_min'] = np.ceil((df['time_end'] - df['time_start']).dt.total_seconds() / 60).astype(int)

    start_offsets_np = df['start_offset_min'].values
    durations_np = df['duration_min'].values
    powers_np = df['power'].values

    _accumulate_load_numba(
        start_offsets_np, durations_np, powers_np,
        total_load_np,
        int(global_start_ts_sec / 60),
        alpha, beta, avg_ratio, min_tail_ratio
    )

    minute_load_scaled = total_load_np * r
    result_series = pd.Series(minute_load_scaled, index=full_time_index, name='scaled_load')

    return result_series

def calculate_load_profile_metrics(normal_profile: pd.Series, ufcs_profile: pd.Series) -> dict:
    """
    Calculates and compares peak-to-valley difference metrics for normal and UFCS load profiles.
    Hourly peak-to-valley difference uses wrapped boundary handling.

    Args:
        normal_profile (pd.Series): Daily load profile for the normal case (index is time, value is load, length should be 1440).
        ufcs_profile (pd.Series): Daily load profile after UFCS deployment (index is time, value is load, length should be 1440).

    Returns:
        dict: Dictionary containing metrics for both cases.
              Structure is:
              {
                  'normal': {
                      'daily_peak_valley_diff': float,
                      'max_hourly_peak_valley_diff': float
                  },
                  'ufcs': {
                      'daily_peak_valley_diff': float,
                      'max_hourly_peak_valley_diff': float
                  }
              }
    """
    results = {}
    profiles = {'normal': normal_profile, 'ufcs': ufcs_profile}
    window_half_width = 30
    window_size = 2 * window_half_width + 1

    for name, profile in profiles.items():
        if not isinstance(profile, pd.Series):
            raise TypeError(f"Input '{name}_profile' must be of type pandas Series")
        if len(profile) != 1440:
             raise ValueError(f"Input '{name}_profile' must have a length of 1440, current is {len(profile)}")
        if profile.isnull().any():
            raise ValueError(f"Input '{name}_profile' contains null values (NaN). Please handle nulls first.")

        daily_max = profile.max()
        daily_min = profile.min()
        daily_peak_valley_diff = daily_max - daily_min

        padded_profile = pd.concat([
            profile.iloc[-window_half_width:],
            profile,
            profile.iloc[:window_half_width]
        ], ignore_index=True)

        rolling_max_padded = padded_profile.rolling(window=window_size, center=True, min_periods=window_size).max()
        rolling_min_padded = padded_profile.rolling(window=window_size, center=True, min_periods=window_size).min()

        valid_rolling_max = rolling_max_padded.iloc[window_half_width : window_half_width + len(profile)]
        valid_rolling_min = rolling_min_padded.iloc[window_half_width : window_half_width + len(profile)]

        rolling_peak_valley_diff = valid_rolling_max - valid_rolling_min

        max_hourly_peak_valley_diff = rolling_peak_valley_diff.max()

        results[name] = {
            'daily_peak_valley_diff': daily_peak_valley_diff,
            'max_hourly_peak_valley_diff': max_hourly_peak_valley_diff
        }

    return results
