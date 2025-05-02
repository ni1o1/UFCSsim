import pandas as pd
from datetime import time
import warnings
from ortools.linear_solver import pywraplp
from tqdm import tqdm
import time as py_time
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np
ETA_RT = 0.90
ETA_CHARGE = np.sqrt(ETA_RT)
ETA_DISCHARGE = np.sqrt(ETA_RT)
ETA_DISCHARGE_SAFE = max(ETA_DISCHARGE, 1e-6)
from functools import lru_cache
def get_city_config(city_name):
    """Retrieves configuration information for the specified city (including prices, period definitions, and peak power price)."""
    configs = {
        'bj': {
            'scaling_factor': 1.0,
            'prices': {'low': 0.4, 'medium': 1.0, 'high': 1.6},
            'peak_power_price': 48.0,
            'periods': [
                {'name': 'low', 'start': time(23, 0), 'end': time(7, 0), 'price_key': 'low'},
                {'name': 'medium1', 'start': time(7, 0), 'end': time(10, 0), 'price_key': 'medium'},
                {'name': 'high1', 'start': time(10, 0), 'end': time(13, 0), 'price_key': 'high'},
                {'name': 'medium2', 'start': time(13, 0), 'end': time(17, 0), 'price_key': 'medium'},
                {'name': 'high2', 'start': time(17, 0), 'end': time(22, 0), 'price_key': 'high'},
                {'name': 'medium3', 'start': time(22, 0), 'end': time(23, 0), 'price_key': 'medium'},
            ]
        },
        'gz': {
            'scaling_factor': 1.0,
            'prices': {'low': 0.4, 'medium': 1.0, 'high': 1.6},
            'peak_power_price': 32.0,
            'periods': [
                {'name': 'low', 'start': time(0, 0), 'end': time(8, 0), 'price_key': 'low'},
                {'name': 'medium1', 'start': time(8, 0), 'end': time(10, 0), 'price_key': 'medium'},
                {'name': 'high1', 'start': time(10, 0), 'end': time(12, 0), 'price_key': 'high'},
                {'name': 'medium2', 'start': time(12, 0), 'end': time(14, 0), 'price_key': 'medium'},
                {'name': 'high2', 'start': time(14, 0), 'end': time(19, 0), 'price_key': 'high'},
                {'name': 'medium3', 'start': time(19, 0), 'end': time(0, 0), 'price_key': 'medium'},
            ]
        },
        'sh': {
            'scaling_factor': 1.0,
            'prices': {'low': 0.4, 'medium': 1.0, 'high': 1.6},
            'peak_power_price': 40.8,
            'periods': [
                {'name': 'low', 'start': time(22, 0), 'end': time(6, 0), 'price_key': 'low'},
                {'name': 'medium1', 'start': time(6, 0), 'end': time(8, 0), 'price_key': 'medium'},
                {'name': 'high1', 'start': time(8, 0), 'end': time(11, 0), 'price_key': 'high'},
                {'name': 'medium2', 'start': time(11, 0), 'end': time(18, 0), 'price_key': 'medium'},
                {'name': 'high2', 'start': time(18, 0), 'end': time(21, 0), 'price_key': 'high'},
                {'name': 'medium3', 'start': time(21, 0), 'end': time(22, 0), 'price_key': 'medium'},
            ]
        }
    }
    config = configs.get(city_name.lower())
    if config:
         for p in config['periods']:
            is_gz_medium3 = city_name.lower() == 'gz' and p['name'] == 'medium3'
            if p['end'] == time(0, 0) and p['start'] != time(0, 0):
                 p['_crosses_midnight'] = True
            elif p['start'] >= p['end'] and p['end'] != time(0, 0):
                 p['_crosses_midnight'] = True
            else:
                 p['_crosses_midnight'] = False
            if is_gz_medium3:
                 p['_crosses_midnight'] = True
         periods_tuple = tuple(frozenset(p.items()) for p in config['periods'])
         prices_tuple = frozenset(config['prices'].items())
         @lru_cache(maxsize=128)
         def _get_period_info_for_time_cached(t_obj, periods_tuple_arg, prices_tuple_arg):
             """Internal helper: Returns period configuration for a time object (cached)."""
             periods_cfg_local = [{k:v for k,v in dict_items} for dict_items in periods_tuple_arg]
             for i, period in enumerate(periods_cfg_local):
                 start, end = period['start'], period['end']
                 crosses_midnight = period.get('_crosses_midnight', False)
                 if crosses_midnight:
                      if t_obj >= start or t_obj < end:
                          return period['price_key'], period['name'], i
                 else:
                     if end == time(0, 0):
                          if t_obj >= start:
                              return period['price_key'], period['name'], i
                     elif t_obj >= start and t_obj < end:
                          return period['price_key'], period['name'], i
             warnings.warn(f"Time {t_obj} did not match any period! (May be at boundary or configuration gap)")
             if periods_cfg_local:
                 last_period = periods_cfg_local[-1]
                 return last_period['price_key'], last_period['name'], len(periods_cfg_local) - 1
             return None, None, -1
         config['_get_period_info'] = lambda t: _get_period_info_for_time_cached(t, periods_tuple, prices_tuple)
    return config
def preprocess_station_load_global(stationload_df):
    """
    Preprocesses station load data, adding an absolute minute index from the start time.
    """
    df = stationload_df.copy()
    if df.empty:
        return df, None
    try:
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            start_time = py_time.time()
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
        if df['time'].isnull().any():
            df = df.dropna(subset=['time'])
            if df.empty:
                return pd.DataFrame(columns=stationload_df.columns.tolist() + ['absolute_minute']), None
    except Exception as e:
        return pd.DataFrame(columns=stationload_df.columns.tolist() + ['absolute_minute']), None
    start_time = py_time.time()
    df = df.sort_values(by=['stationId', 'time']).reset_index(drop=True)
    if df.empty:
        return df, None
    t_start = df['time'].min()
    if pd.isna(t_start):
        return df, None
    start_time = py_time.time()
    df['absolute_minute'] = (df['time'] - t_start).dt.total_seconds() / 60.0
    df['absolute_minute'] = df['absolute_minute'].astype(int)
    start_time = py_time.time()
    df['time_of_day'] = df['time'].dt.time
    return df, t_start
def create_global_time_periods(processed_df, city_config, t_start):
    """
    Generates global time period instances based on the absolute minute timeline and city period configuration.
    """
    if processed_df.empty:
        return pd.Series(dtype=int), {}
    prices = city_config['prices']
    get_period_info_func = city_config['_get_period_info']
    min_abs_minute = processed_df['absolute_minute'].min()
    max_abs_minute = processed_df['absolute_minute'].max()
    total_minutes = max_abs_minute - min_abs_minute + 1
    start_time = py_time.time()
    unique_times_df = processed_df[['absolute_minute', 'time', 'time_of_day']].drop_duplicates(subset=['absolute_minute']).sort_values('absolute_minute')
    start_time = py_time.time()
    period_info = unique_times_df['time_of_day'].apply(get_period_info_func)
    unique_times_df['price_key'] = period_info.apply(lambda x: x[0] if isinstance(x, tuple) and len(x) > 0 else None)
    unique_times_df['period_name'] = period_info.apply(lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else None)
    if unique_times_df['price_key'].isnull().any() or unique_times_df['period_name'].isnull().any():
        warnings.warn("During time to period mapping, some times failed to match. Please check configuration and time data.")
        unique_times_df = unique_times_df.dropna(subset=['price_key', 'period_name'])
        if unique_times_df.empty:
            return pd.Series(dtype=int), {}
    start_time = py_time.time()
    unique_times_df['period_change'] = unique_times_df['price_key'].ne(unique_times_df['price_key'].shift())
    if not unique_times_df.empty:
        unique_times_df.iloc[0, unique_times_df.columns.get_loc('period_change')] = True
    unique_times_df['global_period_index'] = unique_times_df['period_change'].cumsum() - 1
    start_time = py_time.time()
    minute_to_global_period_sparse = unique_times_df.set_index('absolute_minute')['global_period_index']
    full_minute_index = pd.RangeIndex(start=min_abs_minute, stop=max_abs_minute + 1, name='absolute_minute')
    minute_to_global_period = minute_to_global_period_sparse.reindex(full_minute_index).ffill()
    minute_to_global_period = minute_to_global_period.bfill()
    minute_to_global_period = minute_to_global_period.astype(int)
    start_time = py_time.time()
    global_periods = {}
    if not unique_times_df.empty:
        grouped = unique_times_df.groupby('global_period_index')
        for gp_index, group_df in grouped:
            start_minute = group_df['absolute_minute'].min()
            next_gp_start_minute_series = unique_times_df.loc[unique_times_df['global_period_index'] == gp_index + 1, 'absolute_minute']
            if next_gp_start_minute_series.empty:
                end_minute = max_abs_minute + 1
            else:
                end_minute = int(next_gp_start_minute_series.min())
            duration_minutes = end_minute - start_minute
            if not group_df.empty:
                price_key = group_df['price_key'].iloc[0]
                period_name = group_df['period_name'].iloc[0]
            else:
                price_key = "unknown"
                period_name = "unknown"
            price = prices.get(price_key, 0.0)
            if price_key is None or price_key == "unknown":
                 warnings.warn(f"Global period {gp_index} (minutes {start_minute}-{end_minute-1}) failed to match price, price set to 0.")
                 price_key="unknown"
                 period_name="unknown"
            global_periods[gp_index] = {
                'start_minute': start_minute,
                'end_minute': end_minute,
                'duration_minutes': duration_minutes,
                'duration_hours': duration_minutes / 60.0,
                'price_key': price_key,
                'name': period_name,
                'price': price
            }
    num_global_periods = len(global_periods)
    return minute_to_global_period, global_periods
def calculate_ufcs_ess_params(guninfo_df, ufcs_power_threshold_kw=250.0, ess_module_capacity_kwh=501.0, ess_module_power_kw=250.5):
    """Calculates UFCS ESS parameters for each station based on guninfo."""
    df_gun = guninfo_df.copy()
    if df_gun.empty or not all(col in df_gun.columns for col in ['stationId', 'power']):
        return {}
    ufcs_guns = df_gun.loc[df_gun['power'] >= ufcs_power_threshold_kw]
    if ufcs_guns.empty:
        all_station_ids = df_gun['stationId'].unique()
        return {sid: {'soc_max_kwh': 0.0, 'max_charge_kw': 0.0, 'max_discharge_kw': 0.0} for sid in all_station_ids}
    ufcs_counts = ufcs_guns.groupby('stationId').size()
    ess_params = {}
    for station_id, num_guns in ufcs_counts.items():
        soc_max = num_guns * ess_module_capacity_kwh
        max_power = num_guns * ess_module_power_kw
        ess_params[station_id] = {
            'soc_max_kwh': soc_max,
            'max_charge_kw': max_power,
            'max_discharge_kw': max_power
        }
    all_station_ids = df_gun['stationId'].unique()
    for sid in all_station_ids:
        if sid not in ess_params:
             ess_params[sid] = {'soc_max_kwh': 0.0, 'max_charge_kw': 0.0, 'max_discharge_kw': 0.0}
    return ess_params
def optimize_single_station_wrapper(args):
    """Wrapper function for parallel execution (returns Y_net and P_peak)"""
    station_id, station_data_group, minute_to_global_period, global_periods, \
    all_ess_params, initial_soc_map, peak_power_price, scenario = args
    ess_params_for_station = all_ess_params.get(station_id, {'soc_max_kwh': 0.0, 'max_charge_kw': 0.0, 'max_discharge_kw': 0.0})
    initial_soc_for_station = initial_soc_map.get(station_id, 0.0)
    if ess_params_for_station.get('soc_max_kwh', 0) > 1e-6:
        optimal_y_station, p_peak_opt_station = optimize_global_ess_strategy_ortools(
            station_id,
            station_data_group,
            minute_to_global_period,
            global_periods,
            ess_params_for_station,
            initial_soc_for_station,
            peak_power_price=(peak_power_price if scenario == 'capacity_charge' else 0.0),
            eta_charge=ETA_CHARGE,
            eta_discharge=ETA_DISCHARGE_SAFE
        )
        return station_id, optimal_y_station, p_peak_opt_station
    else:
        n_global_periods = len(global_periods)
        return station_id, {i: 0.0 for i in range(n_global_periods)}, 0.0
def optimize_global_ess_strategy_ortools(
    station_id,
    station_data,
    minute_to_global_period,
    global_periods,
    ess_params,
    initial_soc=0.0,
    peak_power_price=0.0,
    eta_charge=ETA_CHARGE,
    eta_discharge=ETA_DISCHARGE_SAFE
):
    """
    (Core optimization logic - with energy loss and optional capacity charge) Uses OR-Tools to optimize the energy storage strategy for a single station.
    Returns: (optimal_Y_net, P_peak_opt)
            optimal_Y_net: dict {period_index: net_ess_energy_kwh}
            P_peak_opt: float, Optimized maximum grid demand (kW)
    """
    n_global_periods = len(global_periods)
    default_return = ({i: 0.0 for i in range(n_global_periods)}, 0.0)
    if ess_params.get('soc_max_kwh', 0) <= 1e-6:
        return default_return
    if station_data is None or station_data.empty:
        return default_return
    try:
        if 'absolute_minute' not in station_data.columns:
            if 'index' in station_data.columns:
                station_data = station_data.reset_index()
            else:
                 warnings.warn(f"Station {station_id}: 'absolute_minute' column missing and cannot be restored from index. Demand aggregation may fail.")
        map_index_name = minute_to_global_period.index.name if minute_to_global_period.index.name else 'absolute_minute'
        if map_index_name not in station_data.columns:
             if station_data.index.name == map_index_name:
                  station_data_map_key = station_data.index
             else:
                  if 'absolute_minute' in station_data.columns:
                      station_data_map_key = station_data['absolute_minute']
                  else:
                      return default_return
        else:
            station_data_map_key = station_data[map_index_name]
        global_period_indices = station_data_map_key.map(minute_to_global_period)
        station_data['global_period_index'] = global_period_indices.astype(int)
        power_col = 'power' if 'power' in station_data.columns else 'original_ev_power_kw'
        if power_col not in station_data.columns:
             return default_return
        demand_kwh_per_global_period = station_data.groupby('global_period_index')[power_col].sum() * (1.0 / 60.0)
        demand_kwh_per_global_period = demand_kwh_per_global_period.reindex(range(n_global_periods), fill_value=0.0).to_dict()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return default_return
    soc_max = ess_params['soc_max_kwh']
    max_charge_kw = ess_params['max_charge_kw']
    max_discharge_kw = ess_params['max_discharge_kw']
    N = n_global_periods
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return default_return
    infinity = solver.infinity()
    G = [solver.NumVar(0.0, infinity, f'G_{i}') for i in range(N)]
    Y_discharge = [solver.NumVar(0.0, infinity, f'Y_discharge_{i}') for i in range(N)]
    Y_charge = [solver.NumVar(0.0, infinity, f'Y_charge_{i}') for i in range(N)]
    SoC = [solver.NumVar(0.0, soc_max, f'SoC_{i}') for i in range(N)]
    P_peak = solver.NumVar(0.0, infinity, 'P_peak')
    for i in range(N):
        P_i = demand_kwh_per_global_period.get(i, 0.0)
        solver.Add(G[i] + Y_discharge[i] == P_i + Y_charge[i], f'energy_balance_{i}')
    for i in range(N):
        duration_h = global_periods[i]['duration_hours']
        if duration_h > 1e-6:
            discharge_limit_kwh = max_discharge_kw * duration_h
            solver.Add(Y_discharge[i] <= discharge_limit_kwh, f'discharge_ac_limit_{i}')
            charge_limit_kwh = max_charge_kw * duration_h
            solver.Add(Y_charge[i] <= charge_limit_kwh, f'charge_ac_limit_{i}')
            solver.Add(G[i] <= P_peak * duration_h, f'peak_power_constraint_{i}')
        else:
            solver.Add(Y_discharge[i] == 0.0, f'no_discharge_zero_duration_{i}')
            solver.Add(Y_charge[i] == 0.0, f'no_charge_zero_duration_{i}')
            solver.Add(G[i] == 0.0, f'no_grid_zero_duration_{i}')
    solver.Add(SoC[0] == initial_soc - (Y_discharge[0] / eta_discharge) + (Y_charge[0] * eta_charge), 'soc_transition_0')
    for i in range(1, N):
        solver.Add(SoC[i] - SoC[i-1] + (Y_discharge[i] / eta_discharge) - (Y_charge[i] * eta_charge) == 0, f'soc_transition_{i}')
    solver.Add(SoC[N-1] == initial_soc, 'periodicity')
    objective = solver.Objective()
    for i in range(N):
        price = global_periods[i]['price']
        objective.SetCoefficient(G[i], price)
    if peak_power_price > 1e-6:
        objective.SetCoefficient(P_peak, peak_power_price)
    objective.SetMinimization()
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        optimal_Y_net = {i: 0.0 for i in range(N)}
        for i in range(N):
            y_discharge_val = Y_discharge[i].solution_value()
            y_charge_val = Y_charge[i].solution_value()
            y_net = y_discharge_val - y_charge_val
            optimal_Y_net[i] = 0.0 if abs(y_net) < 1e-6 else y_net
        p_peak_opt_val = P_peak.solution_value()
        p_peak_opt_val = 0.0 if abs(p_peak_opt_val) < 1e-6 else p_peak_opt_val
        return optimal_Y_net, p_peak_opt_val
    else:
        status_map = {
             pywraplp.Solver.OPTIMAL: 'OPTIMAL',
             pywraplp.Solver.FEASIBLE: 'FEASIBLE',
             pywraplp.Solver.INFEASIBLE: 'INFEASIBLE',
             pywraplp.Solver.UNBOUNDED: 'UNBOUNDED',
             pywraplp.Solver.ABNORMAL: 'ABNORMAL',
             pywraplp.Solver.NOT_SOLVED: 'NOT_SOLVED',
        }
        status_str = status_map.get(status, f'Unknown ({status})')
        return default_return
def distribute_ess_power_globally(
    all_station_loads_processed,
    all_optimal_Y,
    all_optimal_P_peak,
    all_ess_params,
    city_config,
    minute_to_global_period,
    global_periods,
    scenario='unregulated',
    L_mw=None
):
    """
    (Global version) Distributes optimized global period energy to minute-level power (optimized version + capacity_charge scenario).
    """
    if scenario == 'capacity_charge':
        pass
    elif scenario != 'unregulated':
        pass
    else:
        pass
    df = all_station_loads_processed.copy()
    df['ess_power_kw'] = 0.0
    scaling_factor = city_config.get('scaling_factor', 1.0)
    df['scaled_ev_power_kw'] = df['original_ev_power_kw'] * scaling_factor
    if scaling_factor != 1.0:
        pass
    start_time = py_time.time()
    if minute_to_global_period.index.name != 'absolute_minute':
         try: minute_to_global_period.index.name = 'absolute_minute'
         except: warnings.warn("Could not set index name 'absolute_minute' for minute_to_global_period")
    df['global_period_index'] = df['absolute_minute'].map(minute_to_global_period)
    if df['global_period_index'].isnull().any():
         warnings.warn("During distribution, some minutes could not be mapped to global period index! These rows will be skipped.")
         df = df.dropna(subset=['global_period_index'])
         if df.empty:
             return pd.DataFrame(columns=['stationId', 'time', 'original_ev_power_kw', 'scaled_ev_power_kw', 'ess_power_kw', 'grid_power_kw'])
    df['global_period_index'] = df['global_period_index'].astype(int)
    all_station_ids = df['stationId'].unique()
    n_global_periods = len(global_periods)
    remaining_Y = {}
    has_ess_station = set()
    for station_id in all_station_ids:
        station_optimal_y = all_optimal_Y.get(station_id, {})
        ess_p = all_ess_params.get(station_id, {})
        if ess_p.get('soc_max_kwh', 0) > 1e-6:
            has_ess_station.add(station_id)
        for gp_idx in range(n_global_periods):
            remaining_Y[(station_id, gp_idx)] = station_optimal_y.get(gp_idx, 0.0)
    start_time = py_time.time()
    df = df.sort_values(by=['absolute_minute', 'stationId']).reset_index()
    start_time = py_time.time()
    absolute_minutes = df['absolute_minute'].to_numpy()
    station_ids = df['stationId'].to_numpy()
    global_period_indices = df['global_period_index'].to_numpy()
    scaled_ev_power_kw = df['scaled_ev_power_kw'].to_numpy()
    ess_power_kw_array = df['ess_power_kw'].to_numpy()
    start_time_loop = py_time.time()
    unique_minutes, minute_start_indices = np.unique(absolute_minutes, return_index=True)
    progress_bar = tqdm(range(len(unique_minutes)), total=len(unique_minutes), desc=" - Distribution Progress (Global Minutes)")
    for i in progress_bar:
        abs_minute = unique_minutes[i]
        start_idx = minute_start_indices[i]
        end_idx = minute_start_indices[i+1] if i + 1 < len(unique_minutes) else len(df)
        current_station_ids = station_ids[start_idx:end_idx]
        current_gp_indices = global_period_indices[start_idx:end_idx]
        current_scaled_ev_power = scaled_ev_power_kw[start_idx:end_idx]
        current_ess_power_view = ess_power_kw_array[start_idx:end_idx]
        current_global_period_index = current_gp_indices[0]
        discharge_power_this_minute = np.zeros_like(current_ess_power_view)
        total_discharge_kw = 0.0
        for j in range(len(current_station_ids)):
            station_id = current_station_ids[j]
            if station_id in has_ess_station:
                state_key = (station_id, current_global_period_index)
                y_rem_kwh = remaining_Y.get(state_key, 0.0)
                if y_rem_kwh > 1e-6:
                    ess_p = all_ess_params[station_id]
                    max_discharge_kw_station = ess_p['max_discharge_kw']
                    power_limit_from_energy = y_rem_kwh * 60.0
                    potential_discharge_kw = min(max_discharge_kw_station, power_limit_from_energy)
                    current_station_ev_demand_kw = max(0.0, current_scaled_ev_power[j])
                    actual_discharge_kw = max(0.0, min(potential_discharge_kw, current_station_ev_demand_kw))
                    actual_discharge_kw = round(actual_discharge_kw, 6)
                    if actual_discharge_kw > 0:
                        discharge_power_this_minute[j] = actual_discharge_kw
                        total_discharge_kw += actual_discharge_kw
                        remaining_Y[state_key] -= actual_discharge_kw / 60.0
                        if remaining_Y[state_key] < 0: remaining_Y[state_key] = 0.0
        charge_power_this_minute = np.zeros_like(current_ess_power_view)
        desired_charge_requests = {}
        station_id_map_for_charge = {}
        base_grid_load_kw_total = current_scaled_ev_power.sum() - discharge_power_this_minute.sum()
        for j in range(len(current_station_ids)):
            station_id = current_station_ids[j]
            if station_id in has_ess_station:
                state_key = (station_id, current_global_period_index)
                y_rem_kwh = remaining_Y.get(state_key, 0.0)
                if y_rem_kwh < -1e-6:
                    ess_p = all_ess_params[station_id]
                    max_charge_kw_station = ess_p['max_charge_kw']
                    power_limit_from_energy = abs(y_rem_kwh * 60.0)
                    desired_charge_kw = max(0.0, min(max_charge_kw_station, power_limit_from_energy))
                    desired_charge_kw = round(desired_charge_kw, 6)
                    if desired_charge_kw > 0:
                        desired_charge_requests[j] = desired_charge_kw
                        station_id_map_for_charge[j] = station_id
        total_desired_charge_kw = sum(desired_charge_requests.values())
        if total_desired_charge_kw >= 1e-6:
             if scenario == 'unregulated':
                 for j, desired_kw in desired_charge_requests.items():
                     station_id = station_id_map_for_charge[j]
                     state_key = (station_id, current_global_period_index)
                     allocated_kw = desired_kw
                     charge_power_this_minute[j] = -allocated_kw
                     remaining_Y[state_key] += allocated_kw / 60.0
                     if remaining_Y[state_key] > 0: remaining_Y[state_key] = 0.0
             elif scenario == 'capacity_charge':
                 for j, desired_kw in desired_charge_requests.items():
                     station_id = station_id_map_for_charge[j]
                     state_key = (station_id, current_global_period_index)
                     p_peak_opt_station = all_optimal_P_peak.get(station_id, float('inf'))
                     station_base_grid_load = max(0.0, current_scaled_ev_power[j] - discharge_power_this_minute[j])
                     station_charge_headroom = max(0.0, p_peak_opt_station - station_base_grid_load)
                     allocated_kw = min(desired_kw, station_charge_headroom)
                     allocated_kw = round(allocated_kw, 6)
                     if allocated_kw < 1e-4: allocated_kw = 0.0
                     if allocated_kw > 0:
                         charge_power_this_minute[j] = -allocated_kw
                         remaining_Y[state_key] += allocated_kw / 60.0
                         if remaining_Y[state_key] > 0: remaining_Y[state_key] = 0.0
             elif scenario == 'demand_response_curtailment':
                 L_kw_global = L_mw * 1000.0 if L_mw is not None else float('inf')
                 available_grid_for_ess_charge_kw = max(0.0, L_kw_global - base_grid_load_kw_total)
                 current_ess_charge_load_alloc = 0.0
                 requesting_indices = sorted(list(desired_charge_requests.keys()))
                 for j in requesting_indices:
                     station_id = station_id_map_for_charge[j]
                     state_key = (station_id, current_global_period_index)
                     desired_kw = desired_charge_requests[j]
                     if current_ess_charge_load_alloc + desired_kw <= available_grid_for_ess_charge_kw + 1e-6:
                         allocated_kw = desired_kw
                         charge_power_this_minute[j] = -allocated_kw
                         current_ess_charge_load_alloc += allocated_kw
                         remaining_Y[state_key] += allocated_kw / 60.0
                         if remaining_Y[state_key] > 0: remaining_Y[state_key] = 0.0
             elif scenario == 'demand_response_scaling':
                 L_kw_global = L_mw * 1000.0 if L_mw is not None else float('inf')
                 available_grid_for_ess_charge_kw = max(0.0, L_kw_global - base_grid_load_kw_total)
                 scaling_factor_dr = 1.0
                 if total_desired_charge_kw > available_grid_for_ess_charge_kw + 1e-6:
                     scaling_factor_dr = (available_grid_for_ess_charge_kw / total_desired_charge_kw
                                          if total_desired_charge_kw > 1e-9 else 0.0)
                     scaling_factor_dr = max(0.0, min(1.0, scaling_factor_dr))
                 for j, desired_kw in desired_charge_requests.items():
                     station_id = station_id_map_for_charge[j]
                     state_key = (station_id, current_global_period_index)
                     allocated_kw = desired_kw * scaling_factor_dr
                     allocated_kw = round(allocated_kw, 6)
                     if allocated_kw < 1e-4: allocated_kw = 0.0
                     if allocated_kw > 0:
                        charge_power_this_minute[j] = -allocated_kw
                        remaining_Y[state_key] += allocated_kw / 60.0
                        if remaining_Y[state_key] > 0: remaining_Y[state_key] = 0.0
        net_ess_power = discharge_power_this_minute + charge_power_this_minute
        ess_power_kw_array[start_idx:end_idx] = net_ess_power
    progress_bar.close()
    start_time = py_time.time()
    df['ess_power_kw'] = ess_power_kw_array
    df['grid_power_kw'] = df['scaled_ev_power_kw'] - df['ess_power_kw']
    grid_power_np = df['grid_power_kw'].to_numpy()
    neg_grid_mask = grid_power_np < -1e-3
    num_neg_grid = np.sum(neg_grid_mask)
    if num_neg_grid > 0:
        grid_power_np[neg_grid_mask] = 0.0
        df['grid_power_kw'] = grid_power_np
    df['grid_power_kw'] = np.clip(df['grid_power_kw'], 0, None)
    try:
        df = df.rename(columns={'power': 'original_ev_power_kw'}, errors='ignore')
    except Exception as e:
        pass
    output_columns = [
        'stationId', 'time',
        'original_ev_power_kw', 'scaled_ev_power_kw',
        'ess_power_kw', 'grid_power_kw'
    ]
    if 'original_ev_power_kw' not in df.columns and 'power' in all_station_loads_processed.columns:
         output_columns.remove('original_ev_power_kw')
         output_columns.insert(2, 'power')
         df = df.rename(columns={'power': 'power'}, errors='ignore')
    final_df = df[[col for col in output_columns if col in df.columns]]
    return final_df.reset_index(drop=True)
def optimize_ess_strategy(
    stationload_df_orig,
    guninfo_df_orig,
    city_name,
    ufcs_power_threshold_kw=250.0,
    ess_module_capacity_kwh=501.0,
    ess_module_power_kw=250.5,
    scenario='unregulated',
    L_mw=None,
    initial_soc_ratio=0.0,
    num_workers=None
):
    """
    (Global version) Executes the energy storage optimization process based on the global timeline (integrated parallel optimization + capacity charge scenario).
    """
    start_total_time = py_time.time()
    if scenario == 'capacity_charge':
        pass
    elif scenario != 'unregulated':
        pass
    else:
        pass
    stationload = stationload_df_orig.copy()
    guninfo = guninfo_df_orig.copy()
    city_config = get_city_config(city_name)
    if not city_config:
        return None
    peak_power_price = city_config.get('peak_power_price', 0.0)
    if scenario == 'capacity_charge' and peak_power_price <= 0:
        warnings.warn(f"Warning: peak_power_price for city '{city_name}' is not set or 0 in 'capacity_charge' scenario. Proceeding without capacity charge.")
    elif scenario == 'capacity_charge':
         pass
    preprocess_start = py_time.time()
    stationload_processed, t_start = preprocess_station_load_global(stationload)
    if t_start is None or stationload_processed is None or stationload_processed.empty:
        return pd.DataFrame(columns=['stationId', 'time', 'original_ev_power_kw', 'scaled_ev_power_kw', 'ess_power_kw', 'grid_power_kw'])
    if 'power' in stationload_processed.columns:
         stationload_processed = stationload_processed.rename(columns={'power': 'original_ev_power_kw'})
    all_station_ids_in_load = sorted(stationload_processed['stationId'].unique())
    if len(all_station_ids_in_load) == 0:
        return pd.DataFrame(columns=['stationId', 'time', 'original_ev_power_kw', 'scaled_ev_power_kw', 'ess_power_kw', 'grid_power_kw'])
    create_periods_start = py_time.time()
    minute_to_global_period, global_periods = create_global_time_periods(
        stationload_processed, city_config, t_start
    )
    if not global_periods or minute_to_global_period is None or minute_to_global_period.empty:
        return None
    n_global_periods = len(global_periods)
    calc_ess_start = py_time.time()
    all_ess_params = calculate_ufcs_ess_params(
        guninfo,
        ufcs_power_threshold_kw,
        ess_module_capacity_kwh,
        ess_module_power_kw
    )
    for sid in all_station_ids_in_load:
        if sid not in all_ess_params:
             all_ess_params[sid] = {'soc_max_kwh': 0.0, 'max_charge_kw': 0.0, 'max_discharge_kw': 0.0}
    ufcs_station_ids_with_ess = {sid for sid, params in all_ess_params.items() if params.get('soc_max_kwh', 0) > 1e-6}
    initial_soc_map = {}
    for sid in all_station_ids_in_load:
        params = all_ess_params.get(sid, {})
        initial_soc_kwh = params.get('soc_max_kwh', 0.0) * initial_soc_ratio
        initial_soc_map[sid] = initial_soc_kwh
    optimize_start = py_time.time()
    all_optimal_Y = {}
    all_optimal_P_peak = {}
    start_group_time = py_time.time()
    grouped_data = stationload_processed.groupby('stationId')
    start_task_prep_time = py_time.time()
    tasks = []
    for station_id in all_station_ids_in_load:
        try:
            station_data_group = grouped_data.get_group(station_id)
        except KeyError:
            station_data_group = pd.DataFrame()
        tasks.append((
            station_id,
            station_data_group,
            minute_to_global_period,
            global_periods,
            all_ess_params,
            initial_soc_map,
            peak_power_price,
            scenario
        ))
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    try:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(optimize_single_station_wrapper, tasks),
                                total=len(tasks), desc=" - Optimization Progress"))
        for station_id, optimal_y_station, p_peak_station in results:
            all_optimal_Y[station_id] = optimal_y_station
            all_optimal_P_peak[station_id] = p_peak_station
    except Exception as e:
        all_optimal_Y = {}
        all_optimal_P_peak = {}
        station_iterator = tqdm(tasks, total=len(tasks), desc=" - Optimization Progress (Serial)")
        for task_args in station_iterator:
             station_id, optimal_y, p_peak = optimize_single_station_wrapper(task_args)
             all_optimal_Y[station_id] = optimal_y
             all_optimal_P_peak[station_id] = p_peak
    distribute_start = py_time.time()
    final_results_df = distribute_ess_power_globally(
        stationload_processed,
        all_optimal_Y,
        all_optimal_P_peak,
        all_ess_params,
        city_config,
        minute_to_global_period,
        global_periods,
        scenario=scenario,
        L_mw=L_mw
    )
    if final_results_df is None or final_results_df.empty:
        return pd.DataFrame(columns=['stationId', 'time', 'original_ev_power_kw', 'scaled_ev_power_kw', 'ess_power_kw', 'grid_power_kw'])
    end_total_time = py_time.time()
    return final_results_df