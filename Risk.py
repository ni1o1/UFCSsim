import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

def calculate_over_ramping_risk_norm(
    normal_daily_profile: pd.Series,
    ufcs_daily_profile: pd.Series,
    rr_threshold: float,
    window_minutes: int = 30
) -> pd.Series:
    """
    Calculate the per-minute over-ramping risk probability due to UFCS deployment.

    The risk probability is calculated based on the frequency at which the Ramping Load
    exceeds a given RR threshold within a specific window around the time point
    (e.g., +/- 30 minutes).

    Args:
        normal_daily_profile (pd.Series): Original load profile (without UFCS).
                                          Should contain 1440 data points (minutes in a day).
                                          Index should represent time.
        ufcs_daily_profile (pd.Series): Load profile after UFCS deployment.
                                        Format and length should be the same as normal_daily_profile.
        rr_threshold (float):         Ramp Rate (RR) threshold.
                                        Ramping Load exceeding this value is considered an over-ramping event.
        window_minutes (int):         Single-sided size of the time window (in minutes) used for probability calculation.
                                        Defaults to 30 minutes, resulting in a total window size of 2*30+1 = 61 minutes.

    Returns:
        pd.Series: A Pandas Series containing 1440 probability values,
                   representing the P_over_ramping(t, RR) risk probability for each minute of the day.
                   Index is the same as the input Series.

    Raises:
        ValueError: If input Series length is not 1440 or indexes do not match.
        TypeError: If input is not a Pandas Series.
    """

    if not isinstance(normal_daily_profile, pd.Series) or \
       not isinstance(ufcs_daily_profile, pd.Series):
        raise TypeError("Input load profiles must be Pandas Series type.")
    if len(normal_daily_profile) != 1440 or len(ufcs_daily_profile) != 1440:
        raise ValueError("Input load profiles must have a length of 1440 (minutes in a day).")
    if not normal_daily_profile.index.equals(ufcs_daily_profile.index):
        raise ValueError("Input load profiles must have the same index.")
    if not isinstance(rr_threshold, (int, float)) or rr_threshold < 0:
         raise ValueError("rr_threshold must be a non-negative number.")
    if not isinstance(window_minutes, int) or window_minutes < 0:
         raise ValueError("window_minutes must be a non-negative integer.")

    ramping_load = (normal_daily_profile - ufcs_daily_profile).abs()
    ramping_load.name = "ramping_load"

    T = len(ramping_load)
    window_half_size = window_minutes
    window_size = 2 * window_half_size + 1

    over_ramping_probabilities = []

    ramping_load_values = ramping_load.values

    for i in range(T):
        window_indices = []
        for offset in range(-window_half_size, window_half_size + 1):
            idx = (i + offset) % T
            window_indices.append(idx)

        window_ramping_loads = ramping_load_values[window_indices]

        n_over_ramping_in_window = np.sum(window_ramping_loads > rr_threshold)

        if window_size > 0:
            probability = n_over_ramping_in_window / window_size
        else:
            probability = np.nan

        over_ramping_probabilities.append(probability)

    result_series = pd.Series(
        over_ramping_probabilities,
        index=normal_daily_profile.index,
        name=f"P_over_ramping_RR={rr_threshold}_win={window_size}"
    )

    return result_series

def parse_profile_string(profile_str: str) -> pd.Series:
    """Helper function: Parse a specific format string into a Pandas Series"""
    parts = profile_str.split()
    times = parts[0::2]
    values = [float(v) for v in parts[1::2]]
    try:
        index = times
    except ValueError:
        index = times
    name_parts = []
    if "Name:" in parts:
         name_index = parts.index("Name:")
         name_parts = parts[name_index+1:]
         if "Length:" in name_parts:
             name_parts = name_parts[:name_parts.index("Length:")]
         if "dtype:" in name_parts:
              name_parts = name_parts[:name_parts.index("dtype:")]
         series_name = " ".join(name_parts).strip().rstrip(',')
         values = [float(v) for v in parts[1:name_index:2]]
         index = parts[0:name_index:2]
    else:
         series_name = "parsed_profile"

    return pd.Series(values, index=index, name=series_name)

def calculate_over_ramping_risk_kde(
    normal_daily_profile: pd.Series,
    ufcs_daily_profile: pd.Series,
    rr_threshold: float,
    window_minutes: int = 30
) -> pd.Series:
    """
    Calculate the per-minute over-ramping risk probability due to UFCS deployment (using KDE).

    The risk probability is calculated as P(R > RR) based on the Ramping Load data
    within a specific window around the time point (e.g., +/- 30 minutes),
    estimated using Kernel Density Estimation (KDE).

    Args:
        normal_daily_profile (pd.Series): Original load profile (without UFCS).
                                          Should contain 1440 data points (minutes in a day).
                                          Index should represent time.
        ufcs_daily_profile (pd.Series): Load profile after UFCS deployment.
                                        Format and length should be the same as normal_daily_profile.
        rr_threshold (float):         Ramp Rate (RR) threshold.
                                        Probability of exceeding this value is calculated.
        window_minutes (int):         Single-sided size of the time window (in minutes) used for probability calculation.
                                        Defaults to 30 minutes, resulting in a total window size of 2*30+1 = 61 minutes.

    Returns:
        pd.Series: A Pandas Series containing 1440 probability values,
                   representing the P_over_ramping(t, RR) risk probability for each minute of the day (based on KDE).
                   Index is the same as the input Series.

    Raises:
        ValueError: If input Series length is not 1440 or indexes do not match.
        TypeError: If input is not a Pandas Series.
        ImportError: If scipy is not installed.
    """
    if not isinstance(normal_daily_profile, pd.Series) or \
       not isinstance(ufcs_daily_profile, pd.Series):
        raise TypeError("Input load profiles must be Pandas Series type.")
    if len(normal_daily_profile) != 1440 or len(ufcs_daily_profile) != 1440:
        raise ValueError("Input load profiles must have a length of 1440 (minutes in a day).")
    if not normal_daily_profile.index.equals(ufcs_daily_profile.index):
        raise ValueError("Input load profiles must have the same index.")
    if not isinstance(rr_threshold, (int, float)) or rr_threshold < 0:
         raise ValueError("rr_threshold must be a non-negative number.")
    if not isinstance(window_minutes, int) or window_minutes < 0:
         raise ValueError("window_minutes must be a non-negative integer.")

    ramping_load = (normal_daily_profile - ufcs_daily_profile).abs()
    ramping_load.name = "ramping_load"

    T = len(ramping_load)
    window_half_size = window_minutes
    window_size = 2 * window_half_size + 1
    over_ramping_probabilities = []
    ramping_load_values = ramping_load.values

    for i in range(T):
        window_indices = [(i + offset) % T for offset in range(-window_half_size, window_half_size + 1)]
        window_ramping_loads = ramping_load_values[window_indices]

        probability = np.nan

        try:
            if len(window_ramping_loads) >= 2 and np.var(window_ramping_loads) > 1e-9:
                kde = gaussian_kde(window_ramping_loads)
                probability = kde.integrate_box_1d(rr_threshold, np.inf)

            elif len(window_ramping_loads) > 0:
                if np.all(window_ramping_loads == window_ramping_loads[0]):
                     probability = 1.0 if window_ramping_loads[0] > rr_threshold else 0.0
                else:
                    n_over_ramping_in_window = np.sum(window_ramping_loads > rr_threshold)
                    actual_window_len = len(window_ramping_loads)
                    probability = n_over_ramping_in_window / actual_window_len if actual_window_len > 0 else 0.0


            probability = np.clip(probability, 0.0, 1.0)

        except Exception as e:
            probability = np.nan

        over_ramping_probabilities.append(probability)

    result_series = pd.Series(
        over_ramping_probabilities,
        index=normal_daily_profile.index,
        name=f"P_over_ramping_KDE_RR={rr_threshold}_win={window_size}"
    )

    return result_series


def plot_risk(sim_results_pivot,ymin=0,ymax=0.5,unit = 'GW'):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['Arial']

    fig = plt.figure(1, figsize=(4.5, 2.5), dpi=300)
    ax = plt.subplot(111)


    if unit=='GW':
        im = ax.imshow(sim_results_pivot.T, cmap='RdYlBu_r',vmin=0, vmax=1,extent=[0, 24, sim_results_pivot.columns.max()/1000, sim_results_pivot.columns.min()/1000],aspect='auto',label = None)
    if unit=='MW':
        im = ax.imshow(sim_results_pivot.T, cmap='RdYlBu_r',vmin=0, vmax=1,extent=[0, 24, sim_results_pivot.columns.max(), sim_results_pivot.columns.min()],aspect='auto',label = None)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Probability of exceeding reserve", rotation=-90, va="bottom")


    plt.gca().invert_yaxis()
    plt.ylabel(f'Regulating reserves ({unit})')
    plt.xlabel('Time (h)')
    plt.xticks( np.arange(0, 24, 2),rotation=0)
    plt.xlim(0, 23.75)
    plt.ylim(ymin,ymax)
    plt.show()