# Load standard modules
import numpy as np
import math
import datetime

from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs84
from sgp4.propagation import sgp4, sgp4init

from tudatpy.kernel.astro import element_conversion

from utility_functions.time import *

import yaml


def get_tle_ref_time(filename: str) -> [float, np.ndarray]:

    with open(filename, 'r') as metafile:
        metadata = yaml.load(metafile, Loader=yaml.FullLoader)
    time_pps = metadata['Sat']['Record']['time pps']
    rx_time = metadata['Sat']['uhd']['rx_time']
    julian_date_time_pps = jday(time_pps.year, int(time_pps.month), int(time_pps.day), int(time_pps.hour), int(time_pps.minute), int(time_pps.second))
    time_pps = (julian_date_time_pps - j2000_days) * 86400.0

    return time_pps+rx_time


def extract_tle_from_old_yml(filename: str):
    with open(filename, 'r') as metafile:
        metadata = yaml.load(metafile, Loader=yaml.FullLoader)
    line1_tle = metadata["Sat"]["Predict"]["used TLE line1"]
    line2_tle = metadata["Sat"]["Predict"]["used TLE line2"]

    return line1_tle, line2_tle


def extract_tle_from_yml(filename: str):
    with open(filename, 'r') as metafile:
        metadata = yaml.load(metafile, Loader=yaml.FullLoader)
    line1_tle = metadata["satellite"]["tle"]["line1"]
    line2_tle = metadata["satellite"]["tle"]["line2"]

    return line1_tle, line2_tle


def get_tle_initial_conditions(filename: str, old_yml=False) -> [float, np.ndarray, float]:

    # Retrieve TLE
    if old_yml:
        line1_tle, line2_tle = extract_tle_from_old_yml(filename)
    else:
        line1_tle, line2_tle = extract_tle_from_yml(filename)

    # Compute initial Julian date
    year = int('20' + line1_tle[18:20])
    fraction_day = float(str(line1_tle[20:23]) + '.' + str(line1_tle[24:32]))
    mon, day, hr, minute, sec = days2mdhms(year, fraction_day)

    julian_date = jday(year, int(mon), int(day), int(hr), int(minute), 0) + sec / 86400.0
    initial_time = (julian_date - j2000_days) * 86400.0

    # Retrieve initial teme state from TLE
    sat = twoline2rv(line1_tle, line2_tle, wgs84)
    initial_state_sgp4 = sgp4(sat, 0.0)
    b_star_coef = sat.bstar
    initial_state_array = np.concatenate((np.array(initial_state_sgp4[0]), np.array(initial_state_sgp4[1])), axis=None) * 1.0e3

    return initial_time, initial_state_array, b_star_coef


def propagate_sgp4(filename: str, initial_epoch: float, epochs: list[float], old_yml=False) -> np.array:
    # Retrieve TLE
    if old_yml:
        line1_tle, line2_tle = extract_tle_from_old_yml(filename)
    else:
        line1_tle, line2_tle = extract_tle_from_yml(filename)

    sat = twoline2rv(line1_tle, line2_tle, wgs84)

    propagated_states = []
    # Propagate to epoch with sgp4
    for time in epochs:
        state_sgp4 = sgp4(sat, (time-initial_epoch)/60.0)
        state_teme_array = np.concatenate((np.array(state_sgp4[0]), np.array(state_sgp4[1])))*1.0e3
        propagated_states.append(np.concatenate((np.array([time]), element_conversion.teme_state_to_j2000(time, state_teme_array)), axis=None))

    return np.array(propagated_states)
