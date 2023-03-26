# Load standard modules
import numpy as np
import math
import datetime

# # Load Doptrack modules
# from doptrack.recording import Recording
# from doptrack.astro import TLE, TLESatellite
#
# from sgp4.io import twoline2rv
# from sgp4.earth_gravity import wgs84

from tudatpy.kernel import constants

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


def get_tle_initial_conditions(filename: str) -> [float, np.ndarray]:

    # Load recording
    recording = Recording.load(filename, '')

    # Retrieve TLE elements
    line1_tle = recording.meta.satellite.tle.line1
    line2_tle = recording.meta.satellite.tle.line2


    # Compute initial TEME state
    tle_from_meta = TLE(line1=line1_tle, line2=line2_tle)
    delfi_tle = TLESatellite(tle=tle_from_meta)
    year = int('20' + line1_tle[18:20])
    fraction_day = float(str(line1_tle[20:23]) + '.' + str(line1_tle[24:32]))
    mon, day, hr, minute, sec = days2mdhms(year, fraction_day)

    tle_time = datetime.datetime(year, int(mon), int(day), int(hr), int(minute), int(sec))
    state_teme = delfi_tle.state_teme(tle_time)

    # Compute initial Julian date
    julian_date = jday(year, int(mon), int(day), int(hr), int(minute), int(sec))

    initial_state_array = np.array(
        [state_teme.position.x, state_teme.position.y, state_teme.position.z,
         state_teme.velocity.vx, state_teme.velocity.vy, state_teme.velocity.vz])

    initial_time = (julian_date - j2000_days) * 86400.0

    sat = twoline2rv(line1_tle, line2_tle, wgs84)
    b_star = sat.bstar

    return initial_time, initial_state_array, b_star