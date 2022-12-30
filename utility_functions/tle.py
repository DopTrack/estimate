# Load standard modules
import numpy as np
import math
import datetime

# # Load Doptrack modules
from doptrack.recording import Recording
from doptrack.astro import TLE, TLESatellite

from sgp4.io import twoline2rv
from sgp4.earth_gravity import wgs84

from tudatpy.kernel import constants

from utility_functions.time import *


def get_tle_initial_conditions(filename: str) -> [float, np.ndarray]:

    # Load recording
    recording = Recording.load(filename, '')

    # Retrieve TLE elements
    line1_tle = recording.meta.satellite.tle.line1
    line2_tle = recording.meta.satellite.tle.line2

    # Compute initial TEME state
    tle_from_meta = TLE(line1=line1_tle, line2=line2_tle)
    # delfi_tle = TLESatellite(name='delfi', noradid=32789, tle=tle_from_meta)
    delfi_tle = TLESatellite(tle=tle_from_meta)
    year = int('20' + line1_tle[18:20])
    day = line1_tle[20:32]
    fraction_day = float(str(line1_tle[20:23]) + '.' + str(line1_tle[24:32]))
    # print('year', year, 'day', day, 'fraction_day', fraction_day)
    mon, day, hr, minute, sec = days2mdhms(year, fraction_day)
    # print(year,'-',mon,'-',day,' ', hr,':',minute,':',sec)

    tle_time = datetime.datetime(year, int(mon), int(day), int(hr), int(minute), int(sec))
    # print('time TLE', tle_time)
    state_teme = delfi_tle.state_teme(tle_time)

    # Compute initial Julian date
    julian_date = jday(year, int(mon), int(day), int(hr), int(minute), int(sec))
    # print('julian date', julian_date)

    initial_state_array = np.array(
        [state_teme.position.x, state_teme.position.y, state_teme.position.z,
         state_teme.velocity.vx, state_teme.velocity.vy, state_teme.velocity.vz])

    initial_time = (julian_date - j2000_days) * 86400.0

    sat = twoline2rv(line1_tle, line2_tle, wgs84)
    b_star = sat.bstar

    return initial_time, initial_state_array, b_star