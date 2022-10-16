# Load standard modules
import numpy as np
import math
import datetime

# Load Doptrack modules
import doptrack.api as doptrack
from doptrack.astro import coordinates, tle
from doptrack.recording import models
from doptrack.astro import astro as astro
import GroundControl
from GroundControl import sgp4
from sgp4 import ext as util

from tudatpy.kernel import constants


def jday(year, mon, day, hr, minute, sec):

  return (367.0 * year -
          7.0 * (year + ((mon + 9.0) // 12.0)) * 0.25 // 1.0 +
          275.0 * mon // 9.0 +
          day + 1721013.5 +
          ((sec / 60.0 + minute) / 60.0 + hr) / 24.0  #  ut in days
          #  - 0.5*sgn(100.0*year + mon - 190002.5) + 0.5;
          )


def get_start_next_day(time, j2000_days):
    day = math.ceil(time / 86400.0 + j2000_days - 0.5) + 0.5
    return (day - j2000_days) * 86400.0


def get_start_current_day(time, j2000_days):
    day = math.floor(time / 86400.0 + j2000_days - 0.5) + 0.5
    return (day - j2000_days) * 86400.0


def get_tle_initial_conditions(filename: str) -> [float, np.ndarray]:

    # Load recording
    recording = models.Recording.load(filename, '')

    # Retrieve TLE elements
    line1_tle = recording.meta.satellite.tle.line1
    line2_tle = recording.meta.satellite.tle.line2

    # Compute initial TEME state
    tle_from_meta = tle.TLE(line1=line1_tle, line2=line2_tle)
    delfi_tle = astro.TLESatellite(name='delfi', noradid=32789, tle=tle_from_meta)
    year = int('20' + line1_tle[18:20])
    day = line1_tle[20:32]
    fraction_day = float(str(line1_tle[20:23]) + '.' + str(line1_tle[24:32]))
    # print('year', year, 'day', day, 'fraction_day', fraction_day)
    mon, day, hr, minute, sec = util.days2mdhms(year, fraction_day)
    print(year,'-',mon,'-',day,' ', hr,':',minute,':',sec)

    tle_time = datetime.datetime(year, int(mon), int(day), int(hr), int(minute), int(sec))
    # print('time TLE', tle_time)
    state_teme = delfi_tle.state_teme(tle_time)

    # print('initial ', state_teme_time_utc.position)
    # print('initial ', state_teme_time_utc.velocity)

    # Compute initial Julian date
    julian_date = util.jday(year, int(mon), int(day), int(hr), int(minute), int(sec))
    # print('julian date', julian_date)

    initial_state_array = np.array(
        [state_teme.position.x, state_teme.position.y, state_teme.position.z,
         state_teme.velocity.u, state_teme.velocity.v, state_teme.velocity.w])

    return julian_date, initial_state_array


def convert_frequencies_to_range_rate(frequencies):
    return -frequencies * constants.SPEED_OF_LIGHT



