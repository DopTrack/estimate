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


def get_days_starting_times(passes_start_times):
    j2000_days = 2451545.0

    days_start_times = []
    for i in range(len(passes_start_times)):
        current_day = math.floor(passes_start_times[i] / 86400.0 + j2000_days - 0.5) + 0.5
        current_day = (current_day - j2000_days) * 86400.0
        if (current_day in days_start_times) == False:
            days_start_times.append(current_day)

    return days_start_times


def get_days_end_times(days_start_times):
    days_end_times = []
    for i in range(len(days_start_times)):
        days_end_times.append(days_start_times[i] + 86400.0)

    return days_end_times



