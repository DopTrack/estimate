# Load standard modules
import numpy as np
import math
import datetime

from tudatpy.kernel import constants


j2000_days = 2451545.0


def jday(year, mon, day, hr, minute, sec):

  return (367.0 * year -
          7.0 * (year + ((mon + 9.0) // 12.0)) * 0.25 // 1.0 +
          275.0 * mon // 9.0 +
          day + 1721013.5 +
          ((sec / 60.0 + minute) / 60.0 + hr) / 24.0  #  ut in days
          #  - 0.5*sgn(100.0*year + mon - 190002.5) + 0.5;
          )


def _day_of_year_to_month_day(day_of_year, is_leap):
    february_bump = (2 - is_leap) * (day_of_year >= 60 + is_leap)
    august = day_of_year >= 215
    month, day = divmod(2 * (day_of_year - 1 + 30 * august + february_bump), 61)
    month += 1 - august
    day //= 2
    day += 1
    return month, day


def days2mdhms(year, days, round_to_microsecond=6):
    second = days * 86400.0
    if round_to_microsecond:
        second = round(second, round_to_microsecond)

    minute, second = divmod(second, 60.0)
    if round_to_microsecond:
        second = round(second, round_to_microsecond)

    minute = int(minute)
    hour, minute = divmod(minute, 60)
    day_of_year, hour = divmod(hour, 24)

    is_leap = year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)
    month, day = _day_of_year_to_month_day(day_of_year, is_leap)
    if month == 13:  # behave like the original in case of overflow
        month = 12
        day += 31

    return month, day, int(hour), int(minute), second


def get_start_next_day(time):
    day = math.ceil(time / 86400.0 + j2000_days - 0.5) + 0.5
    return (day - j2000_days) * 86400.0


def get_start_current_day(time):
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


