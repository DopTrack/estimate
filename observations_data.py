# Load standard modules
import numpy as np

import util_functions
from estimation import define_link_ends
from util_functions import jday

# Load tudatpy modules
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation.estimation_setup import observation
import tudatpy.kernel.numerical_simulation.estimation as tudat_estimation


def process_observations(filename: str, fraction_discarded: float = 0.1) -> np.array:

    observations = []
    j2000_days = 2451545.0

    f = open(filename, 'r')
    lines = f.readlines()
    nb_points = len(lines)
    # print('nb_points', nb_points)
    nb_discarded_points = int(fraction_discarded / 2.0 * nb_points)
    # print('nb_discarded_points', nb_discarded_points)

    # Compute initial epoch
    result = lines[1].strip().split(',')
    epoch = result[0].split(' ')
    date = epoch[0].split('-')
    time = epoch[1].split(':')
    year = date[0]
    month = date[1]
    day = date[2]
    hour = time[0]
    minute = time[1]
    second = time[2]
    # print(hour, minute, second)

    initial_julian_date = jday(int(year), int(month), int(day), int(hour), int(minute), int(0)) + float(second) / 86400.0
    # print(initial_julian_date)
    initial_epoch = (initial_julian_date - j2000_days) * 86400.0 - float(result[1])
    # print('initial epoch', initial_epoch)

    # Retrieve observations of interest
    for line in lines[nb_discarded_points + 1:nb_points - nb_discarded_points]:
        result = line.strip().split(',')
        observations.append([float(result[1]) + initial_epoch, float(result[2])])
    f.close()

    return np.array(observations)


def load_and_format_observations(data_folder, data, index_files=[]):

    passes_start_times = []

    if len(index_files) == 0:
        for i in range(len(data)):
            index_files.append(i)

    existing_data = process_observations(data_folder + data[index_files[0]])
    passes_start_times.append(existing_data[0, 0] - 10.0)
    for i in range(1, len(index_files)):
        data_set = process_observations(data_folder + data[index_files[i]])
        passes_start_times.append(data_set[0, 0] - 10.0)
        existing_data = np.concatenate((existing_data, data_set))
    obs_times = existing_data[:, 0].tolist()
    obs_values = []
    for i in range(len(existing_data)):
        obs_values.append(np.array([-existing_data[i, 1]/constants.SPEED_OF_LIGHT]))

    # Set existing observations
    existing_observation_set = (define_link_ends(), (obs_values, obs_times))
    observations_input = dict()
    observations_input[observation.one_way_doppler_type] = existing_observation_set

    observations_set = tudat_estimation.set_existing_observations(observations_input, observation.receiver)

    return passes_start_times, obs_times, observations_set