# Load standard modules
import numpy as np
from scipy.interpolate import interp1d # interpolation function

from estimation_functions.estimation import define_link_ends
from utility_functions.time import jday
from utility_functions.tle import get_tle_ref_time

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


def process_observations_new(filename: str, ref_time, fraction_discarded: float = 0.1) -> np.array:

    observations = []
    j2000_days = 2451545.0

    f = open(filename, 'r')
    lines = f.readlines()
    nb_points = len(lines)
    print('nb_points', nb_points)
    nb_discarded_points = int(fraction_discarded / 2.0 * nb_points)
    # print('nb_discarded_points', nb_discarded_points)

    # Retrieve observations of interest
    for line in lines[nb_discarded_points + 1:nb_points - nb_discarded_points]:
        result = line.strip().split(',')
        observations.append([float(result[0])+ref_time, float(result[2])])
    f.close()

    return np.array(observations)


def load_and_format_observations(data_folder, data, index_files=[], metadata=[], new_obs_format=False):

    passes_start_times = []
    passes_end_times = []

    if len(index_files) == 0:
        for i in range(len(data)):
            index_files.append(i)

    if not new_obs_format:
        existing_data = process_observations(data_folder + data[index_files[0]])
    else:
        if len(metadata) == 0:
            raise Exception('Error when using new observation format, metadata should be provided as input to load_and_format_observations')
        ref_time = get_tle_ref_time(data_folder + metadata[0])
        existing_data = process_observations_new(data_folder + data[index_files[0]], ref_time)

    passes_start_times.append(existing_data[0, 0] - 10.0)
    passes_end_times.append(existing_data[np.shape(existing_data)[0]-1, 0]+10.0)
    for i in range(1, len(index_files)):
        if not new_obs_format:
            data_set = process_observations(data_folder + data[index_files[i]])
        else:
            ref_time = get_tle_ref_time(data_folder + metadata[i])
            data_set = process_observations_new(data_folder + data[index_files[i]], ref_time)
        passes_start_times.append(data_set[0, 0] - 10.0)
        passes_end_times.append(data_set[np.shape(data_set)[0]-1, 0] + 10.0)
        existing_data = np.concatenate((existing_data, data_set))
    obs_times = existing_data[:, 0].tolist()
    obs_values = []
    for i in range(len(existing_data)):
        # obs_values.append(np.array([-existing_data[i, 1]/constants.SPEED_OF_LIGHT]))
        obs_values.append(np.array([-existing_data[i, 1]]))

    # Define link ends
    link_ends = dict()
    link_ends[observation.receiver] = observation.body_reference_point_link_end_id("Earth", "DopTrackStation")
    link_ends[observation.transmitter] = observation.body_origin_link_end_id("Delfi")

    # Set existing observations
    existing_observation_set = (link_ends, (obs_values, obs_times))
    observations_input = dict()
    observations_input[observation.one_way_instantaneous_doppler_type] = existing_observation_set

    observations_set = tudat_estimation.set_existing_observations(observations_input, observation.receiver)

    return passes_start_times, passes_end_times, obs_times, observations_set


def convert_frequencies_to_range_rate(frequencies):
    return frequencies #* constants.SPEED_OF_LIGHT


def get_observations_single_pass(single_pass_start_time, single_pass_end_time, observations_set):

    observations = observations_set.concatenated_observations
    times = observations_set.concatenated_times

    selected_obs = []
    selected_times = []

    for i in range(len(observations)):
        if single_pass_start_time <= times[i] <= single_pass_end_time:
            selected_obs.append(observations[i])
            selected_times.append(times[i])

    obs_array = np.zeros((len(selected_obs), 2))
    for i in range(len(selected_obs)):
        obs_array[i, 0] = selected_times[i]
        obs_array[i, 1] = selected_obs[i]

    return obs_array


def interpolate_obs(simulated_obs, real_obs):
    interpolation_function = interp1d(simulated_obs[:, 0], simulated_obs[:, 1], kind='cubic')
    min_time_simulated = min(simulated_obs[:, 0])
    max_time_simulated = max(simulated_obs[:, 0])

    interpolated_real_obs = []

    obs_times_comparison = []
    for i in range(len(real_obs[:, 0])):
        if min_time_simulated < real_obs[i, 0] < max_time_simulated:
            obs_times_comparison.append(real_obs[i, 0])
            interpolated_real_obs.append([real_obs[i, 0], real_obs[i,1]])

    interpolated_real_obs = np.array(interpolated_real_obs)

    interpolated_simulated_obs = np.zeros(np.shape(interpolated_real_obs))
    interpolated_simulated_obs[:,0] = interpolated_real_obs[:,0]
    interpolated_simulated_obs[:,1] = interpolation_function(interpolated_real_obs[:,0])

    return interpolated_simulated_obs, interpolated_real_obs

