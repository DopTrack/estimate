# Load standard modules
import numpy as np
from scipy.interpolate import interp1d  # interpolation function
from utility_functions.time import jday

# Load tudatpy modules
from tudatpy.kernel.numerical_simulation.estimation_setup import observation
import tudatpy.kernel.numerical_simulation.estimation as tudat_estimation
from tudatpy.kernel.numerical_simulation import estimation_setup, estimation

import yaml


# def extract_recording_start_times_old_yml(folder: str, filenames: list[str]) -> list[float]:
#     j2000_days = 2451545.0
#     start_recording_times = []
#
#     for filename in filenames:
#         with open(folder + filename, 'r') as metafile:
#             metadata = yaml.load(metafile, Loader=yaml.FullLoader)
#         time_pps = metadata['Sat']['Record']['time pps']
#         rx_time = metadata['Sat']['uhd']['rx_time']
#         julian_date_time_pps = jday(time_pps.year, int(time_pps.month), int(time_pps.day), int(time_pps.hour),
#                                     int(time_pps.minute), int(time_pps.second))
#         time_pps = (julian_date_time_pps - j2000_days) * 86400.0
#         start_recording_times.append(time_pps + rx_time)
#
#     return start_recording_times


def extract_recording_start_times_yml(folder: str, filenames: list[str], old_yml=False) -> list[float]:
    j2000_days = 2451545.0
    start_recording_times = []

    for filename in filenames:
        with open(folder + filename, 'r') as metafile:
            metadata = yaml.load(metafile, Loader=yaml.FullLoader)

        if old_yml:
            time_pps = metadata['Sat']['Record']['time pps']
            rx_time = metadata['Sat']['uhd']['rx_time']
            julian_date_time_pps = jday(time_pps.year, int(time_pps.month), int(time_pps.day), int(time_pps.hour),
                                        int(time_pps.minute), int(time_pps.second))
            time_pps = (julian_date_time_pps - j2000_days) * 86400.0
            start_recording_times.append(time_pps + rx_time)

        else:
            time = metadata["tracking"]["epoch"]
            julian_date = jday(time.year, int(time.month), int(time.day), int(time.hour), int(time.minute),
                               int(0.0)) + float(time.second) / 86400.0
            start_recording_times.append((julian_date - j2000_days) * 86400.0)

    return start_recording_times


def process_observations_old(filename: str, fraction_discarded: float = 0.1) -> np.array:
    observations = []
    j2000_days = 2451545.0

    f = open(filename, 'r')
    lines = f.readlines()
    nb_points = len(lines)
    nb_discarded_points = int(fraction_discarded / 2.0 * nb_points)

    # Retrieve observations of interest
    for line in lines[nb_discarded_points + 1:nb_points - nb_discarded_points]:
        result = line.strip().split(',')
        observations.append([float(result[1]), float(result[2])])
    f.close()

    return np.array(observations)


def process_observations_new(filename: str, fraction_discarded: float = 0.1) -> np.array:
    observations = []

    f = open(filename, 'r')
    lines = f.readlines()
    nb_points = len(lines)
    nb_discarded_points = int(fraction_discarded / 2.0 * nb_points)

    # Retrieve observations of interest
    for line in lines[nb_discarded_points + 1:nb_points - nb_discarded_points]:
        result = line.strip().split(',')
        observations.append([float(result[0]), float(result[2])])
    f.close()

    return np.array(observations)


def load_and_format_observations(spacecraft_name, data_folder, data, recording_start_times, old_obs_format=False):
    passes_start_times = []
    passes_end_times = []
    existing_data = np.empty((0, 2))

    for k in range(len(data)):
        if old_obs_format:
            data_set = process_observations_old(data_folder + data[k])
        else:
            data_set = process_observations_new(data_folder + data[k])

        data_set[:, 0] += recording_start_times[k]
        passes_start_times.append(data_set[0, 0] - 10.0)
        passes_end_times.append(data_set[np.shape(data_set)[0] - 1, 0] + 10.0)
        existing_data = np.concatenate((existing_data, data_set))

    obs_times = existing_data[:, 0].tolist()
    obs_values = []
    for i in range(len(existing_data)):
        obs_values.append([np.array([-existing_data[i, 1]])])

    # Define link ends
    link_ends = dict()
    link_ends[observation.receiver] = observation.body_reference_point_link_end_id("Earth", "DopTrackStation")
    link_ends[observation.transmitter] = observation.body_origin_link_end_id(spacecraft_name)

    # Set existing observations
    existing_observation_set = (link_ends, (np.array(obs_values), obs_times))
    observations_input = dict()
    observations_input[observation.one_way_instantaneous_doppler_type] = existing_observation_set

    observations_set = tudat_estimation.set_existing_observations(observations_input, observation.receiver)

    return passes_start_times, passes_end_times, obs_times, observations_set


def load_existing_observations(data_folder, data, recording_start_times, new_obs_format=False):
    passes_start_times = []
    passes_end_times = []
    existing_data = np.empty((0, 2))

    for k in range(len(data)):
        if not new_obs_format:
            data_set = process_observations_old(data_folder + data[k])
        else:
            data_set = process_observations_new(data_folder + data[k])

        data_set[:, 0] += recording_start_times[k]
        passes_start_times.append(data_set[0, 0] - 10.0)
        passes_end_times.append(data_set[np.shape(data_set)[0] - 1, 0] + 10.0)
        existing_data = np.concatenate((existing_data, data_set))

    obs_times = existing_data[:, 0].tolist()
    obs_values = []
    for i in range(len(existing_data)):
        # obs_values.append(np.array([-existing_data[i, 1]/constants.SPEED_OF_LIGHT]))
        obs_values.append(np.array([-existing_data[i, 1]]))

    return passes_start_times, passes_end_times, obs_times, obs_values


# def load_simulated_observations(data_folder, indices_simulated_data):
#     nb_passes = len(indices_simulated_data)
#
#     obs_times_per_pass = []
#     obs_values_per_pass = []
#     passes_start_times = []
#     passes_end_times = []
#
#     for i in range(nb_passes):
#
#         obs_times = []
#         obs_values = []
#
#         f = open(data_folder + 'obs_times_pass' + str(indices_simulated_data[i]) + '.txt', 'r')
#         lines = f.readlines()
#         for line in lines:
#             result = line.strip().split(',')
#             obs_times.append(float(result[0]))
#         f.close()
#
#         f = open(data_folder + 'obs_values_pass' + str(indices_simulated_data[i]) + '.txt', 'r')
#         lines = f.readlines()
#         for line in lines:
#             result = line.strip().split(',')
#             obs_values.append(np.array([float(result[0])]))
#         f.close()
#
#         passes_start_times.append(obs_times[0] - 10.0)
#         passes_end_times.append(obs_times[-1] + 10.0)
#
#         obs_times_per_pass.append(obs_times)
#         obs_values_per_pass.append(obs_values)
#
#     return passes_start_times, passes_end_times, obs_times_per_pass, obs_values_per_pass


# def merge_existing_and_simulated_obs(stations_real, stations_fake, existing_obs_times, existing_obs_values,
#                                      simulated_obs_times_per_pass_and_link_end, simulated_obs_values_per_pass_and_link_end):
#     # Define link ends
#     link_ends_real = []
#     for k in range(len(stations_real)):
#         link_ends_real.append(define_link_ends(stations_real[k]))
#
#     # Define fake link ends
#     link_ends_fake = []
#     for k in range(len(stations_fake)):
#         link_ends_fake.append(define_link_ends(stations_fake[k]))
#
#     obs_set = []
#     # Set existing observations
#     for j in range(len(stations_real)):
#         obs_set.append((link_ends_real[j], (existing_obs_values[stations_real[j]], existing_obs_times[stations_real[j]])))
#
#     # Set simulated observations
#     for j in range(len(stations_fake)):
#
#         simulated_obs_times_per_pass = simulated_obs_times_per_pass_and_link_end[stations_fake[j]]
#         simulated_obs_values_per_pass = simulated_obs_values_per_pass_and_link_end[stations_fake[j]]
#
#         all_simulated_obs_times = []
#         all_simulated_obs_values = []
#
#         for k in range(len(simulated_obs_times_per_pass)):
#             all_simulated_obs_times = all_simulated_obs_times + simulated_obs_times_per_pass[k]
#             for i in range(len(simulated_obs_values_per_pass[k])):
#                 all_simulated_obs_values.append([simulated_obs_values_per_pass[k][i]])
#
#         obs_set.append((link_ends_fake[j], (np.array(all_simulated_obs_values), all_simulated_obs_times)))
#
#     observations_input = {observation.one_way_instantaneous_doppler_type: obs_set}
#     observations_set = tudat_estimation.set_existing_observations(observations_input, observation.receiver)
#
#     return observations_set


def get_default_doppler_models() -> dict:
    bias_definition = 'per_pass'
    Doppler_models = dict(
        absolute_bias={
            'activated': True,
            'time_interval': bias_definition
        },
        time_drift={
            'activated': True,
            'time_interval': bias_definition
        }
    )
    return Doppler_models


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


def simulate_ideal_simulations(estimator, bodies, link_ends_def, obs_times, stations, max_elevation):
    link_ends_per_obs = dict()
    link_ends_per_obs[observation.one_way_instantaneous_doppler_type] = link_ends_def
    observation_simulation_settings = observation.tabulated_simulation_settings_list(
        link_ends_per_obs, obs_times, observation.receiver)

    for k in range(len(stations)):
        elevation_condition = observation.elevation_angle_viability(("Earth", stations[k]), np.deg2rad(max_elevation))
        observation.add_viability_check_to_observable_for_link_ends(
            observation_simulation_settings, [elevation_condition], observation.one_way_instantaneous_doppler_type,
            link_ends_def[k])

    return estimation.simulate_observations(observation_simulation_settings, estimator.observation_simulators, bodies)


def get_obs_per_link_end_and_pass(stations, obs_times, obs_values, obs_time_step):
    # Get list of observation times per link end
    obs_times_per_link_end = {}
    obs_values_per_link_end = {}
    prev_obs_time = obs_times[0]

    times_link_end = []
    values_link_end = []
    counter_link_ends = 0
    for k in range(1, len(obs_times)):
        if obs_times[k] >= prev_obs_time:
            times_link_end.append(obs_times[k])
            values_link_end.append(obs_values[k])
        else:
            if counter_link_ends > len(stations) - 1:
                raise Exception('Nb of detected link ends inconsistent with nb of stations.')

            obs_times_per_link_end[stations[counter_link_ends]] = times_link_end
            obs_values_per_link_end[stations[counter_link_ends]] = values_link_end
            times_link_end = []
            values_link_end = []
            counter_link_ends = counter_link_ends + 1

        prev_obs_time = obs_times[k]

    obs_times_per_link_end[stations[counter_link_ends]] = times_link_end
    obs_values_per_link_end[stations[counter_link_ends]] = values_link_end

    # Parse obs. times per link and separate different passes
    passes_start_times_dict = {}
    passes_end_times_dict = {}

    obs_times_per_pass_dict = {}
    obs_values_per_pass_dict = {}

    for k in range(len(obs_times_per_link_end)):

        current_pass_obs_times = []
        current_pass_obs_values = []

        passes_start_times = []
        passes_end_times = []

        obs_times_per_pass = []
        obs_values_per_pass = []

        obs_times_link_end = obs_times_per_link_end[stations[k]]
        obs_values_link_end = obs_values_per_link_end[stations[k]]

        passes_start_times.append(obs_times_link_end[0])
        for i in range(1, len(obs_times_link_end)):
            if (obs_times_link_end[i] - obs_times_link_end[i - 1]) > (3.0 * obs_time_step):
                passes_end_times.append(obs_times_link_end[i - 1])
                passes_start_times.append(obs_times_link_end[i])

                obs_times_per_pass.append(current_pass_obs_times)
                obs_values_per_pass.append(current_pass_obs_values)
                current_pass_obs_times = []
                current_pass_obs_values = []
            else:
                current_pass_obs_times.append(obs_times_link_end[i])
                current_pass_obs_values.append(obs_values_link_end[i])

        passes_end_times.append(obs_times_link_end[-1])
        obs_times_per_pass.append(current_pass_obs_times)
        obs_values_per_pass.append(current_pass_obs_values)

        passes_start_times_dict[stations[k]] = passes_start_times
        passes_end_times_dict[stations[k]] = passes_end_times
        obs_times_per_pass_dict[stations[k]] = obs_times_per_pass
        obs_values_per_pass_dict[stations[k]] = obs_values_per_pass

    return passes_start_times_dict, passes_end_times_dict, obs_times_per_pass_dict, obs_values_per_pass_dict


def get_all_passes_times(real_passes_start_times, real_passes_end_times, simulated_passes_start_times,
                         simulated_passes_end_times):
    passes_start_times = []
    passes_end_times = []

    for k in real_passes_start_times:
        passes_start_times = passes_start_times + real_passes_start_times[k]
        passes_end_times = passes_end_times + real_passes_end_times[k]
    for k in simulated_passes_start_times:
        passes_start_times = passes_start_times + simulated_passes_start_times[k]
        passes_end_times = passes_end_times + simulated_passes_end_times[k]

    ind = sorted(range(len(passes_start_times)), key=passes_start_times.__getitem__)

    passes_start_times = [passes_start_times[i] for i in ind]
    passes_end_times = [passes_end_times[i] for i in ind]

    return passes_start_times, passes_end_times


def interpolate_obs(simulated_obs, real_obs):
    interpolation_function = interp1d(simulated_obs[:, 0], simulated_obs[:, 1], kind='cubic')
    min_time_simulated = min(simulated_obs[:, 0])
    max_time_simulated = max(simulated_obs[:, 0])

    interpolated_real_obs = []

    obs_times_comparison = []
    for i in range(len(real_obs[:, 0])):
        if min_time_simulated < real_obs[i, 0] < max_time_simulated:
            obs_times_comparison.append(real_obs[i, 0])
            interpolated_real_obs.append([real_obs[i, 0], real_obs[i, 1]])

    interpolated_real_obs = np.array(interpolated_real_obs)

    interpolated_simulated_obs = np.zeros(np.shape(interpolated_real_obs))
    interpolated_simulated_obs[:, 0] = interpolated_real_obs[:, 0]
    interpolated_simulated_obs[:, 1] = interpolation_function(interpolated_real_obs[:, 0])

    return interpolated_simulated_obs, interpolated_real_obs
