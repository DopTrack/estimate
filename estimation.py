import math

import numpy as np

# Load tudatpy modules
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation_setup, estimation
from tudatpy.kernel.numerical_simulation.estimation_setup import observation
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.numerical_simulation.environment_setup import ephemeris

import propagation

def define_doptrack_station(bodies):
    station_altitude = 0.0
    delft_latitude = np.deg2rad(52.00667)
    delft_longitude = np.deg2rad(4.35556)

    # Add the ground station to the environment
    environment_setup.add_ground_station(bodies.get_body("Earth"), "DopTrackStation",
                                         [station_altitude, delft_latitude, delft_longitude],
                                         element_conversion.geodetic_position_type)

def define_link_ends():

    # Define the uplink link ends for one-way observable
    link_ends = dict()
    link_ends[observation.receiver] = ("Earth", "DopTrackStation")
    link_ends[observation.transmitter] = ("Delfi", "")

    return link_ends


def define_ideal_doppler_settings():

    # Create observation settings for each link/observable
    observation_settings = [observation.one_way_open_loop_doppler(define_link_ends())]

    return observation_settings

def define_observation_settings(Doppler_models):

    combined_biases = []

    # Define absolute arc-wise biases
    if Doppler_models.get('absolute_bias').get('activated'):
        arc_wise_times = Doppler_models.get('absolute_bias').get('times')

        biases_values = []
        for i in range(len(arc_wise_times)):
            biases_values.append(np.zeros(1))

        arc_wise_absolute_bias = observation.arcwise_absolute_bias(arc_wise_times, biases_values, observation.receiver)

        combined_biases.append(arc_wise_absolute_bias)

    # Define relative arc-wise biases
    if Doppler_models.get('relative_bias').get('activated'):
        arc_wise_times = Doppler_models.get('relative_bias').get('times')

        biases_values = []
        for i in range(len(arc_wise_times)):
            biases_values.append(np.zeros(1))

        arc_wise_relative_bias = observation.arcwise_relative_bias(arc_wise_times, biases_values, observation.receiver)

        combined_biases.append(arc_wise_relative_bias)

    # Define arc-wise time biases
    if Doppler_models.get('time_bias').get('activated'):
        arc_wise_times = Doppler_models.get('time_bias').get('times')

        biases_values = []
        for i in range(len(arc_wise_times)):
            biases_values.append(np.zeros(1))

        arc_wise_time_bias = observation.arc_wise_time_drift_bias(biases_values, arc_wise_times, observation.receiver, arc_wise_times)

        combined_biases.append(arc_wise_time_bias)

    # Define all biases
    biases = observation.combined_bias(combined_biases)

    # Create observation settings for each link/observable
    observation_settings = [observation.one_way_open_loop_doppler(define_link_ends(), bias_settings=biases)]

    return observation_settings


def check_consistency_parameters_observations(parameters_list, obs_models, arc_start_times, passes_start_times):

    # absolute biases
    if parameters_list.get('absolute_bias').get('estimate'):
        if not obs_models.get('absolute_bias').get('activated'):
            raise Exception('Error when estimating absolute biases, such biases not defined in observation model.')
        if parameters_list.get('absolute_bias').get('type') == 'per_pass' and len(obs_models.get('absolute_bias').get('times')) != len(passes_start_times):
            raise Exception('Error when estimating absolute biases once per pass, biases are not defined per pass in observation models.')
        elif parameters_list.get('absolute_bias').get('type') == 'per_arc' and len(obs_models.get('absolute_bias').get('times')) != len(arc_start_times):
            raise Exception('Error when estimating absolute biases once per arc, biases are not defined per arc in observation models.')
        if parameters_list.get('absolute_bias').get('type') == 'global' and len(obs_models.get('absolute_bias').get('times')) != 1:
            raise Exception('Error when estimating absolute biases globally, while arc arc-wise biases are defined in observation models.')

    # relative biases
    if parameters_list.get('relative_bias').get('estimate'):
        if not obs_models.get('relative_bias').get('activated'):
            raise Exception('Error when estimating relative biases, such biases not defined in observation model.')
        if parameters_list.get('relative_bias').get('type') == 'per_pass' and len(
                obs_models.get('relative_bias').get('times')) != len(passes_start_times):
            raise Exception(
                'Error when estimating relative biases once per pass, biases are not defined per pass in observation models.')
        elif parameters_list.get('relative_bias').get('type') == 'per_arc' and len(
                obs_models.get('relative_bias').get('times')) != len(arc_start_times):
            raise Exception(
                'Error when estimating relative biases once per arc, biases are not defined per arc in observation models.')
        if parameters_list.get('relative_bias').get('type') == 'global' and len(
                obs_models.get('relative_bias').get('times')) != 1:
            raise Exception(
                'Error when estimating relative biases globally, while arc arc-wise biases are defined in observation models.')

    # time biases
    if parameters_list.get('time_bias').get('estimate'):
        if not obs_models.get('time_bias').get('activated'):
            raise Exception(
                'Error when estimating time biases, such biases not defined in observation model.')
        if parameters_list.get('time_bias').get('type') == 'per_pass' and len(
                obs_models.get('time_bias').get('times')) != len(passes_start_times):
            raise Exception(
                'Error when estimating time biases once per pass, biases are not defined per pass in observation models.')
        elif parameters_list.get('time_bias').get('type') == 'per_arc' and len(
                obs_models.get('time_bias').get('times')) != len(arc_start_times):
            raise Exception(
                'Error when estimating time biases once per arc, biases are not defined per arc in observation models.')
        if parameters_list.get('time_bias').get('type') == 'global' and len(
                obs_models.get('time_bias').get('times')) != 1:
            raise Exception(
                'Error when estimating time biases globally, while arc arc-wise biases are defined in observation models.')


def define_parameters(parameters_list, bodies, propagator_settings, initial_time, arc_start_times, passes_start_times, obs_models):

    link_ends = define_link_ends()
    parameter_settings = []

    check_consistency_parameters_observations(parameters_list, obs_models, arc_start_times, passes_start_times)

    # Initial states
    if parameters_list.get('initial_state_delfi').get('estimate'):
        if parameters_list.get('initial_state_delfi').get('type') == 'per_arc':
            initial_states_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies, arc_start_times)
        else:
            raise Exception('Error, Delfi initial states have to be estimated for every arc.')
        for settings in initial_states_settings:
            parameter_settings.append(settings)

    # Absolute biases
    if parameters_list.get('absolute_bias').get('estimate'):
        if parameters_list.get('absolute_bias').get('type') == 'per_pass':
            parameter_settings.append(estimation_setup.parameter.arcwise_absolute_observation_bias(
                link_ends, observation.one_way_doppler_type, passes_start_times, observation.receiver))
        elif parameters_list.get('absolute_bias').get('type') == 'per_arc':
            parameter_settings.append(estimation_setup.parameter.arcwise_absolute_observation_bias(
                link_ends, observation.one_way_doppler_type, arc_start_times, observation.receiver))
        elif parameters_list.get('absolute_bias').get('type') == 'global':
            parameter_settings.append(estimation_setup.parameter.absolute_observation_bias(link_ends, observation.one_way_doppler_type))

    # Relative biases
    if parameters_list.get('relative_bias').get('estimate'):
        if parameters_list.get('relative_bias').get('type') == 'per_pass':
            parameter_settings.append(estimation_setup.parameter.arcwise_relative_observation_bias(
                link_ends, observation.one_way_doppler_type, passes_start_times, observation.receiver))
        elif parameters_list.get('relative_bias').get('type') == 'per_arc':
            parameter_settings.append(estimation_setup.parameter.arcwise_relative_observation_bias(
                link_ends, observation.one_way_doppler_type, arc_start_times, observation.receiver))
        elif parameters_list.get('relative_bias').get('type') == 'global':
            parameter_settings.append( estimation_setup.parameter.relative_observation_bias(link_ends, observation.one_way_doppler_type))

    # Time biases
    if parameters_list.get('time_bias').get('estimate'):
        if parameters_list.get('time_bias').get('type') == 'per_pass':
            parameter_settings.append(estimation_setup.parameter.arcwise_time_drift_observation_bias(
                link_ends, observation.one_way_doppler_type, passes_start_times, passes_start_times, observation.receiver))
        elif parameters_list.get('time_bias').get('type') == 'per_arc':
            parameter_settings.append(estimation_setup.parameter.arcwise_time_drift_observation_bias(
                link_ends, observation.one_way_doppler_type, arc_start_times, arc_start_times, observation.receiver))
        elif parameters_list.get('time_bias').get('type') == 'global':
            parameter_settings.append(estimation_setup.parameter.time_drift_observation_bias(link_ends, observation.one_way_doppler_type,
                                      initial_time, observation.receiver))

    # Drag coefficient(s)
    if parameters_list.get('drag_coefficient').get('estimate'):
        if parameters_list.get('drag_coefficient').get('type') == 'per_pass':
            parameter_settings.append(estimation_setup.parameter.arcwise_constant_drag_coefficient("Delfi", passes_start_times))
        elif parameters_list.get('drag_coefficient').get('type') == 'per_arc':
            parameter_settings.append(estimation_setup.parameter.arcwise_constant_drag_coefficient("Delfi", arc_start_times))
        elif parameters_list.get('drag_coefficient').get('type') == 'global':
            parameter_settings.append(estimation_setup.parameter.constant_drag_coefficient("Delfi"))

    # Gravitational parameter
    if parameters_list.get('gravitational_parameter').get('estimate'):
        if parameters_list.get('gravitational_parameter').get('type') == 'global':
            parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Earth"))
        else:
            raise Exception('Error, Earth gravitational parameter can only be estimated globally.')

    # C20 coefficient
    if parameters_list.get('C20').get('estimate'):
        if parameters_list.get('C20').get('type') == 'global':
            parameter_settings.append(estimation_setup.parameter.spherical_harmonics_c_coefficients("Earth", 2,0,2,0))
        else:
            raise Exception('Error, C20 coefficient can only be estimated globally.')

    # C22 coefficient
    if parameters_list.get('C22').get('estimate'):
        if parameters_list.get('C22').get('type') == 'global':
            parameter_settings.append(estimation_setup.parameter.spherical_harmonics_c_coefficients("Earth", 2,2,2,2))
        else:
            raise Exception('Error, C22 coefficient can only be estimated globally.')

    parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)

    return parameters_to_estimate


def simulate_ideal_observations(observation_times, estimator, bodies, min_elevation_angle: float = 10):
    link_ends_per_obs = dict()
    link_ends_per_obs[observation.one_way_doppler_type] = [define_link_ends()]
    observation_simulation_settings = observation.tabulated_simulation_settings_list(
        link_ends_per_obs, observation_times, observation.receiver)

    elevation_condition = observation.elevation_angle_viability(define_link_ends()[observation.receiver], np.deg2rad(min_elevation_angle))
    observation.add_viability_check_to_settings(observation_simulation_settings, [elevation_condition])

    return estimation.simulate_observations(observation_simulation_settings, estimator.observation_simulators, bodies)


def run_estimation(estimator, parameters_to_estimate, observations_set, nb_arcs, nb_iterations):

    truth_parameters = parameters_to_estimate.parameter_vector
    nb_parameters = len(truth_parameters)

    inv_cov = np.zeros((nb_parameters, nb_parameters))
    apriori_covariance_position = 2.0e3
    apriori_covariance_velocity = 2.0
    aPrioriCovarianceSRPCoef = 0.2

    for i in range (nb_arcs):
        for j in range (3):
            inv_cov[i*6+j, i*6+j] = 1.0 / (apriori_covariance_position * apriori_covariance_position)
            inv_cov[i*6+3+j, i*6+3+j] = 1.0 / (apriori_covariance_velocity * apriori_covariance_velocity)

    # Create input object for estimation, adding observations and parameter set information
    pod_input = estimation.PodInput(observations_set, parameters_to_estimate.parameter_set_size,
                                    inverse_apriori_covariance=inv_cov)
    pod_input.define_estimation_settings(reintegrate_variational_equations=True, save_design_matrix=True)

    convergence_check = estimation.estimation_convergence_checker(nb_iterations)

    # define weighting of the observations in the inversion
    noise_level = 5.0 / constants.SPEED_OF_LIGHT
    weights_per_observable = \
        {estimation_setup.observation.one_way_doppler_type: noise_level ** -2}
    pod_input.set_constant_weight_per_observable(weights_per_observable)

    # Perform estimation and return pod_output
    return estimator.perform_estimation(pod_input, convergence_check)


# Function creating a dummy estimator (for 1st part of the tutorial when observations have to be simulated but no estimation
# needs to be run yet)
def create_dummy_estimator(bodies, propagator_settings, integrator_settings, observation_settings):

    initial_states_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
    parameter_settings = []
    for settings in initial_states_settings:
        parameter_settings.append(settings)

    parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)
    estimation_setup.print_parameter_names(parameters_to_estimate)

    # Create the estimator object
    return numerical_simulation.Estimator(bodies, parameters_to_estimate, observation_settings,
                                          integrator_settings, propagator_settings)