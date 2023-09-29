import numpy as np

# Load tudatpy modules
from tudatpy.kernel import constants
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import estimation_setup, estimation
from tudatpy.kernel.numerical_simulation.estimation_setup import observation
from tudatpy.kernel.astro import element_conversion

from propagation_functions.propagation import create_integrator_settings
from utility_functions.time import get_days_starting_times, get_days_end_times


def define_arcs(option, passes_start_times, passes_end_times):

    arc_start_times = []
    arc_end_times = []

    if option == "per_pass":
        for i in range(len(passes_start_times)):
            arc_start_times.append(passes_start_times[i] - 3600.0)
            arc_end_times.append(passes_end_times[i] + 3600.0)

    elif option == "per_day" or option == "per_week" or option == "per_3_days":
        days_start_times = get_days_starting_times(passes_start_times)
        days_end_times = get_days_end_times(days_start_times)

        if option == "per_day":
            arc_start_times = days_start_times
            arc_end_times = days_end_times

        elif option == "per_week":
            arc_start_times.append(days_start_times[0])
            beginning_week = days_start_times[0]
            for i in range(1,len(days_start_times)):
                counter_days = int( (days_start_times[i] - beginning_week)/86400)
                if counter_days == 7:
                    arc_end_times.append(days_end_times[i-1])
                    arc_start_times.append(days_start_times[i])
                    beginning_week = days_start_times[i]

            arc_end_times.append(days_end_times[len(days_end_times)-1])

        elif option == "per_3_days":
            arc_start_times.append(days_start_times[0])
            beginning_week = days_start_times[0]
            for i in range(1, len(days_start_times)):
                counter_days = int((days_start_times[i] - beginning_week) / 86400)
                if counter_days == 3:
                    arc_end_times.append(days_end_times[i - 1])
                    arc_start_times.append(days_start_times[i])
                    beginning_week = days_start_times[i]

            arc_end_times.append(days_end_times[len(days_end_times) - 1])

    else:
        raise Exception('Error when defining tracking arcs, the time interval is not recognised.')

    return arc_start_times, arc_end_times


def define_doptrack_station(bodies):
    station_altitude = 100.0
    delft_latitude = np.deg2rad(51.015663)
    delft_longitude = np.deg2rad(4.0068216)

    # Add the ground station to the environment
    environment_setup.add_ground_station(
        bodies.get_body("Earth"), "DopTrackStation", [station_altitude, delft_latitude, delft_longitude],
        element_conversion.geodetic_position_type)


def define_station(bodies, station, coordinates):
    station_altitude = coordinates[0] # altitude
    station_latitude = coordinates[1] # latitude
    station_longitude = coordinates[2] # longitude

    environment_setup.add_ground_station(bodies.get_body("Earth"), station,
                                         [station_altitude, station_latitude, station_longitude],
                                         element_conversion.geodetic_position_type)


def define_link_ends(station):

    # Define the uplink link ends for one-way observable
    link_ends = dict()
    link_ends[observation.receiver] = observation.body_reference_point_link_end_id("Earth", station)
    link_ends[observation.transmitter] = observation.body_origin_link_end_id("Delfi")

    return link_ends


def get_link_end_def(link_ends):
    return observation.link_definition(link_ends)


def define_all_link_ends(stations):
    link_ends = []
    link_ends_def = []
    for k in range(len(stations)):
        link_ends.append(define_link_ends(stations[k]))
        link_ends_def.append(get_link_end_def(link_ends[k]))

    return link_ends, link_ends_def


def define_ideal_doppler_settings(stations):

    # Create observation settings for each link/observable
    observation_settings = []
    for k in range(len(stations)):
        observation_settings.append(observation.one_way_doppler_instantaneous(get_link_end_def(define_link_ends(stations[k]))))

    return observation_settings


def define_observation_settings(Doppler_models={}, passes_start_times=[], arc_start_times=[]):

    combined_biases = []

    # Define absolute arc-wise biases
    if "absolute_bias" in Doppler_models:
        if Doppler_models.get('absolute_bias').get('activated'):
            time_interval = Doppler_models.get('absolute_bias').get('time_interval')
            arc_wise_times = []
            if time_interval == 'per_pass':
                arc_wise_times = passes_start_times
            elif time_interval == 'per_arc':
                arc_wise_times = arc_start_times
            elif time_interval == 'global':
                arc_wise_times.append(passes_start_times[0])
            else:
                raise Exception('Error when setting up absolute bias, the time interval is not recognised.')

            biases_values = []
            for i in range(len(arc_wise_times)):
                biases_values.append(np.zeros(1))

            if time_interval == 'per_pass' or time_interval == 'per_arc':
                arc_wise_absolute_bias = observation.arcwise_absolute_bias(arc_wise_times, biases_values, observation.receiver)
                combined_biases.append(arc_wise_absolute_bias)
            else:
                absolute_bias = observation.absolute_bias(biases_values)
                combined_biases.append(absolute_bias)

    # Define relative arc-wise biases
    if "relative_bias" in Doppler_models:
        if Doppler_models.get('relative_bias').get('activated'):
            time_interval = Doppler_models.get('relative_bias').get('time_interval')
            arc_wise_times = []
            if time_interval == 'per_pass':
                arc_wise_times = passes_start_times
            elif time_interval == 'per_arc':
                arc_wise_times = arc_start_times
            elif time_interval == 'global':
                arc_wise_times.append(passes_start_times[0])
            else:
                raise Exception('Error when setting up relative bias, the time interval is not recognised.')

            biases_values = []
            for i in range(len(arc_wise_times)):
                biases_values.append(np.zeros(1))

            if time_interval == 'per_pass' or time_interval == 'per_arc':
                arc_wise_relative_bias = observation.arcwise_relative_bias(arc_wise_times, biases_values, observation.receiver)
                combined_biases.append(arc_wise_relative_bias)
            else:
                relative_bias = observation.relative_bias(biases_values)
                combined_biases.append(relative_bias)

    # Define arc-wise time drift biases
    if "time_drift" in Doppler_models:
        if Doppler_models.get('time_drift').get('activated'):
            time_interval = Doppler_models.get('time_drift').get('time_interval')
            arc_wise_times = []
            if time_interval == 'per_pass':
                arc_wise_times = passes_start_times
            elif time_interval == 'per_arc':
                arc_wise_times = arc_start_times
            elif time_interval == 'global':
                arc_wise_times.append(passes_start_times[0])
            else:
                raise Exception('Error when setting up time drift bias, the time interval is not recognised.')

            biases_values = []
            for i in range(len(arc_wise_times)):
                biases_values.append(np.zeros(1))

            if time_interval == 'per_pass' or time_interval == 'per_arc':
                arc_wise_time_drift = observation.arc_wise_time_drift_bias(biases_values, arc_wise_times, observation.receiver, arc_wise_times)
                combined_biases.append(arc_wise_time_drift)
            else:
                time_drift = observation.time_drift_bias(biases_values, observation.receiver, passes_start_times[0])
                combined_biases.append(time_drift)


    # Define arc-wise time biases
    if "time_bias" in Doppler_models:
        if Doppler_models.get('time_bias').get('activated'):
            time_interval = Doppler_models.get('time_bias').get('time_interval')
            arc_wise_times = []
            if time_interval == 'per_pass':
                arc_wise_times = passes_start_times
            elif time_interval == 'per_arc':
                arc_wise_times = arc_start_times
            elif time_interval == 'global':
                arc_wise_times.append(passes_start_times[0])
            else:
                raise Exception('Error when setting up time bias, the time interval is not recognised.')

            biases_values = []
            for i in range(len(arc_wise_times)):
                biases_values.append(np.zeros(1))

            if time_interval == 'per_pass' or time_interval == 'per_arc':
                arc_wise_time_bias = observation.arc_wise_time_drift_bias(biases_values, arc_wise_times, observation.receiver, arc_wise_times)
                combined_biases.append(arc_wise_time_bias)
            else:
                time_bias = observation.time_drift_bias(biases_values, observation.receiver, passes_start_times[0])
                combined_biases.append(time_bias)

    # Define all biases
    biases = observation.combined_bias(combined_biases)

    # Create observation settings for each link/observable
    observation_settings = [observation.one_way_open_loop_doppler(get_link_end_def(define_link_ends("DopTrackStation")), bias_settings=biases)]

    return observation_settings


def define_biases(Doppler_models={}, passes_start_times=[], arc_start_times=[]):

    combined_biases = []

    # Define absolute arc-wise biases
    if "absolute_bias" in Doppler_models:
        if Doppler_models.get('absolute_bias').get('activated'):
            time_interval = Doppler_models.get('absolute_bias').get('time_interval')
            arc_wise_times = []
            if time_interval == 'per_pass':
                arc_wise_times = passes_start_times
            elif time_interval == 'per_arc':
                arc_wise_times = arc_start_times
            elif time_interval == 'global':
                arc_wise_times.append(passes_start_times[0])
            else:
                raise Exception('Error when setting up absolute bias, the time interval is not recognised.')

            biases_values = []
            for i in range(len(arc_wise_times)):
                biases_values.append(np.zeros(1))

            if time_interval == 'per_pass' or time_interval == 'per_arc':
                arc_wise_absolute_bias = observation.arcwise_absolute_bias(arc_wise_times, biases_values, observation.receiver)
                combined_biases.append(arc_wise_absolute_bias)
            else:
                absolute_bias = observation.absolute_bias(biases_values)
                combined_biases.append(absolute_bias)

    # Define relative arc-wise biases
    if "relative_bias" in Doppler_models:
        if Doppler_models.get('relative_bias').get('activated'):
            time_interval = Doppler_models.get('relative_bias').get('time_interval')
            arc_wise_times = []
            if time_interval == 'per_pass':
                arc_wise_times = passes_start_times
            elif time_interval == 'per_arc':
                arc_wise_times = arc_start_times
            elif time_interval == 'global':
                arc_wise_times.append(passes_start_times[0])
            else:
                raise Exception('Error when setting up relative bias, the time interval is not recognised.')

            biases_values = []
            for i in range(len(arc_wise_times)):
                biases_values.append(np.zeros(1))

            if time_interval == 'per_pass' or time_interval == 'per_arc':
                arc_wise_relative_bias = observation.arcwise_relative_bias(arc_wise_times, biases_values, observation.receiver)
                combined_biases.append(arc_wise_relative_bias)
            else:
                relative_bias = observation.relative_bias(biases_values)
                combined_biases.append(relative_bias)

    # Define arc-wise time drift biases
    if "time_drift" in Doppler_models:
        if Doppler_models.get('time_drift').get('activated'):
            time_interval = Doppler_models.get('time_drift').get('time_interval')
            arc_wise_times = []
            if time_interval == 'per_pass':
                arc_wise_times = passes_start_times
            elif time_interval == 'per_arc':
                arc_wise_times = arc_start_times
            elif time_interval == 'global':
                arc_wise_times.append(passes_start_times[0])
            else:
                raise Exception('Error when setting up time drift bias, the time interval is not recognised.')

            biases_values = []
            for i in range(len(arc_wise_times)):
                biases_values.append(np.zeros(1))

            if time_interval == 'per_pass' or time_interval == 'per_arc':
                arc_wise_time_drift = observation.arc_wise_time_drift_bias(biases_values, arc_wise_times, observation.receiver, arc_wise_times)
                combined_biases.append(arc_wise_time_drift)
            else:
                time_drift = observation.time_drift_bias(biases_values, observation.receiver, passes_start_times[0])
                combined_biases.append(time_drift)


    # Define arc-wise time biases
    if "time_bias" in Doppler_models:
        if Doppler_models.get('time_bias').get('activated'):
            time_interval = Doppler_models.get('time_bias').get('time_interval')
            arc_wise_times = []
            if time_interval == 'per_pass':
                arc_wise_times = passes_start_times
            elif time_interval == 'per_arc':
                arc_wise_times = arc_start_times
            elif time_interval == 'global':
                arc_wise_times.append(passes_start_times[0])
            else:
                raise Exception('Error when setting up time bias, the time interval is not recognised.')

            biases_values = []
            for i in range(len(arc_wise_times)):
                biases_values.append(np.zeros(1))

            if time_interval == 'per_pass' or time_interval == 'per_arc':
                arc_wise_time_bias = observation.arc_wise_time_bias(biases_values, arc_wise_times, observation.receiver)
                combined_biases.append(arc_wise_time_bias)
            else:
                time_bias = observation.time_bias(np.zeros(1), observation.receiver)
                combined_biases.append(time_bias)

    return observation.combined_bias(combined_biases)


def define_parameters(parameters_list, bodies, propagator_settings, initial_time, arc_start_times, pass_times_per_linkend, obs_models={}):

    parameter_settings = []

    # Initial states
    if "initial_state_delfi" in parameters_list:
        if parameters_list.get('initial_state_delfi').get('estimate'):
            initial_states_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies, arc_start_times)
            for settings in initial_states_settings:
                parameter_settings.append(settings)

    # Absolute biases
    for k in range(len(pass_times_per_linkend)):
        if "absolute_bias" in parameters_list:
            if parameters_list.get('absolute_bias').get('estimate'):
                if obs_models.get('absolute_bias').get('time_interval') == 'per_pass':
                    parameter_settings.append(estimation_setup.parameter.arcwise_absolute_observation_bias(
                        observation.link_definition(pass_times_per_linkend[k][0]), observation.one_way_instantaneous_doppler_type,
                        pass_times_per_linkend[k][1], observation.receiver))
                elif obs_models.get('absolute_bias').get('time_interval') == 'per_arc':
                    parameter_settings.append(estimation_setup.parameter.arcwise_absolute_observation_bias(
                        observation.link_definition(pass_times_per_linkend[k][0]), observation.one_way_instantaneous_doppler_type, arc_start_times, observation.receiver))
                elif obs_models.get('absolute_bias').get('time_interval') == 'global':
                    parameter_settings.append(estimation_setup.parameter.absolute_observation_bias(
                        observation.link_definition(pass_times_per_linkend[k][0]), observation.one_way_instantaneous_doppler_type))

    # Relative biases
    for k in range(len(pass_times_per_linkend)):
        if "relative_bias" in parameters_list:
            if parameters_list.get('relative_bias').get('estimate'):
                if obs_models.get('relative_bias').get('time_interval') == 'per_pass':
                    parameter_settings.append(estimation_setup.parameter.arcwise_relative_observation_bias(
                        observation.link_definition(pass_times_per_linkend[k][0]), observation.one_way_instantaneous_doppler_type, pass_times_per_linkend[k][1], observation.receiver))
                elif obs_models.get('relative_bias').get('time_interval') == 'per_arc':
                    parameter_settings.append(estimation_setup.parameter.arcwise_relative_observation_bias(
                        observation.link_definition(pass_times_per_linkend[k][0]), observation.one_way_instantaneous_doppler_type, arc_start_times, observation.receiver))
                elif obs_models.get('relative_bias').get('time_interval') == 'global':
                    parameter_settings.append( estimation_setup.parameter.relative_observation_bias(
                        observation.link_definition(pass_times_per_linkend[k][0]), observation.one_way_instantaneous_doppler_type))

    # Time drift biases
    for k in range(len(pass_times_per_linkend)):
        if "time_drift" in parameters_list:
            if parameters_list.get('time_drift').get('estimate'):
                if obs_models.get('time_drift').get('time_interval') == 'per_pass':
                    parameter_settings.append(estimation_setup.parameter.arcwise_time_drift_observation_bias(
                        pass_times_per_linkend[k][0], observation.one_way_instantaneous_doppler_type, pass_times_per_linkend[k][1], pass_times_per_linkend[k][1], observation.receiver))
                elif obs_models.get('time_drift').get('time_interval') == 'per_arc':
                    parameter_settings.append(estimation_setup.parameter.arcwise_time_drift_observation_bias(
                        pass_times_per_linkend[k][0], observation.one_way_instantaneous_doppler_type, arc_start_times, arc_start_times, observation.receiver))
                elif obs_models.get('time_drift').get('time_interval') == 'global':
                    parameter_settings.append(estimation_setup.parameter.time_drift_observation_bias(
                        pass_times_per_linkend[k][0], observation.one_way_instantaneous_doppler_type, pass_times_per_linkend[k][1][0], observation.receiver))

    # Time biases
    for k in range(len(pass_times_per_linkend)):
        if "time_bias" in parameters_list:
            if parameters_list.get('time_bias').get('estimate'):
                if obs_models.get('time_bias').get('time_interval') == 'per_pass':
                    parameter_settings.append(estimation_setup.parameter.arcwise_time_observation_bias(
                        pass_times_per_linkend[k][0], observation.one_way_instantaneous_doppler_type, pass_times_per_linkend[k][1], observation.receiver))
                elif obs_models.get('time_bias').get('time_interval') == 'per_arc':
                    parameter_settings.append(estimation_setup.parameter.arcwise_time_observation_bias(
                        pass_times_per_linkend[k][0], observation.one_way_instantaneous_doppler_type, arc_start_times, observation.receiver))
                elif obs_models.get('time_bias').get('time_interval') == 'global':
                    parameter_settings.append(estimation_setup.parameter.time_observation_bias(
                        pass_times_per_linkend[k][0], observation.one_way_instantaneous_doppler_type, observation.receiver))

    # Drag coefficient(s)
    if "drag_coefficient" in parameters_list:
        if parameters_list.get('drag_coefficient').get('estimate'):
            if parameters_list.get('drag_coefficient').get('type') == 'per_pass':
                parameter_settings.append(estimation_setup.parameter.arcwise_constant_drag_coefficient("Delfi", pass_times_per_linkend[k][1]))
            elif parameters_list.get('drag_coefficient').get('type') == 'per_arc':
                parameter_settings.append(estimation_setup.parameter.arcwise_constant_drag_coefficient("Delfi", arc_start_times))
            elif parameters_list.get('drag_coefficient').get('type') == 'global':
                parameter_settings.append(estimation_setup.parameter.constant_drag_coefficient("Delfi"))

    # Solar radiation pressure coefficient(s)
    if "srp_coefficient" in parameters_list:
        if parameters_list.get('srp_coefficient').get('estimate'):
            if parameters_list.get('srp_coefficient').get('type') == 'per_pass':
                parameter_settings.append(estimation_setup.parameter.arcwise_radiation_pressure_coefficient("Delfi", pass_times_per_linkend[k][1]))
            elif parameters_list.get('srp_coefficient').get('type') == 'per_arc':
                parameter_settings.append(estimation_setup.parameter.arcwise_radiation_pressure_coefficient("Delfi", arc_start_times))
            elif parameters_list.get('srp_coefficient').get('type') == 'global':
                parameter_settings.append(estimation_setup.parameter.radiation_pressure_coefficient("Delfi"))

    # Gravitational parameter
    if "gravitational_parameter" in parameters_list:
        if parameters_list.get('gravitational_parameter').get('estimate'):
            if parameters_list.get('gravitational_parameter').get('type') == 'global':
                parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Earth"))
            else:
                raise Exception('Error, Earth gravitational parameter can only be estimated globally.')

    # C20 coefficient
    if "C20" in parameters_list:
        if parameters_list.get('C20').get('estimate'):
            if parameters_list.get('C20').get('type') == 'global':
                parameter_settings.append(estimation_setup.parameter.spherical_harmonics_c_coefficients("Earth", 2,0,2,0))
            else:
                raise Exception('Error, C20 coefficient can only be estimated globally.')

    # C22 coefficient
    if "C22" in parameters_list:
        if parameters_list.get('C22').get('estimate'):
            if parameters_list.get('C22').get('type') == 'global':
                parameter_settings.append(estimation_setup.parameter.spherical_harmonics_c_coefficients("Earth", 2,2,2,2))
            else:
                raise Exception('Error, C22 coefficient can only be estimated globally.')

    parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies, propagator_settings, [])

    return parameters_to_estimate


def simulate_observations(observation_times, observation_settings, propagator_settings, bodies, initial_time, min_elevation_angle: float = 10):
    link_ends_per_obs = dict()
    link_ends_per_obs[observation.one_way_instantaneous_doppler_type] = [get_link_end_def(define_link_ends("DopTrackStation"))]
    observation_simulation_settings = observation.tabulated_simulation_settings_list(
        link_ends_per_obs, observation_times, observation.receiver)

    # The actual simulation of the observations requires Observation Simulators, which are created automatically by the Estimator object.
    # Therefore, the observations cannot be simulated before the creation of an Estimator object.
    integrator_settings = create_integrator_settings(initial_time)
    estimator = create_dummy_estimator(bodies, propagator_settings, integrator_settings, observation_settings)

    elevation_condition = observation.elevation_angle_viability(("Earth", "DopTrackStation"), np.deg2rad(min_elevation_angle))
    observation.add_viability_check_to_observable_for_link_ends(observation_simulation_settings, [elevation_condition], observation.one_way_instantaneous_doppler_type,
                                                                get_link_end_def(define_link_ends("DopTrackStation")))

    return estimation.simulate_observations(observation_simulation_settings, estimator.observation_simulators, bodies)


def simulate_observations_from_estimator(observation_times, estimator, bodies, min_elevation_angle: float = 10):
    link_ends_per_obs = dict()
    link_ends_per_obs[observation.one_way_instantaneous_doppler_type] = [get_link_end_def(define_link_ends("DopTrackStation"))]
    observation_simulation_settings = observation.tabulated_simulation_settings_list(
        link_ends_per_obs, observation_times, observation.receiver)

    elevation_condition = observation.elevation_angle_viability(("Earth", "DopTrackStation"), np.deg2rad(min_elevation_angle))
    observation.add_viability_check_to_observable_for_link_ends(observation_simulation_settings, [elevation_condition],
                                                                observation.one_way_instantaneous_doppler_type,
                                                                get_link_end_def(define_link_ends("DopTrackStation")))

    return estimation.simulate_observations(observation_simulation_settings, estimator.observation_simulators, bodies)


def run_estimation(estimator, parameters_to_estimate, observations_set, nb_arcs, nb_iterations):

    truth_parameters = parameters_to_estimate.parameter_vector
    nb_parameters = len(truth_parameters)

    inv_cov = np.zeros((nb_parameters, nb_parameters))
    apriori_covariance_position = 5.0e3#e-3
    apriori_covariance_velocity = 5.0#e-6
    aPrioriCovarianceSRPCoef = 0.2

    apriori_time_bias = 1.0e-7

    for i in range (nb_arcs):
        for j in range (3):
            inv_cov[i*6+j, i*6+j] = 1.0 / (apriori_covariance_position * apriori_covariance_position)
            inv_cov[i*6+3+j, i*6+3+j] = 1.0 / (apriori_covariance_velocity * apriori_covariance_velocity)

    # Create input object for estimation_functions, adding observations and parameter set information
    convergence_check = estimation.estimation_convergence_checker(nb_iterations)
    estimation_input = estimation.EstimationInput(observations_set, inv_cov, convergence_check)
    estimation_input.define_estimation_settings(reintegrate_variational_equations=True, save_design_matrix=True)

    # Define observations weights
    noise_level = 5.0
    weights_per_observable = \
        {estimation_setup.observation.one_way_instantaneous_doppler_type: noise_level ** -2}
    estimation_input.set_constant_weight_per_observable(weights_per_observable)

    # Perform estimation_functions and return pod_output
    return estimator.perform_estimation(estimation_input)


# Function creating a dummy estimator (for 1st part of the tutorial when observations have to be simulated but no estimation_functions
# needs to be run yet)
def create_dummy_estimator(bodies, propagator_settings, integrator_settings, observation_settings):

    parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
    parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies, propagator_settings, [])
    estimation_setup.print_parameter_names(parameters_to_estimate)

    # Create the estimator object
    return numerical_simulation.Estimator(bodies, parameters_to_estimate, observation_settings, propagator_settings, True)


def get_residuals_per_pass(obs_times, residuals, passes_start_times):
    residuals_per_pass = []

    current_pass = -1
    current_pass_residuals = []
    for i in range(len(residuals)):
        time = obs_times[i]
        for k in range(len(passes_start_times)):
            if (k < len(passes_start_times)-1 and passes_start_times[k] <= time < passes_start_times[k+1])\
                    or (k == len(passes_start_times)-1 and passes_start_times[-1] <= time):
                updated_pass = k
                if (current_pass > -1) and updated_pass != current_pass:
                    current_pass_residuals = np.array(current_pass_residuals)
                    residuals_per_pass.append(current_pass_residuals)
                    current_pass_residuals = []
                    current_pass_residuals.append(residuals[i, -1])
                    current_pass = -1
                else:
                    current_pass = k
                    current_pass_residuals.append(residuals[i, -1])


    # Save residuals for final pass
    current_pass_residuals = np.array(current_pass_residuals)
    residuals_per_pass.append(current_pass_residuals)

    return residuals_per_pass