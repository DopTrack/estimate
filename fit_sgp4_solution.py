# Load standard modules
import statistics

import numpy as np
from matplotlib import pyplot as plt

from propagation_functions.environment import *
from propagation_functions.propagation import *
from estimation_functions.estimation import *
from estimation_functions.observations_data import *

from utility_functions.tle import *

# Load tudatpy modules
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import estimation_setup


def fit_sgp4_solution(metadata: str, propagation_time_in_days: float, old_yml=False) -> tuple:
    # Retrieve initial epoch and state of the first pass
    initial_epoch, initial_state_teme, b_star = get_tle_initial_conditions(metadata, old_yml)
    start_recording_day = get_start_next_day(initial_epoch)

    # Calculate final propagation_functions epoch
    final_epoch = start_recording_day + propagation_time_in_days * 86400.0
    mid_epoch = (initial_epoch + final_epoch) / 2.0

    epochs = []
    time = mid_epoch
    while time <= final_epoch + 10.0:
        epochs.append(time)
        time += 10.0
    time = mid_epoch - 10.0
    while time >= initial_epoch - 10.0:
        epochs.append(time)
        time += -10.0
    epochs.sort()

    propagated_states_sgp4 = propagate_sgp4(metadata, initial_epoch, epochs, old_yml)
    dict_solution_sgp4 = {}
    for row in propagated_states_sgp4:
        dict_solution_sgp4[row[0]] = row[1:]

    # Define propagation_functions environment
    mass = 2.2
    ref_area = 0.035
    drag_coef = get_drag_coefficient(mass, ref_area, b_star, from_tle=False)
    srp_coef = 1.2
    bodies = define_environment(mass, ref_area, drag_coef, srp_coef, "Delfi",
                                multi_arc_ephemeris=False, tabulated_ephemeris=dict_solution_sgp4)

    # initial state at mid epoch
    initial_state = propagate_sgp4(metadata, initial_epoch, [mid_epoch], old_yml)[0, 1:]

    # Define accelerations exerted on Delfi
    # Warning: point_mass_gravity and spherical_harmonic_gravity accelerations should not be defined simultaneously for a single body
    accelerations = dict(
        Sun={
            'point_mass_gravity': True,
            'solar_radiation_pressure': True
        },
        Moon={
            'point_mass_gravity': True
        },
        Earth={
            'point_mass_gravity': False,
            'spherical_harmonic_gravity': True,
            'drag': True
        },
        Venus={
            'point_mass_gravity': True
        },
        Mars={
            'point_mass_gravity': True
        },
        Jupiter={
            'point_mass_gravity': True
        }
    )

    # Create propagator settings
    propagator_settings = create_propagator_settings(initial_state, initial_epoch, final_epoch, bodies, accelerations)

    # Define ideal position observation settings
    link_ends = dict()
    link_ends[observation.observed_body] = observation.body_origin_link_end_id('Delfi')
    link_def = observation.LinkDefinition(link_ends)

    position_obs_settings = [observation.cartesian_position(link_def)]

    # Define epochs at which the ephemerides shall be checked
    obs_times = np.arange(initial_epoch, final_epoch, 60.0)

    # Create the observation simulation settings per moon
    simulation_settings = list()
    simulation_settings.append(observation.tabulated_simulation_settings(
        observation.position_observable_type, link_def, obs_times, reference_link_end_type=observation.observed_body))

    # Create observation simulators
    obs_simulators = estimation_setup.create_observation_simulators(position_obs_settings, bodies)
    sgp4_states = estimation.simulate_observations(simulation_settings, obs_simulators, bodies)

    parameters_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)
    parameters_settings.append(estimation_setup.parameter.constant_drag_coefficient("Delfi"))
    parameters_to_estimate = estimation_setup.create_parameter_set(parameters_settings, bodies, propagator_settings, [])
    original_parameters = parameters_to_estimate.parameter_vector

    estimator = numerical_simulation.Estimator(bodies, parameters_to_estimate, position_obs_settings, propagator_settings)

    # Create input object for the estimation
    estimation_input = estimation.EstimationInput(sgp4_states)

    # Perform the estimation
    estimation_output = estimator.perform_estimation(estimation_input)
    updated_initial_states = parameters_to_estimate.parameter_vector[:6]
    updated_drag_coef = parameters_to_estimate.parameter_vector[6]

    return initial_epoch, mid_epoch, final_epoch, updated_initial_states, updated_drag_coef
