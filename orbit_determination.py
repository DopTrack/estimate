from propagation_functions.environment import *
from propagation_functions.propagation import *
from estimation_functions.estimation import *
from estimation_functions.observations_data import *
from fit_sgp4_solution import fit_sgp4_solution

from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import estimation_setup


def perform_orbit_determination(data_folder: str, metadata: list[str], data: list[str], process_strategy="per_pass", nb_iterations=10, old_yml=False, old_obs_format=False):

    # initial state at mid epoch
    initial_epoch, mid_epoch, final_epoch, initial_state, drag_coef = fit_sgp4_solution(data_folder + metadata[0], propagation_time_in_days=1.0, old_yml=old_yml)

    # Define propagation_functions environment
    mass = 2.2
    ref_area = 0.035
    srp_coef = 1.2
    bodies = define_environment(mass, ref_area, drag_coef, srp_coef, multi_arc_ephemeris=False)

    # Load and process observations
    recording_start_times = extract_recording_start_times_yml(data_folder, metadata, old_yml=False)
    passes_start_times, passes_end_times, observation_times, observations_set = load_and_format_observations(data_folder, data, recording_start_times, old_obs_format)

    # Define tracking arcs and retrieve the corresponding arc starting times
    arc_start_times, arc_mid_times, arc_end_times = define_arcs(process_strategy, passes_start_times, passes_end_times)

    # Define accelerations exerted on Delfi
    accelerations = get_default_acceleration_models()

    # Propagate dynamics and retrieve Delfi's initial state at the start of each arc
    orbit = propagate_initial_state(initial_state, initial_epoch, final_epoch, bodies, accelerations)
    arc_wise_initial_states = get_initial_states(bodies, arc_mid_times)

    # Redefine environment to allow for multi-arc dynamics propagation_functions
    bodies = define_environment(mass, ref_area, drag_coef, srp_coef, multi_arc_ephemeris=True)

    # Define multi-arc propagator settings
    multi_arc_propagator_settings = define_multi_arc_propagation_settings(arc_wise_initial_states, arc_start_times, arc_end_times, bodies, accelerations)

    # Create the DopTrack station
    define_doptrack_station(bodies)

    # Define default observation settings
    doppler_models = get_default_doppler_models()
    observation_settings = define_observation_settings(doppler_models, passes_start_times, arc_start_times)

    # Define parameters to estimate
    parameters_list = dict(
        initial_state_delfi={
            'estimate': True
        },
        absolute_bias={
            'estimate': True
        },
        time_drift={
            'estimate': True
        }
    )
    parameters_to_estimate = define_parameters(parameters_list, bodies, multi_arc_propagator_settings, initial_epoch,
                                               arc_start_times, arc_mid_times, [(get_link_ends_id("DopTrackStation"), passes_start_times)], doppler_models)
    estimation_setup.print_parameter_names(parameters_to_estimate)

    # Create the estimator object
    estimator = numerical_simulation.Estimator(bodies, parameters_to_estimate, observation_settings, multi_arc_propagator_settings)

    # Save the true parameters to later analyse the error
    original_parameters = parameters_to_estimate.parameter_vector

    # Perform estimation_functions
    nb_arcs = len(arc_start_times)
    pod_output = run_estimation(estimator, parameters_to_estimate, observations_set, nb_arcs, nb_iterations)

    errors = pod_output.formal_errors
    residuals = pod_output.residual_history
    updated_parameters = parameters_to_estimate.parameter_vector

    return original_parameters, updated_parameters, errors, residuals[:, -1]

