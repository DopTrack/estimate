# Load standard modules
import statistics

import numpy as np
from matplotlib import pyplot as plt

from propagation_functions.environment import *
from propagation_functions.propagation import *
from estimation_functions.estimation import *
from estimation_functions.observations_data import *
from fit_sgp4_solution import fit_sgp4_solution

from utility_functions.tle import *

from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import estimation_setup

# Define import folder
data_folder = 'delfiC3/'

# Files to be uploaded
metadata = ['Delfi-C3_32789_202309240829.yml', 'Delfi-C3_32789_202309241900.yml']
data = ['Delfi-C3_32789_202309240829.csv', 'Delfi-C3_32789_202309241900.csv']

# initial state at mid epoch
initial_epoch, mid_epoch, final_epoch, initial_state, drag_coef = fit_sgp4_solution(data_folder + metadata[0], propagation_time_in_days=1.0)

# Define propagation_functions environment
mass = 2.2
ref_area = 0.035
srp_coef = 1.2
bodies = define_environment(mass, ref_area, drag_coef, srp_coef, multi_arc_ephemeris=False)

# Load and process observations
recording_start_times = extract_recording_start_times_yml(data_folder, metadata)
passes_start_times, passes_end_times, observation_times, observations_set = load_and_format_observations(data_folder, data, recording_start_times, new_obs_format=True)

# Define tracking arcs and retrieve the corresponding arc starting times
arc_start_times, arc_mid_times, arc_end_times = define_arcs('per_day', passes_start_times, passes_end_times)

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
Doppler_models = get_default_doppler_models()
observation_settings = define_observation_settings(Doppler_models, passes_start_times, arc_start_times)

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
                                           arc_start_times, arc_mid_times, [(get_link_ends_id("DopTrackStation"), passes_start_times)], Doppler_models)
estimation_setup.print_parameter_names(parameters_to_estimate)

# Create the estimator object
estimator = numerical_simulation.Estimator(bodies, parameters_to_estimate, observation_settings, multi_arc_propagator_settings)

# Save the true parameters to later analyse the error
truth_parameters = parameters_to_estimate.parameter_vector
nb_parameters = len(truth_parameters)

# Perform estimation_functions
nb_iterations = 15
nb_arcs = len(arc_start_times)
pod_output = run_estimation(estimator, parameters_to_estimate, observations_set, nb_arcs, nb_iterations)

errors = pod_output.formal_errors
residuals = pod_output.residual_history
updated_parameters = parameters_to_estimate.parameter_vector
print('diff param', updated_parameters - truth_parameters)

# Compute residuals statistics
mean_residuals = statistics.mean(residuals[:, -1])
std_residuals = statistics.stdev(residuals[:, -1])
rms_residuals = math.sqrt(np.square(residuals[:, -1]).mean())
print('mean_residuals', mean_residuals)
print('std_residuals', std_residuals)
print("rms_residuals", rms_residuals)

# Retrieve estimated parameter values
estimated_state = updated_parameters[0:6]
estimated_bias = updated_parameters[6]
estimated_time_drift = updated_parameters[7]

print('delta state', estimated_state - truth_parameters[0:6])

# Retrieve estimation errors
sig_state = errors[0:6]
sig_bias = errors[6]
sig_time_drift = errors[7]


residuals_per_pass = get_residuals_per_pass(observation_times, residuals, passes_start_times)

# Plot residuals
fig = plt.figure()
fig.tight_layout()
fig.subplots_adjust(hspace=0.3)

for i in range(len(passes_start_times)):
    ax = fig.add_subplot(len(passes_start_times), 1, i+1)
    ax.plot(residuals_per_pass[i], color='blue', linestyle='-.')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Residuals [m/s]')
    ax.set_title(f'Pass '+str(i+1))
    plt.grid()
plt.show()


# Plot residuals histogram
fig = plt.figure()
ax = fig.add_subplot()
# plt.hist(residuals[:,1],100)
plt.hist(residuals[:,nb_iterations-1], 100)
ax.set_xlabel('Doppler residuals')
ax.set_ylabel('Nb occurrences []')
plt.grid()
plt.show()

plt.figure(figsize=(9,5))
plt.imshow(np.abs(pod_output.correlations), aspect='auto', interpolation='none')
plt.colorbar()
plt.tight_layout()
plt.show()
