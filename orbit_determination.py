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
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation_setup
from tudatpy.kernel.astro import element_conversion

j2000_days = 2451545.0

# Define import folder
data_folder = 'delfiC3/'

# Files to be uploaded
metadata = ['Delfi-C3_32789_202309240829.yml', 'Delfi-C3_32789_202309241900.yml']
data = ['Delfi-C3_32789_202309240829.csv', 'Delfi-C3_32789_202309241900.csv']

# Retrieve initial epoch and state of the first pass
initial_epoch, initial_state_teme, b_star_coef = get_tle_initial_conditions(data_folder + metadata[0])
start_recording_day = get_start_next_day(initial_epoch)

# Calculate final propagation_functions epoch
final_epoch = start_recording_day + 1.0 * 86400.0

# Load and process observations
recording_start_times = extract_recording_start_times_yml(data_folder, metadata)
passes_start_times, passes_end_times, observation_times, observations_set = load_and_format_observations(data_folder, data, recording_start_times, new_obs_format=True)

# Define tracking arcs and retrieve the corresponding arc starting times (this will change throughout the assignment)
# Four options: one arc per pass ('per_pass'), one arc per day ('per_day'), one arc every 3 days ('per_3_days') and one arc per week ('per_week')
arc_start_times, arc_mid_times, arc_end_times = define_arcs('per_day', passes_start_times, passes_end_times)

# Define propagation_functions environment
mass_delfi = 2.2
ref_area_delfi = 0.035
drag_coefficient_delfi = get_drag_coefficient(mass_delfi, ref_area_delfi, b_star_coef, from_tle=True)
srp_coefficient_delfi = 1.2
bodies = define_environment(mass_delfi, ref_area_delfi, drag_coefficient_delfi, srp_coefficient_delfi, multi_arc_ephemeris=False)

# Set spacecraft initial state of Delfi
initial_state = element_conversion.teme_state_to_j2000(initial_epoch, initial_state_teme)


# Define accelerations exerted on Delfi
# Warning: point_mass_gravity and spherical_harmonic_gravity accelerations should not be defined simultaneously for a single body
acceleration_models = dict(
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

# Propagate dynamics and retrieve Delfi's initial state at the start of each arc
orbit = propagate_initial_state(initial_state, initial_epoch, final_epoch, bodies, acceleration_models)
arc_wise_initial_states = get_initial_states(bodies, arc_mid_times)
# propagated_states = orbit[0]
# propagated_states_sgp4 = propagate_sgp4(data_folder+metadata, initial_epoch, propagated_states[:, 0].tolist())
#
# diff_tudat_sgp4 = np.linalg.norm(propagated_states_sgp4[:, 1:3] - propagated_states[:, 1:3], axis=1)
#
# # Plot propagated orbit
# plt.figure()
# plt.plot((propagated_states[:, 0]-initial_epoch)/3600.0, diff_tudat_sgp4, color='blue')
# plt.grid()
# plt.ylabel('Difference [km]')
# plt.xlabel('Time [h]')
# plt.show()

# Redefine environment to allow for multi-arc dynamics propagation_functions
bodies = define_environment(mass_delfi, ref_area_delfi, drag_coefficient_delfi, srp_coefficient_delfi, multi_arc_ephemeris=True)
accelerations = create_accelerations(acceleration_models, bodies)

# Define multi-arc propagator settings
multi_arc_propagator_settings = define_multi_arc_propagation_settings(arc_wise_initial_states, arc_start_times, arc_end_times, bodies, accelerations)

# Create the DopTrack station
define_doptrack_station(bodies)

# Define default observation settings
# Specify on which time interval the observation bias(es) should be defined. This will change throughout the assignment (can be 'per_pass', 'per_arc', 'global')
# Noting that the arc duration can vary (see arc definition line 64)
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
plt.hist(residuals[:,nb_iterations-1],100)
ax.set_xlabel('Doppler residuals')
ax.set_ylabel('Nb occurrences []')
plt.grid()
plt.show()

plt.figure(figsize=(9,5))
plt.imshow(np.abs(pod_output.correlations), aspect='auto', interpolation='none')
plt.colorbar()
plt.tight_layout()
plt.show()
