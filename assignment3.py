import math
# import sys
# sys.path.insert(0, 'tudat-bundle/cmake-build-release/tudatpy')

# Load standard modules
import statistics

# Uncomment the following to make plots interactive
# %matplotlib widget
from matplotlib import pyplot as plt

from propagation_functions.environment import *
from propagation_functions.propagation import *
from estimation_functions.estimation import *
from estimation_functions.observations_data import *

from utility_functions.time import *
from utility_functions.tle import *

# Load tudatpy modules
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation_setup
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.numerical_simulation.estimation_setup import observation

j2000_days = 2451545.0

# Define import folder
metadata_folder = 'nayif_data/' # 'metadata/'
data_folder = 'nayif_data/'

# Files to be uploaded
metadata = ['Nayif-1_42017_202101011249.yml', 'Nayif-1_42017_202101012156.yml', 'Nayif-1_42017_202101012331.yml',
            'Nayif-1_42017_202101021051.yml', 'Nayif-1_42017_202101021225.yml', 'Nayif-1_42017_202101022131.yml', 'Nayif-1_42017_202101022305.yml',
            'Nayif-1_42017_202101031026.yml', 'Nayif-1_42017_202101031200.yml', 'Nayif-1_42017_202101032240.yml',
            'Nayif-1_42017_202101041002.yml', 'Nayif-1_42017_202101041135.yml', 'Nayif-1_42017_202101041309.yml', 'Nayif-1_42017_202101042043.yml', 'Nayif-1_42017_202101042215.yml',
            'Nayif-1_42017_202101050938.yml', 'Nayif-1_42017_202101051110.yml', 'Nayif-1_42017_202101051244.yml', 'Nayif-1_42017_202101052020.yml', 'Nayif-1_42017_202101052151.yml', 'Nayif-1_42017_202101052326.yml',
            'Nayif-1_42017_202101061220.yml', 'Nayif-1_42017_202101062300.yml']

data = ['Nayif-1_42017_202101011249.csv', 'Nayif-1_42017_202101012156.csv', 'Nayif-1_42017_202101012331.csv',
        'Nayif-1_42017_202101021051.csv', 'Nayif-1_42017_202101021225.csv', 'Nayif-1_42017_202101022131.csv', 'Nayif-1_42017_202101022305.csv',
        'Nayif-1_42017_202101031026.csv', 'Nayif-1_42017_202101031200.csv', 'Nayif-1_42017_202101032240.csv',
        'Nayif-1_42017_202101041002.csv', 'Nayif-1_42017_202101041135.csv', 'Nayif-1_42017_202101041309.csv', 'Nayif-1_42017_202101042043.csv', 'Nayif-1_42017_202101042215.csv',
        'Nayif-1_42017_202101050938.csv', 'Nayif-1_42017_202101051110.csv', 'Nayif-1_42017_202101051244.csv', 'Nayif-1_42017_202101052020.csv', 'Nayif-1_42017_202101052151.csv', 'Nayif-1_42017_202101052326.csv',
        'Nayif-1_42017_202101061220.csv', 'Nayif-1_42017_202101062300.csv']

# Specify which metadata and data files should be loaded (this will change throughout the assignment)
# indices_files_to_load = [0, 2,
#                          3, 4,
#                          7, 8,
#                          10, 11, 12,
#                          15, 17,
#                          21]

indices_files_to_load = [0,
                         4, 7]#,
                         # 7, 8,
                         # 10, 11, 12,
                         # 15, 17,
                         # 21]


# Retrieve initial epoch and state of the first pass
# initial_epoch, initial_state_teme, b_star_coef = get_tle_initial_conditions(metadata_folder + metadata[0])
# print('initial_epoch', initial_epoch)
# print('initial_state_teme', initial_state_teme)
# print('b_star_coef', b_star_coef)
initial_epoch = 662681610.0000188
initial_state_teme = np.array([2.24031264e6, 6.48506990e6, -1.59193504e2, 9.17424053e2, -3.25519860e2, 7.56370783e3])
b_star_coef = 0.0001178
start_recording_day = get_start_next_day(initial_epoch)


# Calculate final propagation_functions epoch
nb_days_to_propagate = 9
final_epoch = start_recording_day + nb_days_to_propagate * 86400.0

print('initial_epoch', initial_epoch)
print('final_epoch', final_epoch)

# Retrieve recording start epochs
recording_start_times = extract_recording_start_times_old_yml(metadata_folder, [metadata[i] for i in indices_files_to_load])

# Load and process observations
passes_start_times, passes_end_times, observation_times, observations_set = load_and_format_observations(
    data_folder, [data[i] for i in indices_files_to_load], recording_start_times, new_obs_format=True)


# Define tracking arcs and retrieve the corresponding arc starting times (this will change throughout the assignment)
# Four options: one arc per pass ('per_pass'), one arc per day ('per_day'), one arc every 3 days ('per_3_days') and one arc per week ('per_week')
arc_start_times, arc_end_times = define_arcs('per_3_days', passes_start_times, passes_end_times)

print('arc_start_times', arc_start_times)
print('arc_end_times', arc_end_times)

print('passes_start_times', passes_start_times)
print('passes_end_times', passes_end_times)


# Define propagation_functions environment
mass_delfi = 2.2
ref_area_delfi = 0.08
drag_coefficient_delfi = get_drag_coefficient(mass_delfi, ref_area_delfi, b_star_coef, from_tle=False)
print('Cd from TLE', drag_coefficient_delfi)
srp_coefficient_delfi = 1.2
bodies = define_environment(mass_delfi, ref_area_delfi, drag_coefficient_delfi, srp_coefficient_delfi, multi_arc_ephemeris=False)

# Set Delfi's initial state of Delfi
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
arc_wise_initial_states = get_initial_states(bodies, arc_start_times)


# Redefine environment to allow for multi-arc dynamics propagation_functions
bodies = define_environment(mass_delfi, ref_area_delfi, drag_coefficient_delfi, srp_coefficient_delfi, multi_arc_ephemeris=True)
accelerations = create_accelerations(acceleration_models, bodies)

real_mu = bodies.get("Earth").gravity_field_model.gravitational_parameter
bodies.get("Earth").gravity_field_model.gravitational_parameter = 1.0 * bodies.get("Earth").gravity_field_model.gravitational_parameter

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
    relative_bias={
        'activated': True,
        'time_interval': bias_definition
    },
    time_drift={
        'activated': True,
        'time_interval': bias_definition
    },
    time_bias={
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
    relative_bias={
        'estimate': False
    },
    time_drift={
        'estimate': True
    },
    time_bias={
        'estimate': False
    },
    drag_coefficient={
        'estimate': False,
        'type': 'per_arc'
    },
    srp_coefficient={
        'estimate': False,
        'type': 'per_arc'
    },
    gravitational_parameter={
        'estimate': True,
        'type': 'global' # can only be global
    },
    C20={
        'estimate': True,
        'type': 'global' # can only be global
    },
    C22={
        'estimate': False,
        'type': 'global' # can only be global
    }
)
parameters_to_estimate = define_parameters(parameters_list, bodies, multi_arc_propagator_settings, initial_epoch,
                                           arc_start_times, [(get_link_ends_id("DopTrackStation"), passes_start_times)], Doppler_models)
estimation_setup.print_parameter_names(parameters_to_estimate)


# Create the estimator object
estimator = numerical_simulation.Estimator(bodies, parameters_to_estimate, observation_settings, multi_arc_propagator_settings)

# Simulate (ideal) observations
ideal_observations = simulate_observations_from_estimator(observation_times, estimator, bodies)

# Save the initial parameters values to later analyse the error
initial_parameters = parameters_to_estimate.parameter_vector
nb_parameters = len(initial_parameters)

# Perform estimation_functions
nb_iterations = 10
nb_arcs = len(arc_start_times)
pod_output = run_estimation(estimator, parameters_to_estimate, observations_set, nb_arcs, nb_iterations)

residuals = pod_output.residual_history
mean_residuals = statistics.mean(residuals[:,nb_iterations-1])
std_residuals = statistics.stdev(residuals[:,nb_iterations-1])

# Retrieve updated parameters
updated_parameters = parameters_to_estimate.parameter_vector
print('initial parameter values', initial_parameters)
print('updated parameters', updated_parameters)
print('update', updated_parameters - initial_parameters)

original_state = initial_parameters[0:6]
updated_state = updated_parameters[0:6]
print('original_state', original_state)
print('updated_state', updated_state)

original_state_keplerian = element_conversion.cartesian_to_keplerian(original_state, bodies.get("Earth").gravity_field_model.gravitational_parameter)
updated_state_keplerian = element_conversion.cartesian_to_keplerian(updated_state, bodies.get("Earth").gravity_field_model.gravitational_parameter)
print('original_state_keplerian', original_state_keplerian)
print('updated_state_keplerian', updated_state_keplerian)
print('Diff a', updated_state_keplerian[0] - original_state_keplerian[0])
print('Diff e', updated_state_keplerian[1] - original_state_keplerian[1])
print('Diff i', (updated_state_keplerian[2] - original_state_keplerian[2]) * 180.0 / math.pi)
print('Diff w+true anomaly', ((updated_state_keplerian[3]+updated_state_keplerian[5]) - (original_state_keplerian[3]+original_state_keplerian[5])) * 180.0 / math.pi)
print('Diff RAAN', (updated_state_keplerian[4] - original_state_keplerian[4]) * 180.0 / math.pi)

# print('real_mu', real_mu)
# print('initial offset mu', initial_parameters[6] - real_mu)
# print('final offset mu', updated_parameters[6] - real_mu)

# new_b_star = 0.1570*ref_area_delfi*updated_parameters[12]/(2.0*mass_delfi)
# print('b_star', b_star_coef)
# print('new b_star', new_b_star)

# Retrieve formal errors and correlations
formal_errors = pod_output.formal_errors
correlations = pod_output.correlations

print('formal errors', formal_errors)

# Retrieve residuals per pass
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
plt.hist(residuals[:,-1],100)
ax.set_xlabel('Doppler residuals')
ax.set_ylabel('Nb occurrences []')
plt.grid()
plt.show()

plt.figure(figsize=(9,5))
plt.imshow(np.abs(pod_output.correlations), aspect='auto', interpolation='none')
plt.colorbar()
plt.tight_layout()
plt.show()



