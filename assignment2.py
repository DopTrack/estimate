# Load standard modules
import statistics
# Uncomment the following to make plots interactive
# %matplotlib widget
from matplotlib import pyplot as plt

from propagation_functions.environment import *
from propagation_functions.propagation import *
from estimation_functions.estimation import define_doptrack_station, define_parameters, define_observation_settings, simulate_observations_from_estimator, \
    run_estimation
from estimation_functions.observations_data import load_and_format_observations

from utility_functions.time import *
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
metadata_folder = 'metadata/'
data_folder = 'data/'

# Files to be uploaded
metadata = ['Delfi-C3_32789_202004011219.yml'] #['Delfi-C3_32789_202004011044.yml']

data = ['Delfi-C3_32789_202004011219.DOP1C'] #['Delfi-C3_32789_202004011044.DOP1C']


# Retrieve initial epoch and state of the first pass
initial_epoch, initial_state_teme = get_tle_initial_conditions(metadata_folder + metadata[0])
start_recording_day = get_start_next_day(initial_epoch)

# Calculate final propagation_functions epoch
nb_days_to_propagate = 8
final_epoch = start_recording_day + nb_days_to_propagate * 86400.0

print('initial_epoch', initial_epoch)
print('final_epoch', final_epoch)

# Load and process observations
passes_start_times, passes_end_times, observation_times, observations_set = load_and_format_observations(data_folder, data)
print('passes_start_times', passes_start_times)


# Define day-long arcs and retrieve the corresponding arc starting times
arc_start_times = get_days_starting_times(passes_start_times)
arc_end_times = get_days_end_times(arc_start_times)

# Load spice kernels
spice.load_standard_kernels()

# Define propagation_functions environment
mass_delfi = 2.2
reference_area_delfi = 0.035
drag_coefficient_delfi = 1.4
srp_coefficient_delfi = 2.4
bodies = define_environment(mass_delfi, reference_area_delfi, drag_coefficient_delfi, srp_coefficient_delfi)

# Set Delfi's initial state of Delfi
initial_state = element_conversion.teme_state_to_j2000(initial_epoch, initial_state_teme)

# Create numerical integrator settings
integrator_settings = propagation_setup.integrator.runge_kutta_4(initial_epoch, 10.0)

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
    }
)
accelerations = create_accelerations(acceleration_models, bodies)

# Propagate dynamics and retrieve Delfi's initial state at the start of each arc
orbit = propagate_initial_state(initial_state, initial_epoch, final_epoch, bodies, accelerations)
arc_wise_initial_states = get_initial_states(bodies, arc_start_times)


# Redefine environment to allow for multi-arc dynamics propagation_functions
bodies = define_environment(mass_delfi, reference_area_delfi, drag_coefficient_delfi, srp_coefficient_delfi, True)
accelerations = create_accelerations(acceleration_models, bodies)

# Define multi-arc propagator settings
multi_arc_propagator_settings = define_multi_arc_propagation_settings(arc_wise_initial_states, arc_start_times, arc_end_times, bodies, accelerations)

# Create the DopTrack station
define_doptrack_station(bodies)


# Define default observation settings
Doppler_models = dict(
    absolute_bias={
        'activated': True,
        'times': passes_start_times
    },
    relative_bias={
        'activated': True,
        'times': passes_start_times
    },
    time_bias={
        'activated': True,
        'times': passes_start_times
    }
)
observation_settings = define_observation_settings(Doppler_models)

# Define parameters to estimate
parameters_list = dict(
    initial_state_delfi={
        'estimate': True,
        'type': 'per_arc' # can only be per arc
    },
    absolute_bias={
        'estimate': True,
        'type': 'per_pass'
    },
    relative_bias={
        'estimate': True,
        'type': 'per_pass'},
    time_bias={
        'estimate': True,
        'type': 'per_pass'
    }
)
parameters_to_estimate = define_parameters(parameters_list, bodies, multi_arc_propagator_settings, initial_epoch,
                                           arc_start_times, passes_start_times, Doppler_models)
estimation_setup.print_parameter_names(parameters_to_estimate)

# Create the estimator object
estimator = numerical_simulation.Estimator(bodies, parameters_to_estimate, observation_settings, multi_arc_propagator_settings)

# Simulate (ideal) observations
ideal_observations = simulate_observations_from_estimator(observation_times, estimator, bodies)


# Save the true parameters to later analyse the error
truth_parameters = parameters_to_estimate.parameter_vector
nb_parameters = len(truth_parameters)

# Perform estimation_functions
nb_iterations = 10
nb_arcs = len(arc_start_times)
pod_output = run_estimation(estimator, parameters_to_estimate, observations_set, nb_arcs, nb_iterations)

print(pod_output.formal_errors)

residuals = pod_output.residual_history
mean_residuals = statistics.mean(residuals[:,nb_iterations-1])
std_residuals = statistics.stdev(residuals[:,nb_iterations-1])

print('mean', mean_residuals)
print('standard deviation', std_residuals)



# Plot residuals
fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot()
ax.set_title(f'Residuals')

# ax.plot(residuals[:,0], color='red', linestyle='-.')
# ax.plot(residuals[:,1], color='green', linestyle='-.')
# ax.plot(residuals[:,2], color='green', linestyle='-.')
ax.plot(residuals[:,nb_iterations-1],color='blue', linestyle='-.')

# ax.legend()
ax.set_xlabel('Time [s]')
ax.set_ylabel('Doppler [m/s]')
plt.grid()
plt.show()

# Plot residuals histogram
fig = plt.figure()
ax = fig.add_subplot()
plt.hist(residuals[:,1],100)
plt.hist(residuals[:,nb_iterations-1],100)
ax.set_xlabel('Doppler residuals [m/s]')
ax.set_ylabel('Nb occurrences []')
plt.grid()
plt.show()


