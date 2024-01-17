# Load required standard modules
import numpy as np
from matplotlib import pyplot as plt

# Load tudatpy modules
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy.astro.time_conversion import DateTime
from tudatpy.astro import element_conversion

# Import doptrack-estimate functions
from propagation_functions.environment import *
from propagation_functions.propagation import *
from estimation_functions.estimation import *
from utility_functions.tle import *


# Load spice kernels
spice.load_standard_kernels()

# Define initial TLE of Delfi-C3
delfi_tle = environment.Tle("1 32789U 08021G   20090.88491347 +.00001016 +00000-0 +70797-4 0  9997",
                            "2 32789 097.4279 136.4027 0011143 218.6381 141.4051 15.07550601649972")

# Define next TLE update of Delfi-C3
# (useful to derive realistic initial state perturbations based on the quality of our knowledge of Delfi's orbit)
next_delfi_tle = environment.Tle("1 32789U 08021G   20092.14603172 +.00001512 +00000-0 +10336-3 0  9992",
                                 "2 32789 097.4277 137.6209 0011263 214.0075 146.0432 15.07555919650162")

# Define simulation start and end epochs
initial_epoch = delfi_tle.get_epoch()
propagation_time = 1.0 * 86400.0
final_epoch = initial_epoch + propagation_time
mid_epoch = (initial_epoch + final_epoch)/2.0

# Define start, middle, and end times of each arc
arc_duration = 1.0 * 86400.0
arc_start_times = [initial_epoch]
arc_end_times = []
current_arc_time = initial_epoch
while final_epoch - current_arc_time > arc_duration:
    current_arc_time += arc_duration
    arc_end_times.append(current_arc_time)
    arc_start_times.append(current_arc_time)
arc_end_times.append(final_epoch)

arc_mid_times = []
for i in range(len(arc_start_times)):
    arc_mid_times.append((arc_start_times[i] + arc_end_times[i])/2.0)

# Define propagation environment
mass = 2.2
ref_area = (4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
drag_coef = 1.2
srp_coef = 1.2
bodies = define_environment(mass, ref_area, drag_coef, srp_coef, "spacecraft", multi_arc_ephemeris=True)

# Define propagated bodies
bodies_to_propagate = ["spacecraft"]

# Define central bodies
central_bodies = ["Earth"]


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


# Retrieve global initial state at mid-epoch from TLE ephemeris
delfi_ephemeris = environment.TleEphemeris("Earth", "J2000", delfi_tle, False)
initial_state = delfi_ephemeris.cartesian_state(mid_epoch)

# Propagate spacecraft orbit and retrieve arc-wise initial states
orbit = propagate_initial_state(initial_state, initial_epoch, final_epoch, bodies, accelerations, "spacecraft")
state_history = orbit[0]

arc_wise_initial_states = []
for time in arc_mid_times:
    current_initial_state = np.zeros(6)
    for i in range(6):
        current_initial_state[i] = np.interp(np.array(time), state_history[:, 0], state_history[:, i+1])
    arc_wise_initial_states.append(current_initial_state)


# Create propagation settings
multi_arc_propagation_settings = define_multi_arc_propagation_settings(
    arc_wise_initial_states, arc_start_times, arc_end_times, bodies, accelerations, "spacecraft")

# Create the DopTrack ground station
define_doptrack_station(bodies)

# Create artificial ground stations
nb_artificial_stations = 2
stations_lat = [-25.0, -14.0]
stations_long = [134.0, -52.0]

stations_names = ["DopTrackStation"]
for i in range(nb_artificial_stations):
    stations_names.append("Station" + str(i+1))

    environment_setup.add_ground_station(
        bodies.get_body("Earth"), stations_names[i+1], [0.0, np.deg2rad(stations_lat[i]), np.deg2rad(stations_long[i])], element_conversion.geodetic_position_type)


# Define the uplink link ends for one-way Doppler observable
link_definitions = []
for i in range(nb_artificial_stations+1):
    link_ends = dict()
    if i == 0:
        link_ends[observation.transmitter] = observation.body_reference_point_link_end_id("Earth", "DopTrackStation")
    else:
        link_ends[observation.transmitter] = observation.body_reference_point_link_end_id("Earth", "Station"+str(i))
    link_ends[observation.receiver] = observation.body_origin_link_end_id("spacecraft")

    link_definitions.append(observation.LinkDefinition(link_ends))


# Create observation settings for each link/observable
observation_settings_list = []
for link in link_definitions:
    observation_settings_list.append(observation.one_way_doppler_instantaneous(link))

observation_times = np.arange(initial_epoch, final_epoch, 10.0)

observation_simulation_settings = []
for i in range(nb_artificial_stations+1):
    observation_simulation_settings.append(observation.tabulated_simulation_settings(
        observation.one_way_instantaneous_doppler_type, link_definitions[i], observation_times))

# Add Gaussian noise to simulated observations
noise_level = 1.0E-3  # in m/s
observation.add_gaussian_noise_to_observable(observation_simulation_settings, noise_level, observation.one_way_instantaneous_doppler_type)

# Create observation viability settings
for i in range(nb_artificial_stations+1):
    viability_setting = observation.elevation_angle_viability(["Earth", stations_names[i]], np.deg2rad(15))

    observation.add_viability_check_to_observable_for_link_ends(
        [observation_simulation_settings[i]], [viability_setting], observation.one_way_instantaneous_doppler_type, link_definitions[i])


# Define parameters to estimate
parameters_list = dict(
    initial_state={
        'estimate': True
    },
    drag_coefficient={
        'estimate': True,
        'type': 'per_arc'
    },
    gravitational_parameter={
        'estimate': True,
        'type': 'global' # can only be global
    },
    C20={
        'estimate': False,
        'type': 'global' # can only be global
    },
    C22={
        'estimate': False,
        'type': 'global' # can only be global
    }
)
parameters_to_estimate = define_parameters(parameters_list, bodies, multi_arc_propagation_settings, "spacecraft", arc_start_times, arc_mid_times )
estimation_setup.print_parameter_names(parameters_to_estimate)
nb_parameters = parameters_to_estimate.parameter_set_size

# Create estimator
estimator = numerical_simulation.Estimator(
    bodies, parameters_to_estimate, observation_settings_list, multi_arc_propagation_settings)

# Simulate observations
simulated_observations = estimation.simulate_observations(
    observation_simulation_settings, estimator.observation_simulators, bodies)

### Retrieve simulated observations, sorted per ground station
sorted_obs_collection = simulated_observations.sorted_observation_sets
observation_times = np.array(simulated_observations.concatenated_times)
observations_list = np.array(simulated_observations.concatenated_observations)

# Save the true parameter values to later analyse the estimation errors
truth_parameters = parameters_to_estimate.parameter_vector
print('truth_parameters', truth_parameters)


# Use next TLE update to derive realistic initial state perturbation
next_delfi_ephemeris = environment.TleEphemeris("Earth", "J2000", next_delfi_tle, False)
perturbed_initial_state = next_delfi_ephemeris.cartesian_state(mid_epoch)

# Perturb the initial state estimate from the truth (10 m in position; 0.1 m/s in velocity)
perturbed_parameters = truth_parameters.copy( )
perturbed_parameters[:6] = perturbed_initial_state
# perturbed_parameters[7] *= 1.001
# for i in range(3):
#     perturbed_parameters[i] += 1.0e2 #10.0
#     perturbed_parameters[i+3] += 0.1 #0.01
parameters_to_estimate.parameter_vector = perturbed_parameters
initial_parameters_perturbation = perturbed_parameters - truth_parameters


# Create input object for the estimation
convergence_checker = estimation.estimation_convergence_checker(maximum_iterations=10)
estimation_input = estimation.EstimationInput(simulated_observations, convergence_checker=convergence_checker)

# Set methodological options
estimation_input.define_estimation_settings(reintegrate_variational_equations=True)

# Define observation weights
weights_per_observable = {estimation_setup.observation.one_way_instantaneous_doppler_type: noise_level ** -2}
estimation_input.set_constant_weight_per_observable(weights_per_observable)


# Perform the estimation
estimation_input.define_estimation_settings(save_state_history_per_iteration=True)
estimation_output = estimator.perform_estimation(estimation_input)
simulator_object = estimation_output.simulation_results_per_iteration[0]
state_history = result2array(simulator_object.single_arc_results[0].dynamics_results.state_history)

# Retrieve formal errors and (final) true errors
formal_errors = estimation_output.formal_errors
true_errors = parameters_to_estimate.parameter_vector - truth_parameters

# Retrieve correlation matrix
correlations = estimation_output.correlations

print('initial parameters perturbation', initial_parameters_perturbation)
print('true_errors', true_errors)
print('formal errors', formal_errors)

# Compute true error and true-to-formal error ratio for each parameter, at each iteration
# each column corresponds to one LSQ iteration
parameters_history = estimation_output.parameter_history
true_errors_history = np.zeros(parameters_history.shape)
true_to_formal_errors_history = np.zeros(parameters_history.shape)
for i in range(parameters_history.shape[1]):
    true_errors_history[:, i] = np.abs(parameters_history[:, i] - truth_parameters)
    true_to_formal_errors_history[:, i] = np.divide(true_errors_history[:, i], formal_errors)


# Retrieve final residuals and residuals history
final_residuals = estimation_output.final_residuals
residual_history = estimation_output.residual_history


# Plot simulated Doppler observations for each station, as a function of time.
plt.figure()
plt.title("Simulated Doppler observations")
for obs in sorted_obs_collection.values():
    for i in range(len(obs)):
        plt.scatter((np.array(obs[i][0].observation_times) - initial_epoch) / 3600.0, obs[i][0].concatenated_observations, label=stations_names[i])
plt.grid()
plt.xlabel("Time since initial epoch [hr]")
plt.ylabel("Range-rate [m/s]")
plt.legend()
plt.show()


# # Parameters history
# parameters_history = estimation_output.parameter_history
# true_errors_history = np.zeros(parameters_history.shape)
# for i in range(parameters_history.shape[1]):
#     true_errors_history[:, i] = np.abs(parameters_history[:, i] - truth_parameters)

# Plot true-to-formal errors ratio after the last LSQ iteration
plt.figure()
plt.title('True-to-formal errors ratio')
plt.scatter(np.arange(0, nb_parameters), true_to_formal_errors_history[:, -1], color='blue')
plt.plot(np.arange(0, nb_parameters), np.ones(nb_parameters), linestyle='dashed', color='blue', label='true error = formal error')
plt.xlabel('Parameter index [-]')
plt.ylabel('True-to-formal errors ratio [-]')
plt.grid()
plt.legend()
plt.show()


# Plot final residuals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4*2, 4.8))

ax1.scatter((np.array(observation_times) - initial_epoch)/3600.0, residual_history[:, 0], color='blue', label='residuals')
ax1.plot((np.array(observation_times) - initial_epoch)/3600.0, noise_level*np.ones(len(observation_times)), color='blue', label='1sigma noise level')
ax1.plot((np.array(observation_times) - initial_epoch)/3600.0, -noise_level*np.ones(len(observation_times)), color='blue')
ax1.plot((np.array(observation_times) - initial_epoch)/3600.0, 3*noise_level*np.ones(len(observation_times)), color='blue', linestyle='dotted', label='3sigma noise level')
ax1.plot((np.array(observation_times) - initial_epoch)/3600.0, -3*noise_level*np.ones(len(observation_times)), color='blue', linestyle='dotted')
ax1.set_ylabel('Residuals [m/s]')
ax1.set_xlabel('Time since initial epoch [hr]')
ax1.set_title('First iteration')
ax1.grid()
ax1.legend()

ax2.scatter((np.array(observation_times) - initial_epoch)/3600.0, residual_history[:, -1], color='blue', label='residuals')
ax2.plot((np.array(observation_times) - initial_epoch)/3600.0, noise_level*np.ones(len(observation_times)), color='blue', label='1sigma noise level')
ax2.plot((np.array(observation_times) - initial_epoch)/3600.0, -noise_level*np.ones(len(observation_times)), color='blue')
ax2.plot((np.array(observation_times) - initial_epoch)/3600.0, 3*noise_level*np.ones(len(observation_times)), color='blue', linestyle='dotted', label='3sigma noise level')
ax2.plot((np.array(observation_times) - initial_epoch)/3600.0, -3*noise_level*np.ones(len(observation_times)), color='blue', linestyle='dotted')
ax2.set_ylabel('Residuals [m/s]')
ax2.set_xlabel('Time since initial epoch [hr]')
ax2.set_title('Final iteration')
ax2.grid()
ax2.legend()

plt.tight_layout()
plt.show()


# Plot final residuals

print('True-to-formal-error ratio:')
print('\nInitial state')
print(((truth_parameters - parameters_to_estimate.parameter_vector) / estimation_output.formal_errors)[:6])
print('\nPhysical parameters')
print(((truth_parameters - parameters_to_estimate.parameter_vector) / estimation_output.formal_errors)[6:])


print('nb observations', len(observation_times))

plt.figure()
plt.hist(final_residuals, 25, color='blue')
plt.plot(-1.0*noise_level*np.ones(2), [0, len(observation_times)/10.0], color='blue', linestyle='solid', label='1sigma noise level')
plt.plot(1.0*noise_level*np.ones(2), [0, len(observation_times)/10.0], color='blue', linestyle='solid')
plt.plot(-3.0*noise_level*np.ones(2), [0, len(observation_times)/10.0], color='blue', linestyle='dashed', label='3sigma noise level')
plt.plot(3.0*noise_level*np.ones(2), [0, len(observation_times)/10.0], color='blue', linestyle='dashed')
plt.xlabel('Final iteration range-rate residual [m/s]')
plt.ylabel('Occurrences [-]')
plt.title('Final residuals histogram')
plt.tight_layout()
plt.grid()
plt.legend()
plt.show()


# Plot correlations
plt.figure()
plt.imshow(np.abs(correlations), aspect='auto', interpolation='none')
plt.colorbar(label='Absolute correlation [-]')
plt.title('Correlation matrix')
plt.xlabel('Parameter index [-]')
plt.ylabel('Parameter indasheddex [-]')
plt.show()
