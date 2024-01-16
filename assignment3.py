# Load required standard modules
import numpy as np
from matplotlib import pyplot as plt

# Load required tudatpy modules
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

# Retrieve the initial state of Delfi-C3 using Two-Line-Elements (TLEs)
tle_line1 = "1 32789U 08021G   20090.88491347 +.00001016 +00000-0 +70797-4 0  9997"
tle_line2 = "2 32789 097.4279 136.4027 0011143 218.6381 141.4051 15.07550601649972"
delfi_tle = environment.Tle(tle_line1, tle_line2)

next_delfi_tle = environment.Tle(
    "1 32789U 08021G   20092.14603172 +.00001512 +00000-0 +10336-3 0  9992",
    "2 32789 097.4277 137.6209 0011263 214.0075 146.0432 15.07555919650162"
)

# Set simulation start and end epochs
initial_epoch = delfi_tle.get_epoch()
propagation_time = 1.0 * 86400.0
final_epoch = initial_epoch + propagation_time
mid_epoch = (initial_epoch + final_epoch)/2.0

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

# Define bodies that are propagated
bodies_to_propagate = ["spacecraft"]

# Define central bodies of propagation
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


# Retrieve global initial state at mid-epoch
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


# Create numerical integrator settings
integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
    initial_time_step=60.0, coefficient_set=propagation_setup.integrator.CoefficientSets.rkdp_87)

# Create propagation settings
multi_arc_propagation_settings = define_multi_arc_propagation_settings(
    arc_wise_initial_states, arc_start_times, arc_end_times, bodies, accelerations, "spacecraft")

# Create the DopTrack ground station
define_doptrack_station(bodies)

# Create artificial ground stations
nb_artificial_stations = 1
stations_lat = [-25.0]
stations_long = [134.0]

stations_names = []
for i in range(nb_artificial_stations):
    station_name = "Station" + str(i+1)

    environment_setup.add_ground_station(
        bodies.get_body("Earth"),
        station_name,
        [0.0, np.deg2rad(stations_lat[i]), np.deg2rad(stations_long[i])],
        element_conversion.geodetic_position_type)


# Define the uplink link ends for one-way observable
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

# Define observation simulation times for each link
observation_times = np.arange(initial_epoch, final_epoch, 10.0)

observation_simulation_settings = []
for i in range(nb_artificial_stations+1):
    observation_simulation_settings.append(observation.tabulated_simulation_settings(
        observation.one_way_instantaneous_doppler_type, link_definitions[i], observation_times))

# Add Gaussian noise to simulated observations
noise_level = 1.0E-3
observation.add_gaussian_noise_to_observable(
    observation_simulation_settings,
    noise_level,
    observation.one_way_instantaneous_doppler_type
)

# Create viability settings
for i in range(nb_artificial_stations+1):
    if i == 0:
        viability_setting = observation.elevation_angle_viability(["Earth", "DopTrackStation"], np.deg2rad(15))
    else:
        viability_setting = observation.elevation_angle_viability(["Earth", "Station" + str(i)], np.deg2rad(15))

    observation.add_viability_check_to_observable_for_link_ends(
        [observation_simulation_settings[i]],
        [viability_setting],
        observation.one_way_instantaneous_doppler_type,
        link_definitions[i])


# Create parameters to be estimated
parameter_settings = estimation_setup.parameter.initial_states(multi_arc_propagation_settings, bodies, arc_mid_times)

parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Earth"))
parameter_settings.append(estimation_setup.parameter.constant_drag_coefficient("spacecraft"))

parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)



# Create the estimator
estimator = numerical_simulation.Estimator(
    bodies, parameters_to_estimate, observation_settings_list, multi_arc_propagation_settings)

# Simulate required observations
simulated_observations = estimation.simulate_observations(
    observation_simulation_settings, estimator.observation_simulators, bodies)

# Save the true parameters to later analyse the error
truth_parameters = parameters_to_estimate.parameter_vector
print('truth_parameters', truth_parameters)


# Use next TLE update to derive initial state perturbation
next_delfi_ephemeris = environment.TleEphemeris("Earth", "J2000", next_delfi_tle, False)
perturbed_initial_state = next_delfi_ephemeris.cartesian_state(mid_epoch)

# Perturb the initial state estimate from the truth (10 m in position; 0.1 m/s in velocity)
perturbed_parameters = truth_parameters.copy( )
perturbed_parameters[:6] = perturbed_initial_state
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

# Define weighting of the observations in the inversion
weights_per_observable = {estimation_setup.observation.one_way_instantaneous_doppler_type: noise_level ** -2}
estimation_input.set_constant_weight_per_observable(weights_per_observable)
estimation_input.define_estimation_settings(save_state_history_per_iteration=True)


# Perform the estimation
estimation_output = estimator.perform_estimation(estimation_input)
simulator_object = estimation_output.simulation_results_per_iteration[0]
state_history = result2array(simulator_object.single_arc_results[0].dynamics_results.state_history)

true_errors = parameters_to_estimate.parameter_vector - truth_parameters
formal_errors = estimation_output.formal_errors

correlations = estimation_output.correlations
covariances = estimation_output.covariance

# Print the covariance matrix
print('initial parameters perturbation', initial_parameters_perturbation)
print('true_errors', true_errors)
print('formal errors', formal_errors)


### Range-rate over time

observation_times = np.array(simulated_observations.concatenated_times)
observations_list = np.array(simulated_observations.concatenated_observations)

plt.figure(figsize=(9, 5))
plt.title("Observations as a function of time")
plt.scatter(observation_times / 3600.0, observations_list )

plt.xlabel("Time [hr]")
plt.ylabel("Range rate [m/s]")
plt.grid()

plt.tight_layout()
plt.show()


### Residuals history
residual_history = estimation_output.residual_history

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
subplots_list = [ax1, ax2, ax3, ax4]

for i in range(4):
    subplots_list[i].scatter(observation_times, residual_history[:, i])
    subplots_list[i].set_ylabel("Observation Residual [m/s]")
    subplots_list[i].set_title("Iteration "+str(i+1))

ax3.set_xlabel("Time since J2000 [s]")
ax4.set_xlabel("Time since J2000 [s]")

plt.tight_layout()
plt.grid()
plt.show()


### Final residuals

print('True-to-formal-error ratio:')
print('\nInitial state')
print(((truth_parameters - parameters_to_estimate.parameter_vector) / estimation_output.formal_errors)[:6])
print('\nPhysical parameters')
print(((truth_parameters - parameters_to_estimate.parameter_vector) / estimation_output.formal_errors)[6:])

final_residuals = estimation_output.final_residuals

plt.figure(figsize=(9,5))
plt.hist(final_residuals, 25)
plt.xlabel('Final iteration range-rate residual [m/s]')
plt.ylabel('Occurrences [-]')
plt.title('Histogram of residuals on final iteration')
plt.tight_layout()
plt.show()
