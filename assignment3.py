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
from estimation_functions.estimation import *


# Load spice kernels
spice.load_standard_kernels()

# Retrieve the initial state of Delfi-C3 using Two-Line-Elements (TLEs)
delfi_tle = environment.Tle(
    "1 32789U 08021G   20090.88491347 +.00001016 +00000-0 +70797-4 0  9997", #"1 32789U 07021G   08119.60740078 -.00000054  00000-0  00000+0 0  9999",
    "2 32789 097.4279 136.4027 0011143 218.6381 141.4051 15.07550601649972" #"2 32789 098.0082 179.6267 0015321 307.2977 051.0656 14.81417433    68"
)

# Set simulation start and end epochs
simulation_start_epoch = delfi_tle.get_epoch() #DateTime(2020, 4, 1).epoch()
simulation_end_epoch = simulation_start_epoch + 1.0 * 86400.0 #DateTime(2020, 4, 2).epoch()

# Create default body settings for "Sun", "Earth", "Moon", "Mars", and "Venus"
bodies_to_create = ["Sun", "Earth", "Moon", "Mars", "Venus"]

# Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation)

# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)


# Create vehicle objects.
bodies.create_empty_body("spacecraft")
bodies.get("spacecraft").mass = 2.2

# Create aerodynamic coefficient interface settings
reference_area = (4*0.3*0.1+2*0.1*0.1)/4  # Average projection area of a 3U CubeSat
drag_coefficient = 1.2
aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
    reference_area, [drag_coefficient, 0.0, 0.0]
)
# Add the aerodynamic interface to the environment
environment_setup.add_aerodynamic_coefficient_interface(bodies, "spacecraft", aero_coefficient_settings)

# Create radiation pressure settings
reference_area_radiation = 4.0
radiation_pressure_coefficient = 1.2
occulting_bodies_dict = dict()
occulting_bodies_dict["Sun"] = ["Earth"]
radiation_pressure_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    reference_area_radiation, radiation_pressure_coefficient, occulting_bodies_dict)

# Add the radiation pressure interface to the environment
environment_setup.add_radiation_pressure_target_model(bodies, "spacecraft", radiation_pressure_settings)

# Define bodies that are propagated
bodies_to_propagate = ["spacecraft"]

# Define central bodies of propagation
central_bodies = ["Earth"]


# Define the accelerations acting on Delfi-C3
accelerations_settings_delfi_c3 = dict(
    Sun=[
        propagation_setup.acceleration.radiation_pressure(),
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Mars=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Moon=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Earth=[
        propagation_setup.acceleration.spherical_harmonic_gravity(8, 8),
        propagation_setup.acceleration.aerodynamic()
    ])

# Create global accelerations dictionary
acceleration_settings = {"spacecraft": accelerations_settings_delfi_c3}

# Create acceleration models
acceleration_models = propagation_setup.create_acceleration_models(
    bodies,
    acceleration_settings,
    bodies_to_propagate,
    central_bodies)


delfi_ephemeris = environment.TleEphemeris("Earth", "J2000", delfi_tle, False)
initial_state = delfi_ephemeris.cartesian_state(simulation_start_epoch)

# Create numerical integrator settings
integrator_settings = propagation_setup.integrator.\
    runge_kutta_fixed_step_size(initial_time_step=60.0,
                                coefficient_set=propagation_setup.integrator.CoefficientSets.rkdp_87)

# Create termination settings
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

# Create propagation settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_state,
    simulation_start_epoch,
    integrator_settings,
    termination_condition
)


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
observation_times = np.arange(simulation_start_epoch, simulation_end_epoch, 10.0)

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




# Setup parameters settings to propagate the state transition matrix
parameter_settings = estimation_setup.parameter.initial_states(propagator_settings, bodies)

# Add estimated parameters to the sensitivity matrix that will be propagated
parameter_settings.append(estimation_setup.parameter.gravitational_parameter("Earth"))
parameter_settings.append(estimation_setup.parameter.constant_drag_coefficient("spacecraft"))

# Create the parameters that will be estimated
parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies)

# Create the estimator
estimator = numerical_simulation.Estimator(
    bodies,
    parameters_to_estimate,
    observation_settings_list,
    propagator_settings)

# Simulate required observations
simulated_observations = estimation.simulate_observations(
    observation_simulation_settings,
    estimator.observation_simulators,
    bodies)

# Save the true parameters to later analyse the error
truth_parameters = parameters_to_estimate.parameter_vector

# Perturb the initial state estimate from the truth (10 m in position; 0.1 m/s in velocity)
perturbed_parameters = truth_parameters.copy( )
for i in range(3):
    perturbed_parameters[i] += 1.0e2 #10.0
    perturbed_parameters[i+3] += 0.1 #0.01
parameters_to_estimate.parameter_vector = perturbed_parameters

initial_parameters_perturbation = perturbed_parameters - truth_parameters


# Create input object for the estimation
convergence_checker = estimation.estimation_convergence_checker(maximum_iterations=10)
estimation_input = estimation.EstimationInput(
    simulated_observations,
    convergence_checker=convergence_checker)

# Set methodological options
estimation_input.define_estimation_settings(
    reintegrate_variational_equations=False)

# Define weighting of the observations in the inversion
weights_per_observable = {estimation_setup.observation.one_way_instantaneous_doppler_type: noise_level ** -2}
estimation_input.set_constant_weight_per_observable(weights_per_observable)


# Perform the estimation
estimation_output = estimator.perform_estimation(estimation_input)

true_errors = parameters_to_estimate.parameter_vector - truth_parameters
formal_errors = estimation_output.formal_errors

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
plt.ylabel('Occurences [-]')
plt.title('Histogram of residuals on final iteration')
plt.tight_layout()
plt.show()
