# Load standard modules
import statistics
import numpy as np
from matplotlib import pyplot as plt

from environment import *
from propagation import *
from estimation import *
from util_functions import *
from observations_data import *

# Load tudatpy modules
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.astro import element_conversion

# Define import folder
metadata_folder = 'metadata/'

j2000_days = 2451545.0

# For first tutorial, probably directly provide initial time and initial state? or let the students retrieve this information manually
# from previous passes' yml file?
julian_date, initial_state_teme = get_tle_initial_conditions(metadata_folder + 'Delfi-C3_32789_202004020904.yml')

initial_epoch = (julian_date - j2000_days) * 86400.0
start_recording_day = get_start_next_day(initial_epoch, j2000_days)

nb_days_to_propagate = 1
final_epoch = start_recording_day + nb_days_to_propagate * 86400.0

print('initial epoch: ', initial_epoch)
print('initial state TEME: ', initial_state_teme)
print('final epoch', final_epoch)


# 1/ Propagate dynamics of Delfi

# Load spice kernels
spice.load_standard_kernels()

# Define propagation environment
mass_delfi = 2.2
reference_area_delfi = 0.035
drag_coefficient_delfi = 1.4
srp_coefficient_delfi = 2.4
bodies = define_system_bodies(mass_delfi, reference_area_delfi, drag_coefficient_delfi, srp_coefficient_delfi)

# Set the initial state of Delfi
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

# Create propagator settings
propagator_settings = create_propagator_settings(initial_state, final_epoch, accelerations)

# Propagate dynamics
propagated_orbit = propagate_initial_state(initial_state, initial_epoch, final_epoch, bodies, accelerations)

# Plot propagated orbit
fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Delfi-C3 trajectory around Earth')
ax.plot(propagated_orbit[:, 1], propagated_orbit[:, 2], propagated_orbit[:, 3], label='Delfi-C3', linestyle='-.')
ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue')
ax.legend()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
plt.show()


# 2/ Detect passes and simulate Doppler

# Create the DopTrack station
define_doptrack_station(bodies)

# Define observation settings
observation_settings = define_ideal_doppler_settings()

# The actual simulation of the observations requires Observation Simulators, which are created automatically by the Estimator object.
# Therefore, the observations cannot be simulated before the creation of an Estimator object.
dummy_estimator = create_dummy_estimator(bodies, propagator_settings, integrator_settings, observation_settings)

# observation times
possible_obs_times = []
obs_time_step = 10.0
current_time = start_recording_day
while current_time < final_epoch:
    possible_obs_times.append(current_time)
    current_time = current_time + obs_time_step

# Simulate (ideal) observations
simulated_observations = simulate_ideal_observations(possible_obs_times, dummy_estimator, bodies, 5)

simulated_obs_times = np.array(simulated_observations.concatenated_times)
simulated_doppler = simulated_observations.concatenated_observations

fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot()
ax.set_title(f'Doppler')
ax.plot((simulated_obs_times - start_recording_day)/3600, - simulated_doppler * constants.SPEED_OF_LIGHT, label='simulated', color='red', linestyle='none', marker='.')
ax.legend()
ax.set_xlabel('Time [hours since start of day]')
ax.set_ylabel('Doppler [m/s]')
plt.grid()
plt.show()


# 3/ Load real observations

# Load real observations
data_folder = 'data/'

# Files to be uploaded
data = ['Delfi-C3_32789_202004020904.DOP1C', 'Delfi-C3_32789_202004021953.DOP1C']
passes_start_times, observation_times, observations_set = load_and_format_observations(data_folder, data)

# Retrieve observation times and Doppler values
real_doppler = observations_set.concatenated_observations


# Plot simulated vs. real Doppler
fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot()
ax.set_title(f'Doppler')
ax.plot((np.array(simulated_obs_times) - start_recording_day)/3600, convert_frequencies_to_range_rate(simulated_doppler), label='simulated', color='red', linestyle='none', marker='.')
ax.plot((np.array(observation_times) - start_recording_day)/3600, convert_frequencies_to_range_rate(real_doppler), label='recorded', color='blue', linestyle='none', marker='.')
ax.legend()
ax.set_xlabel('Time [hours since start of day]')
ax.set_ylabel('Doppler [m/s]')
plt.grid()
plt.show()