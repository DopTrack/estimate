import math
import sys
sys.path.insert(0, 'tudat-bundle/cmake-build-release/tudatpy')

# Load standard modules
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression # linear regression module

from propagation_functions.environment import *
from propagation_functions.propagation import *
from utility_functions.time import *
from utility_functions.tle import *
from estimation_functions.observations_data import *
from estimation_functions.estimation import *

# Load tudatpy modules
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.numerical_simulation.estimation_setup import observation
from tudatpy.kernel.numerical_simulation import estimation_setup, estimation



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
# indices_files_to_load = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 18, 20]
indices_files_to_load = [0, 1, 2,
                         3, 4, 6]#,
                         # 7, 8,
                         # 10, 12, 13,
                         # 15, 17, 20,
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
nb_days_to_propagate = 2
final_epoch = start_recording_day + nb_days_to_propagate * 86400.0

print('initial_epoch', initial_epoch)
print('final_epoch', final_epoch)

# Load and process observations
passes_start_times, passes_end_times, obs_times, obs_values = load_existing_observations(data_folder, data, indices_files_to_load, metadata, new_obs_format=True)

# Define tracking arcs and retrieve the corresponding arc starting times (this will change throughout the assignment)
# Four options: one arc per pass ('per_pass'), one arc per day ('per_day'), one arc every 3 days ('per_3_days') and one arc per week ('per_week')
arc_start_times, arc_end_times = define_arcs('per_3_days', passes_start_times, passes_end_times)

print('arc_start_times', arc_start_times)
print('arc_end_times', arc_end_times)

print('passes_start_times', passes_start_times)
print('passes_end_times', passes_end_times)


# Define the propagation environment. This function creates a body "Delfi" with the following characteristics.
# The Earth, Sun and Moon are also created, with default settings (gravity field, ephemeris, rotation, etc.)
mass_delfi = 2.2
ref_area_delfi = 0.035
drag_coefficient_delfi = get_drag_coefficient(mass_delfi, ref_area_delfi, b_star_coef, from_tle=True)
srp_coefficient_delfi = 1.2
bodies = define_environment(mass_delfi, ref_area_delfi, drag_coefficient_delfi, srp_coefficient_delfi)

# Set Delfi's initial state to the TLE prediction
initial_state = element_conversion.teme_state_to_j2000(initial_epoch, initial_state_teme)

# Define accelerations exerted on Delfi
# The following can be modified. Warning: point_mass_gravity and spherical_harmonic_gravity accelerations should not be defined simultaneously for a single body
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
accelerations, accelerations_to_save, accelerations_ids = create_accelerations(acceleration_models, bodies, save_accelerations=True)

# Create propagator settings
propagator_settings = create_propagator_settings(initial_state, initial_epoch, final_epoch, accelerations)

# Propagate dynamics of the Delfi satellite from initial_epoch to final_epoch, starting from initial_state
# The propagation output is given in cartesian and keplerian states, and the latitude/longitude of the spacecraft are also saved.
cartesian_states, keplerian_states, latitudes, longitudes, saved_accelerations =\
    propagate_initial_state(initial_state, initial_epoch, final_epoch, bodies, accelerations, True, accelerations_to_save)


# Create one extra station
station1 = "FakeStation1"
coordinates_station1 = np.array([0.0, np.deg2rad(-33.8837), np.deg2rad(151.2007)])
define_station(bodies, station1, coordinates_station1)

# Create a second extra station
station2 = "FakeStation2"
coordinates_station2 = np.array([0.0, np.deg2rad(-33.8837), np.deg2rad(10.0)])
define_station(bodies, station2, coordinates_station2)

stations = [station1, station2]

# Define all downlink link ends for Doppler
link_ends, link_ends_def = define_all_link_ends(stations)

# Define observation settings
observation_settings = define_ideal_doppler_settings(stations)

# Create list of observation times, with one Doppler measurement every 10 seconds
possible_obs_times = []
obs_time_step = 10.0
current_time = start_recording_day
while current_time < final_epoch:
    possible_obs_times.append(current_time)
    current_time = current_time + obs_time_step


integrator_settings = create_integrator_settings(initial_epoch)
estimator = create_dummy_estimator(bodies, propagator_settings, integrator_settings, observation_settings)

# Simulate observations
simulated_observations = simulate_ideal_simulations(estimator, bodies, link_ends_def, possible_obs_times, stations, 0.0)

simulated_obs_times = np.array(simulated_observations.concatenated_times)
simulated_doppler = simulated_observations.concatenated_observations

simulated_passes_start_times, simulated_passes_end_times, obs_times_per_pass, obs_values_per_pass = \
    get_obs_per_link_end_and_pass(stations, simulated_obs_times, simulated_doppler, obs_time_step)


# print('simulated_passes_start_times', simulated_passes_start_times)
# print('nb passes', len(simulated_passes_start_times))
#
# np.savetxt('simulated_nayif_data/obs_times.txt', simulated_obs_times)
# np.savetxt('simulated_nayif_data/obs_values.txt', simulated_doppler)
#
# for k in range(len(obs_times_per_pass)):
#     np.savetxt('simulated_nayif_data/obs_times_pass' + str(k) + '.txt', obs_times_per_pass[k])
#     np.savetxt('simulated_nayif_data/obs_values_pass' + str(k) + '.txt', obs_values_per_pass[k])

# Plot simulated Doppler data
fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot()
ax.set_title(f'Doppler')
ax.plot((simulated_obs_times - start_recording_day)/3600, simulated_doppler, label='simulated', color='red', linestyle='none', marker='.')
ax.legend()
ax.set_xlabel('Time [hours since start of day]')
ax.set_ylabel('Doppler [m/s]')
plt.grid()
plt.show()


