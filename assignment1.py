# Load standard modules
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression # linear regression module

from propagation_functions.environment import *
from propagation_functions.propagation import *
from utility_functions.time import *
from utility_functions.tle import *
from utility_functions.data import extract_tar
from estimation_functions.observations_data import *
from estimation_functions.estimation import *

# Load tudatpy modules
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.astro import element_conversion

# Extract data
extract_tar("./metadata.tar.xz")
extract_tar("./data.tar.xz")

# Define import folders
metadata_folder = 'metadata/'
data_folder = 'data/'

# Metadata input file
metadata_file = metadata_folder + 'Delfi-C3_32789_202004020904.yml'

# Retrieve initial state from TLE and associated time
initial_epoch, initial_state_teme, b_star_coef = get_tle_initial_conditions(metadata_file)

# Define start of the day (initial time of the propagation)
start_recording_day = get_start_next_day(initial_epoch)

# Define end of the day (final time of the propagation)
nb_days_to_propagate = 1
final_epoch = start_recording_day + nb_days_to_propagate * 86400.0

print('initial epoch: ', initial_epoch)
print('initial state TEME: ', initial_state_teme)
print('final epoch', final_epoch)


# --------------------------------------
# 1/ Propagate dynamics of Delfi
# --------------------------------------

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

# Plot propagated orbit
fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Delfi-C3 trajectory around Earth')
ax.plot(cartesian_states[:, 1], cartesian_states[:, 2], cartesian_states[:, 3], label='Delfi-C3', linestyle='-.')
ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue')
ax.legend()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
plt.show()


# Plot accelerations exerted on Delfi
fig = plt.figure()
ax = fig.add_subplot()
ax.set_title(f'Accelerations on Delfi')
for i in range(np.shape(saved_accelerations)[1]-1):
    ax.plot((saved_accelerations[:, 0]-start_recording_day)/86400, saved_accelerations[:, i+1], label=accelerations_ids[i], linestyle='-')
ax.legend()
ax.set_xlabel('Time [Days since first recording day]')
ax.set_ylabel('Acceleration [m/s]')
plt.yscale('log')
plt.grid()
plt.show()


# --------------------------------------
# 2/ Detect passes and simulate Doppler
# --------------------------------------

# Create the DopTrack station
define_doptrack_station(bodies)

# Define observation settings
observation_settings = define_ideal_doppler_settings()

# Create list of observation times, with one Doppler measurement every 10 seconds
possible_obs_times = []
obs_time_step = 10.0
current_time = start_recording_day
while current_time < final_epoch:
    possible_obs_times.append(current_time)
    current_time = current_time + obs_time_step

# Simulate (ideal) observations
simulated_observations = simulate_observations(possible_obs_times, observation_settings, propagator_settings, bodies, initial_epoch, 0)

simulated_obs_times = np.array(simulated_observations.concatenated_times)
simulated_doppler = simulated_observations.concatenated_observations


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


# --------------------------------------
# 3/ Load real observations
# --------------------------------------

# Observation files to be uploaded
data = ['Delfi-C3_32789_202004020904.DOP1C', 'Delfi-C3_32789_202004021953.DOP1C']

# Process observations.
# This loads the recorded observations and retrieve the start of each tracking pass
passes_start_times, passes_end_times, observation_times, observations_set = load_and_format_observations(data_folder, data)

# Retrieve measured Doppler values
real_doppler = observations_set.concatenated_observations


# Plot simulated vs. real Doppler
fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot()
ax.set_title(f'Doppler')
ax.plot((np.array(simulated_obs_times) - start_recording_day)/3600, simulated_doppler, label='simulated', color='red', linestyle='none', marker='.')
ax.plot((np.array(observation_times) - start_recording_day)/3600, convert_frequencies_to_range_rate(real_doppler), label='recorded', color='blue', linestyle='none', marker='.')
ax.legend()
ax.set_xlabel('Time [hours since start of day]')
ax.set_ylabel('Doppler [m/s]')
plt.grid()
plt.show()


# --------------------------------------
# 4/ Compare simulated and recorded data for single pass
# --------------------------------------

# Index of the *recorded* pass of interest (warning: the number of recorded passes might differ from the number of simulated passes)
index_pass = 0
single_pass_start_time = passes_start_times[index_pass]
single_pass_end_time = passes_end_times[index_pass]

# Retrieve recorded Doppler data for single pass
real_obs_single_pass = get_observations_single_pass(single_pass_start_time, single_pass_end_time, observations_set)

# Retrieve simulated Doppler data for single pass
simulated_obs_single_pass = get_observations_single_pass(single_pass_start_time, single_pass_end_time, simulated_observations)

# Interpolate simulated and recorded observations to identical times
interpolated_simulated_obs, interpolated_real_obs = interpolate_obs(simulated_obs_single_pass, real_obs_single_pass)
interpolated_times = interpolated_simulated_obs[:,0]

# Compute first residual between recorded and simulated observations
first_residual_obs = interpolated_real_obs[:,1] - interpolated_simulated_obs[:,1]

# Perform linear regression on first residual
linear_fit = LinearRegression().fit(interpolated_times.reshape((-1, 1)), first_residual_obs)

# Retrieve fit model
fit = linear_fit.predict(np.linspace(interpolated_times[0], interpolated_times[len(interpolated_times)-1]).reshape((-1, 1)))

# Compute second residual after removing linear fit
second_residual_obs = first_residual_obs - linear_fit.predict(interpolated_times.reshape((-1, 1)))



# Plot single pass observations (both recorded and simulated, as well as first and second residuals)

fig = plt.figure()
fig.tight_layout()
fig.subplots_adjust(hspace=0.3)

ax1 = fig.add_subplot(2,2,1)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

ax1.plot((interpolated_times - start_recording_day)/3600, interpolated_real_obs[:,1], label='recorded', color='blue', linestyle='none', marker='.')
ax1.plot((interpolated_times - start_recording_day)/3600, interpolated_simulated_obs[:,1], label='simulated', color='red', linestyle='none', marker='.')
ax1.grid()
ax1.set_title(f'Doppler')
ax1.legend()
ax1.set_xlabel('Time [hours since start of day]')
ax1.set_ylabel('Doppler [m/s]')

ax3.plot((interpolated_times - start_recording_day)/3600, first_residual_obs, label='residual', color='green', linestyle='none', marker='.')
ax3.plot((np.linspace(interpolated_times[0], interpolated_times[len(interpolated_times)-1]) - start_recording_day)/3600, fit, label='linear fit', color='black', linestyle='-')
ax3.grid()
ax3.set_title(f'First residual (recorded - simulated)')
ax3.legend()
ax3.set_xlabel('Time [hours since start of day]')
ax3.set_ylabel('Residual [m/s]')

ax4.plot((interpolated_times - start_recording_day)/3600, second_residual_obs, label='residual', color='green', linestyle='none', marker='.')
ax4.grid()
ax4.set_title(f'Second residual (first residual - linear fit)')
ax4.legend()
ax4.set_xlabel('Time [hours since start of day]')
ax4.set_ylabel('Residual [m/s]')

plt.show()


