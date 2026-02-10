###############################################################
### ASSIGNMENT 2 : STATE ESTIMATION 
###############################################################

# In this assignment you are learning to estimate state from the acquired range-rate data.
# You will capture the data and look at different arc settings and other estimation settings.

#     - Propagation of initial orbit from TLE 
#     - Setup your estimation: data selection, arc length, estimation parameters
#     - Perform estimation
#     - Inspect results
#     - Validate your results


### UNITS AND CONVENTIONS 
# All parameters are represented in SI units or otherwise stated.

### CODE USAGE
# In this course you are using actual tracking data from the DopTrack laboratory (https://doptrack.tudelft.nl) and use the Delft-based orbit determination software Tudat 
# (https://docs.tudat.space/en/stable/#) to perform orbit analysis.


### IMPORT STATEMENTS

# Load standard modules
import sys
sys.path.append("../")

import statistics
from matplotlib import pyplot as plt

# Import doptrack-estimate functions
from propagation_functions.environment import *
from propagation_functions.propagation import *
from estimation_functions.estimation import *
from estimation_functions.observations_data import *

# Import doptrack-estimate functions
from propagation_functions.environment import *
from propagation_functions.propagation import *
from estimation_functions.estimation import *
from estimation_functions.observations_data import *

from utility_functions.time import *
from utility_functions.tle import *
from utility_functions.data import extract_tar

# Load tudatpy modules
from tudatpy import constants
from tudatpy.astro import element_conversion, frame_conversion
from tudatpy.interface import spice
from tudatpy.dynamics import environment
from tudatpy.dynamics import parameters
from tudatpy.estimation import estimation_analysis

# Extract data
extract_tar("./metadata.tar.xz")
extract_tar("./data.tar.xz")

# Define import folders
metadata_folder = 'metadata/'
data_folder = 'data/'

### UPLOAD DATA

# Lets upload Doppler data files and strat setting up the least square fitting. Put here your data files (.csv) and the metadata files (.yml) you want 
# to use in the estimation. The meta files will be used to compute the initial orbit. Here, you can use the doptrack-data.tudelft.nl website to get 
# processed data from the Delft DopTrack tracking station.
# Go to the processed/tracking directory and select a satellite and year you want to use. Than, download the files you want to use for the assignment.
# Download data for one whole week.
# We have made a default data set in data and metadata directories for the Delfi-C3 satellite, but you are more than welcome to use different data for 
# the assignment (including your own satellite pass)


# Files to be uploaded
metadata = ['Delfi-C3_32789_202004011044.yml', 'Delfi-C3_32789_202004011219.yml',
            'Delfi-C3_32789_202004021953.yml', 'Delfi-C3_32789_202004022126.yml',
            'Delfi-C3_32789_202004031031.yml', 'Delfi-C3_32789_202004031947.yml',
            'Delfi-C3_32789_202004041200.yml',

            'Delfi-C3_32789_202004061012.yml', 'Delfi-C3_32789_202004062101.yml',
            'Delfi-C3_32789_202004072055.yml', 'Delfi-C3_32789_202004072230.yml',
            'Delfi-C3_32789_202004081135.yml']

data = ['Delfi-C3_32789_202004011044.csv', 'Delfi-C3_32789_202004011219.csv',
        'Delfi-C3_32789_202004021953.csv', 'Delfi-C3_32789_202004022126.csv',
        'Delfi-C3_32789_202004031031.csv', 'Delfi-C3_32789_202004031947.csv',
        'Delfi-C3_32789_202004041200.csv',

        'Delfi-C3_32789_202004061012.csv', 'Delfi-C3_32789_202004062101.csv', 
        'Delfi-C3_32789_202004072055.csv', 'Delfi-C3_32789_202004072230.csv',
        'Delfi-C3_32789_202004081135.csv']
        
# Specify which metadata and data files should be loaded (this will change throughout the assignment)
# indices_files_to_load = [0, 1]
indices_files_to_load = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]


### SETTING UP AN INITIAL ORBIT DETERMINATION

# For the least square estimator, a first initial guess needs to be performed. Especially for low-quality data and bad geometry, this initial guess 
# should be close to the actual orbit to have the estimation converge. For this, we use the TLE data as this is already something we have and know 
# is close to the actual orbit of the satellite. In Assignment 3, you are going to look at the effect of the error on the initial guess and how it 
# affects the estimation.
# For now, try to set the propagation time large enough that it compasses your complete dataset. This can be done by setting the parameter propagation_time 
# to a number of days that you want to propagate the initial orbit.


# Retrieve initial epoch from TLE
initial_epoch, initial_state_teme, b_star = get_tle_initial_conditions(metadata_folder + metadata[0], old_yml=False)

# Define the propagation time, and compute the final and mid-propagation epochs accordingly.
propagation_time = 10.0 * constants.JULIAN_DAY
final_epoch = get_start_next_day(initial_epoch) + propagation_time
mid_epoch = (initial_epoch + final_epoch) / 2.0

# Retrieve the spacecraft's initial state at mid-epoch from the TLE orbit
initial_state = propagate_sgp4(metadata_folder + metadata[0], initial_epoch, [mid_epoch], old_yml=False)[0, 1:]

# Retrieve recording starting times
recording_start_times = extract_recording_start_times_yml(metadata_folder, [metadata[i] for i in indices_files_to_load], old_yml=False)

# Load and process observations
passes_start_times, passes_end_times, observation_times, observations_set = load_and_format_observations(
    "Delfi", data_folder, [data[i] for i in indices_files_to_load], recording_start_times, old_obs_format=False)


### SETTING YOUR ESTIMATION ARCS

# Here, you need to specify what type of arcs the data is cut into. So, are you calculating a new state after every pass, every day, 3 days or even 
# a week. Here, a trade-off needs to make between the amount of collected data and the unmodelled disturbance forces during the arc. 
# In the assignment, you are going to play with this setting to see what effect it has on the estimation.

# Define tracking arcs and retrieve the corresponding arc starting times (this will change throughout the assignment)
# Four options: one arc per pass ('per_pass'), one arc per day ('per_day'), one arc every 3 days ('per_3_days') and one arc per week ('per_week')
arc_start_times, arc_mid_times, arc_end_times = define_arcs('per_day', passes_start_times, passes_end_times)
print('arc_start_times', arc_start_times)
print('arc_end_times', arc_end_times)

### SETTING THE ESTIMATION SETTINGS 

# Now your initial guess is generated and you selected the estimation arcs, lets take a look at the environment of the satellite 
# (forces acting on the s/c)

# Define propagation_functions environment
mass = 2.2
ref_area = (4 * 0.3 * 0.1 + 2 * 0.1 * 0.1) / 4  # Average projection area of a 3U CubeSat
srp_coef = 1.2
drag_coef = 1.2
bodies = define_environment(mass, ref_area, drag_coef, srp_coef, "Delfi", multi_arc_ephemeris=False)

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

# Propagate dynamics and retrieve Delfi's initial state at the start of each arc
orbit = propagate_initial_state(initial_state, initial_epoch, final_epoch, bodies, accelerations, "Delfi")
arc_wise_initial_states = get_initial_states(bodies, arc_mid_times, "Delfi")


# Redefine environment to allow for multi-arc dynamics propagation_functions
bodies = define_environment(mass, ref_area, drag_coef, srp_coef, "Delfi", multi_arc_ephemeris=True)

# Define multi-arc propagator settings
multi_arc_propagator_settings = define_multi_arc_propagation_settings(arc_wise_initial_states, arc_start_times, arc_end_times,
                                                                      bodies, accelerations, "Delfi")
# Create the DopTrack station
define_doptrack_station(bodies)

# Define default observation settings
# Specify on which time interval the observation bias(es) should be defined. This will change throughout the assignment (can be 'per_pass', 'per_arc', 'global')
# Noting that the arc duration can vary (see arc definition)
bias_definition = 'per_pass'
Doppler_models = dict(
    constant_absolute_bias={
        'activated': True,
        'time_interval': bias_definition
    },
    linear_absolute_bias={
        'activated': True,
        'time_interval': bias_definition
    }
)
observation_settings = define_observation_settings("Delfi", Doppler_models, passes_start_times, arc_start_times)

# Define parameters to estimate
parameters_list = dict(
    initial_state={
        'estimate': True
    },
    constant_absolute_bias={
        'estimate': True
    },
    linear_absolute_bias={
        'estimate': True
    }
)
parameters_to_estimate = define_parameters(parameters_list, bodies, multi_arc_propagator_settings, "Delfi",
                                           arc_start_times, arc_mid_times, [(get_link_ends_id("DopTrackStation", "Delfi"), passes_start_times)], Doppler_models)
parameters.print_parameter_names(parameters_to_estimate)

# Create the estimator object
estimator = estimation_analysis.Estimator(bodies, parameters_to_estimate, observation_settings, multi_arc_propagator_settings)

# Simulate (ideal) observations
ideal_observations = simulate_observations_from_estimator("Delfi", observation_times, estimator, bodies)


### RUN THE ESTIMATION

# Now you are all setup to run the estimation. In the following block the dynamic equations are set and the estimator knows what kind of parameters need to be estimated.
# This can take a while, depending on your amount of data and settings

# Save the true parameters to later analyse the error
truth_parameters = parameters_to_estimate.parameter_vector
nb_parameters = len(truth_parameters)

# Perform estimation_functions
nb_iterations = 10
nb_arcs = len(arc_start_times)
pod_output = run_estimation(estimator, parameters_to_estimate, observations_set, nb_arcs, nb_iterations)

errors = pod_output.formal_errors
residuals = pod_output.residual_history
mean_residuals = statistics.mean(residuals[:,nb_iterations-1])
std_residuals = statistics.stdev(residuals[:,nb_iterations-1])

residuals_per_pass = get_residuals_per_pass(observation_times, residuals, passes_start_times)

print('--------------------------------------------------------------')
for i in range(len(residuals_per_pass)):
    print('size residuals current pass', np.shape(residuals_per_pass[i]))  


### INSPECT THE RESULTS 

# The first number that we look at is final residual. This shows the difference (root mean square) between the observed range-rate and the final orbit model 
# estimated by your program.

# Plot residuals

number_of_passes = len(indices_files_to_load)

fig, axs = plt.subplots(math.ceil(number_of_passes / 3),3, figsize=(12, 8))
for i in range(len(passes_start_times)):
    axs[i//3,i%3].plot(residuals_per_pass[i], color='blue', linestyle='-.')
    axs[i//3,i%3].set_xlabel('Time [s]')
    axs[i//3,i%3].set_ylabel('Residuals [m/s]')
    axs[i//3,i%3].set_title(f'Pass '+str(i+1))
    axs[i//3,i%3].grid()
fig.tight_layout()
plt.show()

# Plot residuals histogram
fig = plt.figure()
ax = fig.add_subplot()
plt.hist(residuals[:,nb_iterations-1],100)
ax.set_xlabel('Doppler residuals [m/s]')
ax.set_ylabel('Nb occurrences []')
plt.grid()
plt.show()


### ORBIT VALIDATION: some comparison suggestions
updated_parameters = parameters_to_estimate.parameter_vector
gravitational_parameter = bodies.get("Earth").gravity_field_model.gravitational_parameter
print('ALL ESTIMATED PARAMETERS')
print(updated_parameters)

for arc in range(nb_arcs):
    print('-------------ARC #', str(arc+1), '---------------')

    print('INITIAL STATE from TLE')
    print(arc_wise_initial_states[arc])
    print('UPDATED STATE from DOPTRACK')
    print(updated_parameters[arc*6:(arc+1)*6])

    # Distance between the two orbits
    pos_error = np.sqrt((updated_parameters[arc*6+0]-arc_wise_initial_states[arc][0])**2+(updated_parameters[arc*6+1]-arc_wise_initial_states[arc][1])**2+(updated_parameters[arc*6+2]-arc_wise_initial_states[arc][2])**2)
    print('Distance [km] between TLE initial state and estimated state: ', pos_error/1000)

    state_keplerian = element_conversion.cartesian_to_keplerian(updated_parameters[arc*6:(arc+1)*6], gravitational_parameter)

    print('-------------Estimated state---------------')
    print('Semi-major axis = \t\t\t',state_keplerian[0]/1000, '\t km')
    print('Eccentricity = \t\t\t\t',state_keplerian[1])
    print('Inclination = \t\t\t\t',np.rad2deg(state_keplerian[2]), '\t deg')
    print('Argument of Perigee = \t\t\t',np.rad2deg(state_keplerian[3]), '\t deg')
    print('Right Ascension of Ascending Node = \t',np.rad2deg(state_keplerian[4]), '\t deg')
    print('True anomaly = \t\t\t\t',np.rad2deg(state_keplerian[5]), '\t deg')
    print('True longitude = \t\t\t',np.mod(np.rad2deg(state_keplerian[5])+np.rad2deg(state_keplerian[3]),360), '\t deg')
    print('Altitude = \t\t\t\t',state_keplerian[0]/1000-6371.360, '\t km')

    TLE_keplerian = element_conversion.cartesian_to_keplerian(arc_wise_initial_states[arc], gravitational_parameter)

    print('---------------TLE state----------------')
    print('Semi-major axis = \t\t\t',TLE_keplerian[0]/1000, '\t km')
    print('Eccentricity = \t\t\t\t',TLE_keplerian[1])
    print('Inclination = \t\t\t\t',np.rad2deg(TLE_keplerian[2]), '\t deg')
    print('Argument of Perigee = \t\t\t',np.rad2deg(TLE_keplerian[3]), '\t deg')
    print('Right Ascension of Ascending Node = \t',np.rad2deg(TLE_keplerian[4]), '\t deg')
    print('True anomaly = \t\t\t\t',np.rad2deg(TLE_keplerian[5]), '\t deg')
    print('True longitude = \t\t\t',np.mod(np.rad2deg(TLE_keplerian[5])+np.rad2deg(TLE_keplerian[3]),360), '\t deg')
    print('Altitude = \t\t\t\t',TLE_keplerian[0]/1000-6371.360, '\t km')

    

print('----------------------------------------')
print('BIASES ESTIMATES')
print('ABSOLUTE CONSTANT BIASES ESTIMATES')
print(updated_parameters[6*nb_arcs:6*nb_arcs+number_of_passes])
print('LINEAR CONSTANT BIASES ESTIMATES')
print(updated_parameters[6*nb_arcs+number_of_passes+1:6*nb_arcs+number_of_passes*2])


# Comparing estimated vs TLE orbit. First redefine the dynamical environment (multi-arc ephemeris disabled) 
bodies = define_environment(mass, ref_area, drag_coef, srp_coef, "Delfi",multi_arc_ephemeris=False)

# Specify which the index of the arc you want to investigate further
arc_index = 0

# Retrieve estimated and TLE states for the arc under consideration
estimated_state = updated_parameters[6*arc_index:(arc_index+1)*6]
TLE_state = arc_wise_initial_states[arc_index]

# Propagate estimated and TLE orbits
estimated_orbit = propagate_initial_state(estimated_state, arc_start_times[arc_index], arc_end_times[arc_index], bodies, accelerations, "Delfi")[0]
TLE_orbit = propagate_initial_state(TLE_state, arc_start_times[arc_index], arc_end_times[arc_index], bodies, accelerations, "Delfi")[0]



# Plot differences between the TLE and estimated orbits

fig = plt.figure(figsize=(10, 8)) 
ax = fig.add_subplot(3, 2, 1)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],(TLE_orbit[:,1]-estimated_orbit[:,1])/1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Diff X [km]')
ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

ax = fig.add_subplot(3, 2, 3)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],(TLE_orbit[:,2]-estimated_orbit[:,2])/1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Diff Y [km]')
# ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

ax = fig.add_subplot(3, 2, 5)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],(TLE_orbit[:,3]-estimated_orbit[:,3])/1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Diff Z [km]')
# ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

ax = fig.add_subplot(3, 2, 2)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],(TLE_orbit[:,4]-estimated_orbit[:,4])/1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Diff VX [km/s]')
# ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

ax = fig.add_subplot(3, 2, 4)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],(TLE_orbit[:,5]-estimated_orbit[:,5])/1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Diff VY [km/s]')
# ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

ax = fig.add_subplot(3, 2, 6)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],(TLE_orbit[:,6]-estimated_orbit[:,6])/1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Diff VZ [km/s]')
# ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

fig.tight_layout()
plt.show()


# Plot propagated (estimated and TLE) orbits
fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Satellite trajectory around Earth')
ax.plot(TLE_orbit[:, 1], TLE_orbit[:, 2], TLE_orbit[:, 3], label='TLE orbit', linestyle='-.')
ax.plot(estimated_orbit[:, 1], estimated_orbit[:, 2], estimated_orbit[:, 3], label='estimated orbit', linestyle='-.')
ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue')
ax.legend()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
plt.show()

# Compute distance and velocity magnitude for both TLE and estimated orbits
range_TLE = np.sqrt(TLE_orbit[:,1]**2+TLE_orbit[:,2]**2+TLE_orbit[:,3]**2)
range_estimated = np.sqrt(estimated_orbit[:,1]**2+estimated_orbit[:,2]**2+estimated_orbit[:,3]**2)

Vmag_TLE = np.sqrt(TLE_orbit[:,4]**2+TLE_orbit[:,5]**2+TLE_orbit[:,6]**2)
Vmag_estimated = np.sqrt(estimated_orbit[:,4]**2+estimated_orbit[:,5]**2+estimated_orbit[:,6]**2)

# Plot difference in distance and velocity magnitude between TLE and estimated orbits
fig = plt.figure() # plt.figure(figsize=(10,2*5.0), dpi=125)

ax = fig.add_subplot(2, 1, 1)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],(range_TLE-range_estimated)/1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Residuals range [km]')
ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

ax = fig.add_subplot(2, 1, 2)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],(Vmag_TLE-Vmag_estimated)/1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Residuals Vmag [km/s]')
plt.grid()
fig.tight_layout()
plt.show()


# Compute distance between the TLE and estimated orbits
distance_orbits = np.sqrt((TLE_orbit[:,1]-estimated_orbit[:,1])**2+(TLE_orbit[:,2]-estimated_orbit[:,2])**2+(TLE_orbit[:,3]-estimated_orbit[:,3])**2)

fig = plt.figure() 
fig.tight_layout()
fig.subplots_adjust(hspace=0.3)
ax = fig.add_subplot(1, 1, 1)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],(distance_orbits)/1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Distance between orbits [km]')
ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()
plt.show()


# Compute difference in RSW and keplerian coordinates
rsw_difference_wrt_tle = np.zeros((len(TLE_orbit[:,0]),7))
keplerian_difference_wrt_tle = np.zeros((len(TLE_orbit[:,0]),7))
for i in range(len(TLE_orbit[:,0])): 

    current_epoch = TLE_orbit[i,0]
    rsw_difference_wrt_tle[i,0] = current_epoch
    keplerian_difference_wrt_tle[i,0] = current_epoch

    # Retrieve current TLE and estimated states
    current_tle_state = TLE_orbit[i,1:]
    current_estimated_state = estimated_orbit[i,1:]

    # Compute Keplerian elements from TLE and estimated orbits
    current_tle_keplerian = element_conversion.cartesian_to_keplerian(current_tle_state, bodies.get("Earth").gravitational_parameter)
    current_estimated_keplerian = element_conversion.cartesian_to_keplerian(current_estimated_state, bodies.get("Earth").gravitational_parameter)
    keplerian_difference_wrt_tle[i, 1:7] = current_estimated_keplerian - current_tle_keplerian

    # Compute difference in the inertial frame
    current_state_difference = current_estimated_state - current_tle_state
    current_position_difference = current_state_difference[0:3]
    current_velocity_difference = current_state_difference[3:6]

    # Compute the rotation matrix from inertial to RSW frames
    rotation_to_rsw = frame_conversion.inertial_to_rsw_rotation_matrix(current_tle_state)

    # Convert the state difference from inertial to RSW frames
    rsw_difference_wrt_tle[i, 1:4] = rotation_to_rsw @ current_position_difference
    rsw_difference_wrt_tle[i, 4:7] = rotation_to_rsw @ current_velocity_difference


# Plot differences between the TLE and estimated orbits in RSW

fig = plt.figure(figsize=(10, 8)) 
ax = fig.add_subplot(3, 2, 1)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],rsw_difference_wrt_tle[:,1]/1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Diff R [km]')
ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

ax = fig.add_subplot(3, 2, 3)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],rsw_difference_wrt_tle[:,2]/1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Diff S [km]')
# ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

ax = fig.add_subplot(3, 2, 5)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],rsw_difference_wrt_tle[:,3]/1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Diff W [km]')
# ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

ax = fig.add_subplot(3, 2, 2)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],rsw_difference_wrt_tle[:,4]/1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Diff Vr [km/s]')
# ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

ax = fig.add_subplot(3, 2, 4)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],rsw_difference_wrt_tle[:,5]/1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Diff Vs [km/s]')
# ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

ax = fig.add_subplot(3, 2, 6)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],rsw_difference_wrt_tle[:,6]/1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Diff Vw [km/s]')
# ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

fig.tight_layout()
plt.show()


# Plot differences between the TLE and estimated orbits in Keplerian elements

fig = plt.figure(figsize=(10, 8)) 
ax = fig.add_subplot(3, 2, 1)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],keplerian_difference_wrt_tle[:,1]/1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$\Delta a$ [km]')
ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

ax = fig.add_subplot(3, 2, 3)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],keplerian_difference_wrt_tle[:,2], color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$\Delta e$ [-]')
# ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

ax = fig.add_subplot(3, 2, 5)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],np.degrees(keplerian_difference_wrt_tle[:,3]), color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$\Delta i$ [deg]')
# ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

ax = fig.add_subplot(3, 2, 2)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],np.degrees(keplerian_difference_wrt_tle[:,4]), color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$\Delta \omega$ [deg]')
# ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

ax = fig.add_subplot(3, 2, 4)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],np.degrees(keplerian_difference_wrt_tle[:,5]), color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$\Delta \Omega$ [deg]')
# ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

ax = fig.add_subplot(3, 2, 6)
ax.plot(TLE_orbit[:,0]-TLE_orbit[0,0],np.degrees(keplerian_difference_wrt_tle[:,6]), color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$\Delta \theta$ [deg]')
# ax.set_title(f'Pass '+str(arc_index+1))
plt.grid()

fig.tight_layout()
plt.show()