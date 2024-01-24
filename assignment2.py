# Load standard modules
import statistics
from matplotlib import pyplot as plt

# Import doptrack-estimate functions
from propagation_functions.environment import *
from propagation_functions.propagation import *
from estimation_functions.estimation import *
from estimation_functions.observations_data import *

from utility_functions.time import *
from utility_functions.tle import *
from utility_functions.data import extract_tar
from fit_sgp4_solution import fit_sgp4_solution

# Load tudatpy modules
from tudatpy.numerical_simulation import environment
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import estimation_setup
from tudatpy.kernel.astro import element_conversion

# Extract data
extract_tar("./metadata.tar.xz")
extract_tar("./data.tar.xz")

# Define import folders
metadata_folder = 'metadata/'
data_folder = 'data/'

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
indices_files_to_load = [0, 1]
# indices_files_to_load = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


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

# Define tracking arcs and retrieve the corresponding arc starting times (this will change throughout the assignment)
# Four options: one arc per pass ('per_pass'), one arc per day ('per_day'), one arc every 3 days ('per_3_days') and one arc per week ('per_week')
arc_start_times, arc_mid_times, arc_end_times = define_arcs('per_day', passes_start_times, passes_end_times)

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
    }
)
observation_settings = define_observation_settings("Delfi", Doppler_models, passes_start_times, arc_start_times)

# Define parameters to estimate
parameters_list = dict(
    initial_state={
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
    }
)
parameters_to_estimate = define_parameters(parameters_list, bodies, multi_arc_propagator_settings, "Delfi",
                                           arc_start_times, arc_mid_times, [(get_link_ends_id("DopTrackStation", "Delfi"), passes_start_times)], Doppler_models)
estimation_setup.print_parameter_names(parameters_to_estimate)

# Create the estimator object
estimator = numerical_simulation.Estimator(bodies, parameters_to_estimate, observation_settings, multi_arc_propagator_settings)

# Simulate (ideal) observations
ideal_observations = simulate_observations_from_estimator("Delfi", observation_times, estimator, bodies)


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
