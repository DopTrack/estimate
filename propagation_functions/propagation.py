# Load standard modules
import numpy as np

# Load tudatpy modules
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation.environment_setup import ephemeris
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.util import result2array


def create_accelerations(acceleration_models, bodies):
    # Define bodies that are propagated
    bodies_to_propagate = ["Delfi"]

    # Define central bodies of propagation_functions
    central_bodies = ["Earth"]

    # Define the accelerations acting on Delfi
    accelerations_due_to_sun = []
    if acceleration_models['Sun']["point_mass_gravity"]:
        accelerations_due_to_sun.append(propagation_setup.acceleration.point_mass_gravity())
    if acceleration_models['Sun']["solar_radiation_pressure"]:
        accelerations_due_to_sun.append(propagation_setup.acceleration.cannonball_radiation_pressure())

    accelerations_due_to_moon = []
    if acceleration_models['Moon']["point_mass_gravity"]:
        accelerations_due_to_moon.append(propagation_setup.acceleration.point_mass_gravity())

    accelerations_due_to_earth = []
    if acceleration_models['Earth']["point_mass_gravity"]:
        accelerations_due_to_earth.append(propagation_setup.acceleration.point_mass_gravity())
    if acceleration_models['Earth']["spherical_harmonic_gravity"]:
        accelerations_due_to_earth.append(propagation_setup.acceleration.spherical_harmonic_gravity(12, 12))
    if acceleration_models['Earth']["drag"]:
        accelerations_due_to_earth.append(propagation_setup.acceleration.aerodynamic())

    accelerations_settings_delfi = dict(
        Sun=accelerations_due_to_sun,
        Moon=accelerations_due_to_moon,
        Earth=accelerations_due_to_earth
    )

    # Create global accelerations dictionary
    acceleration_settings = {"Delfi": accelerations_settings_delfi}

    return propagation_setup.create_acceleration_models(bodies, acceleration_settings, bodies_to_propagate,
                                                        central_bodies)


def create_integrator_settings(initial_time, time_step: float = 10.0):
    return propagation_setup.integrator.runge_kutta_4(initial_time, time_step)


def create_propagator_settings(initial_state, initial_time, final_time, accelerations):

    # Define bodies that are propagated
    bodies_to_propagate = ["Delfi"]

    # Define central bodies of propagation_functions
    central_bodies = ["Earth"]

    # Create termination settings
    termination_condition = propagation_setup.propagator.time_termination(final_time)

    # Define integrator settings
    integrator_settings = create_integrator_settings(initial_time)

    # Define dependent variables
    dependent_variables = [
        propagation_setup.dependent_variable.keplerian_state("Delfi", "Earth"),
        propagation_setup.dependent_variable.latitude("Delfi", "Earth"),
        propagation_setup.dependent_variable.longitude("Delfi", "Earth")
    ]

    return propagation_setup.propagator.translational(
        central_bodies, accelerations, bodies_to_propagate, initial_state, initial_time, integrator_settings, termination_condition, output_variables=dependent_variables)


def propagate_initial_state(initial_state, initial_time, final_time, bodies, accelerations):

    # Create numerical integrator settings
    integrator_settings = create_integrator_settings(initial_time)

    # Create propagator settings
    single_arc_propagator_settings = create_propagator_settings(initial_state, initial_time, final_time, accelerations)

    # Propagate dynamics
    simulator = numerical_simulation.create_dynamics_simulator(bodies, single_arc_propagator_settings, 1)

    cartesian_states = result2array(simulator.state_history)
    dependent_variables = result2array(simulator.dependent_variable_history)
    keplerian_states = dependent_variables[:, 0:7]
    latitudes = dependent_variables[:, [0, 7]]
    longitudes = dependent_variables[:, [0, 8]]

    return cartesian_states, keplerian_states, latitudes, longitudes


def get_initial_states(bodies, arc_start_times):
    arc_initial_states = []
    for i in range(len(arc_start_times)):
        arc_initial_states.append(bodies.get("Delfi").ephemeris.cartesian_state(arc_start_times[i])
                                  - bodies.get("Earth").ephemeris.get_cartesian_state(arc_start_times[i]))
    return arc_initial_states


def define_multi_arc_propagation_settings(arc_wise_initial_states, arc_end_times, bodies, accelerations):

    bodies_to_propagate = ["Delfi"]
    central_bodies = ["Earth"]

    nb_arcs = len(arc_wise_initial_states)
    propagator_settings_list = []
    # concatenated_initial_states = np.zeros(6 * nb_arcs)
    for i in range(nb_arcs):
        arc_initial_state = arc_wise_initial_states[i]
        # bodies.get("Delfi").ephemeris.cartesian_state(arc_start_times[i]) \
        #                     - bodies.get("Earth").ephemeris.cartesian_state(arc_start_times[i])
        # concatenated_initial_states[i * 6:(i + 1) * 6] = arc_initial_state
        arc_termination_condition = propagation_setup.propagator.time_termination(arc_end_times[i])
        propagator_settings_list.append(propagation_setup.propagator.translational(
            central_bodies, accelerations, bodies_to_propagate, arc_initial_state, arc_termination_condition))

    multi_arc_propagator_settings = propagation_setup.propagator.multi_arc(propagator_settings_list)

    return multi_arc_propagator_settings