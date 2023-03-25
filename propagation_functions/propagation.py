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


def create_accelerations(acceleration_models, bodies, save_accelerations=False):
    # Define bodies that are propagated
    bodies_to_propagate = ["Delfi"]

    # Define central bodies of propagation_functions
    central_bodies = ["Earth"]

    # Define dependent variables for all accelerations if necessary
    dependent_variables = []
    accelerations_ids = []

    # Define the accelerations acting on Delfi
    accelerations_due_to_sun = []
    if "Sun" in acceleration_models:
        if acceleration_models['Sun']["point_mass_gravity"]:
            accelerations_due_to_sun.append(propagation_setup.acceleration.point_mass_gravity())
            if save_accelerations:
                dependent_variables.append(propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.point_mass_gravity_type, "Delfi", "Sun"))
                accelerations_ids.append("point mass gravity Sun")
        if acceleration_models['Sun']["solar_radiation_pressure"]:
            accelerations_due_to_sun.append(propagation_setup.acceleration.cannonball_radiation_pressure())
            if save_accelerations:
                dependent_variables.append(propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.cannonball_radiation_pressure_type, "Delfi", "Sun"))
                accelerations_ids.append("solar radiation pressure Sun")

    accelerations_due_to_moon = []
    if "Moon" in acceleration_models:
        if acceleration_models['Moon']["point_mass_gravity"]:
            accelerations_due_to_moon.append(propagation_setup.acceleration.point_mass_gravity())
            if save_accelerations:
                dependent_variables.append(propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.point_mass_gravity_type, "Delfi", "Moon"))
                accelerations_ids.append("point mass gravity Moon")

    accelerations_due_to_earth = []
    if "Earth" in acceleration_models:
        if acceleration_models['Earth']["point_mass_gravity"]:
            accelerations_due_to_earth.append(propagation_setup.acceleration.point_mass_gravity())
            if save_accelerations:
                dependent_variables.append(propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.point_mass_gravity_type, "Delfi", "Earth"))
                accelerations_ids.append("point mass gravity Earth")
        if acceleration_models['Earth']["spherical_harmonic_gravity"]:
            accelerations_due_to_earth.append(propagation_setup.acceleration.spherical_harmonic_gravity(2, 2))
            if save_accelerations:
                dependent_variables.append(propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.spherical_harmonic_gravity_type, "Delfi", "Earth"))
                accelerations_ids.append("spherical harmonics gravity Earth")
        if acceleration_models['Earth']["drag"]:
            accelerations_due_to_earth.append(propagation_setup.acceleration.aerodynamic())
            if save_accelerations:
                dependent_variables.append(propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.aerodynamic_type, "Delfi", "Earth"))
                accelerations_ids.append("drag Earth")

    accelerations_due_to_venus = []
    if "Venus" in acceleration_models:
        if acceleration_models['Venus']["point_mass_gravity"]:
            accelerations_due_to_venus.append(propagation_setup.acceleration.point_mass_gravity())
            if save_accelerations:
                dependent_variables.append(propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.point_mass_gravity_type, "Delfi", "Venus"))
                accelerations_ids.append("point mass gravity Venus")

    accelerations_due_to_mars = []
    if "Mars" in acceleration_models:
        if acceleration_models['Mars']["point_mass_gravity"]:
            accelerations_due_to_mars.append(propagation_setup.acceleration.point_mass_gravity())
            if save_accelerations:
                dependent_variables.append(propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.point_mass_gravity_type, "Delfi", "Mars"))
                accelerations_ids.append("point mass gravity Mars")

    accelerations_due_to_jupiter = []
    if "Jupiter" in acceleration_models:
        if acceleration_models['Jupiter']["point_mass_gravity"]:
            accelerations_due_to_jupiter.append(propagation_setup.acceleration.point_mass_gravity())
            if save_accelerations:
                dependent_variables.append(propagation_setup.dependent_variable.single_acceleration_norm(
                    propagation_setup.acceleration.point_mass_gravity_type, "Delfi", "Jupiter"))
                accelerations_ids.append("point mass gravity Jupiter")

    accelerations_settings_delfi = dict(
        Sun=accelerations_due_to_sun,
        Moon=accelerations_due_to_moon,
        Earth=accelerations_due_to_earth,
        Venus=accelerations_due_to_venus,
        Mars=accelerations_due_to_mars,
        Jupiter=accelerations_due_to_jupiter
    )

    # Create global accelerations dictionary
    acceleration_settings = {"Delfi": accelerations_settings_delfi}

    return propagation_setup.create_acceleration_models(bodies, acceleration_settings, bodies_to_propagate,
                                                        central_bodies), dependent_variables, accelerations_ids


def create_integrator_settings(initial_time, time_step: float = 10.0):
    return propagation_setup.integrator.runge_kutta_4(initial_time, time_step)


def create_propagator_settings(initial_state, initial_time, final_time, accelerations, save_accelerations=False, accelerations_to_save=[]):

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
    if save_accelerations:
        for i in range(len(accelerations_to_save)):
            dependent_variables.append(accelerations_to_save[i])

    return propagation_setup.propagator.translational(
        central_bodies, accelerations, bodies_to_propagate, initial_state, initial_time, integrator_settings,
        termination_condition, output_variables=dependent_variables)


def propagate_initial_state(initial_state, initial_time, final_time, bodies, accelerations, save_accelerations=False, accelerations_to_save=[]):

    # Create numerical integrator settings
    integrator_settings = create_integrator_settings(initial_time)

    # Create propagator settings
    single_arc_propagator_settings = create_propagator_settings(initial_state, initial_time, final_time, accelerations, save_accelerations, accelerations_to_save)

    # Propagate dynamics
    simulator = numerical_simulation.SingleArcSimulator(bodies, integrator_settings, single_arc_propagator_settings, 1, 0, 1)
    # simulator = numerical_simulation.create_dynamics_simulator(bodies, single_arc_propagator_settings, 1)

    cartesian_states = result2array(simulator.state_history)
    dependent_variables = result2array(simulator.dependent_variable_history)
    keplerian_states = dependent_variables[:, 0:7]
    latitudes = dependent_variables[:, [0, 7]]
    longitudes = dependent_variables[:, [0, 8]]
    saved_accelerations = np.zeros((np.shape(dependent_variables)[0], np.shape(dependent_variables)[1]-8))
    if save_accelerations:
        saved_accelerations[:, 0] = dependent_variables[:, 0]
        for i in range(np.shape(dependent_variables)[1]-9):
            saved_accelerations[:, i+1] = dependent_variables[:, i+9]

    return cartesian_states, keplerian_states, latitudes, longitudes, saved_accelerations


def get_initial_states(bodies, arc_start_times):
    arc_initial_states = []
    for i in range(len(arc_start_times)):
        arc_initial_states.append(bodies.get("Delfi").ephemeris.cartesian_state(arc_start_times[i])
                                  - bodies.get("Earth").ephemeris.get_cartesian_state(arc_start_times[i]))
    return arc_initial_states


def define_multi_arc_propagation_settings(arc_wise_initial_states, arc_start_times, arc_end_times, bodies, accelerations):

    bodies_to_propagate = ["Delfi"]
    central_bodies = ["Earth"]

    nb_arcs = len(arc_wise_initial_states)
    propagator_settings_list = []
    # concatenated_initial_states = np.zeros(6 * nb_arcs)
    for i in range(nb_arcs):
        arc_initial_state = arc_wise_initial_states[i]
        arc_initial_time = arc_start_times[i]
        # bodies.get("Delfi").ephemeris.cartesian_state(arc_start_times[i]) \
        #                     - bodies.get("Earth").ephemeris.cartesian_state(arc_start_times[i])
        # concatenated_initial_states[i * 6:(i + 1) * 6] = arc_initial_state

        integrator_settings = create_integrator_settings(arc_initial_time)

        arc_termination_condition = propagation_setup.propagator.time_termination(arc_end_times[i])

        dependent_variables = []
        dependent_variables.append(propagation_setup.dependent_variable.total_acceleration("Delfi"))

        propagator_settings_list.append(propagation_setup.propagator.translational(
            central_bodies, accelerations, bodies_to_propagate, arc_initial_state, arc_initial_time, integrator_settings, arc_termination_condition,
            output_variables=dependent_variables))

    multi_arc_propagator_settings = propagation_setup.propagator.multi_arc(propagator_settings_list)

    return multi_arc_propagator_settings