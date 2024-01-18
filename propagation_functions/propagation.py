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


def get_arc_times_definition(initial_epoch, final_epoch, arc_duration):

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
        arc_mid_times.append((arc_start_times[i] + arc_end_times[i]) / 2.0)

    return arc_start_times, arc_mid_times, arc_end_times


def retrieve_arc_wise_states_from_orbit(orbit, arc_times):
    state_history = orbit[0]

    arc_wise_initial_states = []
    for time in arc_times:
        current_initial_state = np.zeros(6)
        for i in range(6):
            current_initial_state[i] = np.interp(np.array(time), state_history[:, 0], state_history[:, i + 1])
        arc_wise_initial_states.append(current_initial_state)

    return arc_wise_initial_states


def get_default_acceleration_models() -> dict:
    acceleration_models = dict(
        Sun={
            'point_mass_gravity': True, 'solar_radiation_pressure': True
        },
        Moon={
            'point_mass_gravity': True
        },
        Earth={
            'point_mass_gravity': False, 'spherical_harmonic_gravity': True, 'drag': True
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
    return acceleration_models


def create_accelerations(acceleration_models, bodies, spacecraft_name):
    # Define bodies that are propagated
    bodies_to_propagate = [spacecraft_name]

    # Define central bodies of propagation_functions
    central_bodies = ["Earth"]

    # Define the accelerations acting on the spacecraft
    accelerations_due_to_sun = []
    if "Sun" in acceleration_models:
        if acceleration_models['Sun']["point_mass_gravity"]:
            accelerations_due_to_sun.append(propagation_setup.acceleration.point_mass_gravity())
        if acceleration_models['Sun']["solar_radiation_pressure"]:
            accelerations_due_to_sun.append(propagation_setup.acceleration.radiation_pressure())

    accelerations_due_to_moon = []
    if "Moon" in acceleration_models:
        if acceleration_models['Moon']["point_mass_gravity"]:
            accelerations_due_to_moon.append(propagation_setup.acceleration.point_mass_gravity())

    accelerations_due_to_earth = []
    if "Earth" in acceleration_models:
        if acceleration_models['Earth']["point_mass_gravity"]:
            accelerations_due_to_earth.append(propagation_setup.acceleration.point_mass_gravity())
        if acceleration_models['Earth']["spherical_harmonic_gravity"]:
            accelerations_due_to_earth.append(propagation_setup.acceleration.spherical_harmonic_gravity(2, 2))
        if acceleration_models['Earth']["drag"]:
            accelerations_due_to_earth.append(propagation_setup.acceleration.aerodynamic())

    accelerations_due_to_venus = []
    if "Venus" in acceleration_models:
        if acceleration_models['Venus']["point_mass_gravity"]:
            accelerations_due_to_venus.append(propagation_setup.acceleration.point_mass_gravity())

    accelerations_due_to_mars = []
    if "Mars" in acceleration_models:
        if acceleration_models['Mars']["point_mass_gravity"]:
            accelerations_due_to_mars.append(propagation_setup.acceleration.point_mass_gravity())

    accelerations_due_to_jupiter = []
    if "Jupiter" in acceleration_models:
        if acceleration_models['Jupiter']["point_mass_gravity"]:
            accelerations_due_to_jupiter.append(propagation_setup.acceleration.point_mass_gravity())

    accelerations_settings = dict(
        Sun=accelerations_due_to_sun,
        Moon=accelerations_due_to_moon,
        Earth=accelerations_due_to_earth,
        Venus=accelerations_due_to_venus,
        Mars=accelerations_due_to_mars,
        Jupiter=accelerations_due_to_jupiter
    )

    # Create global accelerations dictionary
    acceleration_settings = {spacecraft_name: accelerations_settings}

    return propagation_setup.create_acceleration_models(bodies, acceleration_settings, bodies_to_propagate,
                                                        central_bodies)


def retrieve_accelerations_to_save(acceleration_models, spacecraft_name):
    dependent_variables = []
    accelerations_ids = []

    # Check accelerations acting on the spacecraft
    if "Sun" in acceleration_models:
        if acceleration_models['Sun']["point_mass_gravity"]:
            dependent_variables.append(propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, spacecraft_name, "Sun"))
            accelerations_ids.append("point mass gravity Sun")
        if acceleration_models['Sun']["solar_radiation_pressure"]:
            dependent_variables.append(propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.radiation_pressure_type, spacecraft_name, "Sun"))
            accelerations_ids.append("solar radiation pressure Sun")

    if "Moon" in acceleration_models:
        if acceleration_models['Moon']["point_mass_gravity"]:
            dependent_variables.append(propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, spacecraft_name, "Moon"))
            accelerations_ids.append("point mass gravity Moon")

    if "Earth" in acceleration_models:
        if acceleration_models['Earth']["point_mass_gravity"]:
            dependent_variables.append(propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, spacecraft_name, "Earth"))
            accelerations_ids.append("point mass gravity Earth")
        if acceleration_models['Earth']["spherical_harmonic_gravity"]:
            dependent_variables.append(propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.spherical_harmonic_gravity_type, spacecraft_name, "Earth"))
            accelerations_ids.append("spherical harmonics gravity Earth")
        if acceleration_models['Earth']["drag"]:
            dependent_variables.append(propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.aerodynamic_type, spacecraft_name, "Earth"))
            accelerations_ids.append("drag Earth")

    if "Venus" in acceleration_models:
        if acceleration_models['Venus']["point_mass_gravity"]:
            dependent_variables.append(propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, spacecraft_name, "Venus"))
            accelerations_ids.append("point mass gravity Venus")

    if "Mars" in acceleration_models:
        if acceleration_models['Mars']["point_mass_gravity"]:
            dependent_variables.append(propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, spacecraft_name, "Mars"))
            accelerations_ids.append("point mass gravity Mars")

    if "Jupiter" in acceleration_models:
        if acceleration_models['Jupiter']["point_mass_gravity"]:
            dependent_variables.append(propagation_setup.dependent_variable.single_acceleration_norm(
                propagation_setup.acceleration.point_mass_gravity_type, spacecraft_name, "Jupiter"))
            accelerations_ids.append("point mass gravity Jupiter")

    return dependent_variables, accelerations_ids


def create_integrator_settings(time_step: float = 10.0):
    return propagation_setup.integrator.runge_kutta_4(initial_time_step=time_step)


def create_propagator_settings(initial_state, initial_time, final_time, bodies, acceleration_models, spacecraft_name,
                               save_accelerations=False, accelerations_to_save=[]):
    # Define mid time
    mid_time = (initial_time + final_time) / 2.0

    # Define bodies that are propagated
    bodies_to_propagate = [spacecraft_name]

    # Define central bodies of propagation_functions
    central_bodies = ["Earth"]

    # Create termination settings
    termination_condition = propagation_setup.propagator.non_sequential_termination(
        propagation_setup.propagator.time_termination(final_time),
        propagation_setup.propagator.time_termination(initial_time))

    # Define integrator settings
    integrator_settings = create_integrator_settings()

    # Define dependent variables
    dependent_variables = [
        propagation_setup.dependent_variable.keplerian_state(spacecraft_name, "Earth"),
        propagation_setup.dependent_variable.latitude(spacecraft_name, "Earth"),
        propagation_setup.dependent_variable.longitude(spacecraft_name, "Earth")
    ]
    if save_accelerations:
        for i in range(len(accelerations_to_save)):
            dependent_variables.append(accelerations_to_save[i])

    accelerations = create_accelerations(acceleration_models, bodies, spacecraft_name)

    return propagation_setup.propagator.translational(
        central_bodies, accelerations, bodies_to_propagate, initial_state, mid_time, integrator_settings,
        termination_condition, output_variables=dependent_variables)


def propagate_initial_state(initial_state, initial_time, final_time, bodies, acceleration_models, spacecraft_name,
                            save_accelerations=False):
    # mid time
    mid_time = (initial_time + final_time) / 2.0

    # Create accelerations
    accelerations = create_accelerations(acceleration_models, bodies, spacecraft_name)
    accelerations_to_save = []
    if save_accelerations:
        accelerations_to_save, accelerations_ids = retrieve_accelerations_to_save(acceleration_models, spacecraft_name)

    # Create numerical integrator settings
    integrator_settings = create_integrator_settings()

    # Create propagator settings
    single_arc_propagator_settings = create_propagator_settings(initial_state, initial_time, final_time, bodies, acceleration_models,
                                                                spacecraft_name, save_accelerations, accelerations_to_save)

    # Propagate dynamics
    simulator = numerical_simulation.create_dynamics_simulator(bodies, single_arc_propagator_settings)

    cartesian_states = result2array(simulator.state_history)
    dependent_variables = result2array(simulator.dependent_variable_history)
    keplerian_states = dependent_variables[:, 0:7]
    latitudes = dependent_variables[:, [0, 7]]
    longitudes = dependent_variables[:, [0, 8]]
    saved_accelerations = np.zeros((np.shape(dependent_variables)[0], np.shape(dependent_variables)[1] - 8))
    if save_accelerations:
        saved_accelerations[:, 0] = dependent_variables[:, 0]
        for i in range(np.shape(dependent_variables)[1] - 9):
            saved_accelerations[:, i + 1] = dependent_variables[:, i + 9]

    return cartesian_states, keplerian_states, latitudes, longitudes, saved_accelerations


def get_initial_states(bodies, arc_start_times, spacecraft_name):
    arc_initial_states = []
    for i in range(len(arc_start_times)):
        arc_initial_states.append(bodies.get(spacecraft_name).ephemeris.cartesian_state(arc_start_times[i])
                                  - bodies.get("Earth").ephemeris.get_cartesian_state(arc_start_times[i]))
    return arc_initial_states


def define_multi_arc_propagation_settings(arc_wise_initial_states, arc_start_times, arc_end_times, bodies,
                                          acceleration_models, spacecraft_name):
    bodies_to_propagate = [spacecraft_name]
    central_bodies = ["Earth"]

    nb_arcs = len(arc_wise_initial_states)
    propagator_settings_list = []
    for i in range(nb_arcs):
        arc_initial_state = arc_wise_initial_states[i]
        arc_mid_time = (arc_start_times[i] + arc_end_times[i]) / 2.0

        integrator_settings = create_integrator_settings()

        arc_termination_condition = propagation_setup.propagator.non_sequential_termination(
            propagation_setup.propagator.time_termination(arc_end_times[i]),
            propagation_setup.propagator.time_termination(arc_start_times[i]))

        dependent_variables = []
        dependent_variables.append(propagation_setup.dependent_variable.total_acceleration(spacecraft_name))

        accelerations = create_accelerations(acceleration_models, bodies, spacecraft_name)

        propagator_settings_list.append(propagation_setup.propagator.translational(
            central_bodies, accelerations, bodies_to_propagate, arc_initial_state, arc_mid_time, integrator_settings,
            arc_termination_condition,
            output_variables=dependent_variables))

    multi_arc_propagator_settings = propagation_setup.propagator.multi_arc(propagator_settings_list)

    return multi_arc_propagator_settings
