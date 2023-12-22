# Load tudatpy modules

import numpy as np

from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation.environment_setup import ephemeris


def get_drag_coefficient(mass, ref_area, b_star, from_tle):
    if from_tle:
        return 2.0 * mass / (0.1570 * ref_area) * b_star
    else:
        return 1.4


def define_body_settings(spacecraft_name, multi_arc_ephemeris=False):

    bodies_to_create = ["Earth", "Sun", "Moon", "Venus", "Mars", "Jupiter"]

    # Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
    global_frame_origin = "SSB"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin, global_frame_orientation)

    body_settings.add_empty_settings(spacecraft_name)
    body_state_history = dict()
    body_settings.get(spacecraft_name).ephemeris_settings = ephemeris.tabulated(dict(), global_frame_origin, global_frame_orientation)
    if multi_arc_ephemeris:
        body_settings.get(spacecraft_name).ephemeris_settings.make_multi_arc_ephemeris = 1


    return body_settings


def define_environment(mass, reference_area, drag_coefficient, srp_coefficient, spacecraft_name, multi_arc_ephemeris=False, tabulated_ephemeris={ }):

    # Load spice kernels
    spice.load_standard_kernels()

    # Define body settings
    body_settings = define_body_settings(spacecraft_name, multi_arc_ephemeris)
    if tabulated_ephemeris:
        body_settings.get(spacecraft_name).ephemeris_settings = environment_setup.ephemeris.tabulated(tabulated_ephemeris, "Earth", "J2000")

    # Create system of bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Create Delfi body
    bodies.get(spacecraft_name).mass = mass

    # Create aerodynamic coefficient interface settings
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area, np.array([drag_coefficient, 0.0, 0.0]))

    # Add the aerodynamic interface to the environment
    environment_setup.add_aerodynamic_coefficient_interface(bodies, spacecraft_name, aero_coefficient_settings)

    # Create radiation pressure settings
    occulting_bodies_dict = dict()
    occulting_bodies_dict["Sun"] = ["Earth"]
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
        reference_area, srp_coefficient, occulting_bodies_dict)

    # Add the radiation pressure interface to the environment
    environment_setup.add_radiation_pressure_target_model(bodies, spacecraft_name, radiation_pressure_settings)

    return bodies
