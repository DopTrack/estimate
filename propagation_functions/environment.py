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


def define_body_settings(multi_arc_ephemeris=False):

    bodies_to_create = ["Earth", "Sun", "Moon", "Venus", "Mars", "Jupiter"]

    # Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
    global_frame_origin = "SSB"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin, global_frame_orientation)

    body_settings.add_empty_settings("Delfi")
    body_state_history = dict()
    body_settings.get("Delfi").ephemeris_settings = ephemeris.tabulated(dict(), global_frame_origin, global_frame_orientation)
    if multi_arc_ephemeris:
        body_settings.get("Delfi").ephemeris_settings.make_multi_arc_ephemeris = 1

    # Define the spherical harmonics gravity model
    gravitational_parameter = 3.986004415e14
    reference_radius = 6378136.3
    normalized_cosine_coefficients = [
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [-0.484165371736E-03*1.0, -0.186987635955E-09, 0.243914352398E-05, 0],
        [0.957254173792E-06, 0.202998882184E-05, 0.904627768605E-06, 0.721072657057E-06]
    ]
    normalized_sine_coefficients = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0.119528012031E-08, -0.140016683654E-05, 0],
        [0, 0.248513158716E-06, -0.619025944205E-06, 0.141435626958E-05]
    ]
    associated_reference_frame = "IAU_Earth"
    # Create the gravity field settings and add them to the body "Earth"
    body_settings.get("Earth").gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(
        gravitational_parameter,
        reference_radius,
        normalized_cosine_coefficients,
        normalized_sine_coefficients,
        associated_reference_frame)

    return body_settings


def define_environment(mass, reference_area, drag_coefficient, srp_coefficient, multi_arc_ephemeris=False, tabulated_ephemeris={ }):

    # Load spice kernels
    spice.load_standard_kernels()

    # Define body settings
    body_settings = define_body_settings(multi_arc_ephemeris)
    if tabulated_ephemeris:
        body_settings.get("Delfi").ephemeris_settings = environment_setup.ephemeris.tabulated(tabulated_ephemeris, "Earth", "J2000")

    # Create system of bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Create Delfi body
    bodies.get("Delfi").mass = mass

    # Create aerodynamic coefficient interface settings
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
        reference_area, np.array([drag_coefficient, 0.0, 0.0]))

    # Add the aerodynamic interface to the environment
    environment_setup.add_aerodynamic_coefficient_interface(bodies, "Delfi", aero_coefficient_settings)

    # Create radiation pressure settings
    occulting_bodies_dict = dict()
    occulting_bodies_dict["Sun"] = ["Earth"]
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
        reference_area, srp_coefficient, occulting_bodies_dict)

    # Add the radiation pressure interface to the environment
    environment_setup.add_radiation_pressure_target_model(bodies, "Delfi", radiation_pressure_settings)

    return bodies
