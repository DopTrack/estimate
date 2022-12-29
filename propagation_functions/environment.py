# Load tudatpy modules
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation.environment_setup import ephemeris

def define_body_settings(multi_arc_ephemeris=False):

    bodies_to_create = ["Earth", "Sun", "Moon", "Venus", "Mars", "Jupiter"]

    # Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
    global_frame_origin = "SSB"
    global_frame_orientation = "J2000"
    body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin,
                                                                global_frame_orientation)

    body_settings.add_empty_settings("Delfi")
    body_state_history = dict()
    body_settings.get("Delfi").ephemeris_settings = ephemeris.tabulated(dict(), global_frame_origin, global_frame_orientation)
    if multi_arc_ephemeris:
        body_settings.get("Delfi").ephemeris_settings.make_multi_arc_ephemeris = 1

    return body_settings

def define_environment(mass, reference_area, drag_coefficient, srp_coefficient, multi_arc_ephemeris = False):

    # Load spice kernels
    spice.load_standard_kernels()

    # Define body settings
    body_settings = define_body_settings(multi_arc_ephemeris)

    # Create system of bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Create Delfi body
    bodies.get("Delfi").mass = mass

    # Create aerodynamic coefficient interface settings
    aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(reference_area,
                                                                                    [drag_coefficient, 0.0, 0.0])

    # Add the aerodynamic interface to the environment
    environment_setup.add_aerodynamic_coefficient_interface(bodies, "Delfi", aero_coefficient_settings)

    # Create radiation pressure settings
    occulting_bodies = ["Earth"]
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area, srp_coefficient, occulting_bodies)

    # Add the radiation pressure interface to the environment
    environment_setup.add_radiation_pressure_interface(bodies, "Delfi", radiation_pressure_settings)

    return bodies
