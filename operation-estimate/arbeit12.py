import os
import shutil
import pandas as pd
from pathlib import Path
import datetime as dt
import sys
sys.path.append("../")
from docx import Document
from docx.shared import Inches
import textwrap
doc = Document()
import statistics
from matplotlib import pyplot as plt
import pickle
import json
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
import subprocess
RUN_ID = os.environ.get("RUN_ID")

if RUN_ID is None:
    RUN_ID = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["RUN_ID"] = RUN_ID
results_folder = Path(__file__).parent / "results" / RUN_ID
pkl_folder = results_folder / "pkl_files"

results_folder.mkdir(parents=True, exist_ok=True)
pkl_folder.mkdir(parents=True, exist_ok=True)
#The purpose of the code block is to record the results for later analysis. The final version of the code is extractable.
if os.environ.get("TEE_CAPTURE") != "1" and "--run-single-chunk" not in sys.argv:
    env = os.environ.copy()
    env["TEE_CAPTURE"] = "1"

    with open("terminal_output.txt", "w", encoding="utf-8", errors="replace") as log:       
        process = subprocess.Popen(
            [sys.executable] + sys.argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env
        )
        sys.__stdout__.write("Main script launched. Waiting for output...\n")
        sys.__stdout__.flush()
        log.write("Main script launched. Waiting for output...\n")
        log.flush()

        for line in process.stdout:
            sys.__stdout__.write(line)
            sys.__stdout__.flush()

            log.write(line)
            log.flush()

        sys.exit(process.wait())

def read_config(filename="config.txt"):
    config = {}

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()

    return config


config = read_config("config.txt")

satellite_name = config["satellite_name"]
arc_definition = config["arc_definition"]

start_time = dt.datetime.strptime(
    config["start_time"],
    "%Y-%m-%d %H.%M"
)

estimate_days = int(config["estimate_days"])
def datetime_from_dataid(dataid):
    timestamp = dataid.split("_")[-1]
    return dt.datetime.strptime(timestamp, "%Y%m%d%H%M")

def select_data(path, satellite_name, start_time, estimate_days):
    end_time = start_time + dt.timedelta(days=estimate_days)
    filtered_files = []

    for year in range(start_time.year, end_time.year + 1):
        folder = path / satellite_name / str(year)

        if not folder.exists():
            continue

        files = os.listdir(folder)

        dataids = {
            file.rsplit(".", 1)[0]
            for file in files
            if file.endswith(".dat")
        }

        for dataid in dataids:
            time = datetime_from_dataid(dataid)

            if start_time <= time < end_time:
                dat_path = folder / f"{dataid}.dat"
                yml_path = folder / f"{dataid}.yml"

                if dat_path.exists() and yml_path.exists():
                    filtered_files.append({
                        "dataid": dataid,
                        "time": time,
                        "dat_path": dat_path,
                        "yml_path": yml_path
                    })

    return sorted(filtered_files, key=lambda x: x["time"]), end_time
# --- CHUNK DURATION MAP ---
arc_to_days = {
    "per_pass": 1,
    "per_day": 1,
    "per_3_days": 3,
    "per_week": 7,
}

chunk_days = arc_to_days[arc_definition]
chunk_length = chunk_days * constants.JULIAN_DAY
def define_pass_grouped_chunks(passes_start_times, passes_end_times, chunk_length, margin=1800.0):
    chunk_start_times = []
    chunk_mid_times = []
    chunk_end_times = []

    current_chunk_start = passes_start_times[0]
    current_chunk_end = passes_end_times[0]

    for pass_start, pass_end in zip(passes_start_times[1:], passes_end_times[1:]):

        if pass_end - current_chunk_start <= chunk_length:
            current_chunk_end = pass_end

        else:
            chunk_start_times.append(current_chunk_start - margin)
            chunk_mid_times.append((current_chunk_start + current_chunk_end) / 2.0)
            chunk_end_times.append(current_chunk_end + margin)

            current_chunk_start = pass_start
            current_chunk_end = pass_end

    chunk_start_times.append(current_chunk_start - margin)
    chunk_mid_times.append((current_chunk_start + current_chunk_end) / 2.0)
    chunk_end_times.append(current_chunk_end + margin)

    return chunk_start_times, chunk_mid_times, chunk_end_times

# Data source selection:
# True  -> Use the existing data1/ and metadata1/ folders.
# False -> Rebuild the local folders by copying the required files

satellite_name = config["satellite_name"]
arc_definition = config["arc_definition"]

start_time = dt.datetime.strptime(
    config["start_time"],
    "%Y-%m-%d %H.%M"
)

estimate_days = int(config["estimate_days"])
IS_CHILD_MODE = "--run-single-chunk" in sys.argv
USE_LOCAL_DATA = True
if not IS_CHILD_MODE:
    if USE_LOCAL_DATA:
        print("Using existing local data folders (data1/ and metadata1/).", flush=True)
    else:
        print("Collecting data from network path and rebuilding local folders.", flush=True)
PATH = Path(config["data_path"])
selected_files, end_time = select_data(
    PATH,
    satellite_name,
    start_time,
    estimate_days
)

def prepare_chunk_local_folders(selected_files, chunk_indices, chunk_number):
    current_dir = Path(__file__).parent

    data_chunk_path = current_dir / "data1" / f"chunk_{chunk_number}"
    metadata_chunk_path = current_dir / "metadata1" / f"chunk_{chunk_number}"

    for folder in [data_chunk_path, metadata_chunk_path]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

    for i in chunk_indices:
        shutil.copy2(
            selected_files[i]["dat_path"],
            data_chunk_path / selected_files[i]["dat_path"].name
        )
        shutil.copy2(
            selected_files[i]["yml_path"],
            metadata_chunk_path / selected_files[i]["yml_path"].name
        )

    return str(data_chunk_path) + "/", str(metadata_chunk_path) + "/"

metadata_folder = "metadata1/"
data_folder = "data1/"

metadata = [f"{item['dataid']}.yml" for item in selected_files]
data = [f"{item['dataid']}.dat" for item in selected_files]
indices_files_to_load = list(range(len(metadata)))

# Load all observations once only to define chunks according to arc_definition
recording_start_times_all = extract_recording_start_times_yml(
    metadata_folder,
    [metadata[i] for i in indices_files_to_load],
    old_yml=False
)

passes_start_times_all, passes_end_times_all, observation_times_all, observations_set_all = load_and_format_observations(
    satellite_name,
    data_folder,
    [data[i] for i in indices_files_to_load],
    recording_start_times_all,
    old_obs_format=False
)

if arc_definition == "per_pass":
    chunk_arc_start_times, chunk_arc_mid_times, chunk_arc_end_times = define_arcs(
        arc_definition,
        passes_start_times_all,
        passes_end_times_all
    )
else:
    chunk_arc_start_times, chunk_arc_mid_times, chunk_arc_end_times = define_pass_grouped_chunks(
        passes_start_times_all,
        passes_end_times_all,
        chunk_length,
        margin=1800.0
    )

chunks = []

if arc_definition == "per_pass":
    chunks = [[i] for i in indices_files_to_load]

else:
    chunk_days = arc_to_days[arc_definition]

    current_start = start_time
    current_end = start_time + dt.timedelta(days=chunk_days)

    while current_start < end_time:
        current_chunk = []

        for i in indices_files_to_load:
            file_time = selected_files[i]["time"]

            if current_start <= file_time < current_end:
                current_chunk.append(i)

        if current_chunk:
            chunks.append(current_chunk)

        current_start = current_end
        current_end = current_start + dt.timedelta(days=chunk_days)


if not IS_CHILD_MODE:
    print("\nSatellite:", satellite_name)
    print("Arc definition:", arc_definition)
    print("Estimate days:", estimate_days)
    print("Start time:", start_time)
    print("End time:", end_time)
    print("Number of matching files:", len(selected_files))
    print("Selected indices:", indices_files_to_load)
    print("Chunks based on arc definition:")

    for i, chunk in enumerate(chunks, start=1):
        print(f"Chunk {i}: {chunk}")

    print("Chunks:", chunks)
    print("metadata =", metadata)
    print("data =", data)


def run_one_chunk(chunk_indices, chunk_number, nb_iterations=10, verbose=True,
                  data_folder_override=None, metadata_folder_override=None):
    local_data_folder = data_folder_override if data_folder_override is not None else data_folder
    local_metadata_folder = metadata_folder_override if metadata_folder_override is not None else metadata_folder    
    tle_global_index = chunk_indices[0]
    initial_epoch, initial_state_teme, b_star = get_tle_initial_conditions(
        local_metadata_folder + metadata[tle_global_index],
        old_yml=False
)
    print(f"\n==============================")
    print(f"RUNNING CHUNK {chunk_number}")
    print(f"Indices: {chunk_indices}")
    print(f"==============================\n")

    # Retrieve recording starting times
    recording_start_times = extract_recording_start_times_yml(local_metadata_folder, [metadata[i] for i in chunk_indices], old_yml=False)

    spacecraft_name = satellite_name
    # Load and process observations
    passes_start_times, passes_end_times, observation_times, observations_set = load_and_format_observations(
        spacecraft_name, local_data_folder, [data[i] for i in chunk_indices], recording_start_times, old_obs_format=False)

    ### SETTING YOUR ESTIMATION ARCS

    if arc_definition == "per_pass":
        arc_start_times, arc_mid_times, arc_end_times = define_arcs(
            arc_definition,
            passes_start_times,
            passes_end_times
        )
    else:
        arc_start_times, arc_mid_times, arc_end_times = define_pass_grouped_chunks(
            passes_start_times,
            passes_end_times,
            chunk_length,
            margin=1800.0
        )

    if verbose:
        print('arc_start_times', arc_start_times)
        print('arc_end_times', arc_end_times)

    arc_pass_indices = []

    for arc in range(len(arc_start_times)):
        current_indices = []

        for i, pass_start in enumerate(passes_start_times):
            if arc_start_times[arc] <= pass_start <= arc_end_times[arc]:
                current_indices.append(chunk_indices[i])

        arc_pass_indices.append(current_indices)

    if verbose:
        print("arc_pass_indices =", arc_pass_indices)

    chunk_drag_days = max(
        1,
        int(np.ceil((arc_end_times[-1] - arc_start_times[0]) / constants.JULIAN_DAY))
    )

    drag_start_times = [
        arc_start_times[0] + i * constants.JULIAN_DAY
        for i in range(chunk_drag_days)
    ]

    plot_arc_start_times = arc_start_times.copy()
    plot_arc_end_times = arc_end_times.copy()

    for i in range(len(plot_arc_end_times) - 1):
        plot_arc_end_times[i] = plot_arc_start_times[i + 1]

    if len(chunk_indices) == 1:
        requested_final_epoch = arc_end_times[-1]
    else:
        requested_final_epoch = arc_start_times[0] + chunk_length
    plot_arc_end_times[-1] = requested_final_epoch

    propagation_start = get_start_next_day(initial_epoch)
    final_epoch = plot_arc_end_times[-1] 
    mid_epoch = (initial_epoch + final_epoch) / 2.0

    # Retrieve the spacecraft's initial state at mid-epoch from the TLE orbit
    initial_state = propagate_sgp4(
        local_metadata_folder + metadata[tle_global_index],
        initial_epoch,
        [mid_epoch],
        old_yml=False
    )[0, 1:]

    # Define propagation_functions environment
    mass = 2.2
    ref_area = (4 * 0.3 * 0.1 + 2 * 0.1 * 0.1) / 4  # Average projection area of a 3U CubeSat
    srp_coef = 1.2
    drag_coef = 1.2
    bodies = define_environment(mass, ref_area, drag_coef, srp_coef, spacecraft_name, multi_arc_ephemeris=False)

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
    orbit = propagate_initial_state(initial_state, initial_epoch, final_epoch, bodies, accelerations, spacecraft_name)
    arc_wise_initial_states = get_initial_states(bodies, arc_mid_times, spacecraft_name)


    # Redefine environment to allow for multi-arc dynamics propagation_functions
    bodies = define_environment(mass, ref_area, drag_coef, srp_coef, spacecraft_name, multi_arc_ephemeris=True)

    # Define multi-arc propagator settings
    multi_arc_propagator_settings = define_multi_arc_propagation_settings(arc_wise_initial_states, arc_start_times, arc_end_times,
                                                                        bodies, accelerations, spacecraft_name)
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
    observation_settings = define_observation_settings(spacecraft_name, Doppler_models, passes_start_times, arc_start_times)

    # Define parameters to estimate
    parameters_list = dict(
        initial_state={
            'estimate': True
        },
            drag_coefficient={
            'estimate': True,
            'type': 'per_day'
        },
        constant_absolute_bias={
            'estimate': True
        },
        linear_absolute_bias={
            'estimate': True
        }
    )
    parameters_to_estimate = define_parameters(parameters_list, bodies, multi_arc_propagator_settings, spacecraft_name,
                                            arc_start_times, arc_mid_times, [(get_link_ends_id("DopTrackStation", spacecraft_name), passes_start_times)], Doppler_models, drag_start_times=drag_start_times)
    if verbose:
        parameters.print_parameter_names(parameters_to_estimate)

    # Create the estimator object
    estimator = estimation_analysis.Estimator(bodies, parameters_to_estimate, observation_settings, multi_arc_propagator_settings)

    # Simulate (ideal) observations
    ideal_observations = simulate_observations_from_estimator(spacecraft_name, observation_times, estimator, bodies)


    ### RUN THE ESTIMATION

    # Now you are all setup to run the estimation. In the following block the dynamic equations are set and the estimator knows what kind of parameters need to be estimated.
    # This can take a while, depending on your amount of data and settings

    # Save the true parameters to later analyse the error
    truth_parameters = parameters_to_estimate.parameter_vector
    nb_parameters = len(truth_parameters)

    number_of_passes = len(chunk_indices)

    #nb_iterations = 10
    nb_arcs = len(arc_start_times)

    pod_output = run_estimation(
        estimator,
        parameters_to_estimate,
        observations_set,
        nb_arcs,
        nb_iterations,
        parameters_list=parameters_list,
        obs_models=Doppler_models,
        number_of_passes=number_of_passes,
        number_of_drag_estimates=len(drag_start_times)
    )

    updated_parameters = parameters_to_estimate.parameter_vector.copy()
    # --------------------------------------------------------
    # Store drag estimates directly in result
    # Parameter order:
    # states -> constant biases -> linear biases -> drag
    # --------------------------------------------------------

    state_parameter_count = 6 * nb_arcs

    constant_bias_start = state_parameter_count
    constant_bias_end = constant_bias_start + number_of_passes

    linear_bias_start = constant_bias_end
    linear_bias_end = linear_bias_start + number_of_passes

    drag_start_index = linear_bias_end
    drag_end_index = drag_start_index + len(drag_start_times)

    drag_estimates_for_excel = updated_parameters[drag_start_index:drag_end_index]

    residuals = pod_output.residual_history

    residuals_per_pass = get_residuals_per_pass(
        observation_times,
        residuals,
        passes_start_times
    )
    return {
        "chunk_number": chunk_number,
        "chunk_indices": chunk_indices,
        "arc_start_times": arc_start_times,
        "arc_mid_times": arc_mid_times,
        "arc_end_times": arc_end_times,
        "arc_wise_initial_states": arc_wise_initial_states,
        "updated_parameters": updated_parameters,
        "residuals": residuals[:, nb_iterations - 1],
        "residuals_per_pass": residuals_per_pass,
        "passes_start_times": passes_start_times,
        "passes_end_times": passes_end_times,
        "nb_arcs": nb_arcs,
        "number_of_passes": number_of_passes,
        "plot_arc_start_times": plot_arc_start_times,
        "plot_arc_end_times": plot_arc_end_times,
        "mass": mass,
        "ref_area": ref_area,
        "drag_coef": drag_coef,
        "srp_coef": srp_coef,
        "spacecraft_name": spacecraft_name,
        "accelerations": accelerations,
        "arc_pass_indices": arc_pass_indices,
        "parameters_list": parameters_list,
        "Doppler_models": Doppler_models,
        "drag_start_times": drag_start_times,
        "residual_history": residuals,
        "drag_estimates": drag_estimates_for_excel       
    }
def save_chunk_result(result, chunk_number):
    result_path = pkl_folder / f"chunk_result_{chunk_number}.pkl"
    with open(result_path, "wb") as f:
        pickle.dump(result, f)
    return result_path


def load_chunk_result(chunk_number):
    result_path = pkl_folder / f"chunk_result_{chunk_number}.pkl"
    with open(result_path, "rb") as f:
        return pickle.load(f)

def is_valid_result(result):
    if result is None:
        return False

    updated_parameters = result.get("updated_parameters", None)
    residuals = result.get("residuals", None)

    if updated_parameters is None or residuals is None:
        return False

    if not np.all(np.isfinite(updated_parameters)):
        return False

    if not np.all(np.isfinite(residuals)):
        return False

    state = updated_parameters[:6]
    r_norm = np.linalg.norm(state[:3])
    v_norm = np.linalg.norm(state[3:6])

    if not (6.0e6 < r_norm < 9.0e6):
        return False

    if not (6.0e3 < v_norm < 9.0e3):
        return False

    return True

def is_bad_pass_first_iteration(result, threshold=270.0):
    if result is None:
        return True
    residual_history = result.get("residual_history", None)

    if residual_history is None:
        return True

    first_iteration_residuals = residual_history[:, 0]

    if not np.all(np.isfinite(first_iteration_residuals)):
        return True

    first_rms = np.sqrt(np.mean(first_iteration_residuals**2))

    if not np.isfinite(first_rms):
        return True

    return first_rms > threshold

all_chunk_results = []
failed_chunks = []

# ------------------------------------------------------------
# CHILD MODE: only run one chunk, save result, exit
# ------------------------------------------------------------
if "--run-single-chunk" in sys.argv:
    arg_index = sys.argv.index("--run-single-chunk")
    chunk_number = sys.argv[arg_index + 1]
    chunk_indices = json.loads(sys.argv[arg_index + 2])
    chunk_data_folder = sys.argv[arg_index + 3]
    chunk_metadata_folder = sys.argv[arg_index + 4]

    is_pass_check = str(chunk_number).startswith("PASS_")

    result = run_one_chunk(
        chunk_indices,
        chunk_number,
        nb_iterations=1 if is_pass_check else 10,
        verbose=not is_pass_check,
        data_folder_override=chunk_data_folder,
        metadata_folder_override=chunk_metadata_folder
    )

    if not is_valid_result(result):
        raise ValueError("Invalid chunk result")

    save_chunk_result(result, chunk_number)
    sys.exit(0)

# ------------------------------------------------------------
# PARENT MODE: run each chunk in a separate subprocess
# ------------------------------------------------------------
for chunk_number, chunk_indices in enumerate(chunks, start=1):
    chunk_data_folder, chunk_metadata_folder = prepare_chunk_local_folders(
        selected_files,
        chunk_indices,
        chunk_number
    )
    print(f"\nStarting chunk {chunk_number} in separate subprocess...")

    result_path = pkl_folder / f"chunk_result_{chunk_number}.pkl"

    if result_path.exists():
        result_path.unlink()

    cmd = [
        sys.executable,
        __file__,
        "--run-single-chunk",
        str(chunk_number),
        json.dumps(chunk_indices),
        chunk_data_folder,
        chunk_metadata_folder
    ]

    process = subprocess.run(cmd)

    if process.returncode != 0:
        print(f"Chunk {chunk_number} failed. Searching bad passes...")

        good_passes = []
        bad_passes = []

        for idx in chunk_indices:
            pass_cmd = [
                sys.executable,
                __file__,
                "--run-single-chunk",
                f"PASS_{idx}",
                json.dumps([idx]),
                chunk_data_folder,
                chunk_metadata_folder
            ]

            pass_process = subprocess.run(pass_cmd)

            if pass_process.returncode != 0:
                print(f"PASS {idx} failed.")
                bad_passes.append(idx)
                continue

            try:
                pass_result = load_chunk_result(f"PASS_{idx}")

                if (not is_valid_result(pass_result)) or is_bad_pass_first_iteration(pass_result, threshold=270.0):
                    bad_passes.append(idx)
                else:
                    good_passes.append(idx)

            except BaseException as e:
                print(f"PASS {idx} result could not be loaded:", repr(e))
                bad_passes.append(idx)

        print("Good passes:", good_passes)
        print("Bad passes:", bad_passes)

        MINIMUM_GOOD_PASSES = math.ceil(len(chunk_indices) / 2)

        if len(good_passes) < MINIMUM_GOOD_PASSES:
            print(
                f"Chunk {chunk_number}: insufficient valid passes "
                f"({len(good_passes)}/{MINIMUM_GOOD_PASSES}) -> skipped"
                f"  good passes = {len(good_passes)}, "
                f"required = {MINIMUM_GOOD_PASSES}"
            )
            failed_chunks.append({
                "chunk_number": chunk_number,
                "chunk_indices": chunk_indices,
                "good_passes": good_passes,
                "bad_passes": bad_passes,
                "reason": (
                    f"Not enough good passes. "
                    f"Found {len(good_passes)}, "
                    f"required at least {MINIMUM_GOOD_PASSES}"
                )
            })
            continue

        clean_cmd = [
            sys.executable,
            __file__,
            "--run-single-chunk",
            f"{chunk_number}_CLEAN",
            json.dumps(good_passes),
            chunk_data_folder,
            chunk_metadata_folder
        ]

        clean_process = subprocess.run(clean_cmd)

        if clean_process.returncode != 0:
            failed_chunks.append({
                "chunk_number": chunk_number,
                "chunk_indices": chunk_indices,
                "good_passes": good_passes,
                "bad_passes": bad_passes,
                "reason": f"Clean rerun failed with return code {clean_process.returncode}"
            })
            continue

        try:
            clean_result = load_chunk_result(f"{chunk_number}_CLEAN")

            if not is_valid_result(clean_result):
                raise ValueError("Clean chunk still invalid")

            all_chunk_results.append(clean_result)
            print(f"Chunk {chunk_number} completed after removing bad passes.")

        except BaseException as e:
            failed_chunks.append({
                "chunk_number": chunk_number,
                "chunk_indices": chunk_indices,
                "good_passes": good_passes,
                "bad_passes": bad_passes,
                "reason": "Clean result could not be loaded: " + repr(e)
            })

        continue

    try:
        result = load_chunk_result(chunk_number)

        if not is_valid_result(result):
            raise ValueError("Invalid chunk result after subprocess")

        all_chunk_results.append(result)
        print(f"Chunk {chunk_number} completed successfully.")

    except Exception as e:
        failed_chunks.append({
            "chunk_number": chunk_number,
            "chunk_indices": chunk_indices,
            "reason": str(e)
        })
        print(f"Chunk {chunk_number} result could not be loaded. Skipping.")
        continue
valid_validation_results = []

for result in all_chunk_results:

    try:

        if is_valid_result(result):
            valid_validation_results.append(result)

        else:
            failed_chunks.append({
                "chunk_number": result.get("chunk_number", "unknown"),
                "chunk_indices": result.get("chunk_indices", []),
                "reason": "Invalid result"
            })

    except BaseException  as e:

        failed_chunks.append({
            "chunk_number": result.get("chunk_number", "unknown"),
            "chunk_indices": result.get("chunk_indices", []),
            "reason": str(e)
        })
if len(valid_validation_results) == 0:
    raise RuntimeError("All chunks failed. No valid estimation result available.")

print("\nFAILED CHUNKS:")
for item in failed_chunks:
    print(item)

all_residuals = np.concatenate([
    result["residuals"]
    for result in valid_validation_results
])

mean_residuals = statistics.mean(all_residuals)
std_residuals = statistics.stdev(all_residuals)

print('--------------------------------------------------------------')

for result in valid_validation_results:
    print(f"CHUNK {result['chunk_number']}")
    print("indices:", result["chunk_indices"])

    for i in range(len(result["residuals_per_pass"])):
        print(
            "size residuals current pass",
            np.shape(result["residuals_per_pass"][i])
        )

### INSPECT THE RESULTS 

# The first number that we look at is final residual. This shows the difference (root mean square) between the observed range-rate and the final orbit model 
# estimated by your program.
# Excel filename
timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

results_folder = Path(__file__).parent / f"results_{satellite_name}_{arc_definition}_{timestamp}"
results_folder.mkdir(exist_ok=True)
csv_folder = results_folder / "csv"
csv_folder.mkdir(parents=True, exist_ok=True)

word_doc = Document()

word_doc.add_heading("Orbit Estimation Chunk Results", level=1)
word_doc.add_paragraph(f"Satellite: {satellite_name}")
word_doc.add_paragraph(f"Arc definition: {arc_definition}")
word_doc.add_paragraph(f"Number of chunks: {len(valid_validation_results)}")

word_doc.add_page_break()
word_doc.add_heading("Terminal Output", level=1)

with open("terminal_output.txt", "r", encoding="utf-8", errors="replace") as f:
    terminal_text = f.read()

word_doc.add_paragraph(terminal_text)

# Plot residuals histogram
all_residuals = np.concatenate([
    result["residuals"]
    for result in valid_validation_results
])

mean_residuals = statistics.mean(all_residuals)
std_residuals = statistics.stdev(all_residuals)

print("Mean residual:", mean_residuals)
print("Std residual:", std_residuals)

fig = plt.figure()
ax = fig.add_subplot()

plt.hist(all_residuals, 100)

ax.set_xlabel('Doppler residuals [m/s]')
ax.set_ylabel('Nb occurrences []')
ax.set_title('Residual histogram - all chunks')

plt.grid()

fig.savefig(
    results_folder / "residual_histogram_all_chunks.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

for result in valid_validation_results:

    chunk_folder = results_folder / f"CHUNK_{result['chunk_number']}"
    chunk_folder.mkdir(exist_ok=True)

    residuals_per_pass = result["residuals_per_pass"]
    number_of_passes = len(residuals_per_pass)

    fig, axs = plt.subplots(
        math.ceil(number_of_passes / 3),
        3,
        figsize=(12, 8)
    )

    axs = np.array(axs).reshape(-1)

    for i in range(number_of_passes):
        ax = axs[i]
        ax.plot(residuals_per_pass[i], linestyle='-.')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Residuals [m/s]')
        ax.set_title(f"Chunk {result['chunk_number']} - Pass {result['chunk_indices'][i]}")
        ax.grid()

    for j in range(number_of_passes, len(axs)):
        axs[j].axis("off")

    fig.tight_layout()

    fig.savefig(
        chunk_folder / "residuals_per_pass.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

### ORBIT VALIDATION: some comparison suggestions
print("ORBIT VALIDATION FOR EACH CHUNK")

for result in valid_validation_results:

    chunk_number = result["chunk_number"]
    updated_parameters = result["updated_parameters"]
    arc_wise_initial_states = result["arc_wise_initial_states"]
    plot_arc_start_times = result["plot_arc_start_times"]
    plot_arc_end_times = result["plot_arc_end_times"]
    nb_arcs = result["nb_arcs"]
    residuals_per_pass = result["residuals_per_pass"]
    arc_pass_indices = result["arc_pass_indices"]
    parameters_list = result["parameters_list"]
    Doppler_models = result["Doppler_models"]
    drag_type = parameters_list["drag_coefficient"]["type"]

    bodies = define_environment(
        result["mass"],
        result["ref_area"],
        result["drag_coef"],
        result["srp_coef"],
        result["spacecraft_name"],
        multi_arc_ephemeris=False
    )

    chunk_folder = results_folder / f"CHUNK_{chunk_number}"
    chunk_folder.mkdir(exist_ok=True)

    for arc_index in range(nb_arcs):

        estimated_state = updated_parameters[6*arc_index:(arc_index+1)*6]
        TLE_state = arc_wise_initial_states[arc_index]

    gravitational_parameter = bodies.get("Earth").gravity_field_model.gravitational_parameter

    print(f"========== CHUNK {chunk_number} ==========")
    print("ALL ESTIMATED PARAMETERS")
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
        arc_folder = chunk_folder / f"ARC_{arc+1}"
        arc_folder.mkdir(exist_ok=True)
        arc_text = f"""
    ARC #{arc+1}

    Used chunk index count: {len(result["chunk_indices"])}
    Used chunk indices: {result["chunk_indices"]}

    INITIAL STATE from TLE
    {arc_wise_initial_states[arc]}

    UPDATED STATE from DOPTRACK
    {updated_parameters[arc*6:(arc+1)*6]}

    Distance [km] between TLE initial state and estimated state:
    {pos_error/1000}

    Estimated state:
    Semi-major axis [km]: {state_keplerian[0]/1000}
    Eccentricity: {state_keplerian[1]}
    Inclination [deg]: {np.rad2deg(state_keplerian[2])}
    Argument of Perigee [deg]: {np.rad2deg(state_keplerian[3])}
    RAAN [deg]: {np.rad2deg(state_keplerian[4])}
    True anomaly [deg]: {np.rad2deg(state_keplerian[5])}
    True longitude [deg]: {np.mod(np.rad2deg(state_keplerian[5]) + np.rad2deg(state_keplerian[3]), 360)}
    Altitude [km]: {state_keplerian[0]/1000 - 6371.360}

    TLE state:
    Semi-major axis [km]: {TLE_keplerian[0]/1000}
    Eccentricity: {TLE_keplerian[1]}
    Inclination [deg]: {np.rad2deg(TLE_keplerian[2])}
    Argument of Perigee [deg]: {np.rad2deg(TLE_keplerian[3])}
    RAAN [deg]: {np.rad2deg(TLE_keplerian[4])}
    True anomaly [deg]: {np.rad2deg(TLE_keplerian[5])}
    True longitude [deg]: {np.mod(np.rad2deg(TLE_keplerian[5]) + np.rad2deg(TLE_keplerian[3]), 360)}
    Altitude [km]: {TLE_keplerian[0]/1000 - 6371.360}
    """

        # Add content to Word document
        word_doc.add_heading(f"ARC #{arc+1}", level=2)
        word_doc.add_paragraph(f"Used chunk index count: {len(result['chunk_indices'])}")
        word_doc.add_paragraph(f"Used chunk indices: {result['chunk_indices']}")


        table = word_doc.add_table(rows=1, cols=3)
        table.style = "Table Grid"

        hdr = table.rows[0].cells
        hdr[0].text = "Parameter"
        hdr[1].text = "Estimated"
        hdr[2].text = "TLE"

        rows = [
            ("Semi-major axis [km]", state_keplerian[0]/1000, TLE_keplerian[0]/1000),
            ("Eccentricity", state_keplerian[1], TLE_keplerian[1]),
            ("Inclination [deg]", np.rad2deg(state_keplerian[2]), np.rad2deg(TLE_keplerian[2])),
            ("Argument of Perigee [deg]", np.rad2deg(state_keplerian[3]), np.rad2deg(TLE_keplerian[3])),
            ("RAAN [deg]", np.rad2deg(state_keplerian[4]), np.rad2deg(TLE_keplerian[4])),
            ("True anomaly [deg]", np.rad2deg(state_keplerian[5]), np.rad2deg(TLE_keplerian[5])),
            ("True longitude [deg]",
            np.mod(np.rad2deg(state_keplerian[5]) + np.rad2deg(state_keplerian[3]), 360),
            np.mod(np.rad2deg(TLE_keplerian[5]) + np.rad2deg(TLE_keplerian[3]), 360)),
            ("Altitude [km]", state_keplerian[0]/1000 - 6371.360, TLE_keplerian[0]/1000 - 6371.360),
        ]

        for name, estimated, tle in rows:
            cells = table.add_row().cells
            cells[0].text = name
            cells[1].text = str(estimated)
            cells[2].text = str(tle)

        word_doc.add_paragraph(f"Distance between TLE and estimated state [km]: {pos_error/1000}")

        # Create PNG summary image
        fig = plt.figure(figsize=(10, 8))
        plt.axis("off")
        plt.text(
            0.01,
            0.99,
            arc_text,
            va="top",
            ha="left",
            family="monospace",
            fontsize=8
        )

        arc_image_path = arc_folder / f"arc_{arc+1}_summary.png"
        fig.savefig(arc_image_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Insert PNG into Word document
        word_doc.add_picture(str(arc_image_path), width=Inches(6.5))

        estimated_values = [
            state_keplerian[0]/1000,
            state_keplerian[1],
            np.rad2deg(state_keplerian[2]),
            state_keplerian[0]/1000 - 6371.360
        ]

        tle_values = [
            TLE_keplerian[0]/1000,
            TLE_keplerian[1],
            np.rad2deg(TLE_keplerian[2]),
            TLE_keplerian[0]/1000 - 6371.360
        ]


print('----------------------------------------')
print('BIASES ESTIMATES')
print('ABSOLUTE CONSTANT BIASES ESTIMATES')
print(updated_parameters[6*nb_arcs:6*nb_arcs+number_of_passes])
print('LINEAR CONSTANT BIASES ESTIMATES')
print(updated_parameters[6*nb_arcs+number_of_passes+1:6*nb_arcs+number_of_passes*2])

# ============================================================
# ORBIT VALIDATION FOR ALL CHUNKS COMBINED
# ============================================================

all_time = []

all_diff_x = []
all_diff_y = []
all_diff_z = []

all_diff_vx = []
all_diff_vy = []
all_diff_vz = []

all_range_residual = []
all_vmag_residual = []
all_distance_orbits = []

all_rsw = []
all_keplerian = []

all_tle_x = []
all_tle_y = []
all_tle_z = []

all_estimated_x = []
all_estimated_y = []
all_estimated_z = []

for result in valid_validation_results:
    chunk_number = result["chunk_number"]
    chunk_csv_folder = csv_folder / f"chunk_{chunk_number}"
    chunk_csv_folder.mkdir(exist_ok=True)
    updated_parameters = result["updated_parameters"]
    arc_wise_initial_states = result["arc_wise_initial_states"]

    plot_arc_start_times = result["plot_arc_start_times"]
    plot_arc_end_times = result["plot_arc_end_times"]

    nb_arcs = result["nb_arcs"]
    accelerations = result["accelerations"]
    spacecraft_name = result["spacecraft_name"]
    residuals_per_pass = result["residuals_per_pass"]
    arc_pass_indices = result["arc_pass_indices"]

    bodies = define_environment(
        result["mass"],
        result["ref_area"],
        result["drag_coef"],
        result["srp_coef"],
        result["spacecraft_name"],
        multi_arc_ephemeris=False
    )

    chunk_folder = results_folder / f"CHUNK_{chunk_number}"
    chunk_folder.mkdir(exist_ok=True)
    for arc_index in range(nb_arcs):

        estimated_state = updated_parameters[6*arc_index:(arc_index+1)*6]
        TLE_state = arc_wise_initial_states[arc_index]

        estimated_orbit = propagate_initial_state(
            estimated_state,
            plot_arc_start_times[arc_index],
            plot_arc_end_times[arc_index],
            bodies,
            accelerations,
            spacecraft_name
        )[0]

        TLE_orbit = propagate_initial_state(
            TLE_state,
            plot_arc_start_times[arc_index],
            plot_arc_end_times[arc_index],
            bodies,
            accelerations,
            spacecraft_name
        )[0]

        time_current = TLE_orbit[:, 0]

        all_time.append(time_current)

        all_tle_x.append(TLE_orbit[:, 1])
        all_tle_y.append(TLE_orbit[:, 2])
        all_tle_z.append(TLE_orbit[:, 3])

        all_estimated_x.append(estimated_orbit[:, 1])
        all_estimated_y.append(estimated_orbit[:, 2])
        all_estimated_z.append(estimated_orbit[:, 3])

        all_diff_x.append((TLE_orbit[:, 1] - estimated_orbit[:, 1]) / 1000)
        all_diff_y.append((TLE_orbit[:, 2] - estimated_orbit[:, 2]) / 1000)
        all_diff_z.append((TLE_orbit[:, 3] - estimated_orbit[:, 3]) / 1000)

        all_diff_vx.append((TLE_orbit[:, 4] - estimated_orbit[:, 4]) / 1000)
        all_diff_vy.append((TLE_orbit[:, 5] - estimated_orbit[:, 5]) / 1000)
        all_diff_vz.append((TLE_orbit[:, 6] - estimated_orbit[:, 6]) / 1000)

        range_TLE = np.sqrt(TLE_orbit[:,1]**2 + TLE_orbit[:,2]**2 + TLE_orbit[:,3]**2)
        range_estimated = np.sqrt(estimated_orbit[:,1]**2 + estimated_orbit[:,2]**2 + estimated_orbit[:,3]**2)

        Vmag_TLE = np.sqrt(TLE_orbit[:,4]**2 + TLE_orbit[:,5]**2 + TLE_orbit[:,6]**2)
        Vmag_estimated = np.sqrt(estimated_orbit[:,4]**2 + estimated_orbit[:,5]**2 + estimated_orbit[:,6]**2)

        all_range_residual.append((range_TLE - range_estimated) / 1000)
        all_vmag_residual.append((Vmag_TLE - Vmag_estimated) / 1000)

        distance_orbits = np.sqrt(
            (TLE_orbit[:,1] - estimated_orbit[:,1])**2 +
            (TLE_orbit[:,2] - estimated_orbit[:,2])**2 +
            (TLE_orbit[:,3] - estimated_orbit[:,3])**2
        )

        all_distance_orbits.append(distance_orbits / 1000)
        
        rsw_difference_current = np.zeros((len(TLE_orbit[:,0]), 7))
        keplerian_difference_current = np.zeros((len(TLE_orbit[:,0]), 7))

        for i in range(len(TLE_orbit[:,0])):

            current_epoch = TLE_orbit[i, 0]

            current_tle_state = TLE_orbit[i, 1:]
            current_estimated_state = estimated_orbit[i, 1:]

            current_tle_keplerian = element_conversion.cartesian_to_keplerian(
                current_tle_state,
                bodies.get("Earth").gravitational_parameter
            )

            current_estimated_keplerian = element_conversion.cartesian_to_keplerian(
                current_estimated_state,
                bodies.get("Earth").gravitational_parameter
            )

            keplerian_difference_current[i, 0] = current_epoch
            keplerian_difference_current[i, 1:7] = (
                current_estimated_keplerian - current_tle_keplerian
            )

            current_state_difference = current_estimated_state - current_tle_state
            current_position_difference = current_state_difference[0:3]
            current_velocity_difference = current_state_difference[3:6]

            rotation_to_rsw = frame_conversion.inertial_to_rsw_rotation_matrix(
                current_tle_state
            )

            rsw_difference_current[i, 0] = current_epoch
            rsw_difference_current[i, 1:4] = rotation_to_rsw @ current_position_difference
            rsw_difference_current[i, 4:7] = rotation_to_rsw @ current_velocity_difference

        all_rsw.append(rsw_difference_current)
        all_keplerian.append(keplerian_difference_current)
        arc_folder = chunk_folder / f"ARC_{arc_index+1}"
        arc_folder.mkdir(exist_ok=True)

        arc_excel_path = arc_folder / f"comparison_results_arc_{arc_index+1}.xlsx"

        time_arc = TLE_orbit[:, 0] - TLE_orbit[0, 0]

        range_TLE_arc = np.sqrt(TLE_orbit[:, 1]**2 + TLE_orbit[:, 2]**2 + TLE_orbit[:, 3]**2)
        range_estimated_arc = np.sqrt(estimated_orbit[:, 1]**2 + estimated_orbit[:, 2]**2 + estimated_orbit[:, 3]**2)

        Vmag_TLE_arc = np.sqrt(TLE_orbit[:, 4]**2 + TLE_orbit[:, 5]**2 + TLE_orbit[:, 6]**2)
        Vmag_estimated_arc = np.sqrt(estimated_orbit[:, 4]**2 + estimated_orbit[:, 5]**2 + estimated_orbit[:, 6]**2)

        distance_orbits_arc = np.sqrt(
            (TLE_orbit[:, 1] - estimated_orbit[:, 1])**2 +
            (TLE_orbit[:, 2] - estimated_orbit[:, 2])**2 +
            (TLE_orbit[:, 3] - estimated_orbit[:, 3])**2
        )

        with pd.ExcelWriter(arc_excel_path, engine="openpyxl") as writer:

            for pass_index in range(len(residuals_per_pass)):
                df_pass = pd.DataFrame({
                    "Residuals_mps": residuals_per_pass[pass_index]
                })

                df_pass.to_excel(
                    writer,
                    sheet_name=f"Residuals_Pass_{pass_index+1}",
                    index=False
                )

            df_hist = pd.DataFrame({
                "Residuals_all": np.concatenate(residuals_per_pass)
            })

            df_hist.to_excel(
                writer,
                sheet_name="Residuals_Histogram",
                index=False
            )

            df_xyz = pd.DataFrame({
                "Time_s": time_arc,

                "Diff_X_km": (TLE_orbit[:, 1] - estimated_orbit[:, 1]) / 1000,
                "Diff_Y_km": (TLE_orbit[:, 2] - estimated_orbit[:, 2]) / 1000,
                "Diff_Z_km": (TLE_orbit[:, 3] - estimated_orbit[:, 3]) / 1000,

                "Diff_VX_kms": (TLE_orbit[:, 4] - estimated_orbit[:, 4]) / 1000,
                "Diff_VY_kms": (TLE_orbit[:, 5] - estimated_orbit[:, 5]) / 1000,
                "Diff_VZ_kms": (TLE_orbit[:, 6] - estimated_orbit[:, 6]) / 1000
            })
            # Backup CSV
            df_xyz.to_csv(
                chunk_csv_folder / "XYZ_Differences_backup.csv",
                index=False
            )

            df_xyz.to_excel(
                writer,
                sheet_name="XYZ_Differences",
                index=False
            )

            df_range = pd.DataFrame({
                "Time_s": time_arc,
                "Residual_Range_km": (range_TLE_arc - range_estimated_arc) / 1000,
                "Residual_Vmag_kms": (Vmag_TLE_arc - Vmag_estimated_arc) / 1000
            })

            df_range.to_csv(
                chunk_csv_folder / "Range_Vmag.csv",
                index=False
            )
            df_range.to_excel(
                writer,
                sheet_name="Range_Vmag",
                index=False
            )

            df_distance = pd.DataFrame({
                "Time_s": time_arc,
                "Distance_km": distance_orbits_arc / 1000
            })

            df_distance.to_csv(
                chunk_csv_folder / "Orbit_Distance.csv",
                index=False
            )

            df_distance.to_excel(
                writer,
                sheet_name="Orbit_Distance",
                index=False
            )

            df_rsw = pd.DataFrame({
                "Time_s": rsw_difference_current[:, 0] - rsw_difference_current[0, 0],

                "Diff_R_km": rsw_difference_current[:, 1] / 1000,
                "Diff_S_km": rsw_difference_current[:, 2] / 1000,
                "Diff_W_km": rsw_difference_current[:, 3] / 1000,

                "Diff_Vr_kms": rsw_difference_current[:, 4] / 1000,
                "Diff_Vs_kms": rsw_difference_current[:, 5] / 1000,
                "Diff_Vw_kms": rsw_difference_current[:, 6] / 1000
            })

            df_rsw.to_csv(
                chunk_csv_folder / "RSW_Differences.csv",
                index=False
            )

            df_rsw.to_excel(
                writer,
                sheet_name="RSW_Differences",
                index=False
            )

            df_kepler = pd.DataFrame({
                "Time_s": keplerian_difference_current[:, 0] - keplerian_difference_current[0, 0],

                "Diff_a_m": keplerian_difference_current[:, 1],
                "Diff_e": keplerian_difference_current[:, 2],
                "Diff_i_deg": np.rad2deg(keplerian_difference_current[:, 3]),
                "Diff_omega_deg": np.rad2deg(keplerian_difference_current[:, 4]),
                "Diff_RAAN_deg": np.rad2deg(keplerian_difference_current[:, 5]),
                "Diff_theta_deg": np.rad2deg(keplerian_difference_current[:, 6])
            })

            df_kepler.to_csv(
                chunk_csv_folder / "Keplerian_Diff.csv",
                index=False
            )

            df_kepler.to_excel(
                writer,
                sheet_name="Keplerian_Diff",
                index=False
            )

        print(f"ARC {arc_index+1} Excel saved at: {arc_excel_path}")

# Concatenate all chunks
all_time = np.concatenate(all_time)
#plot_time = np.arange(len(all_time)) * 10.0
plot_time = all_time - all_time[0]

all_tle_x = np.concatenate(all_tle_x)
all_tle_y = np.concatenate(all_tle_y)
all_tle_z = np.concatenate(all_tle_z)

all_estimated_x = np.concatenate(all_estimated_x)
all_estimated_y = np.concatenate(all_estimated_y)
all_estimated_z = np.concatenate(all_estimated_z)

all_diff_x = np.concatenate(all_diff_x)
all_diff_y = np.concatenate(all_diff_y)
all_diff_z = np.concatenate(all_diff_z)

all_diff_vx = np.concatenate(all_diff_vx)
all_diff_vy = np.concatenate(all_diff_vy)
all_diff_vz = np.concatenate(all_diff_vz)

all_range_residual = np.concatenate(all_range_residual)
all_vmag_residual = np.concatenate(all_vmag_residual)
all_distance_orbits = np.concatenate(all_distance_orbits)

all_rsw = np.vstack(all_rsw)
all_keplerian = np.vstack(all_keplerian)


# Sort by time across all chunks
sort_idx = np.argsort(all_time)
all_keplerian = all_keplerian[sort_idx]
all_time = all_time[sort_idx]
#plot_time = np.arange(len(all_time)) * 10.0
plot_time = all_time - all_time[0]
all_rsw = all_rsw[sort_idx]

all_diff_x = all_diff_x[sort_idx]
all_diff_y = all_diff_y[sort_idx]
all_diff_z = all_diff_z[sort_idx]

all_diff_vx = all_diff_vx[sort_idx]
all_diff_vy = all_diff_vy[sort_idx]
all_diff_vz = all_diff_vz[sort_idx]

all_range_residual = all_range_residual[sort_idx]
all_vmag_residual = all_vmag_residual[sort_idx]
all_distance_orbits = all_distance_orbits[sort_idx]

all_tle_x = all_tle_x[sort_idx]
all_tle_y = all_tle_y[sort_idx]
all_tle_z = all_tle_z[sort_idx]

all_estimated_x = all_estimated_x[sort_idx]
all_estimated_y = all_estimated_y[sort_idx]
all_estimated_z = all_estimated_z[sort_idx]

# Plot differences between the TLE and estimated orbits

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(3, 2, 1)
ax.plot(plot_time, all_diff_x, color='blue', linestyle='-.')
ax.set_ylabel('Diff X [km]')
ax.grid()

ax = fig.add_subplot(3, 2, 3)
ax.plot(plot_time, all_diff_y, color='blue', linestyle='-.')
ax.set_ylabel('Diff Y [km]')
ax.grid()

ax = fig.add_subplot(3, 2, 5)
ax.plot(plot_time, all_diff_z, color='blue', linestyle='-.')
ax.set_ylabel('Diff Z [km]')
ax.set_xlabel('Time [s]')
ax.grid()

ax = fig.add_subplot(3, 2, 2)
ax.plot(plot_time, all_diff_vx, color='blue', linestyle='-.')
ax.set_ylabel('Diff VX [km/s]')
ax.grid()

ax = fig.add_subplot(3, 2, 4)
ax.plot(plot_time, all_diff_vy, color='blue', linestyle='-.')
ax.set_ylabel('Diff VY [km/s]')
ax.grid()

ax = fig.add_subplot(3, 2, 6)
ax.plot(plot_time, all_diff_vz, color='blue', linestyle='-.')
ax.set_ylabel('Diff VZ [km/s]')
ax.set_xlabel('Time [s]')
ax.grid()

fig.tight_layout()
fig.savefig(
    results_folder / "xyz_position_velocity_differences_all_chunks.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# Plot propagated (estimated and TLE) orbits
fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot(111, projection='3d')
ax.set_title(f'Satellite trajectory around Earth')
ax.plot(all_tle_x, all_tle_y, all_tle_z, label='TLE orbit', linestyle='-.')
ax.plot(all_estimated_x, all_estimated_y, all_estimated_z, label='estimated orbit', linestyle='-.')
ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue')
ax.legend()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
fig.savefig(
    results_folder / "tle_vs_estimated_orbit_3d.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# Plot only TLE orbit
fig = plt.figure(figsize=(6,6), dpi=125)
ax = fig.add_subplot(111, projection='3d')

ax.set_title('TLE Orbit')
ax.plot(all_tle_x, all_tle_y, all_tle_z,
        label='TLE orbit', linestyle='-.')
ax.scatter(0.0, 0.0, 0.0,
           label='Earth', marker='o', color='blue')

ax.legend()
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')

fig.savefig(
    results_folder / "tle_orbit_3d.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)
# Plot difference in distance and velocity magnitude between TLE and estimated orbits - ALL CHUNKS

fig = plt.figure()

ax = fig.add_subplot(2, 1, 1)
ax.plot(plot_time, all_range_residual, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Residuals range [km]')
ax.set_title('Range residuals - all chunks')
plt.grid()

ax = fig.add_subplot(2, 1, 2)
ax.plot(plot_time, all_vmag_residual, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Residuals Vmag [km/s]')
plt.grid()

fig.tight_layout()
fig.savefig(
    results_folder / "range_velocity_magnitude_residuals_all_chunks.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# Plot distance between the TLE and estimated orbits - ALL CHUNKS

fig = plt.figure()
fig.tight_layout()
fig.subplots_adjust(hspace=0.3)

ax = fig.add_subplot(1, 1, 1)
ax.plot(plot_time, all_distance_orbits, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Distance between orbits [km]')
ax.set_title('Distance between orbits - all chunks')
plt.grid()

fig.savefig(
    results_folder / "orbit_distance_all_chunks.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)
# Plot differences between the TLE and estimated orbits in RSW - ALL CHUNKS

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(3, 2, 1)
ax.plot(plot_time, all_rsw[:, 1] / 1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Diff R [km]')
ax.set_title('RSW differences - all chunks')
plt.grid()

ax = fig.add_subplot(3, 2, 3)
ax.plot(plot_time, all_rsw[:, 2] / 1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Diff S [km]')
plt.grid()

ax = fig.add_subplot(3, 2, 5)
ax.plot(plot_time, all_rsw[:, 3] / 1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Diff W [km]')
plt.grid()

ax = fig.add_subplot(3, 2, 2)
ax.plot(plot_time, all_rsw[:, 4] / 1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Diff Vr [km/s]')
plt.grid()

ax = fig.add_subplot(3, 2, 4)
ax.plot(plot_time, all_rsw[:, 5] / 1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Diff Vs [km/s]')
plt.grid()

ax = fig.add_subplot(3, 2, 6)
ax.plot(plot_time, all_rsw[:, 6] / 1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Diff Vw [km/s]')
plt.grid()

fig.tight_layout()
fig.savefig(
    results_folder / "rsw_differences_all_chunks.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# Plot differences between the TLE and estimated orbits in Keplerian elements - ALL CHUNKS

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(3, 2, 1)
ax.plot(plot_time, all_keplerian[:, 1] / 1000, color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$\Delta a$ [km]')
ax.set_title('Keplerian differences - all chunks')
plt.grid()

ax = fig.add_subplot(3, 2, 3)
ax.plot(plot_time, all_keplerian[:, 2], color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$\Delta e$ [-]')
plt.grid()

ax = fig.add_subplot(3, 2, 5)
ax.plot(plot_time, np.degrees(all_keplerian[:, 3]), color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$\Delta i$ [deg]')
plt.grid()

ax = fig.add_subplot(3, 2, 2)
ax.plot(plot_time, np.degrees(all_keplerian[:, 4]), color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$\Delta \omega$ [deg]')
plt.grid()

ax = fig.add_subplot(3, 2, 4)
ax.plot(plot_time, np.degrees(all_keplerian[:, 5]), color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$\Delta \Omega$ [deg]')
plt.grid()

ax = fig.add_subplot(3, 2, 6)
ax.plot(plot_time, np.degrees(all_keplerian[:, 6]), color='blue', linestyle='-.')
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'$\Delta \theta$ [deg]')
plt.grid()

fig.tight_layout()
fig.savefig(
    results_folder / "keplerian_differences_all_chunks.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# ============================================================
# SAVE ALL COMBINED GRAPH DATA TO EXCEL
# ============================================================
EXCEL_MAX_ROWS = 1_000_000

def write_large_df_to_excel(writer, df, base_sheet_name, index=False):
    for part, start in enumerate(range(0, len(df), EXCEL_MAX_ROWS), start=1):
        end = start + EXCEL_MAX_ROWS

        df.iloc[start:end].to_excel(
            writer,
            sheet_name=f"{base_sheet_name}_{part}"[:31],
            index=index
        )
excel_path = results_folder / "comparison_results_all_chunks.xlsx"

# Collect all residuals from all chunks
all_residuals_combined = []

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

    # --------------------------------------------------------
    # 1. Residuals per chunk and pass
    # --------------------------------------------------------
    for result in valid_validation_results:

        chunk_number = result["chunk_number"]
        residuals_per_pass_chunk = result["residuals_per_pass"]

        for pass_index in range(len(residuals_per_pass_chunk)):

            residual_current = residuals_per_pass_chunk[pass_index]
            all_residuals_combined.append(residual_current)

            df_pass = pd.DataFrame({
                "Residuals_mps": residual_current
            })

            sheet_name = f"Residuals_Pass_{len(all_residuals_combined)}"
            sheet_name = sheet_name[:31]

            df_pass.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )


    # --------------------------------------------------------
    # 2. Residual histogram data for all chunks
    # --------------------------------------------------------
    df_hist = pd.DataFrame({
        "Residuals_all_chunks": np.concatenate(all_residuals_combined)
    })

    df_hist.to_excel(
        writer,
        sheet_name="Residuals_Histogram",
        index=False
    )
    combined_csv_folder = results_folder / "combined_csv"
    combined_csv_folder.mkdir(exist_ok=True)
    # --------------------------------------------------------
    # 3. XYZ position/velocity differences - all chunks
    # --------------------------------------------------------
    df_xyz = pd.DataFrame({
        "Time_s": plot_time,

        "Diff_X_km": all_diff_x,
        "Diff_Y_km": all_diff_y,
        "Diff_Z_km": all_diff_z,

        "Diff_VX_kms": all_diff_vx,
        "Diff_VY_kms": all_diff_vy,
        "Diff_VZ_kms": all_diff_vz
    })

    df_xyz.to_csv(
        combined_csv_folder / "XYZ_Differences_All_Chunks.csv",
        index=False
    )

    write_large_df_to_excel(
        writer,
        df_xyz,
        "XYZ_Differences",
        index=False
    )

    # --------------------------------------------------------
    # 4. Range and velocity magnitude residuals - all chunks
    # --------------------------------------------------------
    df_range = pd.DataFrame({
        "Time_s": plot_time,
        "Residual_Range_km": all_range_residual,
        "Residual_Vmag_kms": all_vmag_residual
    })

    df_range.to_csv(
        combined_csv_folder / "Range_Vmag_All_Chunks.csv",
        index=False
    )

    write_large_df_to_excel(
        writer,
        df_range,
        "Range_Vmag",
        index=False
    )

    # --------------------------------------------------------
    # 5. Distance between orbits - all chunks
    # --------------------------------------------------------
    df_distance = pd.DataFrame({
        "Time_s": plot_time,
        "Distance_km": all_distance_orbits
    })

    df_distance.to_csv(
        combined_csv_folder / "Orbit_Distance_All_Chunks.csv",
        index=False
    )

    write_large_df_to_excel(
        writer,
        df_distance,
        "Orbit_Distance",
        index=False
    )

    # --------------------------------------------------------
    # 6. RSW differences - all chunks
    # --------------------------------------------------------
    df_rsw = pd.DataFrame({
        "Time_s": plot_time,

        "Diff_R_km": all_rsw[:, 1] / 1000,
        "Diff_S_km": all_rsw[:, 2] / 1000,
        "Diff_W_km": all_rsw[:, 3] / 1000,

        "Diff_Vr_kms": all_rsw[:, 4] / 1000,
        "Diff_Vs_kms": all_rsw[:, 5] / 1000,
        "Diff_Vw_kms": all_rsw[:, 6] / 1000
    })

    df_rsw.to_csv(
        combined_csv_folder / "RSW_Differences_All_Chunks.csv",
        index=False
    )

    write_large_df_to_excel(
        writer,
        df_rsw,
        "RSW_Differences",
        index=False
    )

    # --------------------------------------------------------
    # 7. Keplerian differences - all chunks
    # --------------------------------------------------------
    df_kepler = pd.DataFrame({
        "Time_s": plot_time,

        "Diff_a_m": all_keplerian[:, 1],
        "Diff_e": all_keplerian[:, 2],
        "Diff_i_deg": np.rad2deg(all_keplerian[:, 3]),
        "Diff_omega_deg": np.rad2deg(all_keplerian[:, 4]),
        "Diff_RAAN_deg": np.rad2deg(all_keplerian[:, 5]),
        "Diff_theta_deg": np.rad2deg(all_keplerian[:, 6])
    })

    df_kepler.to_csv(
        combined_csv_folder / "Keplerian_Diff_All_Chunks.csv",
        index=False
    )

    write_large_df_to_excel(
        writer,
        df_kepler,
        "Keplerian_Diff",
        index=False
    )
    # --------------------------------------------------------
    # 8. Keplerian state, drag and bias estimates - all chunks
    # --------------------------------------------------------

    kepler_state_rows = []
    drag_rows = []
    bias_rows = []

    for result in valid_validation_results:

        chunk_number = result["chunk_number"]
        updated_parameters = result["updated_parameters"]
        nb_arcs = result["nb_arcs"]
        chunk_indices = result["chunk_indices"]
        number_of_passes = len(chunk_indices)

        arc_wise_initial_states = result["arc_wise_initial_states"]

        # ----------------------------
        # Keplerian states
        # ----------------------------
        for arc_index in range(nb_arcs):

            estimated_cartesian = updated_parameters[6 * arc_index:6 * (arc_index + 1)]
            tle_cartesian = arc_wise_initial_states[arc_index]

            state_keplerian = element_conversion.cartesian_to_keplerian(
                estimated_cartesian,
                gravitational_parameter
            )

            tle_keplerian = element_conversion.cartesian_to_keplerian(
                tle_cartesian,
                gravitational_parameter
            )

            rows = [
                ("Semi-major axis [km]", state_keplerian[0] / 1000, tle_keplerian[0] / 1000),
                ("Eccentricity", state_keplerian[1], tle_keplerian[1]),
                ("Inclination [deg]", np.rad2deg(state_keplerian[2]), np.rad2deg(tle_keplerian[2])),
                ("Argument of Perigee [deg]", np.rad2deg(state_keplerian[3]), np.rad2deg(tle_keplerian[3])),
                ("RAAN [deg]", np.rad2deg(state_keplerian[4]), np.rad2deg(tle_keplerian[4])),
                ("True anomaly [deg]", np.rad2deg(state_keplerian[5]), np.rad2deg(tle_keplerian[5])),
                (
                    "True longitude [deg]",
                    np.mod(np.rad2deg(state_keplerian[5]) + np.rad2deg(state_keplerian[3]), 360),
                    np.mod(np.rad2deg(tle_keplerian[5]) + np.rad2deg(tle_keplerian[3]), 360),
                ),
                ("Altitude [km]", state_keplerian[0] / 1000 - 6371.360, tle_keplerian[0] / 1000 - 6371.360),
            ]

            for parameter, estimated, tle in rows:
                kepler_state_rows.append({
                    "Chunk": chunk_number,
                    "Arc": arc_index + 1,
                    "Parameter": parameter,
                    "Estimated": estimated,
                    "TLE": tle
                })

        # ----------------------------
        # Bias estimates
        # ----------------------------
        constant_bias_start = 6 * nb_arcs
        constant_bias_end = constant_bias_start + number_of_passes

        linear_bias_start = constant_bias_end
        linear_bias_end = linear_bias_start + number_of_passes

        constant_biases = updated_parameters[constant_bias_start:constant_bias_end]
        linear_biases = updated_parameters[linear_bias_start:linear_bias_end]

        for i in range(number_of_passes):
            bias_rows.append({
                "Chunk": chunk_number,
                "Local_Pass_Index": i,
                "Global_Pass_Index": chunk_indices[i],
                "Absolute_Constant_Bias": constant_biases[i] if i < len(constant_biases) else "",
                "Linear_Constant_Bias": linear_biases[i] if i < len(linear_biases) else ""
            })

        # ----------------------------
        # Drag estimates
        # ----------------------------
        drag_type = result["parameters_list"]["drag_coefficient"]["type"]
        drag_estimates = result.get("drag_estimates", [])
        drag_start_times = result.get("drag_start_times", [])

        for i, value in enumerate(drag_estimates):
            drag_rows.append({
                "Chunk": chunk_number,
                "Local_Index": i,
                "Drag_Type": drag_type,
                "Drag_Epoch": drag_start_times[i] if i < len(drag_start_times) else "",
                "Estimated_Cd": value
            })
    # --------------------------------------------------------
    # Estimated states, biases and drag estimates
    # --------------------------------------------------------

    df_kepler_states = pd.DataFrame(kepler_state_rows)
    df_biases = pd.DataFrame(bias_rows)
    df_drag = pd.DataFrame(drag_rows)

    df_kepler_states.to_excel(
        writer,
        sheet_name="Estimated_States",
        index=False
    )

    df_biases.to_excel(
        writer,
        sheet_name="Bias_Estimates",
        index=False
    )

    df_drag.to_excel(
        writer,
        sheet_name="Drag_Estimates",
        index=False
    )

    # --------------------------------------------------------
    # Separate sheets for each Keplerian parameter
    # With arc-to-arc change and TLE comparison
    # --------------------------------------------------------

    kepler_sheet_names = {
        "Semi-major axis [km]": "SMA",
        "Eccentricity": "Eccentricity",
        "Inclination [deg]": "Inclination",
        "Argument of Perigee [deg]": "Arg_Perigee",
        "RAAN [deg]": "RAAN",
        "True anomaly [deg]": "True_Anomaly",
        "True longitude [deg]": "True_Longitude",
        "Altitude [km]": "Altitude"
    }

    for parameter_name, sheet_name in kepler_sheet_names.items():

        df_param = df_kepler_states[
            df_kepler_states["Parameter"] == parameter_name
        ].copy()

        df_param = df_param.sort_values(["Chunk", "Arc"]).reset_index(drop=True)

        # Compare each arc with the next arc inside the same chunk
        df_param["Next_Estimated"] = df_param.groupby("Chunk")["Estimated"].shift(-1)

        df_param["Delta_to_Next"] = (
            df_param["Next_Estimated"] - df_param["Estimated"]
        )

        df_param["Percent_Delta_to_Next"] = np.where(
            df_param["Estimated"] != 0,
            100.0 * df_param["Delta_to_Next"] / df_param["Estimated"],
            np.nan
        )

        # Compare estimated value with TLE value
        df_param["Delta_vs_TLE"] = df_param["Estimated"] - df_param["TLE"]

        df_param["Percent_Error_vs_TLE"] = np.where(
            df_param["TLE"] != 0,
            100.0 * df_param["Delta_vs_TLE"] / df_param["TLE"],
            np.nan
        )

        df_param = df_param[[
            "Chunk",
            "Arc",
            "Estimated",
            "TLE",
            "Next_Estimated",
            "Delta_to_Next",
            "Percent_Delta_to_Next",
            "Delta_vs_TLE",
            "Percent_Error_vs_TLE"
        ]]

        df_param.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )

        for parameter_name, sheet_name in kepler_sheet_names.items():

            df_param = df_kepler_states[
                df_kepler_states["Parameter"] == parameter_name
            ].copy()

            df_param = df_param.sort_values(["Chunk", "Arc"]).reset_index(drop=True)

            # Compare each arc with the next arc inside the same chunk
            df_param["Next_Estimated"] = df_param.groupby("Chunk")["Estimated"].shift(-1)

            df_param["Delta_to_Next"] = (
                df_param["Next_Estimated"] - df_param["Estimated"]
            )

            df_param["Percent_Delta_to_Next"] = np.where(
                df_param["Estimated"] != 0,
                100.0 * df_param["Delta_to_Next"] / df_param["Estimated"],
                np.nan
            )

            # Compare estimated value with TLE value
            df_param["Delta_vs_TLE"] = df_param["Estimated"] - df_param["TLE"]

            df_param["Percent_Error_vs_TLE"] = np.where(
                df_param["TLE"] != 0,
                100.0 * df_param["Delta_vs_TLE"] / df_param["TLE"],
                np.nan
            )

            df_param = df_param[[
                "Chunk",
                "Arc",
                "Estimated",
                "TLE",
                "Next_Estimated",
                "Delta_to_Next",
                "Percent_Delta_to_Next",
                "Delta_vs_TLE",
                "Percent_Error_vs_TLE"
            ]]

            df_param.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )


print(f"\nCombined Excel file saved at:\n{excel_path}")
print(f"\nResults folder:\n{results_folder}")

word_path = results_folder / "arc_results_summary_all_chunks.docx"

word_doc.add_heading("Drag Coefficient Estimates - All Chunks", level=1)

for result in valid_validation_results:

    chunk_number = result["chunk_number"]
    updated_parameters = result["updated_parameters"]
    nb_arcs = result["nb_arcs"]
    number_of_passes = result["number_of_passes"]
    chunk_indices = result["chunk_indices"]

    drag_type = parameters_list["drag_coefficient"]["type"]

    if drag_type == "global":
        n_drag = 1
    elif drag_type == "per_arc":
        n_drag = nb_arcs
    elif drag_type == "per_day":
        n_drag = estimate_days
    elif drag_type == "per_pass":
        n_drag = number_of_passes
    else:
        raise ValueError(f"Unknown drag coefficient type: {drag_type}")

    state_parameter_count = 6 * nb_arcs
    drag_start_index = state_parameter_count

    if drag_type != "global":

        if parameters_list["constant_absolute_bias"]["estimate"]:
            interval = Doppler_models["constant_absolute_bias"]["time_interval"]

            if interval == "global":
                drag_start_index += 1
            elif interval == "per_arc":
                drag_start_index += nb_arcs
            elif interval == "per_pass":
                drag_start_index += number_of_passes

        if parameters_list["linear_absolute_bias"]["estimate"]:
            interval = Doppler_models["linear_absolute_bias"]["time_interval"]

            if interval == "global":
                drag_start_index += 1
            elif interval == "per_arc":
                drag_start_index += nb_arcs
            elif interval == "per_pass":
                drag_start_index += number_of_passes

    drag_estimates = updated_parameters[
        drag_start_index : drag_start_index + n_drag
    ]

    word_doc.add_heading(f"Chunk {chunk_number}", level=2)

    drag_table = word_doc.add_table(rows=1, cols=4)
    drag_table.style = "Table Grid"

    hdr = drag_table.rows[0].cells
    hdr[0].text = "Local Index"
    hdr[1].text = "Parameter Index"
    hdr[2].text = "Definition"
    hdr[3].text = "Estimated Cd"

    for i, value in enumerate(drag_estimates):
        cells = drag_table.add_row().cells
        cells[0].text = str(i)
        cells[1].text = str(drag_start_index + i)

        if drag_type == "global":
            cells[2].text = f"Chunk {chunk_number} global Cd"
        elif drag_type == "per_arc":
            cells[2].text = f"Chunk {chunk_number}, Arc {i + 1}"
        elif drag_type == "per_day":
            cells[2].text = f"Chunk {chunk_number}, Day {i + 1}"
        elif drag_type == "per_pass":
            cells[2].text = f"Chunk {chunk_number}, Pass {chunk_indices[i]}"

        cells[3].text = f"{value:.10g}"


word_doc.add_heading("Bias Estimates - All Chunks", level=1)

for result in valid_validation_results:

    chunk_number = result["chunk_number"]
    updated_parameters = result["updated_parameters"]
    nb_arcs = result["nb_arcs"]
    number_of_passes = result["number_of_passes"]
    chunk_indices = result["chunk_indices"]
    parameters_list = result["parameters_list"]
    Doppler_models = result["Doppler_models"]

    drag_type = parameters_list["drag_coefficient"]["type"]
    constant_bias_start = 6 * nb_arcs
    constant_bias_end = constant_bias_start + number_of_passes

    linear_bias_start = constant_bias_end
    linear_bias_end = linear_bias_start + number_of_passes

    constant_biases = updated_parameters[constant_bias_start:constant_bias_end]
    linear_biases = updated_parameters[linear_bias_start:linear_bias_end]

    word_doc.add_heading(f"Chunk {chunk_number}", level=2)

    bias_table = word_doc.add_table(rows=1, cols=3)
    bias_table.style = "Table Grid"

    hdr = bias_table.rows[0].cells
    hdr[0].text = "Global Pass Index"
    hdr[1].text = "Absolute Constant Bias"
    hdr[2].text = "Linear Constant Bias"

    for i in range(number_of_passes):
        cells = bias_table.add_row().cells
        cells[0].text = str(chunk_indices[i])
        cells[1].text = str(constant_biases[i])

        if i < len(linear_biases):
            cells[2].text = str(linear_biases[i])
        else:
            cells[2].text = ""

word_doc.save(word_path)
shutil.copy2(
    "terminal_output.txt",
    results_folder / "terminal_output.txt"
)

print(f"Word ARC summary saved at:\n{word_path}")