import os
import sys
import argparse
import copy
import time
from pathlib import Path
from datetime import datetime
import random
from itertools import combinations
from collections import defaultdict
import subprocess
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle

import io
from contextlib import redirect_stdout, contextmanager

from boolnetlab import BN_Realisation as BN # The core class for Boolean network analysis
from boolnetlab import state2bin, int2bin, bin2state, parse_cabean_attractors


# ==== Settings ====

ATTRACTOR_TIMEOUT = 30 * 60

CABEAN_PATH = os.path.join(Path.home(), "cabean")

# ==================


@contextmanager
def timer():
    res = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield res
    finally:
        res["elapsed"] = time.perf_counter() - start


def match(state: str, state_template: str):

    assert len(state) == len(state_template)

    for i in range(len(state)):
        if state_template[i] != '-':
            if state[i] != state_template[i]:
                return False

    return True


def run_CABEAN_seq(model_file_path: str, verbose: bool=False):

    if verbose:
        print(f"Running CABEAN on {model_file_path} ...")

    cabean_path = CABEAN_PATH
        
    try:
        args = cabean_path + " -compositional 2 " + model_file_path
        output = subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, shell=True, timeout=ATTRACTOR_TIMEOUT)
        return((output, 0))
    except subprocess.CalledProcessError as e:
        print(f"Error occured with code {e.returncode}!")
        partial_output = e.output
        return((partial_output, e.returncode))
    except subprocess.TimeoutExpired as e:
        print(f"Attractor computation timed out after {e.timeout} seconds.")
        partial_output = e.output
        return((partial_output, None))


def computeAttractorsWithCabean(bn, log_folder: str, log_filename: str, verbose: bool=False):

    ### === Run CABEAN for all possible environmental conditions ===
    
    failed_env_conditions = []
    total_num_attractors = 0

    input_nodes = bn.getInputNodeNames()
    num_input_nodes = len(input_nodes)
    if verbose:
        print(f"Input nodes: {input_nodes}")
        print(f"Number of input nodes: {num_input_nodes}")

    # Preparing models
    for env_condition in range(2**num_input_nodes):

        if verbose:
            print(f"Making model file for environmental condition {env_condition} ... ", flush=True)
         
        env_condition_str = '\t'
        if num_input_nodes != 0:
            for input_node, node_value in zip(input_nodes, bin2state(int2bin(env_condition,num_input_nodes))):
                env_condition_str = env_condition_str + 'M.' + input_node
                if node_value == 1:
                    env_condition_str += '=true and '
                else:
                    env_condition_str += '=false and '
            env_condition_str = env_condition_str[:-5]
        else:
            env_condition_str += f"M.{bn.node_names[0]}=false or M.{bn.node_names[0]}=true"
        env_condition_str += ';'
        
        model_file_path = os.path.join(log_folder, f"{log_filename}_env_cond_{env_condition}.ispl")
        bn.save_ispl(model_file_path, env_condition_str, verbose=False)

    # Running CABEAN for each environmental condition
    results = [run_CABEAN_seq(os.path.join(log_folder, f"{log_filename}_env_cond_{env_condition}.ispl")) for env_condition in range(2**num_input_nodes)]

    attractors = dict()
    
    attractor_key_index, attractor = 0, []
    total_cabean_runtime = 0.0
    for env_condition, output_tuple in enumerate(results):

        output = output_tuple[0]
        exit_code = output_tuple[1]

        if verbose:
            print(output, flush=True)
        with open(os.path.join(log_folder, f"cabean_attractors_{env_condition}.txt"), 'w') as file:
            file.write(str(output))

        if exit_code is None:
            # Timeout took place
            failed_env_conditions.append(env_condition)
            if verbose:
                print(f"WARNING: Attractor computation timed out for environmental condition {env_condition}!")
        elif exit_code == 0:
            parts = output.split("number of attractors =")
            num_attractors = int(parts[1].split('\n')[0])
            total_num_attractors += num_attractors
            # print(f"\tNumber of attractors: {num_attractors}", flush=True)

            # Extract individual attractors
            for line in output.split('\n'):
                if len(line) > 2:
                    if "find attractor #" in line:
                        if len(attractor) > 0: # Check whether this is not the first attractor
                            attractors['A' + str(attractor_key_index)] = set(attractor)
                            attractor_key_index += 1
                            attractor = []
                    elif line[0] == '0' or line[0] == '1' or line[0] == '-':
                        i = 0
                        state_template = ''
                        while i < len(line[:-4]):
                            state_template += line[i]
                            i += 2
                        attractor.append(state_template)
                    elif "time for attractor detection" in line:
                        print(f"Cabean runtime: {float(line.split('=')[1].split(' ')[0])}s")
                        total_cabean_runtime += float(line.split('=')[1].split(' ')[0])

            if len(attractor) > 0:
                attractors['A' + str(attractor_key_index)] = set(attractor)
                attractor_key_index += 1
                attractor = [] # Prepare for next attractor or free memory
        else:
            failed_env_conditions.append(env_condition)
            if verbose:
                print(f"Attractor computation failed on model with environmental condition {env_condition}")
                print(f"  Error message: {output}")
                print(f"  Exit code: {exit_code}")


    msg = str(f"Total number of attractors found for all environmental conditions: {total_num_attractors}")
    print(msg, flush=True)
    msg = str(f"Attractor computation failed for the following environmental conditions: {failed_env_conditions}")
    print(msg, flush=True)

    ever_failed = (len(failed_env_conditions) != 0)

    return attractors, ever_failed, total_cabean_runtime


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model-file', type=str, help='ISPL file with the BN model.')
    parser.add_argument('--log-folder', type=str, help='Folder where auxiliary outputs and results will be saved.')
    parser.add_argument('--filename-stem', type=str)
    parser.add_argument('--bn-size', type=int, help='Size of the BN to be generated (optional); used only if --model-file is not specified.')
    parser.add_argument('--max-parents', type=int)
    parser.add_argument('--cabean-file', type=str, help='Optional Cabean output file with a list of attractors.')
    parser.add_argument('--no-cabean', action='store_false', dest='run_cabean', help='Disable computing exact attractors using cabean.')
    parser.add_argument('--num-mc-repetitions', type=int, default=1, help='Number of times the Monte Carlo attractor search is run.')
    parser.add_argument('--verbose', action='store_true', dest='verbose', help='Enable verbose mode.')

    args = parser.parse_args()

    if args.log_folder is not None:
        if not os.path.exists(args.log_folder):
            os.mkdir(args.log_folder)

    if args.model_file is not None:
        bn = BN.load_ispl(args.model_file)

    elif args.bn_size is not None:

        if args.max_parents is not None:
            max_parents = args.max_parents
        else:
            print(f"Option --max-parents not specified. Setting maximum number of parents to 2.")
            max_parents = 2
    
        bn = BN.generate_random_bn(num_nodes=args.bn_size, max_parent_nodes=max_parents, min_parent_nodes=1, allow_self_loops=True, allow_input_nodes=True, mode='asynchronous', verbose=True)
        bn.save_ispl(os.path.join(args.log_folder, args.filename_stem + 'bn.ispl'))

    else:
        raise ValueError("Model file (--model-file) or BN size (--bn_size) needs to be specified!")

    # === Running exact attractor detection using Cabean or reading attractors from Cabean output file ===
    if args.run_cabean and args.cabean_file is None:
        attractor_states, ever_failed, total_cabean_runtime = computeAttractorsWithCabean(bn, log_folder=args.log_folder, log_filename=args.filename_stem)

        if ever_failed:
            print(f"Attractor computation failed!")
            return None
    elif args.cabean_file is not None:
        attractor_states, total_cabean_runtime = parse_cabean_attractors(args.cabean_file)

    attractor_sizes = []
    for a_key in attractor_states.keys():
        attractor_sizes.append(sum([2**state_template.count("-") for state_template in attractor_states[a_key]]))

    num_fp_attr = sum(np.array(attractor_sizes) == 1)
    
    print(f"Exact attractors:\n{attractor_states}")
    print(f"Exact attractor sizes: {attractor_sizes}")


    # === Monte Carlo attractor search ===
    pseudoattractor_states_list = []
    fixed_point_pa_states_list = []
    mc_runtime_list = []

    for mc_rep in range(args.num_mc_repetitions):

        with timer() as t:
            pa_states = bn.getAttractorsMonteCarlo(n_parallel = min(3200, 100*bn.num_nodes, 2**bn.num_nodes-1),
                                                   burn_in_len = 10000,
                                                   history_len = 2000,
                                                   threshold = 0.15)

        # Just for testing code correctness - to be removed later.
        for pas in pa_states:
            assert len(pas) == bn.num_nodes

        pseudoattractor_states = set([state2bin(state) for state in pa_states])
        fixed_point_pa_states = set([state2bin(state) for state in pa_states if bn.is_fixpoint_attractor(state)])

        pseudoattractor_states_list.append(pseudoattractor_states)
        fixed_point_pa_states_list.append(fixed_point_pa_states)
        mc_runtime_list.append(t["elapsed"])

        if not args.run_cabean:

            print(f"----------------------------------------------------------------")
            print(f"\tResults of Monte Carlo attractor detection (run {mc_rep + 1})")
            print(f"----------------------------------------------------------------")
            # print(f"Pseudo-attractor states: {pseudoattractor_states}")
            print(f"Number of pseudo-attractor states: {len(pseudoattractor_states_list[mc_rep])}")
            print(f"Number of fixed-point pseudo-attractor states: {len(fixed_point_pa_states_list[mc_rep])}")


    ### ==== Computing statistics ====
    if args.run_cabean or args.cabean_file is not None:

        missed_attractor_number_list = []
        false_positive_number_list = []
        found_attractor_number_list = []
        fixed_point_pa_state_number_list = []

        for mc_rep in range(args.num_mc_repetitions):

            attractor_states_c = copy.deepcopy(attractor_states)

            discovered_attractors = set()

            false_positives = set()

            state_templates_coverage = dict()
            for a_key in attractor_states_c.keys():
                for attr_state_template in list(attractor_states_c[a_key]):
                    if '-' in attr_state_template:
                        state_templates_coverage[attr_state_template] = [0, 2**attr_state_template.count("-")]
                    else:
                        state_templates_coverage[attr_state_template] = [0, 1]

            for pa_state in list(pseudoattractor_states_list[mc_rep]):
                true_pa_state = False
                for a_key in attractor_states.keys():
                    for attr_state_template in list(attractor_states[a_key]):
                        if match(pa_state, attr_state_template):
                            true_pa_state = True
                            discovered_attractors.add(a_key)
                            state_templates_coverage[attr_state_template][0] += 1
                            if '-' in attr_state_template:
                                attractor_states_c[a_key].discard(attr_state_template) # Does not raise error if element does not exist.
                            else:
                                attractor_states_c[a_key].remove(attr_state_template)
                            break
                    if true_pa_state:
                        break
                if not true_pa_state:
                    false_positives.add(pa_state)

            missed_attractors = set(attractor_states.keys()).difference(discovered_attractors)
                    
            if (args.num_mc_repetitions == 1) or (args.verbose):
                print(f"----------------------------------------------------------------")
                print(f"\tResults of Monte Carlo attractor detection (run {mc_rep + 1})")
                print(f"----------------------------------------------------------------")
                print(f"Number of pseudo-attractor states: {len(pseudoattractor_states_list[mc_rep])}")
                print(f"Number of fixed-point pseudo-attractor states: {len(fixed_point_pa_states_list[mc_rep])}")
                print(f"Discovered attractors: {discovered_attractors}")
                print(f"Discovered fixed-point attractors: {len(fixed_point_pa_states_list[mc_rep])}")
                print(f"Missed attractors: {missed_attractors}")
                print(f"Number of missed attractors: {len(missed_attractors)} out of {len(attractor_states.keys())}")
                print(f"Coverage of attractor states: {state_templates_coverage}")
                print(f"False positivies: {false_positives}")
                print(f"Number of false positivies: {len(false_positives)}")
                print(f"Not found attractor states: {attractor_states_c}")

            missed_attractor_number_list.append(len(missed_attractors))
            false_positive_number_list.append(len(false_positives))
            found_attractor_number_list.append(len(discovered_attractors))
            l = len(fixed_point_pa_states_list[mc_rep])
            fixed_point_pa_state_number_list.append(l)

            # For code correctness testing purposes - to be deleted
            assert l == num_fp_attr - sum([1 if sum([2**state_template.count("-") for state_template in attractor_states[a_key]]) == 1 else 0 for a_key in missed_attractors])


    print(f"--- Summary of performance over {args.num_mc_repetitions} runs ---")

    if args.run_cabean:

        print(f"CABEAN runtime on all environmental conditions: {total_cabean_runtime:.2f}s")

        print(f"Monte Carlo attractor identification runtime(s): {', '.join(f'{t:.2f}s' for t in mc_runtime_list)}")
        if args.num_mc_repetitions > 1:
            print(f"Mean Monte Carlo attractor identification runtime: {np.mean(mc_runtime_list):.2f}s ({np.std(mc_runtime_list):.2f}s std)")

        if total_cabean_runtime != 0:
            print(f"Runtime ratio(s): {np.array(mc_runtime_list) / total_cabean_runtime}")

            if args.num_mc_repetitions > 1:
                print(f"Mean ratio: {np.mean(np.array(mc_runtime_list) / total_cabean_runtime):.2f} ({np.std(np.array(mc_runtime_list) / total_cabean_runtime):.2f} std)")

        print(f"Number of exact attractors: {len(attractor_states.keys())}")
        print(f"Number of exact fixed-point attractors: {num_fp_attr}")

        print(f"Mean number of attractors found by MC: {np.mean(found_attractor_number_list):.2f} ({np.std(found_attractor_number_list):.2f} std)")
        print(f"Mean number of fixed-point attractors found by MC: {np.mean(fixed_point_pa_state_number_list):.2f} ({np.std(fixed_point_pa_state_number_list):.2f} std)")

        print(f"Mean number of MC false positivies: {np.mean(false_positive_number_list):.2f} ({np.std(false_positive_number_list):.2f} std)")

    else:

        print(f"Mean number of pseudo-attractor states: {np.mean([len(el) for el in pseudoattractor_states_list])} ({np.std([len(el) for el in pseudoattractor_states_list])})")
        print(f"Mean number of fixed-point pseudo-attractor states: {np.mean([len(el) for el in fixed_point_pa_states_list])} ({np.std([len(el) for el in fixed_point_pa_states_list])})")


main()
