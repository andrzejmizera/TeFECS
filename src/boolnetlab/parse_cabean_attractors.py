from itertools import product
import pickle

from boolnetlab import bin2state


def _expand(s):
    n = s.count("-")
    for replacements in product("01", repeat=n):
        it = iter(replacements)
        yield ''.join(next(it) if c == '-' else c for c in s)


def parse_cabean_attractors(input_file: str, verbose: bool = False):

    # parser = argparse.ArgumentParser()

    # parser.add_argument('--file-name', type=str, required=True)

    # args = parser.parse_args()

    # input_file = args.file_name

    with open(input_file, 'r') as file:
        cabean_output = file.read()

    attractors = dict()
    attractor_key_index, attractor = 0, []
    cabean_runtime = 0.0

    parts = cabean_output.split("number of attractors =")
    num_attractors = int(parts[1].split('\n')[0])
    if verbose:
        print(f"Number of attractors: {num_attractors}", flush=True)

    # Extract individual attractors
    for line in cabean_output.split('\n'):
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
                if verbose:
                    print(f"Cabean runtime: {float(line.split('=')[1].split(' ')[0])}s")
                cabean_runtime = float(line.split('=')[1].split(' ')[0])

    if len(attractor) > 0:
        attractors['A' + str(attractor_key_index)] = set(attractor)

    return attractors, cabean_runtime


def extract_attractor_states(attractors: dict[str, set[str]], output_file: str | None = None):

    attractor_states = []

    for a_key in attractors.keys():
        for state_template in attractors[a_key]:
            if state_template.count('-') == 0:
                attractor_states.append([bin2state(state_template)])
            else:
                attractor_states.extend([[bin2state(s)] for s in list(_expand(state_template))])

    print(attractor_states)
    print(f"Number of attractor states: {len(attractor_states)}")

    if output_file is not None:
        with open(output_file, 'wb') as file:
            pickle.dump(attractor_states, file)
