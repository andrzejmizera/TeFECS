# TeFECS - Temporary Forward Edgetics Control Scheme
This is an implementation of a novel algorithm for identifying edge perturbation control strategies that drive a Boolean Network from a source attractor to a target attractor.

# Installation

Download the TeFECS by clicking on "Full repo ZIP" in the top right corner and unzip the downloaded file.

```bash
cd TeFECS
```

Prepare and activate the virtual environment:

```bash
python3 -m venv <folder of the environment>
source <folder of the environment>/bin/activate
```

For the last line, use \<folder of the environment\>\Scripts\activate if on Windows.

Install the **BANG** package after downloading it from https://anonymous.4open.science/r/bang-3506/ as follows:
```bash
unzip bang-3506.zip
cd bang-3506
pip install -e .
cd ..
```

Continue the installation:

```bash 
pip install -e .
```

# Running the Monte Carlo method for scalable identification of attractor states

The Monte Carlo method can be re-run for the ABA model as follows:

> python evaluation_MC_attractors_identification.py --model-file biological_models/aba/aba81-gattaca_ec.ispl --cabean-file cabean_output/cabean_aba-gattaca_ec.txt --num-mc-repetitions 10 --verbose

To re-run the method for other models, provide the appropriate files from the `biological_models` and `cabean_output` subfolders.

The code will run simulations on GPU if available, otherwise falling back to CPU.

# Re-running the Temporary Forward Edgetics Control Scheme (TeFECS)

To re-run TeFECS on the saved models, i.e. learned Attractor Landscape Control Graphs (ALCGs):

> python tefecs.py --model-file biological_models/aba/aba81-gattaca_ec.ispl --control_graph_file results/tefecs_experiments/experiment_aba81-gattaca_ec/aba81-gattaca_ec_control_graph.pkl --cabean-file cabean_output/cabean_aba-gattaca_ec.txt --num_random_pairs 5 --verbose

To re-run the method for other models, provide the appropriate files from the `biological_models`, and `results/tefecs_experiments`, and `cabean_output` subfolders. 

As the Boolean Networks are simulated with the asynchronous update mode, their dynamics are non-deterministic and thus can differ from run to run.

By default, the results will be saved to the `tefecs_experiments` subfolder.

# Running TeFECS from scratch:

To run from scratch, please download CABEAN from https://satoss.uni.lu/software/CABEAN/ and set the `CABEAN_PATH` constant in tefecs.py to the path of the downloaded `cabean` executable. 

> python tefecs.py --model-file \<ISPL file with the BN model specification\>

or

> python tefecs.py --model-file \<ISPL file with the BN model specification\> --cabean-file \<Cabean output file with a list of attractors.\>
