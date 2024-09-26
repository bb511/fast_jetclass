[![Email Badge](https://img.shields.io/badge/blah-podagiu%40ethz.ch-blue?style=flat-square&logo=minutemailer&logoColor=white&label=%20&labelColor=grey)](mailto:podagiu@ethz.ch)
[![Zenodo Badge](https://img.shields.io/badge/blah-10.5281%2Fzenodo.10553804-blue?style=flat-square&label=Zenodo&labelColor=grey)](https://zenodo.org/records/10553805)

# Ultrafast Jet Classification using Geometric Learning for the HL-LHC

Code repository corresponding to the [Ultrafast Jet Classification at the HL-LHC](https://arxiv.org/abs/2402.01876) paper.

Three machine learning models are used to perform jet origin classification. 
These models are optimized for deployment on a field-programmable gate array device. 
In this context, we demonstrate how latency and resource consumption scale with the input size and choice of algorithm. 
Through quantization-aware training and efficient synthetization for a specific field programmable gate array, we show that O(100) ns inference of geometric learning architectures such as Deep Sets and Interaction Networks is feasible at a relatively low computational resource cost.

## Installation
The main dependencies can be installed using `conda` by running the following command in the terminal while in this repository's directory
```
conda env create -f fast_jetclass.yml
```
Alternatively, one can install the following packages manually:
```
```

Then, after installing the dependencies, install this repository using
```
pip install .
```
while still in this repository's directory.

### Additional dependencies for synthesis
The synthesis of the models for the FPGA are done using `hls4ml` 

### GLIBCxx library errorA
In case you get an error that your machine does not have the right `GLIBCxx` library, i.e., something like the following
```
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
```
then configure the `LD_LIBRARY_PATH` to point to the conda `lib` of the `conda` environment, if you used `conda`.
To do this quickly, run the following command in your terminal:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/deodagiu/miniconda/envs/fast_jetclass/lib/
```

To do this gracefully, open or create the following file
```
[your_conda_dir]/envs/fast_jetclass/etc/conda/activate.d/env_vars.sh
```
and paste the following inside
```
#!/bin/sh

export OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/deodagiu/miniconda/envs/fast_jetclass/lib/
```
This will modify your `LD_LIBRARY_PATH` only when inside the conda environment. To restore this path to its original form, open or create the following file
```
[your_conda_dir]/envs/fast_jetclass/etc/conda/deactivate.d/env_vars.sh
```
and paste the following inside
```
#!/bin/sh

export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH
```

