# Installation

Setup a python virtual environment with 
```
python -m venv ./venv
```

Then install dependencies with
```
pip install -r requirements.txt
```

# Run

There are two entry points for all the experiments contained in the paper. 

`python3 ./src/run_gbif.py` runs all experiments for the GBIF dataset

`python3 ./src/run_glc23.py` runs all experiments for the GLC23 dataset

Both scripts will read source data from the `./data` folder, run experiments, and save results in the `./results` folder for future use. If a result file already presents in the `./results` folder, the script will simply re-use it to save time.

# Repo Structure

### `Data` folder

The `./data` folder contains processed GBIF and GLC23 presence-absence data, concatenated with other features like environmental data, maxent predictions on top 5 species, xgboost predictions on top 5 species, gold standard of top 5 species etc, for easier experiment streamlining.

It also contains maxent predictions of all species seperately to evaluate maxent-only performance. 

Please extract the `data.zip` to obtain the data files used in our experiments.

### `Results` Folder

The `results` folder stores prediction results of our various approaches detailed in the paper draft.

Each file stores the scores (from 0-1) of presence probabilities, for all species across all data. When evaluating, we only take the rows that represents sequestered data.

The `figures` subfolder stores the bar graph used in the paper. These bar graphs are also generated when running the scripts at the end.

### `src` folder

The `src` folder contains our source code that generates the experimental results in the paper. The two sets of experiments presented in the paper draft are run through
```
src/run_gbif.py
src/run_glc23.py
```
respectively

# Docker Image

A Docker image containing our source code and data has been published at https://hub.docker.com/r/jiexunxu1/plant-presence

To build image, run
```
docker build -t jiexunxu1/plant-presence:latest .
```

To push to dockerhub, run
```
docker push jiexunxu1/plant-presence:latest
```