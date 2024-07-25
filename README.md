# Installation

Setup a python virtual environment with 
```
python -m venv ./venv
```

Then install dependencies with
```
pip install -r requirements.txt
```

# Structure

The `data` folder contains both the GBIF and GLC23 processed presence/absence data. They are concatenated with other features like environmental data, maxent predictions on top 5 species, xgboost predictions on top 5 species, gold standard of top 5 species etc. It also contains maxent predictions of all species seperately to evaluate maxent-only performance. Please decompress the `data.zip` to obtain the data files used in our experiments.

The `results` folder stores prediction results of our various approaches detailed in the paper draft.

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