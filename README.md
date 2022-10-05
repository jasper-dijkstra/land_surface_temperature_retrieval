# land_surface_temperature_retrieval
This repository contains the source code used for master thesis project: *Land surface temperature retrieval at high spatio-temporal resolution*. The repository contains the code and data (from June 1, 2020 until June 1, 2022) to render the interactive Jupyter Notebook to validate different Land Surface Temperature (LST) products. In addition, the repository contains the JavaScript scripts to retrieve the ancillary datasets from Google Earth Engine.

# Installation
1. Clone the project and change directory
```
git@github.com:jasper-dijkstra/land_surface_temperature_retrieval.git
cd land_surface_temperature_retrieval
```

2. Build the conda environment
```
conda env create --file conda_environment.yml
source activate land_surface_temperature_retrieval
pip install -e .
```

# Opening the notebook
After installing the repository the jupyter notebook can be opened via a linux terminal (in the land_surface_temperature_retrieval direcotry) with the following command:
```
jupyter notebook scripts/ValidationDashboard.ipynb
```

