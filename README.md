# singapore-soundscape-site-selection-survey

Raw data and code for replication of all results in the Singapore Soundscape Site Selection Survey (S5).

This repository stores the responses obtained as part of the S5 project and replication code for the identification of characteristic soundscapes in Singapore that are "full of life and exciting", "chaotic and restless", "calm and tranquil", and "boring and lifeless". For more details on the dataset, please refer to our publication:

> Ooi, K.; Lam, B.; Hong, J.; Watcharasupat, K. N.; Ong, Z.-T.; Gan, W.-S. Singapore Soundscape Site Selection Survey (S5): Identification of Characteristic Soundscapes of Singapore via Weighted k-means Clustering. Sustainability 2022. [Under review]

# Getting started

Firstly, clone this repository by manually downloading it from https://github.com/ntudsp/singapore-soundscape-site-selection-survey, or enter the following line from a terminal (you need to have <a href="https://github.com/git-guides/install-git">git</a> installed first, of course):

    git clone https://github.com/ntudsp/singapore-soundscape-site-selection-survey

If you are using <a href="https://docs.conda.io/en/latest/">conda</a> as your package manager, you may enter the following line into a terminal to install the required packages into a conda environment (or you may install them manually using the requirements stated in `s5.yml`):

    conda env create -f s5.yml

Activate the conda environment by entering the following line into a terminal:

    conda activate s5

Then, to run the replication code and basic data analysis as described in our publication, you may enter the following line into a terminal (this opens a Jupyter notebook in your default browser):

    jupyter lab --notebook-dir . ./code/replication_code.ipynb

The Jupyter notebook contains further details and comments for the replication code.
    
# Directory structure

## This repository
    .
    ├── code                         
    │   ├── replication_code.ipynb   # Code used to process the raw data and output the results (run this first).
    │   └── s5_utils.py              # Auxiliary functions used in replication_code.ipynb.
    │
    ├── data
    │   └── latlng.csv               # Responses obtained from the 67 participants in the S5 project.
    │
    ├── figures                      # Figures used in our publication & replication_code.ipynb.
    │   ├── circumplex.svg
    │   ├── ...
    │   └── weights.svg
    │
    ├── .gitignore
    ├── LICENSE
    ├── README.md                    # This file.
    └── s5.yml                       # The Anaconda environment containing required packages to run all the code in ./code.
