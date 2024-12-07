[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=17236106&assignment_repo_type=AssignmentRepo)
Project Instructions
==============================

This repo contains the instructions for a machine learning project.

Project Organization
------------
This repository is structured specifically for an ML project. Explanation of what it organizes and what it is for:

***

### What's in this Project?

1. **README.md**
Project Overview Summing up the project, portraying an overview of how it can be used: The Front Page for anyone intending to interact with this project.

2. **notebooks
Contains Jupyter Notebooks. Jupyter notebooks are files in which you can document, experiment, and visualize your data analysis or ML processes interactively. Such files should be named clearly for easy identification.

3. **accounts**
A folder for the presentation of results and insights from the project:
- **graphs**: Graphs and chart developed during analysis.
-**README.md**: The address of a YouTube video or summary report about the project.
- final_project_report : the polished and formal report in PDF format summarizing the findings.
- **presentation**: A PowerPoint document summarizing the main results.

4. requirements.txt
A list of all the used Python libraries and versions. That makes other's life easier to repeat the working environment.

5. src
This folder has all main code for the project, divided into logical parts:
- `__init__.py`: A special file in the `src` folder making that directory a Python module.
- **data**:
- **raw**: Original, unaltered datasets.
- **processed**: Data that has been cleaned and formatted ready for analysis.
-  **preprocessing_data**: Cleaning and preparation scripts for raw data. Example: `pre-processing.py` for deletion of duplicate values or filling a missing value.
- **feature_engineering**: Scripts that convert raw data to useful inputs for machine learning models. For example: `build_features.py` that generates features like averages or encodings.
- **models**:
- **train_model.py**: A script to train ML models using the data.
- **predict_model.py**: code to make predictions with the trained models.
- **Visualization**: Scripts like `visualize.py` to produce graphs and charts in order to understand the data and results.
- **main.py**: Essentially the "control center" of the project. Here, one runs all the different parts (preprocessing, training, etc.) in the right order.

6. LICENSE
Legal document which states how others can use or share your code.

---

How is it all supposed to work together?

**Start** by reading through `README.md` for a conceptual overview.
Use **notebooks** to explore data interactively or try out ideas.
The scripts in the folder `preprocessing_data` accomplish the preprocessing of raw data. - **Generate Features** and persist cleaned datasets in `data/processed`.

Train the models with scripts from the `models` folder and run evaluation.

Visualize your results and insights using the scripts in the folder `visualization`.

Share: End consolidation of ALL findings in the folder called **Reports. ### **Why This Structure?** This organization assists Keep your project modular and organized. Make it easier to collaborate by clearly separating tasks and roles. - Let someone easily reproduce your result by providing good documentation and environments. It's almost like having a well-organized toolbox for any ML project!

    ├── README.md          <- The top-level README for describing highlights for using this ML project.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention should snake case.
    │
    ├── reports            
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │   └── README.md      <- Youtube Video Link
    │   └── final_project_report <- final report .pdf format and supporting files
    │   └── presentation   <-  final power point presentation 
    |
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
       ├── __init__.py    <- Makes src a Python module
       ├── data
       │   ├── processed      <- The final, canonical data sets for modeling.
       │   └── raw            <- The original, immutable data dump.
       │
       ├── preprocessing_data           <- Scripts to download or generate data and pre-process the data
       │   └── pre-processing.py
       │
       ├── feature_engineering       <- Scripts to turn raw data into features for modeling
       │   └── build_features.py
       │
       ├── models         <- Scripts to train models and then use trained models to make
       │   │                 predictions
       │   ├── predict_model.py
       │   └── train_model.py
       │
       └── visualization  <- Scripts to create exploratory and results oriented visualizations
       │   └── visualize.py  
       │
       └── main.py  <- main script to run all the models and call appropriate functions
       |
       ├── LICENSE  <- LICENSE terms to be included for the use of the source code distribution



