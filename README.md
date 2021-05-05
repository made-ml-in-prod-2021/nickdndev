![CI workflow](https://github.com/made-ml-in-prod-2021/nickdndev/actions/workflows/python-app.yml/badge.svg?branch=homework1)

# Made Production ML

Installation:

~~~
git clone https://github.com/made-ml-in-prod-2021/nickdndev.git
cd ml_project/

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~

Fit model:

~~~
 provide path to dataset in app_config.yaml or use default path like:"data/raw/dataset.zip"
 
 python -m src.model.fit
~~~

Predict:

~~~
 provide path to prediton and dataset path in app_config.yaml"

 python -m src.model.predict  
 
~~~

Test:

~~~
 pytest tests/
~~~

Data profiling:

~~~
 provide path to report path and dataset path in profiling.yaml"

 python -m src.data_report.report
~~~

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── conf               <- Configuration files.
    │
    ├── data
    │   ├── predictions    <- Predictions from the model.
    │   ├── profiling      <- Data profiling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── outputs            <- Outputs from hydra.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── report         <- Data profillig by Pandas
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment.
    │
    └── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── configs         <- configuration dataclasses for type checking
    │   │
    │   ├── data           <- code to download or generate data
    │   │
    │   ├── features       <- code to turn raw data into features for modeling
    │   │
    │   ├── model          <- code to train models and then use trained models to make predictions
    │   │
    │   ├── data report    <- code for eda generation
    │   │
    │   └── utils          <- miscellaneous util functions
    │
    └── tests              <- unit tests

--------

### References:

- [Pandas Profiling](https://towardsdatascience.com/exploratory-data-analysis-with-pandas-profiling-de3aae2ddff3)
- [Generate dataset](https://www.caktusgroup.com/blog/2020/04/15/quick-guide-generating-fake-data-with-pandas/)
- [CI Github](https://docs.github.com/en/actions/guides/setting-up-continuous-integration-using-workflow-templates)
- [Github Badge for CI](https://docs.github.com/en/actions/managing-workflow-runs/adding-a-workflow-status-badge#using-the-workflow-file-name)

