![CI workflow](https://github.com/made-ml-in-prod-2021/nickdndev/actions/workflows/python-app.yml/badge.svg?branch=homework1)

# Made Production ML

Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
Usage:
~~~
python ml_example/fit.py configs/train_config.yaml
~~~

Test:
~~~
pytest ml_project/tests/
~~~
Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── ml_project                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- code to download or generate data
    │   │
    │   ├── features       <- code to turn raw data into features for modeling
    │   │
    │   ├── models         <- code to train models and then use trained models to make
    │   │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

### References:
- [Generate dataset](https://www.caktusgroup.com/blog/2020/04/15/quick-guide-generating-fake-data-with-pandas/)
- [CI Github](https://docs.github.com/en/actions/guides/setting-up-continuous-integration-using-workflow-templates)
- [Github Badge for CI](https://docs.github.com/en/actions/managing-workflow-runs/adding-a-workflow-status-badge#using-the-workflow-file-name)

