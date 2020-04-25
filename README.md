# Kaggle - Tweet Sentiment Extraction

## Getting started

### Requirements
Following tools must be install to setup this project:
* `python >= 3.7`
* `poetry >= 0.12` (poetry installation guide could be found on their website)

### Setup environment
Following command lines could be used to setup the project.
```
$ git clone git@github.com:vpham-gia/2020-04_kaggle_tweet_sentiment_extraction.git
$ poetry install  # Install virtual environment with packages from poetry.lock file
$ python -m spacy download en_core_web_md
```

In order to use `settings` folder accordingly, one must copy `settings/.env_template` to `settings/.env` and fill `TWEET_SE_SETTINGS_MODULE` variable.

### Run script
In order to run a script, following steps could be performed:
```
$ source activate.sh
$ python3 tweet_sentiment_extraction/application/run_baseline.py
```

## Useful poetry commands
```
$ poetry new folder --name package_name

$ poetry env use 3.7 # creates .venv with accurate version of python
$ poetry env info
```

Install from `poetry.lock` file that already exists
```
poetry install --no-root # --no-root option skips installation of the project package
```

Update the latest versions of the dependencies and update `poetry.lock` file
```
poetry update
```

```
poetry add pandas
```

