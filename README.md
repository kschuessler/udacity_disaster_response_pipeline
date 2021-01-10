# Disaster Response Pipeline Project

## Table of Contents

1. [Project Motivation](#motivation)
2. [File Descriptions](#files)
3. [Installation](#installation)
4. [Licensing, Authors, and Acknowledgements](#licensing)


## Project Motivation<a name="motivation"></a>

This project is part of the Udacity Data Scientist Nanodegree. 
The goal is to classify disaster response messages using machine learning algorithms.

To achieve this, the project contains an ETL pipeline script to prepare the raw data as well as a ML pipeline to build a classification model.
The resulting model can then be displayed in a Flask web app that allows the user to enter a new message and get classification results.

## File Descriptions <a name="files"></a>

The project contains the following scripts:
  
```
- \app
    - run.py: script to run the Flask web app
- \data
    - disaster_categories.csv: categories data set
    - disaster_messages.csv: messages data set
    - DisasterResponse.db: disaster response data base (result of ETL pipeline)
    - process_data.py: script to read the data from raw csv files, transforms and cleans them and stores result in a SQLite database
- \models
    - classifier.pkl: model pickle file (result of ML pipeline)
      NOTE: file too big to be stored on Github --> execute script to generate file
    - train_classifier.py: script to build, train, evaluate and save the model
```

## Installation <a name="installation"></a>

To run the scripts, Python 3.7 and the following libraries are necessary:

* json
* nltk
* pandas
* pickle
* plotly
* sqlalchemy
* sklearn

To set up the project, the following steps are necessary:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Template files for the solution and general instructions how to approach the challenge were provided by Udacity as part of the Data Science Nanodegree.
The data were provided by Figure Eight.