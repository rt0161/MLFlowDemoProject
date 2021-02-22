import pandas as pd
import numpy as np
import os, sys
import random
from pathlib import Path
import logging
import mlflow

# set up base file path
base_path=Path().resolve().parents[0]
data_path = os.path.join(base_path,'02-data/')
allcsvs = [x for x in os.listdir(data_path) if '.csv' in x ]
allparquet = [x for x in os.listdir(data_path) if '.par' in x]
dataname = 'auto_clean.csv'

#logger config
def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('map_columns.log', mode='a')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

# tool functions
def create_mapping_rand(df, col):
    """ This is a tool to create random mapping of the text columns to integer"""

    keylist = df[col].unique()
    random.shuffle(keylist)
    map_dict = {key: x for x,key in enumerate(keylist)}
    return map_dict



def map_rank(df,col,mapdict):
    """ This function maps the text column content into numerical"""
    map_num={col:mapdict}
    df.replace(map_num,inplace=True)
    return df


def trans_dataframe_car(df, logger):
    """ this function transforms the data frame of car type data into inputs to ML model,
    (1) fill in null values with approprieate means
    (2) convert text/categorical data into encoding by mapping"""

    # print null columns
    print("all columns and the number of null cells:")
    print(df_carmod.isnull().sum())

    # delete all no-brand cars
    df = df[df['make'].notnull()]

    # arrange datatypes
    df['engine-size']=df['engine-size'].astype('int16')
    df['normalized-losses']=df['normalized-losses'].astype('int16')
    df['horsepower']=df['horsepower'].fillna(0).astype('int16')
    df['gas']=df['gas'].astype('bool')
    df['diesel']=df['diesel'].astype('bool') 
    df['highway-mpg']=df['highway-mpg'].fillna(0).astype('int16') 

    # deal with null cells(and inappropriate 0s)

    df.fillna({'stroke':df['stroke'].mean()}, inplace=True)
    horspb = df[df['make']==df[df['horsepower-binned'].isnull()]['make'].values[0]]['horsepower-binned'].values
    df.fillna({'horsepower-binned':horspb.any()}, inplace= True)
    #df['normalized-losses']=df['normalized-losses'].fillna(int(df['normalized-losses'].mean()))
    #df['aspiration']=df['aspiration'].fillna('std')

    # mapping string columns 
    # particular columns which can go random encoded

    rand_enc_cols =['make','engine-type','fuel-system','body-style']
    for col in rand_enc_cols:
        mapdict=create_mapping_rand(df,col)
        df = map_rank(df, col,mapdict)
        # writes mapdict to log file
        logger.info("%s column mapping is: %s",col, mapdict)
    
    # hardcode some mapping of text columns based on intuitive knowledge 
    map_asp = {'std':0,'turbo':1}
    df=map_rank(df,'aspiration',map_asp)
    logger.info("%s column mapping is: %s",'aspiration', map_asp)
    map_numdoor = {'four':4, 'two':2}
    df=map_rank(df,'num-of-doors',map_numdoor)
    logger.info("%s column mapping is: %s",'num-of-doors', map_numdoor)
    map_drw = {'fwd':1, 'rwd': 0, '4wd': 4}
    df=map_rank(df, 'drive-wheels', map_drw)
    logger.info("%s column mapping is: %s",'drive-wheels', map_drw)
    map_numcyl = {'four':4,'six': 6, 'five':5, 'eight':8, 'two':2,'three': 3,'twelve':12}
    df=map_rank(df, 'num-of-cylinders',map_numcyl)
    logger.info("%s column mapping is: %s",'num-of-cylinders', map_numcyl)
    map_engloc = {'front':0,'rear':1}
    df=map_rank(df,'engine-location',map_engloc)
    logger.info("%s column mapping is: %s",'num-of-cylinders', map_numcyl)
    map_hrsp = {'Low':0,'Medium':1, 'High':2}
    df=map_rank(df,'horsepower-binned',map_hrsp)
    logger.info("%s column mapping is: %s",'horsepower-binned', map_hrsp)


    return df 


if __name__ == '__main__':


        # initiate mlflow PROJECT
    #tracking_uri_path = "file:"+str(base_path)+"02-data/mlrun_store"
    #mlflow.set_tracking_uri(tracking_uri_path)
    mlflow.set_experiment("data_prep")
    experiment = mlflow.get_experiment_by_name("data_prep")
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Tags: {}".format(experiment.tags))
    print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

        #print("run_id: {}; status: {}".format(run.info.run_id, run.info.status))
        #print("--")
        # log initial stage data
    mlflow.log_artifact(data_path+'auto_clean.csv')
    df_carmod = pd.read_csv(data_path+dataname)
    # prepare logger
    try:
        if (logger.hasHandlers()):logger.handlers.clear()
    except:
        logger = setup_custom_logger('maplog')

    df_carmod = trans_dataframe_car(df_carmod, logger)

    # save new dataframe to input to ML training 
    df_carmod.to_csv('../02-data/prep_car_input.csv', index=None)

    # log stage of data after 
    mlflow.log_artifact(data_path+'prep_car_input.csv')
    mlflow.log_artifact('map_columns.log')

    #mlflow.end_run()
    #print("Active run : {}".format(mlflow.active_run()))
