import pandas as pd
import numpy as np
import SQLTool
import argparse

from Utils import read_json, write_json

def main():

    # load in config and tags list
    config = read_json('../data/download_config.json')
    tags = read_json('../data/download_tags.json')
    process = config['process_name']

    # add suffix to tags
    tags = [tag + config['tag_suffix'] for tag in tags['model_tags']]

    # download a tags
    hist_df, missing_tags = SQLTool.download_and_format(
        tags,
        config['client'],
        config['server'], 
        -6,
        start_date =  config['start_date'],
        end_date =  config['end_date'],
        rate = config['rate'],
        unit = config['unit'])

    # save a tags
    hist_df.to_csv(f'../outputs/{process}_hist_df.csv')
    pd.DataFrame({'missing_tags': missing_tags}).to_csv(f'../outputs/{process}_missing_SQL_tags.json', index = False)

    return

if __name__ == '__main__':
    main()