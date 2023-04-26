import datetime
import multiprocessing as mp
import logging
import pandas as pd
import argparse
import os.path as osp

from Utils import read_pickle, read_json
from opt_tools import mp_optimization

def setup_logger():
    
    formatstr = '%(asctime)s: %(levelname)s: %(funcName)s Line: %(lineno)d %(message)s'
    datestr = '%m/%d/%Y %H:%M:%S'
    logging.basicConfig(
        level=logging.INFO, 
        format=formatstr, 
        datefmt=datestr, 
        handlers=[
            logging.FileHandler('mp.log'),
            logging.StreamHandler()
            ]
        )

    return 

def parse_args() -> argparse.Namespace:
    """Function to parse command line arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument('c_val', type=float, help='C value for optimization')
    parser.add_argument('output_dir', type=str, help='Path to the output folder')
    args = parser.parse_args()

    return args

def main():
    """Main Function"""

    process = "b21"

    start = datetime.datetime.now()
    mp.set_start_method("spawn")

    args = parse_args()

    # set up logger
    setup_logger()

    # mp set up
    n_cores = mp.cpu_count()
    assert n_cores > 1 and n_cores <= mp.cpu_count(), f"NumberOfCoresError: Number of cores must be greater than 1 but less than {mp.cpu_count()}. Recieved {n_cores}"

    # load in data, model, configs
    logging.info("Reading Config and Data")
    model = read_pickle(f'../models/{process}_optimization_model.pkl')
    optimizer = read_json(f'../outputs/{process}_optimizer.json')
    test_df = pd.read_csv(f'../outputs/{process}_test_df.csv').set_index('Date')
    tags_dict = read_json(f'../outputs/{process}_tags_dict.json')
    test_df.index = pd.to_datetime(test_df.index)

    ctrl_tags = list(optimizer['controllable'].keys())
    indct_tags = [tag for tag in tags_dict['noncontrollable'] if tag in test_df.columns]

    bounds_df = test_df.loc[:,ctrl_tags]

    rate_change = [ctrl_info['rate'] for ctrl_info in optimizer['controllable'].values()]

    min_dynamic_bounds = bounds_df - (bounds_df * rate_change)
    max_dynamic_bounds = bounds_df + (bounds_df * rate_change)

    min_global_bounds = [tag_info['bounds'][0] for tag_info in optimizer['controllable'].values()]
    max_global_bounds = [tag_info['bounds'][1] for tag_info in optimizer['controllable'].values()]

    for i, control_tag in enumerate(ctrl_tags):
        min_dynamic_bounds.loc[min_dynamic_bounds[control_tag] < min_global_bounds[i], control_tag] = min_global_bounds[i]
        min_dynamic_bounds.loc[min_dynamic_bounds[control_tag] > max_global_bounds[i], control_tag] = max_global_bounds[i] - 0.0001
        max_dynamic_bounds.loc[max_dynamic_bounds[control_tag] < min_global_bounds[i], control_tag] = min_global_bounds[i] + 0.0001
        max_dynamic_bounds.loc[max_dynamic_bounds[control_tag] > max_global_bounds[i], control_tag] = max_global_bounds[i]

    max_dynamic_bounds = max_dynamic_bounds + 0.001

    bounds = [
            [(min_dynamic_bounds.at[timestamp, col], max_dynamic_bounds.at[timestamp, col]) for col in ctrl_tags] 
            for timestamp in test_df.index.tolist()
            ]

    outlet = test_df.loc[:, 'RESPONSE']

    logging.info("Formatting Data for multiprocessing")
    max_iter = 100
    out = mp_optimization(test_df, 'Date', ctrl_tags, indct_tags, bounds, model, max_iter, n_cores, outlet, args.c_val)


    logging.info(f"Saving results")
    file_path = args.output_dir
    out.to_csv(file_path)


    run_time = datetime.datetime.now() - start
    logging.info(f"Total run time: {run_time.total_seconds() / 60:.3f} (minutes)")

    return


if __name__ == '__main__':
    main()