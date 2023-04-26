

import numpy as np
import pandas as pd
from scipy.optimize import dual_annealing
import logging
from tqdm import tqdm
import multiprocessing as mp
from typing import List, Tuple, Protocol
from datetime import datetime

class Model(Protocol):
    def predict(self, formatted_data: np.ndarray) -> np.ndarray:
        """Function to predict the denominator of the Process KPI based on controllable and noncontrollable data"""

def objective(controls, noncontrols, model: Model, outlet, c):
    data = np.array(np.concatenate([controls, noncontrols]).reshape(1, -1))
    oil = data[0][1]
    gas = data[0][2]
    model_output = model.predict(data)[0]
    
    return (oil + gas) - (c * model_output)

def format_for_pool(
    df:pd.DataFrame, 
    date_label:str, 
    control_variables: List[str], 
    noncontrol_variables: List[str], 
    bounds : List[List[float]], 
    model : Model, 
    maxiter : int,
    outlet,
    c) -> Tuple[Tuple[datetime, List[float], List[float]]]:
    """
    Function to format historical format of data into tuples for the pool
    Parameters
    ----------
    df : pd.DataFrame
        historical variable
    date_label : str
        label of the date column
    control_variables : List[str]
        control variables (column names)
    noncontrol_variables : List[str]
        noncontrol variables (column names)
    bounds : List[List[float, float]]
        lower and upper bound for each variable
    model : Model
        model that implements predict method
    maxiter : int
        max iterations
    Returns
    -------
    out :  Tuple[Tuple[datetime, List[float], List[float]]]
        output in the format for the pool.starmap ... elements of the inner tuple will be: timestamp, controllable variables (values) and noncontrollable variables (values)
    """
    # create a copy for manipulation
    df_copy = df.copy().reset_index()

    dates = df_copy.pop(date_label)
    controls = df_copy[control_variables].values
    noncontrols = df_copy[noncontrol_variables].values

    result = [None] * len(dates)

    for i, row in enumerate(controls):
        result[i] = (dates[i], row, noncontrols[i, :], model, bounds[i], maxiter, outlet[i], c)

    return result


def run_optimization(timestamp, controllable: List[float], noncontrollable: List[float], model: Model, bounds: List[List[float]], maxiter: int, outlet, c_value):
    """
    High level api call to run the optimization procedure --- this will be the function passed to mp.Pool().map()
    Parameters
    ----------
    controllable : List[float] | np.ndarray
        controllable variables 
    noncontrollable : List[float] | np.ndarray
        noncontrollable variables
    model : Protocol
        Must implement a predict method
    timestamp : datetime 
        date
    bounds : Tuple[Tuple[float, float]]
        lower and upper bound for each variable
    maxiter : int
        max iterations for the dual_annealing
    Returns
    -------
    timestamp : datetime
        timestamp
    nonoptimized_predicted_kpi_denominator : float
        value of objective function without the optimized controls (no directive)
    optimized_predicted_kpi_denominator : float
        value of objective function with the optimal controls
    optimal_controls : List[float]
        optimal control values for the control variables
    success : bool
        whether the optimization was successful or not
    """
    result = dual_annealing(objective, bounds, args=(noncontrollable, model, outlet, c_value), x0=controllable, maxiter=maxiter)    
    optimal_controls = result.x

    # NOTE: the timestamp will be used to verify the order of the result but it shouldn't be needed --- check on this later ...
    return timestamp, optimal_controls, result.success

def bind_optimization_results(results, date_label:str, controls:List[str]):
    """
    Function to bind the optimization results
    Parameters
    ----------
    results : Tuple[Tuple[datetime, List[float], bool]]
        results from the optimization procedure: timestamp, controls values, success
    date_label : str
        date label 
    controls : List[str]
        control variable labels
    Returns
    -------
    out : pd.DataFrame
        results bound into a dataframe: columns: date_label (date), optimized_kpi, nonopt_kpi, success
    """
    # preallocate result variables
    dates = [None] * len(results)
    control_values = [None] * len(results)
    success = [None] * len(results)

    # unpack results
    for i, row in enumerate(results):
        dates[i], control_values[i], success[i] = row

    # TODO: check once if the result is in order ...

    # bind to df
    out = pd.DataFrame(control_values, columns=[name+'_Optimized' for name in controls], index=dates)
    out.index.name = date_label
    # NOTE: check if this line below is needed
    out.index = pd.to_datetime(out.index)

    out['Success'] = success

    return out

def mp_optimization(
    data:pd.DataFrame, 
    date_label:str, 
    controllable_vars:List[str], 
    noncontrollable_vars:List[str], 
    control_bounds:List[List[float]], 
    model:Model, 
    maxiter:int, 
    n_process:int,
    outlet,
    c):
    """
    Function to run the optimizaion in the multiprocessing format
    Parameters
    ----------
    data : pd.DataFrame
        data of interest
    date_label : str
        label of the date 
    controllable_vars : List[str]
        list of the names of the controllable columns
    noncontrollable_vars : List[str]
        list of the noncontrollable variables that are still used in the model
    control_bounds : List[List[float, float]]
        list of controls (lower, upper) for each controllable variables --- assumes order is the same as controllable_vars
    model : Model
        model that implements .predict
    maxiter : int
        max iterations of each step of the optimization
    n_process : int
        number of processes to use
    Returns
    -------
    result : pd.DataFrame
        optimization results 
    """
    logger = logging.getLogger(__name__)
    reformatted = format_for_pool(data, date_label, controllable_vars, noncontrollable_vars, control_bounds, model, maxiter, outlet, c)

    logger.info(f"Running multiprocessing with {n_process} cores")
    with mp.get_context("spawn").Pool(processes=n_process) as pool:
        out = pool.starmap(run_optimization, tqdm(reformatted, total=len(reformatted[0])))

    logger.info("Bind results to historical format")
    result = bind_optimization_results(out, date_label, controllable_vars)

    return result