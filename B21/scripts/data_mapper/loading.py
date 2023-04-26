import logging
import numpy as np
import os
import os.path as osp
import pandas as pd
from shutil import move

from typing import List

def get_data(incoming, object_ids, property_ids, id_to_name, sep='___', sampling_rate=None):
    """
    Function to read in data from live deployment format and map to an easier format to be processed (run through ds algorithm)

    Parameters
    ----------
    incoming : str
        path to the incoming directory
    object_ids : List[int]
        object ids
    property_ids : List[int]
        property ids
    id_to_name : Dict[Tuple[int, int]] -> Tuple[str, str]
        mapping from (object id, property id) to (object name, property name)
    sampling_rate : str
        the resamping rate --- put link here for options ...
    
    Returns
    -------
    df : pd.DataFrame
        dataframe with the data requested
        --- index is the timestamp
    """
    # NOTE: these args and the column names can be hardcoded since they do not change (live deployment format)
    data = pd.read_csv(incoming, delimiter=';',
                       header=None, names=['ObjectId', 'PropertyId',
                                           'Value', 'TimeStamp', '0'])
    data = data[['ObjectId', 'PropertyId', 'Value', 'TimeStamp']]

    # preallocate the lst 
    lst = [None] * len(object_ids)
    for i, (objectID, propertyID) in enumerate(zip(object_ids, property_ids)):
        objectName, propertyName = id_to_name[objectID, propertyID]

        temp = data.loc[(data['ObjectId'] == objectID) & (data['PropertyId'] == propertyID), ['TimeStamp', 'Value']].copy()
        temp.columns = ['TimeStamp', sep.join([objectName, propertyName])]
        temp['TimeStamp'] = pd.to_datetime(temp['TimeStamp'], unit='ms')
        temp = temp.set_index('TimeStamp', drop=True)
        lst[i] = temp

    # TODO: should we put a check here to make sure all the tags that are expected are there ...
    df = pd.concat(lst, axis=1)

    if sampling_rate:
        df = df.resample(sampling_rate).mean()
        
    return df

def read_input_folder(input_dir, ext, *args, **kwargs):
    """
    Function to read all the files in the input folder 

    Parameters
    ---------- 
    input_dir : str
        path to the input folder where the input files are stored
    ext : str
        valid file extension
    
    Yields
    ------
    data : pd.DataFrame
        data with columns ObjectName_PropertyName
    """
    incoming_files: List[str] = [incoming_file for incoming_file in os.listdir(input_dir) if incoming_file.endswith(ext)]

    for incoming_file in incoming_files:
        print(f"starting file: {incoming_file}")
        try:
            data = get_data(osp.join(input_dir, incoming_file), *args, **kwargs)
        except:
            logging.exception("Error in getting data")
            move(osp.join(input_dir, incoming_file), osp.join('..', 'Error', incoming_file))
            continue
        if data.shape[0] == 0:
            logging.exception("Input file empty")
            move(osp.join(input_dir, incoming_file), osp.join('..', 'Error', incoming_file))
            continue

        yield incoming_file, data

# NOTE: this function will be the trickiest since the output of models can be very different ....
def put_data(df, output_dir, name_to_id, sep='___', output_property_id=None):
    """
    This section saves the result to a file in a specific directory

    Parameters
    ----------
    df : pd.DataFrame
        processed data: index is the timestamp, columns names are objectname{sep}propertyname
        Assuming this dataframe has n columns, n lines will be written into the output file in the live deployment format for this timestamp
    output_dir : str
        path to save the output
    name_to_id : Dict[Tuple[str, str]] -> Tuple[int, int]
        mapping from (object id, property id) to (object name, property name)
    sep : str, default='___'
        separator between the object and the property
    output_property_id : int | Dict[int] -> int | None
        output property id: None if no adjustment is needed, int if it is the same across object ids, 
        dictionary if it differs across object ids (object id is key and final property id is value)

    Returns
    -------
    """
    # NOTE: should the output_property_id be part of a filtering process outside this function??
    timestamp = df.index[0]
    # parse time into yyyy-mm-dd_HH_mm_ss
    folderdate = '_'.join([timestamp.strftime('%Y'), timestamp.strftime('%m'), timestamp.strftime('%d')])
    filetime = '_'.join([folderdate, timestamp.strftime('%H'), timestamp.strftime('%M'), timestamp.strftime('%S')])
    
    # out is a list that holds the values 
    out = [None] * df.shape[1]
    columns=['DatapointID', 'PropertyID', 'Value', 'TimeStamp', 'Status']

    # updated function: moved rounding to the end of the operation, 1e3 converts to ms
    epoch = int(timestamp.timestamp() * 1e3)
    # Update: there is a recommended way to do this: https://stackoverflow.com/questions/54313463/pandas-datetime-to-unix-timestamp-seconds
    # this method also makes it clear that we are in milliseconds
    # NOTE: 2 things when converting the timestamps 
    # 1. Do conversions as little as possible, according to the link above rounding error can throw stuff off 
    # 2. Keep the conversions within the same package (pick datetime or pandas, don't mix them! Again precision issues)
    # Don't need to subtract since that is the beginning for unix time (hence we are subtracting zero)
    # epoch = (timestamp - pd.Timestamp("1970-01-01")) // pd.Timedelta('1ms')

    for c, col in enumerate(df.columns.tolist()):
        objectName, propertyName = col.split(sep)
        objectID, propertyID = name_to_id[objectName, propertyName]
        val = df.at[timestamp, col]
        # NOTE: add a check for a missing value
        if np.isnan(val):
            raise TypeError(f"Value was missing for col: {col} at time: {timestamp}\tobjectId: {objectID}, propertyId: {propertyID}, epoch: {epoch} (ms)")

        if output_property_id:
            if isinstance(output_property_id, int):
                propertyID = output_property_id
            elif isinstance(output_property_id, dict):
                propertyID = output_property_id[objectID]
            else:
                raise TypeError(f"{type(output_property_id)} is not supported. Supported types are NoneType, int or Dict[int] -> int")

        out[c] = [objectID, propertyID, val, epoch, 0]

    out_path = osp.join(output_dir, f"{filetime}.csv")
    # return the intermediate format bec there is additional info that needs to be written out here ... convert epoch back to s format
    # return out, columns, f"{filetime}.csv" , epoch / 1000

    final = pd.DataFrame(out, columns=columns)
    out_path = osp.join(output_dir, f"{filetime}.csv")
    final.to_csv(out_path, sep=';', index=False, header=False)
#     # TODO: return out so that additional info can be added to it before writing it out ... also return the colum names 

    return