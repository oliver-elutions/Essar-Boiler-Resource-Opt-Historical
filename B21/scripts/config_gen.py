import shutil
import pandas as pd

from Utils import read_json, write_json
from config_helpers import calculate_cont_bounds
import data_mapper
from data_mapper import conversion

def main():
    """"Main Function"""
    process = "b21"

    # Read Files
    print('Reading in Files')
    config = read_json('../data/config_gen_config.json')
    hist_df = pd.read_csv(f'../outputs/{process}_processed_df.csv').set_index('Date')
    hist_df.index = pd.to_datetime(hist_df.index)
    mapper_df = pd.read_excel('../data/Essar Mapping.xlsx', sheet_name = 'Datapoint Mappings')
    tags_dict = read_json(f'../outputs/{process}_tags_dict.json')


    # OPTIMIZER.JSON
    # controllables
    ctrl_mod_tags = [tag for tag in tags_dict['controllable'] if tag in hist_df.columns]
    ctrl_df = hist_df.loc[:, ctrl_mod_tags]
    ctrl_bounds = calculate_cont_bounds(ctrl_df)
    ctrl_dict = {tag:{'rate': rate, 'bounds':bound} for tag,rate,bound in zip(ctrl_mod_tags, config['control_change_rates'], ctrl_bounds)}

    # noncontrollables
    indct_mod_tags = [tag for tag in tags_dict['noncontrollable'] if tag in hist_df.columns]

    # make dictionary
    opt_dict = dict()
    opt_dict['kwargs'] = {'maxiter': 10}
    opt_dict['controllable'] = ctrl_dict
    opt_dict['noncontrollable'] = indct_mod_tags


    # INFO.JSON (add in feature engineering tags here if needed)
    map_ctrl_tags = [tag + " Stanlow" for tag in ctrl_mod_tags]
    map_indct_tags = [tag + " Stanlow" for tag in indct_mod_tags]
    map_resp_tags = [tag + " Stanlow" for tag in tags_dict['response']]
    map_kpi_tags = [tag + " Stanlow" for tag in tags_dict['kpi']]

    print(f"Control tags: {len(map_ctrl_tags)}")
    print(f"Non Control tags: {len(map_indct_tags)}")
    print(f"Response tags: {len(map_resp_tags)}")
    print(f"KPI tags: {len(map_kpi_tags)}")
    print("")

    controllable_ids = list(set(conversion.get_ids_from_names(mapper_df, map_ctrl_tags)))
    noncontrollable_ids = list(set(conversion.get_ids_from_names(mapper_df, map_indct_tags)))
    response_ids = list(set(conversion.get_ids_from_names(mapper_df, map_resp_tags)))
    kpi_ids = list(set(conversion.get_ids_from_names(mapper_df, map_kpi_tags)))

    print(f"Mapped Control tags: {len(controllable_ids)}")
    print(f"Mapped Non Control tags: {len(noncontrollable_ids)}")
    print(f"Mapped Response tags: {len(response_ids)}")
    print(f"Mapped KPI tags: {len(kpi_ids)}")

    info = dict()
    info["read_params"] = {
        'args': {},
        'kwargs': {
            'sheet_name': "Datapoint Mappings"
        }}
    info['mapper_path'] = "../Model/Essar Mapping.xlsx"
    info["output_property_id"] = -180
    info['failure_label'] = -99
    info['property_ids'] = -6
    info['controllable_ids'] = controllable_ids
    info['noncontrollable_ids'] = noncontrollable_ids 
    info['response_ids'] = response_ids
    info['kpi_ids'] = kpi_ids


    # USED DP NAMES (add in feature engineering tags here)
    live_tags = [*map_ctrl_tags, *map_indct_tags, *map_resp_tags, *map_kpi_tags]
    live_tags_df = pd.DataFrame({'tags': live_tags})
    print(live_tags_df.shape)


    # PREVIOUS TAG VALS
    hist_df.columns = [tag + " Stanlow" for tag in hist_df.columns]
    mod_df = hist_df.loc[:, [*map_ctrl_tags, *map_indct_tags]].tail(4)
    extra_tags = [tag for tag in live_tags if tag not in mod_df.columns]

    for tag in extra_tags:
        tag_df = pd.DataFrame({f'{tag}': [1,1,1,1]})
        tag_df.index = mod_df.index
        mod_df = pd.concat([mod_df, tag_df], axis = 1)

    previous_tag_vals = mod_df.copy()


    # LIVE DATA
    mh = data_mapper.MapperHandler(*config['data_mapper_columns'])
    live_format, sub, name_to_id, id_to_name = conversion.convert_to_ld(previous_tag_vals, mapper_df, mh, config['property_ids'])
    all_object_ids = [*info['controllable_ids'], *info['noncontrollable_ids'], *info['response_ids'], *info['kpi_ids']]
    all_object_names = ['___'.join(id_to_name[object_id, -6]) for object_id in all_object_ids]

    for i in range(3):
        data_mapper.put_data(live_format.loc[:, all_object_names].iloc[[i], :], '../outputs/', name_to_id)   


    # save configs
    print("Saving Results")
    write_json(opt_dict, f'../outputs/{process}_optimizer.json')
    write_json(info, f'../outputs/{process}_info.json')
    live_tags_df.to_csv(f'../outputs/{process}_UsedDatapointNames.csv', index = False)
    previous_tag_vals.to_csv(f'../outputs/{process}_previous_tag_vals.csv')

    return

if __name__ == '__main__':
    main()






