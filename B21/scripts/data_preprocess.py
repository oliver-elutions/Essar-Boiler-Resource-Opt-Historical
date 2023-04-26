import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# local imports
from Utils import write_json

def main():
    """Main Function"""
    # process
    process = 'b21'
    HIST_DF_PATH = f'../outputs/{process}_hist_df.csv'
    TAGS_DICT_PATH = f'../outputs/{process}_tags_dict.json'

    # load in data, configs
    df = pd.read_csv(HIST_DF_PATH).set_index('Date')
    df.index = pd.to_datetime(df.index)

    # delete suffix from column name
    df.columns = [tag.replace(' Stanlow', '') for tag in df.columns]

    # load in tags list
    sheet = "B21"
    tags_list_df = pd.read_excel('../data/Boiler Resource Efficiency Tag List v0.xlsx', sheet_name = sheet)

    # map each tag to controllable, noncontrollable, response
    used_tags = df.columns.tolist()

    # controllable
    ctrl_df = tags_list_df.loc[tags_list_df['Type '] == 'Control', :]
    ctrl_tags = ctrl_df['Name'].tolist()

    # response
    response_df = tags_list_df.loc[tags_list_df['Type '] == 'Response', :]
    response_tags = response_df['Name'].tolist()

    # kpi tags
    kpi_df = tags_list_df.loc[tags_list_df['Type '] == 'KPI', :]
    kpi_tags = kpi_df['Name'].tolist()
    kpi_names = kpi_df['Description'].tolist()

    # noncontrollable
    indct_tags = [tag for tag in used_tags if tag not in ctrl_tags and tag not in response_tags and tag not in kpi_tags]

    # make config to track this
    tag_types_dict = dict()
    tag_types_dict['control'] = ctrl_tags
    tag_types_dict['noncontrol'] = indct_tags
    tag_types_dict['response'] = response_tags

    # map each tag to more meaningful name

    # get descriptions of used tags
    names_dict = tags_list_df.loc[tags_list_df['Name'].isin(used_tags), :].set_index('Name')['Description'].to_dict()

    # tags dict
    tags_dict = {}
    tags_dict['controllable'] = ctrl_tags
    tags_dict['noncontrollable'] = indct_tags
    tags_dict['response'] = response_tags
    tags_dict['kpi'] = kpi_tags

    # split into seperate dfs
    ctrl_df = df[ctrl_tags]
    response_df = df[response_tags]
    indct_df = df[indct_tags]


    # # map column names of df to meaningful name
    ctrl_df.columns = [v for k,v in names_dict.items() if k in ctrl_tags]
    response_df.columns = [v for k,v in names_dict.items() if k in response_tags]
    indct_df.columns = [v for k,v in names_dict.items() if k in indct_tags]

    ctrl_cols = ctrl_df.columns
    indct_cols = indct_df.columns
    resp_cols = response_df.columns

    all_df = pd.concat([ctrl_df, indct_df, response_df], axis = 1)

    def fix_bad_time_period(start_date, end_date):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        good_time_df = all_df[(all_df.index < start_date) | (all_df.index > end_date)]
        
        return good_time_df

    all_df = fix_bad_time_period("2022-01-01 00:00:00", "2022-01-24 00:00:00")
    all_df = fix_bad_time_period("2022-02-17 00:00:00", "2022-04-01 00:00:00")
    all_df = fix_bad_time_period("2022-09-25 00:00:00", "2022-10-20 00:00:00")
    all_df = fix_bad_time_period("2021-09-25 00:00:00", "2022-01-01 00:00:00")

    all_df = all_df.dropna(thresh = 8)

    ctrl_df = all_df.loc[:, ctrl_cols]
    indct_df = all_df.loc[:, indct_cols]
    response_df = all_df.loc[:, resp_cols]

    indct_df = indct_df.drop(['LCAP AIR IN B SIDE TEMP', 'FUEL TEMP', 'FUEL GAS'], axis = 1)

    def lin_interp(df, thresh):
        
        df_int = df.interpolate(method = 'linear', limit = thresh)
        print("Missing values after interpolation")
        print(round(df_int.isna().sum()/df_int.shape[0],3)*100)
        print("")
        
        return df_int

    ctrl_df = lin_interp(ctrl_df,6)
    indct_df = lin_interp(indct_df,6)
    response_df = lin_interp(response_df,6)

    def delete_low_vals(df):
        return df.loc[response_df['MP STEAM TO BOILER 21 AUXS'] >= 75, :]

    ctrl_df = delete_low_vals(ctrl_df)
    indct_df = delete_low_vals(indct_df)
    response_df = delete_low_vals(response_df)

    all_df = pd.concat([ctrl_df, indct_df, response_df], axis = 1)
    all_df = all_df.dropna()

    all_df = all_df.assign(RESPONSE=lambda x: x['MP STEAM TO BOILER 21 AUXS'] + x['VHP STEAM EX BOILER'])
    all_df = all_df.drop(['MP STEAM TO BOILER 21 AUXS', 'VHP STEAM EX BOILER'], axis = 1)
    all_df = all_df.drop(['LCAP AIR OUT B SIDE TEMP'], axis = 1)
    all_df = all_df.loc[all_df['FINAL STEAM TEMP CONTROL'] > 300, :]
    all_df.loc[all_df['COMBUSTION AIR CONTROL'] < 0, 'COMBUSTION AIR CONTROL'] = 0
    all_df.loc[all_df['FUEL OIL CONTROL'] < 0, 'FUEL OIL CONTROL'] = 0
    all_df.loc[all_df['B21 Gas Load'] < 0, 'B21 Gas Load'] = 0

    all_df = all_df.rename({v:k for k,v in names_dict.items()}, axis = 1)

    all_df.to_csv(f'../outputs/{process}_processed_df.csv')

    opt_df = all_df.loc[all_df.index.year == 2022, :]
    opt_df.to_csv(f'../outputs/{process}_test_df.csv')

    pre_opt_df = all_df[(all_df.index < pd.to_datetime("2022-01-01 00:00:00"))]
    pre_opt_df.to_csv(f'../outputs/{process}_train_df.csv')


    write_json(tags_dict, TAGS_DICT_PATH)

    return


if __name__ == '__main__':
    main()