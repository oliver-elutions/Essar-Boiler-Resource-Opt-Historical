import pandas as pd

def calculate_cont_bounds(ctrl_df: pd.DataFrame):
    """
    Function calcualtes bounds for controllable tags
    Parameters
    ----------
    cont_tags: List[str]
        list of cont tags in model
    hist_df: pd.DataFrame
        historical data
    Returns
    -------
    bounds: List[(float, float)]
        bounds for each controllable tag
    """
    cont_tags = list(ctrl_df.columns)
    quantile_lower = [ctrl_df[tag].min() for tag in cont_tags]
    quantile_upper = [ctrl_df[tag].max() for tag in cont_tags]
    bounds = [list(a) for a in zip(quantile_lower, quantile_upper)]

    return bounds