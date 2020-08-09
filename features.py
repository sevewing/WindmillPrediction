import pandas as pd
import numpy as np
# from math import log
import time
from multiprocessing import  Pool

from sklearn.preprocessing import LabelBinarizer
from scipy.interpolate import interp1d

from tools import pow_exponent, pow_law, rn_exponent


get_ahead = None
get_by_grid_wu10 = None
get_by_grid_wv10 = None
get_by_grid_wu100 = None
get_by_grid_wv100 = None
get_weather = lambda x, fun: fun(x["grid"], x["TIME_CET"])

# get_ws_hub_r_u = lambda x :pow_law(x["wu10"], 10, x["Navhub_height"], rn_exponent(x["Roughness"]))
# get_ws_hub_r_v = lambda x :pow_law(x["wv10"], 10, x["Navhub_height"], rn_exponent(x["Roughness"]))
# get_ws_hub_wsr_u = lambda x :pow_law(x["wu10"], 10, x["Navhub_height"], rn_exponent(x["wsr_u"]))
# get_ws_hub_wsr_v = lambda x :pow_law(x["wv10"], 10, x["Navhub_height"], rn_exponent(x["wsr_v"]))

get_ws_hub_r_u = lambda x :pow_law(x["wu10"], 10, x["Navhub_height"], x["Roughness"])
get_ws_hub_r_v = lambda x :pow_law(x["wv10"], 10, x["Navhub_height"], x["Roughness"])
get_ws_hub_wsr_u = lambda x :pow_law(x["wu10"], 10, x["Navhub_height"], x["wsr_u"])
get_ws_hub_wsr_v = lambda x :pow_law(x["wv10"], 10, x["Navhub_height"], x["wsr_v"])

get_ws_by_uv = lambda u, v : (u ** 2 + v ** 2) ** 0.5
get_ws_by_uv = lambda u, v : (u ** 2 + v ** 2) ** 0.5 

get_by_grid_tmp2 = None
get_by_grid_tmp100 = None
get_tmp_hub = lambda x :pow_law(x["tmp2"], 2, x["Navhub_height"], x["exp_tmp"])


def get_by_grid(df, g, t):
    """
    Get weather data by grid
    """
    try:
        return df[df['TIME_CET'] == t][g].tolist()[0]
    except:
        return 0

def fun_register(wu10, wv10, wu100, wv100, tmp2=None, tmp100=None):
    """
    Register weather data extract functions
    """
    global get_ahead
    global get_by_grid_wu10
    global get_by_grid_wv10
    global get_by_grid_wu100
    global get_by_grid_wv100
    global get_by_grid_tmp2
    global get_by_grid_tmp100
    
    get_ahead = lambda t: get_by_grid(wu10, "predicted_ahead", t)
    get_by_grid_wu10 = lambda g, t: get_by_grid(wu10, g, t)
    get_by_grid_wv10 = lambda g, t: get_by_grid(wv10, g, t)
    get_by_grid_wu100 = lambda g, t: get_by_grid(wu100, g, t)
    get_by_grid_wv100 = lambda g, t: get_by_grid(wv100, g, t)

    get_by_grid_tmp2 = lambda g, t: get_by_grid(tmp2, g, t)
    get_by_grid_tmp100 = lambda g, t: get_by_grid(tmp100, g, t) 


# def _boundary_limit(df, f, boundary):
#     df[f] = df[f].apply(lambda x: 0 if x < boundary[0] else x)
#     # df[f] = df[f].apply(lambda x: 0 if x < boundary[0] else boundary[1] if x > boundary[1] else x)
#     # df[f] = df[f].apply(lambda x: 0 if x < boundary[0] else boundary[1] if x > boundary[1] else x)
#     try:
#         index = df[df[f] == boundary[0]].index.values.astype(int)[0]
#         for i in range(index):
#             df.iloc[i][f] = boundary[0]
#     except:
#         pass

#     # try:
#     #     index = df[df[f] == boundary[1]].index.values.astype(int)[0]
#     #     for i in range(index, len(temp)):
#     #         df.iloc[i][f] = boundary[1]
#     # except:
#     #     pass

#     return df



def _extract_basic(df):
    """
    Extract Basic wind info
    """
    df["wu10"] = df.apply(lambda x: get_weather(x, get_by_grid_wu10), axis=1)
    df["wv10"] = df.apply(lambda x: get_weather(x, get_by_grid_wv10), axis=1)
    df["wu100"] = df.apply(lambda x: get_weather(x, get_by_grid_wu100), axis=1)
    df["wv100"] = df.apply(lambda x: get_weather(x, get_by_grid_wv100), axis=1)
    df["tmp2"] = df.apply(lambda x: get_weather(x, get_by_grid_tmp2), axis=1)
    df["tmp100"] = df.apply(lambda x: get_weather(x, get_by_grid_tmp100), axis=1)
    return df


def _extract_rn_intep(df):
    """
    Extract Roughness based interpolation
    """
    df["hws_u_rn"] = df.apply(lambda x: get_ws_hub_r_u(x), axis=1)
    df["hws_v_rn"] = df.apply(lambda x: get_ws_hub_r_v(x), axis=1)
    df["hws_uv_rn"] = df.apply(lambda x: get_ws_by_uv(x["hws_u_rn"], x["hws_v_rn"]), axis=1)
    df["hws_uv_rn^2"] = df.apply(lambda x: x["hws_uv_rn"] ** 2 , axis=1)
    df["hws_uv_rn^3"] = df.apply(lambda x: x["hws_uv_rn"] ** 3 , axis=1)
    return df

def _extract_wsr_intep(df):
    """
    Extract 2-Layers windshear based interpolation
    """
    df["wsr_u"] = df.apply(lambda x: pow_exponent(x["wu10"], x["wu100"], 10, 100), axis=1)
    df["wsr_v"] = df.apply(lambda x: pow_exponent(x["wv10"], x["wv100"], 10, 100), axis=1)
    df["hws_u_wsr"] = df.apply(lambda x: get_ws_hub_wsr_u(x), axis=1)
    df["hws_v_wsr"] = df.apply(lambda x: get_ws_hub_wsr_v(x), axis=1)
    df["hws_uv_wsr"] = df.apply(lambda x: get_ws_by_uv(x["hws_u_wsr"], x["hws_v_wsr"]), axis=1)
    df["hws_uv_wsr^2"] = df.apply(lambda x: x["hws_uv_wsr"] ** 2 , axis=1)
    df["hws_uv_wsr^3"] = df.apply(lambda x: x["hws_uv_wsr"] ** 3 , axis=1)
    return df

def _extract_tmp_intep(df):
    """
    Extract Temperature
    """
    
    df["exp_tmp"] = df.apply(lambda x: pow_exponent(x["tmp2"], x["tmp100"], 2, 100), axis=1)
    df["htmp_exp"] = df.apply(lambda x: get_tmp_hub(x), axis=1)
    return df

def _extract_time(df):
    """
    Extract month and hour as independent columns
    """
    df["month"] = df.apply(lambda x: int(x["TIME_CET"][5:7]), axis=1)
    temp = pd.DataFrame(columns=["m" + str(x) for x in range(1, 13)])
    df = pd.concat([df, temp], axis=1)
    for i, v in df.iterrows():
        df.loc[i, "m" + str(v["month"])] = 1

    df["hour"] = df.apply(lambda x: int(x["TIME_CET"][11:13]), axis=1)
    temp = pd.DataFrame(columns=["h" + str(x) for x in range(0, 24)])
    df = pd.concat([df, temp], axis=1)
    for i, v in df.iterrows():
        df.loc[i, "h" + str(v["hour"])] = 1
    return df

def extract_quantiles(df, feature):
    """
    Extract quantiles by interpolation 1d
    """
    ws_qual = df[[feature, "VAERDI"]]
    ws_qual[feature] = round(ws_qual[feature], 1)
    quantiles = {"q0.1":0.1, "q0.5":0.5, "q0.9":0.9}
    models = []
    aggs = {"VAERDI" : [lambda x :x.quantile(0.1), 
                        # lambda x :x.quantile(0.3),
                        lambda x :x.quantile(0.5),
                        # lambda x :x.quantile(0.7),
                        lambda x :x.quantile(0.9)]}   
    ws_qual = ws_qual.groupby(feature, as_index=False).agg(aggs).fillna(0).sort_values(by=feature)
    ws_qual = pd.DataFrame(ws_qual.values, columns = [feature] + list(quantiles.keys()))

    X_1 = ws_qual[feature].values.reshape(-1)
    X_2 = df[feature].values.reshape(-1)
    for k in quantiles.keys():
        y = ws_qual[k]
        model = interp1d(X_1, y, bounds_error=False, fill_value='extrapolate')
        # model = make_pipeline(PolynomialFeatures(3), Ridge())
        # model.fit(X_1, y)
        models.append(model)
        ws_qual[k] = pd.Series(model(X_1))
        df[feature + "_" + k] = pd.Series(model(X_2))

    return df, model, ws_qual


def extract(df):
    """
    Main extraction
    """
    df = _extract_basic(df)
    df = _extract_wsr_intep(df)
    df = _extract_rn_intep(df)
    df = _extract_tmp_intep(df)
    df = _extract_time(df)

    df = df.fillna(0)
    return df

# def extract_park(df):
#     """
#     Main extraction
#     """
#     df = _extract_basic(df)
#     df = _extract_time(df)

#     df = df.fillna(0)
#     return df

def parallelize_extract(df, n_cores=4):
    """
    Parallelized computation
    """
    start = time.time()

    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    # if tp == 'single':
    df = pd.concat(pool.map(extract, df_split))
    # elif tp == 'park':
    #     df = pd.concat(pool.map(extract_park, df_split))
    pool.close()
    pool.join()

    end = time.time()
    print("parallelize_extract time: ", end - start)
    return df