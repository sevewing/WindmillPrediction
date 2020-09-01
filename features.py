"""
@ features.py: Feature Engineering
@ Thesis: Geographical Data and Predictions of Windmill Energy Production
@ Weisi Li
@ liwb@itu.dk, liweisi8121@hotmail.com
"""

import pandas as pd
import numpy as np
import time
from math import log,sqrt,atan2,pi,cos,sin
from multiprocessing import  Pool

from sklearn.preprocessing import LabelBinarizer
from scipy.interpolate import interp1d


get_ahead = None
get_by_grid_wu10 = None
get_by_grid_wv10 = None
get_by_grid_wu100 = None
get_by_grid_wv100 = None
get_weather = lambda x, fun: fun(x["grid"], x["TIME_CET"])

get_ws_hub_pow_r_u = lambda x :pow_law(x["wu10"], 10, x["Navhub_height"], rn_exponent(x["Roughness"]) if x["Roughness"]>=0.001 else rn_exponent(0.001))
get_ws_hub_pow_r_v = lambda x :pow_law(x["wv10"], 10, x["Navhub_height"], rn_exponent(x["Roughness"]) if x["Roughness"]>=0.001 else rn_exponent(0.001))
get_ws_hub_log_r_u = lambda x :log_law(x["wu10"], 10, x["Navhub_height"], x["Roughness"])
get_ws_hub_log_r_v = lambda x :log_law(x["wv10"], 10, x["Navhub_height"], x["Roughness"])

get_ws_hub_wsr_u = lambda x :pow_law(x["wu10"], 10, x["Navhub_height"], x["wsr_u"])
get_ws_hub_wsr_v = lambda x :pow_law(x["wv10"], 10, x["Navhub_height"], x["wsr_v"])

get_ws_by_uv = lambda u, v : (u ** 2 + v ** 2) ** 0.5
get_ws_by_uv = lambda u, v : (u ** 2 + v ** 2) ** 0.5 

get_by_grid_tmp2 = None
get_by_grid_tmp100 = None
get_tmp_hub = lambda x :thermal_interpolation(x["tmp2"], x["tmp100"], 2, 100,  x["Navhub_height"])

def to_vector(s, d):
    """
    Convert Wind speed and wind direction to u, v component.
    s: Wind speed
    d: Wind direction
    u: x-axis component
    v: y-axis component
    """
    u = s * cos(pi/180 * (270-d))
    v = s * sin(pi/180 * (270-d))
    return u, v

thermal_interpolation = lambda t1, t2, z1, z2, z_hat : (z_hat - z1) * (t2 - t1)  / (z2 - z1) + t1
pow_law = lambda v, z, z_hat, a : v * ( (z_hat/z) ** a)
log_law = lambda v, z, z_hat, rl : v * np.log(z_hat/rl) / np.log(z/rl)
rn_exponent = lambda rn : 0.096 * np.log10(rn) + 0.016 * np.log10(rn) ** 2 + 0.24

def pow_exponent(v1, v2, z1, z2):
    """
    Calaulate windshear in reverse by two layers
    """
    a1 = abs(v1)
    a2 = abs(v2)
    if v1 < 0 and v2 >= 0 or v1 >= 0 and v2 < 0:
        a2 = a1 + a2
    exp = np.log(a2/(a1+1e-06)+1e-06) / np.log(z2/z1)
    return exp


def wind_vinterp_exp(u1, v1, u2, v2, z_hat):
    """
    Wind u,v component vertical interpolation by a invert calculation of exponent.
    u1, v1 at height(AGL) 10m
    u2, v2 at height(AGL) 100m
    z_hat: a target height
    """
    exp_u = pow_exponent(u1, u2, 10, 100)
    exp_v = pow_exponent(v1, v2, 10, 100)
    u_i = round(pow_law(u1, 10, z_hat, exp_u),3)
    v_i = round(pow_law(v1, 10, z_hat, exp_v),3)
    return u_i, v_i

def wind_vinterp_rn(u, v, z_hat, rl):
    """
    Wind u,v component vertical interpolation by roughness length.
    u, v at height(AGL) 10m
    z_hat: a target height
    rl: roughness length
    """
    u_i = round(pow_law(u, 10, z_hat, rl),3)
    v_i = round(pow_law(v, 10, z_hat, rl),3)
    return u_i, v_i


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


def _extract_pow_rn_intep(df):
    """
    Extract Roughness based interpolation
    """
    df["hws_u_pow_rn"] = df.apply(lambda x: get_ws_hub_pow_r_u(x), axis=1)
    df["hws_v_pow_rn"] = df.apply(lambda x: get_ws_hub_pow_r_v(x), axis=1)
    df["hws_uv_pow_rn"] = df.apply(lambda x: get_ws_by_uv(x["hws_u_pow_rn"], x["hws_v_pow_rn"]), axis=1)
    df["hws_uv_pow_rn^2"] = df.apply(lambda x: x["hws_uv_pow_rn"] ** 2 , axis=1)
    df["hws_uv_pow_rn^3"] = df.apply(lambda x: x["hws_uv_pow_rn"] ** 3 , axis=1)
    return df

def _extract_log_rn_intep(df):
    """
    Extract Roughness based interpolation
    """
    df["hws_u_log_rn"] = df.apply(lambda x: get_ws_hub_log_r_u(x), axis=1)
    df["hws_v_log_rn"] = df.apply(lambda x: get_ws_hub_log_r_v(x), axis=1)
    df["hws_uv_log_rn"] = df.apply(lambda x: get_ws_by_uv(x["hws_u_log_rn"], x["hws_v_log_rn"]), axis=1)
    df["hws_uv_log_rn^2"] = df.apply(lambda x: x["hws_uv_log_rn"] ** 2 , axis=1)
    df["hws_uv_log_rn^3"] = df.apply(lambda x: x["hws_uv_log_rn"] ** 3 , axis=1)
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
    df["htmp_inp"] = df.apply(lambda x: get_tmp_hub(x), axis=1)
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
    (invalid)
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
        models.append(model)
        ws_qual[k] = pd.Series(model(X_1))
        df[feature + "_" + k] = pd.Series(model(X_2))

    return df, model, ws_qual


def extract(df):
    """
    Main extraction groups
    """
    df = _extract_basic(df)
    df = _extract_wsr_intep(df)
    df = _extract_pow_rn_intep(df)
    df = _extract_log_rn_intep(df)
    df = _extract_tmp_intep(df)
    df = _extract_time(df)

    df = df.fillna(0)
    return df


def parallelize_extract(df, n_cores=4):
    """
    Parallelized computation
    """
    start = time.time()

    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(extract, df_split))
    pool.close()
    pool.join()

    end = time.time()
    print("parallelize_extract time: ", end - start)
    return df