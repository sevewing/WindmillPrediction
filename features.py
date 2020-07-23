import pandas as pd
import numpy as np
# from math import log
import time
from multiprocessing import  Pool

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

pow_law = lambda v, z, z_hat, a : v * ( (z_hat/z) ** a) + 1e-06
get_ahead = None
get_by_grid_wu10 = None
get_by_grid_wv10 = None
get_by_grid_wu100 = None
get_by_grid_wv100 = None
get_weather = lambda x, fun: fun(x["grid"], x["TIME_CET"])

get_ws_hub_r_u = lambda x :pow_law(x["wu10"], 10, x["Navhub_height"], x["Roughness"])
get_ws_hub_r_v = lambda x :pow_law(x["wv10"], 10, x["Navhub_height"], x["Roughness"])
get_ws_hub_wsr_u = lambda x :pow_law(x["wu10"], 10, x["Navhub_height"], x["wsr_u"])
get_ws_hub_wsr_v = lambda x :pow_law(x["wv10"], 10, x["Navhub_height"], x["wsr_v"])
get_ws_by_uv = lambda u, v : (u ** 2 + v ** 2) ** 0.5

get_by_grid_tmp2 = None
get_by_grid_tmp100 = None
get_tmp_hub = lambda x :pow_law(x["tmp2"], 2, x["Navhub_height"], x["exp_tmp"])

def fun_register(wu10, wv10, wu100, wv100, tmp2=None, tmp100=None):
    global get_ahead
    global get_by_grid_wu10
    global get_by_grid_wv10
    global get_by_grid_wu100
    global get_by_grid_wv100
    global get_by_grid_tmp2
    global get_by_grid_tmp100
    global tmp

    get_ahead = lambda t: get_by_grid(wu10, "predicted_ahead", t)
    get_by_grid_wu10 = lambda g, t: get_by_grid(wu10, g, t)
    get_by_grid_wv10 = lambda g, t: get_by_grid(wv10, g, t)
    get_by_grid_wu100 = lambda g, t: get_by_grid(wu100, g, t)
    get_by_grid_wv100 = lambda g, t: get_by_grid(wv100, g, t)

    tmp = True if tmp2 is not None and tmp100 is not None else False
    if tmp:
        get_by_grid_tmp2 = lambda g, t: get_by_grid(tmp2, g, t)
        get_by_grid_tmp100 = lambda g, t: get_by_grid(tmp100, g, t)
        
def get_by_grid(df, g, t):
    try:
        return df[df['TIME_CET'] == t][g].tolist()[0]
    except:
        return 0


def windshear(v1, v2, z1, z2):
    """
    Calaulate windshear in reverse by two layers
    """
    a1 = abs(v1)
    a2 = abs(v2)
    if v1 < 0 and v2 >= 0 or v1 >= 0 and v2 < 0:
        a2 = a1 + a2
    a = np.log(a2/(a1+1e-06)+1e-06) / np.log(z2/z1)
    return a

def _boundary_limit(df, f, boundary):
    df[f] = df[f].apply(lambda x: 0 if x < boundary[0] else x)
    # df[f] = df[f].apply(lambda x: 0 if x < boundary[0] else boundary[1] if x > boundary[1] else x)
    # df[f] = df[f].apply(lambda x: 0 if x < boundary[0] else boundary[1] if x > boundary[1] else x)
    try:
        index = df[df[f] == boundary[0]].index.values.astype(int)[0]
        for i in range(index):
            df.iloc[i][f] = boundary[0]
    except:
        pass

    # try:
    #     index = df[df[f] == boundary[1]].index.values.astype(int)[0]
    #     for i in range(index, len(temp)):
    #         df.iloc[i][f] = boundary[1]
    # except:
    #     pass

    return df


def extract_quantiles(df, feature, boundary=[]):
    temp = df[[feature, "VAERDI"]]
    temp[feature] = round(temp[feature],0)
    quantiles = {"q0.1":0.1, "q0.3":0.3, "q0.5":0.5, "q0.7":0.7, "q0.9":0.9}

    aggs = {"VAERDI" : [lambda x :x.quantile(0.1), 
                        lambda x :x.quantile(0.3),
                        lambda x :x.quantile(0.5),
                        lambda x :x.quantile(0.7),
                        lambda x :x.quantile(0.9)]}   
    temp = temp.groupby(feature, as_index=False).agg(aggs).fillna(0).sort_values(by=feature)

    temp = pd.DataFrame(temp.values, columns = [feature] + list(quantiles.keys()))

    X_1 = temp[feature].values.reshape(-1, 1)
    X_2 = df[feature].values.reshape(-1, 1)
    for k in quantiles.keys():
        y = temp[k]
        model = make_pipeline(PolynomialFeatures(3), Ridge())
        model.fit(X_1, y)
        temp[k] = pd.Series(model.predict(X_1))
        df[feature + "_" + k] = pd.Series(model.predict(X_2))

        if len(boundary) > 0:
            temp = _boundary_limit(temp, k, boundary)
            df = _boundary_limit(df, feature + "_" + k, boundary)

    return df, temp


def extract(df):
    df["predicted_ahead"] = df.apply(lambda x: get_ahead(x["TIME_CET"]), axis=1)

    df["wu10"] = df.apply(lambda x: get_weather(x, get_by_grid_wu10), axis=1)
    df["wv10"] = df.apply(lambda x: get_weather(x, get_by_grid_wv10), axis=1)
    df["wu100"] = df.apply(lambda x: get_weather(x, get_by_grid_wu100), axis=1)
    df["wv100"] = df.apply(lambda x: get_weather(x, get_by_grid_wv100), axis=1)


    df["wsr_u"] = df.apply(lambda x: windshear(x["wu10"], x["wu100"], 10, 100), axis=1)
    df["wsr_v"] = df.apply(lambda x: windshear(x["wv10"], x["wv100"], 10, 100), axis=1)

    df["hws_u_rn"] = df.apply(lambda x: get_ws_hub_r_u(x), axis=1)
    df["hws_v_rn"] = df.apply(lambda x: get_ws_hub_r_v(x), axis=1)
    df["hws_uv_rn"] = df.apply(lambda x: get_ws_by_uv(x["hws_u_rn"], x["hws_v_rn"]), axis=1)

    df["hws_u_wsr"] = df.apply(lambda x: get_ws_hub_wsr_u(x), axis=1)
    df["hws_v_wsr"] = df.apply(lambda x: get_ws_hub_wsr_v(x), axis=1)
    df["hws_uv_wsr"] = df.apply(lambda x: get_ws_by_uv(x["hws_u_wsr"], x["hws_v_wsr"]), axis=1)

    df["hws_uv_rn^2"] = df.apply(lambda x: x["hws_uv_rn"] ** 2 , axis=1)
    df["hws_uv_rn^3"] = df.apply(lambda x: x["hws_uv_rn"] ** 3 , axis=1)

    df["hws_uv_wsr^2"] = df.apply(lambda x: x["hws_uv_wsr"] ** 2 , axis=1)
    df["hws_uv_wsr^3"] = df.apply(lambda x: x["hws_uv_wsr"] ** 3 , axis=1)

    if tmp:
        df["tmp2"] = df.apply(lambda x: get_weather(x, get_by_grid_tmp2), axis=1)
        df["tmp100"] = df.apply(lambda x: get_weather(x, get_by_grid_tmp100), axis=1)
        df["exp_tmp"] = df.apply(lambda x: windshear(x["tmp2"], x["tmp100"], 2, 100), axis=1)
        df["htmp_exp"] = df.apply(lambda x: get_tmp_hub(x), axis=1)

    boundary = [0, df["Capacity_kw"].values[0]]
    df,_ = extract_quantiles(df, "hws_uv_rn", boundary)
    df,_ = extract_quantiles(df, "hws_uv_wsr", boundary)

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

    df = df.fillna(0)

    return df

def parallelize_extract(df, n_cores=4):
    start = time.time()

    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(extract, df_split))
    pool.close()
    pool.join()

    end = time.time()
    print(end - start)
    return df