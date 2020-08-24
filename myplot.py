import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
cmap = plt.cm.winter
from constant import plot_path

coldic = {"Original":"#394a6d", "Windshear":"#3c9d9b", "Geo":"#52de97", "Semigeo":"#9ACC8F", "Measured Power":"#DE5299"}


def model_loss(train_test_hist, path=None):
    plt.figure(figsize=(20,7))
    for label, th in train_test_hist.items():
        plt.plot(th, label=label)
    plt.xlabel("Echos Times")
    plt.legend()
    # plt.show()
    if path is not None:
        plt.savefig(path, dpi=150, bbox_inches='tight')


def timelines(time, ys, xlabel="", ylabel="", tp="line", fulltime=True, figsize=(30,7), path=None):
    if fulltime:
        time = time.apply(str)
        time = time.apply(lambda x: x[-11:-6])
        time_gap = [x for x in time if x[-2:]=='23']

    if tp=="line":
        plt.figure(figsize=figsize)
        for label, th in ys.items():
            plt.plot(time, th, label=label, marker='o', linewidth=2, color=coldic[label])
    elif tp=="scatter":
        plt.figure(figsize=figsize)
        for label, th in ys.items():
            plt.scatter(time, th, label=label, marker='o', color=coldic[label])
    if fulltime and len(time_gap) > 1:
        for i in range(0, len(time_gap)-1):    
            plt.axvline(x=time_gap[i], color='gray', linestyle='--')
    plt.xticks(rotation=75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, axis='y')
    plt.legend()
    # plt.show()
    if path is not None:
        plt.savefig(path, dpi=150, bbox_inches='tight')


def ploynomial_quantile_windpower_curves(X, ys, scatterplt=[], boundary=[], degree=3):
    plt.figure(figsize=(10,10))
    if len(scatterplt) > 0:
        plt.scatter(scatterplt[0], scatterplt[1], alpha=0.1, color="k")

    for y in ys:
        model= make_pipeline(PolynomialFeatures(degree), Ridge())
        model.fit(X, y)
        pred = model.predict(X)

        if len(boundary) > 0:
            pred[pred < boundary[0]] = boundary[0]
            pred[pred > boundary[1]] = boundary[1]
            pred = pred.tolist()
            try:
                for i in range(pred.index(boundary[0])):
                    pred[i] = boundary[0]
            except:
                pass

            try:
                for i in range(pred.index(boundary[1]), len(pred)):
                    pred[i] = boundary[1]
            except:
                pass

        plt.plot(X, pred, linewidth=2)


def wind_power_scatter(df_1_wsr,df_1_rn,df_2_wsr,df_2_rn, feature, xlim):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))

    axs[0, 0].set_title("Windmill_1 Wind Shear")
    axs[0, 0].scatter(df_1_wsr["hws_uv_wsr"], df_1_wsr["VAERDI"], color='C0')
    axs[0, 0].set_xlim(xlim)
    axs[0, 0].set_xlabel("Wind Speed (m/s)")
    axs[0, 0].set_ylabel("Generation Power (kw)")

    axs[1, 0].set_title("Windmill_1 Roughness")
    axs[1, 0].scatter(df_1_rn["hws_uv_rn"], df_1_rn["VAERDI"], color='C1')
    axs[1, 0].set_xlim(xlim)
    axs[1, 0].set_xlabel("Wind Speed (m/s)")
    axs[1, 0].set_ylabel("Generation Power (kw)")

    axs[0, 1].set_title("Windmill_2 Wind Shear")
    axs[0, 1].scatter(df_2_wsr["hws_uv_wsr"], df_2_wsr["VAERDI"], color='C0')
    axs[0, 1].set_xlim(xlim)
    axs[0, 1].set_xlabel("Wind Speed (m/s)")
    axs[0, 1].set_ylabel("Generation Power (kw)")

    axs[1, 1].set_title("Windmill_2 Roughness")
    axs[1, 1].scatter(df_2_rn["hws_uv_rn"], df_2_rn["VAERDI"], color='C1')
    axs[1, 1].set_xlim(xlim)
    axs[1, 1].set_xlabel("Wind Speed (m/s)")
    axs[1, 1].set_ylabel("Generation Power (kw)")


def geo_precentage(df, features, path=None):
    labels = df.index.to_list()
    data = df[features].values 
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, len(features)))

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(features, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.6,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.4 else 'grey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c))+"%", ha='center', va='center',
                    color=text_color, fontsize=13)
    
    ncol = len(features) if len(features) <= 10 else len(features)//2+1
    ax.legend(ncol=ncol, bbox_to_anchor=(0, 1),
              loc='lower left', fontsize=11)
    # plt.show()
    if path is not None:
        plt.savefig(path, dpi=150, bbox_inches='tight')


def roughness_simulation(ls, h_range, path=None):
    plt.figure(figsize=(10, 10))
    plt.xlim(0, 10.5)
    plt.ylim(0, 110)
    x_new = np.linspace(0,11,100)
    for rn, ws in ls.iterrows():
        if rn == 0.000001:
            plt.plot(ws, h_range, label='{:f}'.format(rn).rstrip('0'), linewidth=4)
        else:
            plt.plot(ws, h_range, label='{:f}'.format(rn).rstrip('0'))
    # plt.axvline(x=6, color='gray', linestyle='--')
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("AGL (m)")
    plt.legend(loc="upper left")
    # plt.show()
    if path is not None:
        plt.savefig(path, dpi=150, bbox_inches='tight')


def k_fold_validation(n_groups, k_scores, path=None):
    k_range = range(1, n_groups) 
    plt.figure(figsize=(10, 7))
    plt.plot(k_range, k_scores, marker='o')  
    plt.xticks(k_range)
    plt.xlabel('Value of K')
    plt.ylabel('Cross-Validated SmoothL1Loss')  
    # plt.show()
    if path is not None:
        plt.savefig(path, dpi=150, bbox_inches='tight')