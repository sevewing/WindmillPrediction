import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
cmap = plt.cm.winter
from constant import plot_path


def model_loss(train_test_hist, save_name=None):
    plt.figure(figsize=(20,7))
    for label, th in train_test_hist.items():
        plt.plot(th, label=label)
    plt.legend()
    # plt.show()
    if save_name is not None:
        plt.savefig(plot_path + save_name, dpi=150)


def timelines(time, ys, tp="line", figsize=(30,7), save_name=None):
    time = time.apply(str)
    time = time.apply(lambda x: x[-11:-6])
    if tp=="line":
        plt.figure(figsize=figsize)
        for label, th in ys.items():
            plt.plot(time, th, label=label, marker='o', linewidth=2)
    elif tp=="scatter":
        plt.figure(figsize=figsize)
        for label, th in ys.items():
            plt.scatter(time, th, label=label, marker='o')
    plt.grid(True, axis='y')
    plt.legend()
    # plt.show()
    if save_name is not None:
        plt.savefig(plot_path + save_name, dpi=150)

# def timelines(time, ys, line=True, save_name=None):
#     time = time.apply(str)
#     time = time.apply(lambda x: x[-11:-6])
#     plt.figure(figsize=(20,7))
#     plt.grid(True, axis='y')
#     for label, th in ys.items():
#         plt.plot(time, th, label=label, marker='o', linewidth=2)
#     plt.legend()
#     # plt.show()
#     if save_name is not None:
#         plt.savefig(plot_path + save_name, dpi=150)


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


def geo_precentage(df, features, save=True):
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
    ax.legend(ncol=len(features), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize=11)
    # plt.show()
    if save:
        plt.savefig(plot_path + "geo_precentage.png", dpi=150)