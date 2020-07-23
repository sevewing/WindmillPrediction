import matplotlib.pyplot as plt
from matplotlib import rcParams, cycler
cmap = plt.cm.winter



def Model_Loss(train_test_hist):
    plt.figure(figsize=(20,7))
    for label, th in train_test_hist.items():
        plt.plot(th, label=label)
    plt.legend()
    plt.show()

def timelines_plot(time, ys):
    time = time.apply(str)
    plt.figure(figsize=(30,7))
    for label, th in ys.items():
        plt.plot(time.apply(lambda x: x[-11:-6]), th, label=label)
    plt.legend()
    plt.show()


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