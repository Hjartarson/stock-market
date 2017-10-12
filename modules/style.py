import matplotlib.pyplot as plt
from cycler import cycler

def set_style():
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    plt.rc('legend', frameon=True,fancybox=True,fontsize=14)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('grid', linestyle="--", color='grey',alpha = 0.2)
    plt.rc('font', size=14, family='sans-serif',style='normal',weight='normal')
    plt.rc('axes', labelsize=14, titlesize=14,
           titleweight='bold',prop_cycle=(cycler('color',tableau20)))
    print('Style set')