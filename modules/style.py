import matplotlib.pyplot as plt
from cycler import cycler
import pandas as pd

def set_style():
    tableau20 = get_tableau()
    plt.rc('legend', frameon=True,fancybox=True,fontsize=14)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('lines', linewidth=1)
    plt.rc('grid', linestyle="--", color='grey',alpha = 0.2)
    plt.rc('font', size=14, family='sans-serif',style='normal',weight='normal')
    plt.rc('axes', labelsize=14, titlesize=14,linewidth=0,
           titleweight='bold',prop_cycle=(cycler('color',tableau20)))
    plt.rc('figure', figsize=(14, 8), titlesize=16)
    print('Style set')

def get_tableau():
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    return tableau20


def create_axis(figures = (1,1)):
    # Such function is used by the following projects:
    #      \bids-analysis-platform\analyses\monthly-report\smt-report
    #       \bids-analysis-platform\analyses\freerounds\acquisition
    slide_width = 15
    slide_height = 10
    f, ax = plt.subplots(figures[0], figures[1], sharex=False, figsize=(slide_width, slide_height))
    return f, ax


def fix_legend(ax,cols=4,handles='none',labels='none'):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.8])
    if handles == 'none':
        handles, labels = ax.get_legend_handles_labels()
    # Put a legend below current axis
    ax.legend(handles=handles,labels=labels,loc='upper center', bbox_to_anchor=(0.5, -0.08),
              fancybox=True, ncol=cols)
    return ax

def set_month_xticks(df_date,ax,rot=0):
    major_ticks = pd.to_datetime((df_date + pd.tseries.offsets.MonthBegin()).unique())
    major_tick_labels = list(map(lambda x: x.strftime('%b'), major_ticks))

    minor_ticks = list(filter(lambda x: x.month == 1, major_ticks))
    minor_tick_labels = list(map(lambda x: x.year, minor_ticks))

    ax.set_xticks(major_ticks, minor=False)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticklabels(major_tick_labels, minor=False, rotation=rot, ha='right',y=-0.03)
    ax.set_xticklabels(minor_tick_labels, minor=True, rotation=0, ha='right', weight='bold',alpha=1)
    ax.set_xlabel('')
    return ax