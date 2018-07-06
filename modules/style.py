import matplotlib.pyplot as plt
from cycler import cycler

def set_style():
    tableau20 = get_tableau()
    plt.rc('legend', frameon=True,fancybox=True,fontsize=14)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('lines', linewidth=0.5)
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
    slide_width = 25.4
    slide_height = 15
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