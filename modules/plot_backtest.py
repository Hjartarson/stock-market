

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

from PIL import Image

def create_axis(figures = (1,1)):
    # Such function is used by the following projects:
    #      \bids-analysis-platform\analyses\monthly-report\smt-report
    #       \bids-analysis-platform\analyses\freerounds\acquisition
    slide_width = 20
    slide_height = 14
    f, ax = plt.subplots(figures[0], figures[1], sharex=False,
                         #figsize=(slide_width, slide_height)
                         )
    return f, ax

def save_fig(f, path):
    f.savefig(path, dpi=100, bbox_inches='tight')
    img = Image.open(path)
    img.show()


