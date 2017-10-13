import matplotlib.patches as patches
import pandas as pd
import style as style
import os



def color_quantiles(ax,x_width,y_width):
    tableau = style.get_tableau()
    for nr, x, y in zip([0, 1, 2, 3], [-x_width, -x_width, 0, 0], [0, -y_width, 0, -y_width]):
        ax.add_patch(
            patches.Rectangle(
                (x, y),  # (x,y)
                x_width,  # width
                y_width,  # height
                alpha=0.1,
                color=tableau[nr]
            ))
    ax.set_xlim([-x_width, x_width])
    ax.set_ylim([-y_width, y_width])
    add_cross(ax)

def add_cross(ax):
    ax.axhline(0, ls='--', color='black')
    ax.axvline(0, ls='--', color='black')

def save_fig(f, filename):
    filepath = os.path.join('fig',filename)
    f.savefig(filepath, dpi=100,bbox_inches='tight')

def set_daily_xticks(df_date, ax, rot=0):
    major_ticks = pd.to_datetime(df_date.index.unique())
    major_tick_labels = list(map(lambda x: x.strftime('%a'), major_ticks))

    minor_ticks = list(filter(lambda x: x.dayofweek == 0, major_ticks))
    minor_tick_labels = list(map(lambda x: x.week, minor_ticks))

    ax.set_xticks(major_ticks, minor=False)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_xticklabels(major_tick_labels, minor=False, rotation=rot, ha='center', y=-0.03)
    ax.set_xticklabels(minor_tick_labels, minor=True, rotation=0, ha='center', weight='normal', alpha=1)

    ax.set_xlabel('')
    ax.grid()
    return ax

def add_measurement(df,x,m,ax,text=False,color='g'):
    y1 = df.loc[x[0]][m[0]]
    y2 = df.loc[x[1]][m[1]]
    if y2 < y1:
        text_color='r'
        color = 'r'
    else:
        text_color='g'
        color = 'g'
    d = {'datetime':x, text:[y1,y2]}
    df = pd.DataFrame(data=d).set_index('datetime')
    df.plot(marker='o', linestyle=':', color=color,ax=ax, legend=False,ms=6,alpha=1,lw=1)
    if text:
        text = str(round((y2/y1-1)*100,1))+'%'
        days = (x[1]-x[0]).days
        x_text = x[1]-pd.DateOffset(days=days/2)
        y_text = (y2+y1)/2
        ax.text(x_text, y_text, text,ha="center", va="center", color=text_color, bbox=dict(facecolor='white',
                                                                                           ec=color,ls=':'))
    return ax