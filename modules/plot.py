import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

def feature_importance(xgb,x_var,stock):
    # Plot feature importance
    f, ax = plt.subplots(1,1,figsize=(10,24))
    feature_importance = xgb.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)

    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(x_var)[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title(stock+' | Feature Importance')

def normalize_df(df,scale):
    #df_norm = (df-df.min().min())/(df-df.min().min()).max()  #All numbers 0-1
    df_norm = (df-0.5)*scale+0.5
    return df_norm

def render_mpl_table(data, ax, info='None', **kwargs):
    cmap=plt.get_cmap('RdYlGn')
    cmap.set_under('white')
    data_norm = data.pipe(normalize_df, 0.8)

    data = data.reset_index()
    data_norm = data_norm.reset_index()
    ax.grid(False)
    ax.axis('off')
    bbox = [0, 0, 1, 1]  # Table full screen
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns,
                         cellLoc='center', **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(16)
    row_label_height = 0.05
    cell_width = 0.055
    col_label_width = 0.4
    table_props = mpl_table.properties()
    table_cells = table_props['celld']
    cell_game = (-1, -1)
    cell_kpi = (-1, -1)
    for k, cell in sorted(table_cells.items()):
        cell_text = cell.get_text().get_text()
        #cell.set_edgecolor('black')
        #cell.set_facecolor('black')
        if k[0] == 0 and k[1] == 0:
            cell.set_text_props(fontsize=20, weight='bold', alpha=0.5)
            cell._text.set_text(info)
            cell.set_width(col_label_width)
            cell.set_height(row_label_height)
        elif k[1] == 0:  # HEADER COLUMN (gamename)
            cell.set_text_props(weight='bold', fontsize=17)
            cell.set_width(col_label_width)
            cell.set_linewidth(2)
        elif k[0] == 0:  # HEADER ROW (KPI)
            cell.set_height(row_label_height)
            cell.set_width(cell_width)
            cell.set_text_props(weight='normal', color='w', rotation=60, fontsize=16)
        else:  # Cells
            cell._text.set_text(cell_text[:4])
            cell.set_width(cell_width)
            #cell.set_text_props(color='black')
            cell.set_facecolor(cmap(data_norm.values[k[0] - 1, k[1]]))
            if cell_text == 'nan':
                cell.set_text_props(color='white')
    return mpl_table