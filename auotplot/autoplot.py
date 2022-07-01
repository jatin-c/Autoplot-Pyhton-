import json
from dataprofiler import Data, Profiler
from pywaffle import Waffle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import uuid
from pywaffle import Waffle
from sqlalchemy import create_engine
from dateutil.parser import parse
engine = create_engine("mysql+mysqlconnector://root:sh00j0000@localhost/flaskapp", echo=True)

def histogram(df, count_json):
    n_bins=20
    # Creating distribution
    n_col=count_json['n_name']
    x = df[n_col]
    legend = ['distribution']
    
    # Creating histogram
    fig, axs = plt.subplots(1, 1,
                            figsize =(10, 7),
                            tight_layout = True)
    # Remove axes splines
    for s in ['top', 'bottom', 'left', 'right']:
        axs.spines[s].set_visible(False)
    
    # Remove x, y ticks
    axs.xaxis.set_ticks_position('none')
    axs.yaxis.set_ticks_position('none')
    
    # Add padding between axes and labels
    axs.xaxis.set_tick_params(pad = 5)
    axs.yaxis.set_tick_params(pad = 10)
    
    # Add x, y gridlines
    axs.grid(b = True, color ='grey',
            linestyle ='-.', linewidth = 0.5,
            alpha = 0.6)
    # Creating histogram
    N, bins, patches = axs.hist(x, bins = n_bins)
    
    # Setting color
    fracs = ((N**(1 / 5)) / N.max())
    norm = colors.Normalize(fracs.min(), fracs.max())
    
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    
    # Adding extra features   
    plt.xlabel("X-axis")
    plt.ylabel("y-axis")
    plt.legend(legend)
    plt.title('Customized histogram')
    plt.savefig('histogram.png',bbox_inches='tight')


def correllogram(df, count_json):
    plt.figure(figsize=(12,10), dpi= 80)
    sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)
    # Decorations
    plt.title('Correlogram of mtcars', fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig('correllogram.png',bbox_inches='tight')
    plt.clf()

def pie_chart(df, count_json):
    if count_json['n_count']==4:
        sumIt=round(df[count_json['n_name'][0]].sum(),2)
        sumprod=round(df[count_json['n_name'][1]].sum(),2)
        sumunprod=round(df[count_json['n_name'][2]].sum(),2)
        sumunprod1=round(df[count_json['n_name'][3]].sum(),2)
        labels = [count_json['n_name'][0], count_json['n_name'][1], count_json['n_name'][2],count_json['n_name'][3]]
        sizes = [sumIt, sumprod, sumunprod,sumunprod1]
        #colors
        colors = ['#ff9999','#66b3ff','#99ff99','#708090']

        #plot_analys=str(uuid.uuid1())
        #file_analys=name+"analyse.png"
        filepath_analyse="G:/VScode workspace/Autoplot/{path}".format(path=str(uuid.uuid1())+"analyse.png")
        plt.figure(figsize=(8, 8)) 
        explode = (0.05,0.05,0.05,0.05)
        fig, ax1 = plt.subplots()
        ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90,pctdistance=0.85, explode = explode)
        #draw circle
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')  
        plt.tight_layout()
        plt.legend(loc='lower left')
        plt.savefig(filepath_analyse,bbox_inches='tight')
        plt.close(fig)
    elif count_json['n_count']==3:
        sumIt=round(df[count_json['n_name'][0]].sum(),2)
        sumprod=round(df[count_json['n_name'][1]].sum(),2)
        sumunprod=round(df[count_json['n_name'][2]].sum(),2)
        labels = [count_json['n_name'][0], count_json['n_name'][1], count_json['n_name'][2]]
        sizes = [sumIt, sumprod, sumunprod]
        #colors
        colors = ['#ff9999','#66b3ff','#99ff99']
        filepath_analyse="G:/VScode workspace/Autoplot/{path}".format(path=str(uuid.uuid1())+"analyse.png")
        plt.figure(figsize=(8, 8)) 
        explode = (0.05,0.05,0.05)
        fig, ax1 = plt.subplots()
        ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90,pctdistance=0.85, explode = explode)
        #draw circle
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')  
        plt.tight_layout()
        plt.legend(loc='lower left')
        plt.savefig(filepath_analyse,bbox_inches='tight')
        plt.close(fig)

    return filepath_analyse


def company(df,engine):
    filepath="G:/VScode workspace/Autoplot/{path}".format(path=str(uuid.uuid1())+"lines.png")
    fig=df.plot(kind="line",stacked=False)
    plt.title('Company Statistics')
    fig.figure.savefig(filepath,bbox_inches='tight')
    plt.close(fig)

    return filepath

def pairplot(df, count_json):
    if count_json['c_count']==False:
        sns.set(style="ticks", color_codes=True)
        plot1=sns.pairplot(df[[count_json['n_name'][0],count_json['n_name'][1],count_json['n_name'][2], count_json['n_name'][3]]])
        plot1.savefig('pairplot.png',bbox_inches='tight')
        plt.clf()
    else:
        sns.set(style="ticks", color_codes=True)
        plot1=sns.pairplot(df[[count_json['n_name'][0],count_json['n_name'][1],count_json['n_name'][2], count_json['n_name'][3]]],hue=count_json['c_name'][0])
        plot1.savefig('pairplot.png',bbox_inches='tight')
        plt.clf()

def waffle_chart(df_raw,count_json): 
    groupby_col=count_json['c_name'][0]
    df = df_raw.groupby(groupby_col).size().reset_index(name='counts')
    n_categories = df.shape[0]
    colors = [plt.cm.inferno_r(i/float(n_categories)) for i in range(n_categories)]
    ttl="categoies distribution of "+groupby_col
    # Draw Plot and Decorate
    fig = plt.figure(
        FigureClass=Waffle,
        plots={
            '111': {
                'values': df['counts'],
                'labels': ["{0} ({1})".format(n[0], n[1]) for n in df[[groupby_col, 'counts']].itertuples()],
                'legend': {'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'fontsize': 12},
                'title': {'label': ttl, 'loc': 'center', 'fontsize':18}
            },
        },
        rows=7,
        colors=colors,
        figsize=(16, 9)
    )
    plt.savefig('waffle.png',bbox_inches='tight')


def scatterplot(df, count_json):
    cat_name=count_json['c_name'][0]
    categories = np.unique(df[cat_name])
    colors = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]

    # Draw Plot for Each Category
    plt.figure(figsize=(16, 10), dpi= 80, facecolor='w', edgecolor='k')

    for i, category in enumerate(categories):
        plt.scatter(count_json['n_name'][0], count_json['n_name'][1], 
                    data=df.loc[df['cat_name']==category, :], 
                    s=20, c=colors[i], label=str(category))
    # Decorations
    plt.gca().set(xlim=(0.0, 0.1), ylim=(0, 90000),
                xlabel=count_json['n_name'][0], ylabel=count_json['n_name'][1])
    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.legend(fontsize=12)    
    plt.savefig("scatter.png",bbox_inches='tight')


def stacked_chart(df, count_json):
    if count_json['n_count']==3:
        con_col=count_json['c_name'][0]
        df.set_index(con_col, inplace=True)
        fig, ax = plt.subplots(1, figsize=(16, 8))
        plt.rc('xtick', labelsize=13) 
        x = np.arange(0, len(df.index))
        plt.bar(x - 0.3, df[count_json['n_name'][0]], width = 0.4, color = '#BC8F8F')
        plt.bar(x - 0.1, df[count_json['n_name'][1]], width = 0.4, color = '#8B4513')
        plt.bar(x + 0.1, df[count_json['n_name'][2]], width = 0.4, color = '#708090')
        #plt.bar(x + 0.3, df_grouped['Other_Sales'], width = 0.2, color = '#FAC748')
        # remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # x y details
        plt.xticks(x, df.index)
        plt.xlim(-0.5, 6.5)
        # grid lines
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.2)
        # title and legend
        plt.legend([count_json['n_name'][0], count_json['n_name'][1], count_json['n_name'][2]], loc='upper left', ncol = 4)
        #fig=plt.show()
        #fig=res1.plot(kind="bar",stacked=True)
        filepath_stacked="G:/VScode workspace/Autoplot/{path}".format(path=str(uuid.uuid1())+"stacked.png")
        fig.savefig(filepath_stacked,bbox_inches='tight')
        plt.clf()
    elif count_json['n_count']==2:
        con_col=count_json['c_name'][0]
        df.set_index(con_col, inplace=True)
        fig, ax = plt.subplots(1, figsize=(16, 8))
        plt.rc('xtick', labelsize=13) 
        x = np.arange(0, len(df.index))
        plt.bar(x - 0.3, df[count_json['n_name'][0]], width = 0.4, color = '#BC8F8F')
        plt.bar(x - 0.1, df[count_json['n_name'][1]], width = 0.4, color = '#8B4513')
        #plt.bar(x + 0.3, df_grouped['Other_Sales'], width = 0.2, color = '#FAC748')
        # remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # x y details
        plt.xticks(x, df.index)
        plt.xlim(-0.5, 6.5)
        # grid lines
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.2)
        # title and legend
        plt.legend([count_json['n_name'][0], count_json['n_name'][1]], loc='upper left', ncol = 3)
        #fig=plt.show()
        #fig=res1.plot(kind="bar",stacked=True)
        filepath_stacked="G:/VScode workspace/Autoplot/{path}".format(path=str(uuid.uuid1())+"stacked.png")
        fig.savefig(filepath_stacked,bbox_inches='tight')
        plt.clf()

def bars(df, count_json):
    con_col=count_json['c_name'][0]
    df.set_index(con_col, inplace=True)
    fig, ax = plt.subplots(1, figsize=(16, 8))
    plt.rc('xtick', labelsize=13) 
    x = np.arange(0, len(df.index))
    plt.bar(x - 0.3, df[count_json['n_name'][0]], width = 0.4, color = '#BC8F8F')
    #plt.bar(x + 0.3, df_grouped['Other_Sales'], width = 0.2, color = '#FAC748')
    # remove spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # x y details
    plt.ylabel(count_json['n_name'][0])
    plt.xticks(x, df.index)
    plt.xlim(-0.5, 6.5)
    # grid lines
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.2)
    # title and legend
    plt.legend([count_json['n_name'][0]], loc='upper left', ncol = 2)
    #fig=plt.show()
    #fig=res1.plot(kind="bar",stacked=True)
    filepath_stacked="G:/VScode workspace/Autoplot/{path}".format(path=str(uuid.uuid1())+"stacked.png")
    fig.savefig(filepath_stacked,bbox_inches='tight')
    plt.clf()



def categorical_plots(df,count_json):
    x_var = count_json['c_name'][0]
    groupby_var = count_json['c_name'][1]
    df_agg = df.loc[:, [x_var, groupby_var]].groupby(groupby_var)
    vals = [df[x_var].values.tolist() for i, df in df_agg]

    # Draw
    plt.figure(figsize=(16,9), dpi= 80)
    colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
    n, bins, patches = plt.hist(vals, df[x_var].unique().__len__(), stacked=True, density=False, color=colors[:len(vals)])

    # Decoration
    plt.legend({group:col for group, col in zip(np.unique(df[groupby_var]).tolist(), colors[:len(vals)])})
    plt.title(f"Stacked Histogram of ${x_var}$ colored by ${groupby_var}$", fontsize=22)
    plt.xlabel(x_var)
    plt.ylabel("Frequency")
    plt.ylim(0, 40)
    plt.xticks(ticks=bins, labels=np.unique(df[x_var]).tolist(), rotation=90, horizontalalignment='left')
    plt.savefig("cat_plots.png",bbox_inches="tight")


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False
        
def profiler(df,count_json):
    if count_json['no_column']==1:
        if count_json['n_count']==False:
            waffle_chart(df,count_json)
        else:
            histogram(df,count_json)
    elif count_json['no_column']==2:
        if count_json['c_count']==False:
            scatterplot()
        elif count_json['n_count']==False:
            categorical_plots(df, count_json)
        else:
            bars(df, count_json)
    elif count_json['no_column']==3:
        if count_json['c_count']==1:
            scatterplot()
            stacked_chart()
        elif count_json['n_count']==3:
            pie_chart()
            pairplot()
    if count_json['no_column']==4:
        if count_json['n_count']==4:
            pairplot(df, count_json)
            pie_chart(df, count_json)
            correllogram(df, count_json)
        elif count_json['c_count']==1:
            stacked_chart(df, count_json)
    else:
        return "Can't visualize more than 4 dimensions"



    