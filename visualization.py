#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
- File: visualization.py
- Description: 
- Author: Qi Liu
- Time: Created: 2021/8/18 11:48
        Last Update: 2021/8/18 11:48
- History: 
- TODO: 
- License: MIT
"""

# Futures
from __future__ import print_function

# Built-in/Generic Imports
import os
import sys
from os.path import join as pjoin

# Libs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# magic commands (jupyter notebook/lab)
# %matplotlib inline
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg')
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Chinese fonts
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.dpi'] = 100
plt.rcParams['axes.axisbelow'] = True

# Own modules

__author__ = 'Qi Liu'
__copyright__ = 'Copyright 2021, Ligand based VS'
__credits__ = ['Qi Liu']
__license__ = 'MIT'
__version__ = '0.0.1'
__maintainer__ = 'Qi Liu'
__email__ = 'qi.liu@ghddi.org'
__status__ = 'dev'


class ResultsVisualizer:

    def __init__(self, source_root='./',  files=None):
        self.source_root = source_root
        if files is None:
            files = {
                'mfold_logs': pjoin(self.source_root, "output/202108181435/mfold_logs_all.csv")
            }
        self.files = files
        self.df_mfold_logs = self.get_mfold_logs()

    def get_df(self, kw, **kwargs):
        return pd.read_csv(self.files[kw], **kwargs)

    def get_mfold_logs(self, kw='mfold_logs'):
        return self.get_df(kw=kw)

    def display_mfold_logs(
        self,
        phase='global_only',
        title=None,
        xlabel='Data Type',
        ylabel='Best AuPR',
        *,
        flag_display=True,
        flag_save_plot=False,
        save_to="results/images",
        filename=None,
        ylim=(0.5, 1)
    ):
        if filename is None:
            filename = phase
        df = self.df_mfold_logs
        query = f"phase == '{phase}'"
        df_selected = df.query(query)
        g = sns.boxplot(x=df_selected['data_type'], y=df_selected['best_aupr'])
        sns.swarmplot(x=df_selected['data_type'], y=df_selected['best_aupr'], color=".25")
        g.set_title(title)
        g.set_xlabel(xlabel)
        g.set_ylabel(ylabel)
        g.set_ylim(ylim)

        if flag_save_plot:
            save_to = save_to if save_to is not None else "results/images"
            for ext in ['pdf', 'svg', 'jpeg']:
                path = pjoin(save_to, ext)
                if not os.path.exists(path):
                    os.makedirs(path)
                file = pjoin(path, f"{filename}.{ext}")
                plt.savefig(file, bbox_inches='tight')
            file = pjoin(save_to, f"{filename}.png")
            plt.savefig(file, bbox_inches='tight')

        if flag_display:
            plt.show()


def main():
    rv = ResultsVisualizer()
    phases = ['global_only', 'global_fixed', 'mixed', 'common']
    params = {
        0: {
            "phase": 'global_only',
            "title": "Training on Global Data Only",
            "ylim": (0.65, 0.95),
            "flag_save_plot": True
        },
        1: {
            "phase": 'mixed',
            "title": "Training on Global add Local Data",
            "ylim": (0.65, 0.95),
            "flag_save_plot": True
        },
        2: {
            "phase": 'global_fixed',
            "title": "Training on Global Fixed",
            "ylim": (0.4, 1),
            "flag_save_plot": True
        },
        3: {
            "phase": 'common',
            "title": "Training on Global add Local Same Portion",
            "ylim": (0.6, 0.95),
            "flag_save_plot": True
        },
    }
    for i in range(4):
        rv.display_mfold_logs(
            phase=params[i]['phase'],
            title=params[i]['title'],
            ylim=params[i]['ylim'],
            flag_save_plot=params[i]["flag_save_plot"]
        )


if __name__ == '__main__':
    main()
