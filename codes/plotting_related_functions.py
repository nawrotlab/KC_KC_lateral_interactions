#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 16:51:39 2023

@author: kundakci_1789
"""

import matplotlib.pyplot as plt

#adjust the plotting rc parameters

def set_rcparams(posterfig=False):
    """"
    This function sets the rc parameters for plotting.

    Parameters
    ----------
    posterfig : bool, optional
        If True, the rc parameters are set for a poster figure to make everything bigger. The default is False.

    Returns
    -------
    None.
    """

    if posterfig:
        figdict = {'axes.titlesize': 35,
                   'axes.labelsize': 35,
                   'xtick.labelsize': 35,
                   'ytick.labelsize': 35,
                   'legend.fontsize': 35,
                   'figure.titlesize': 40,
                   'font.size': 35,
                   'axes.spines.top': False,
                   'axes.spines.right': False,
                   'axes.linewidth': 3,
                   'ytick.major.size': 5,
                   'xtick.major.size': 5,
                   'ytick.major.width': 2,
                   'xtick.major.width': 2,
                   'lines.linewidth': 3,
                   'lines.markersize': 12,}

    else:
        figdict = {'axes.titlesize' : 25,
                    'axes.labelsize' : 25,
                    'xtick.labelsize' : 25,
                    'ytick.labelsize' : 25,
                    'legend.fontsize' : 25,
                    'figure.titlesize' : 30,
                    'image.cmap' : 'gray',
                    'axes.formatter.limits' : [-7, 7],
                    'font.size' : 25,
                    'axes.spines.top' : False,
                    'axes.spines.right' : False,
                    'axes.linewidth' : 2,
                    'ytick.major.size' : 4,
                    'xtick.major.size' : 4,
                    'ytick.major.width' : 1.5,
                    'xtick.major.width' : 1.5,
                    'legend.handlelength' : 1.5,
                    'legend.columnspacing' : 0.75,
                    'legend.handletextpad' : 0.4,
                    'legend.frameon' : False,
                    'hatch.linewidth' : 2}


    plt.rcParams.update(figdict)
    return

set_rcparams()