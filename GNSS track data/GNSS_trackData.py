#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 20:10:23 2021

@author: demust
"""

import sqlite3 as sql
import pandas as pd
import numpy as np

from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d, UnivariateSpline

from matplotlib import pyplot
from matplotlib.gridspec import GridSpec

from colorama import Fore, Style

import os

# ? What to do with the NaNs coming from the np.gradient at altitude?

"""Simulation input parameters"""

database_dir = "Database/"
working_dir = "Results/Test/"
graph_dir = "Graphs/"
destination_dir = "ToCopy/"
name_tag = ""
palette = pyplot.cm.tab10
palette = palette(range(palette.N))
pyplot.style.use('mplstyle.work')

"""PATH"""

working_path = working_dir


"""Main simulation"""


def main():
    """Calling simulation model."""

    Model.queryRealizations(working_path)
    Model.averageRealization()
    Model.filterRealization()
    Model.calcAltitude(graph=False)
    Model.calcvMax()
    Model.calcTrackResistance(graph=False)
    Model.saveToDatabase(working_path)


"""Simulation model"""

class Realizations:
    """Class definition for storing GNSS data and calculating track parameters."""

    def __init__(self):
        self.query = pd.DataFrame()
        self.rawRealizations = []
        self.condRealizations = []
        self.sumRealization = pd.DataFrame()
        self.avRealization = pd.DataFrame()

    def queryRealizations(self, wPath):
        con = sql.connect(wPath + "query.db")
        self.query = pd.read_sql("SELECT * FROM query", con)
        print("\nFollowing realizations selected:")
        pd.set_option('display.max_columns', None)
        print(self.query)
        con.close()

        con = sql.connect(wPath + "rawRealizations.db")
        for each in self.query.fileName:
            self.rawRealizations.append(
                pd.read_sql(f"SELECT * FROM \"{each}\"", con))
        con.close()

        con = sql.connect(wPath + "condRealizations.db")
        for each in self.query.fileName:
            self.condRealizations.append(
                pd.read_sql(f"SELECT * FROM \"{each}\"", con))
        con.close()

        con = sql.connect(wPath + "sumRealization.db")
        self.sumRealization = pd.read_sql("SELECT * FROM \"sum\"", con)
        con.close()

        con = sql.connect(wPath + "avRealization.db")
        self.avRealization = pd.read_sql("SELECT * FROM \"av\"", con)
        con.close()
        print(f"{Fore.GREEN}\nRealizations loaded.\n{Style.RESET_ALL}")

    def averageRealization(self):
        """Clean sum GNSS data."""
        self.avRealization.s = self.sumRealization.s

        """Calculate rolling median of GNSS data based on distance."""
        cols = ["lat", "lon", "alt", "v", "a"]
        N = [16, 16, 16, 16, 16]

        for n, col in enumerate(cols):
            self.avRealization[col] = (
                self.sumRealization[col]
                .rolling(N[n], center=True, min_periods=1)
                .mean()
            )
            if col in ["alt", "v", "a"]:
                centered = self.sumRealization[col] - self.avRealization[col]
                self.avRealization[col + "_std"] = centered.rolling(
                    N[n], center=True, min_periods=1
                ).std()

        print("Mean and standard deviation values calculated.\n")

    def filterRealization(self):
        """Smooth the average Realization."""
        cols = ["alt", "v", "a"]
        order = [5, 5, 5]
        freq = [0.001, 0.001, 0.001]

        for idx, col in enumerate(cols):
            f = interp1d(self.avRealization.s, self.avRealization[col])
            x = np.linspace(
                self.avRealization.s.iloc[0], self.avRealization.s.iloc[-1], len(self.avRealization.s))
            resampled = f(x)
            b, a = butter(order[idx], freq[idx], output='ba')
            filtered = filtfilt(b, a, resampled)
            f = interp1d(x, filtered)
            backSampled = f(self.avRealization.s)
            self.avRealization[col + "_filt"] = backSampled
            if np.isnan(backSampled).any():
                print("Filtering failed!")
            else:
                print(f"Filtering successful for {col}.")

        print("Average values filtered.\n")

    def calcAltitude(self, graph=False):
        """Approximate altitude with piece-wise linear fit and calculate gradient."""

        spl = UnivariateSpline(self.avRealization.s,
                               self.avRealization.alt_filt, k=1)
        self.avRealization['alt_lin'] = spl(self.avRealization.s)

        print("Altitude piece-wise approximation done.")

        cond = self.avRealization.s.shift() != self.avRealization.s
        self.avRealization['alt_grad'] = np.nan
        grad = self.avRealization.alt_grad.copy()
        grad.loc[cond] = np.gradient(
            self.avRealization.alt_lin.loc[cond], self.avRealization.s.loc[cond])
        grad.fillna(method='ffill', inplace=True)
        self.avRealization.alt_grad = grad

        print("Altitude gradient calculated.")

    def calcvMax(self):
        """Calculate v max from route profile."""

        N = len(self.condRealizations) * 25

        v = self.sumRealization.v.rolling(
            N, center=True, min_periods=1).apply(
                lambda x: np.max(x))

        v = np.around(v+3, decimals=-1)

        S_treshold = 750 / 1000 / 120  # A lassújel hossza legyen arányos a sebességgel

        v = v.to_numpy()

        idxSteps = np.concatenate((np.array([0]), np.where(
            v[:-1] != v[1:])[0]+1), axis=0)

        v_i = v[idxSteps]
        s_i = self.avRealization.s[idxSteps].to_numpy()
        dist_i = s_i[1:] - s_i[:-1]
        dist_i = np.append(dist_i, S_treshold*130)
        v_i[dist_i < S_treshold*v_i] = 0

        for i in range(1, len(v_i)-1):
            if v_i[i] == 0:
                v_i[i] = np.maximum(v_i[i-1], v_i[i+1])

        v.fill(np.nan)
        v[idxSteps] = v_i

        self.avRealization['v_max'] = v
        self.avRealization.v_max.fillna(method='ffill', inplace=True)

        fig, ax = pyplot.subplots(1, 1)
        for each in self.condRealizations:
            ax.plot(each.s, each.v, c='#1f77b4', linewidth=0.2)
        ax.plot(self.avRealization.s, self.avRealization.v_max,
                c='#d62728', linewidth=0.6)
        ax.set_xlabel('Distance [km]')
        ax.set_ylabel('Vehicle speed [km/h]')

        print("\nCalculation of vmax done.")

    def calcTrackResistance(self, graph=False):
        """Calculate track resistance."""

        M = 124
        g = 9.80665
        self.avRealization['track_resistance'] = self.avRealization.alt_grad * M * g
        for each in self.condRealizations:
            each['track_resistance'] = self.avRealization.track_resistance.iloc[np.searchsorted(
                self.avRealization.s, each.s)].reset_index(drop=True)

        if graph:
            fig, ax = pyplot.subplots(1, 2)
            # fig = pyplot.figure()
            # gs = GridSpec(2, 1, height_ratios=[3, 1])

            # ax = []
            # ax.append(fig.add_subplot(gs[0]))
            # ax.append(fig.add_subplot(gs[1]))

            for each in self.condRealizations:
                ax[0].plot(each.s, each.alt, c='#2ca02c', linewidth=0.2)
            ax[0].plot(self.avRealization.s,
                       self.avRealization.alt, c='purple', alpha=0.5, label='Average altitude')
            ax[0].plot(self.avRealization.s,
                       self.avRealization.alt_filt, c='purple', label='Filtered altitude')
            ax[0].plot(self.avRealization.s,
                       self.avRealization.alt_filt, c='#d62728', label='Linear approximation', linewidth=0.6)
            ax[1].plot(self.avRealization.s, self.avRealization.alt_grad,
                       c='#1f77b4', label='Track gradient')
            ax[1].yaxis.set_ticks([-20, -10, 0, 10, 20])
            ax[0].set_xlabel('Distance [km]')
            ax[1].set_xlabel('Distance [km]')
            ax[0].set_ylabel('Altitude [m]')
            ax[1].set_ylabel('Track gradient\n[1/1000]')
            # ax[2].set_ylabel('Track gradient')
            ax[0].legend()
            # ax[1].legend()
            # ax[2].legend()

            # fig, ax = pyplot.subplots(1, 1)
            # ax.plot(self.avRealization.s, self.avRealization.alt_grad,
            #         c='#1f77b4', label='Track gradient')
            # ax.set_xlabel('Distance travelled [km]')
            # ax.set_ylabel('Track gradient [1/1000]')
            x = [0, 4.6, 7.7, 20.6, 30.2, 35.8, 50.5, 58.7,
                 66.3, 77.4, 87.4, 100.7, 105.7, 112.7, 125]

            ref_grade = [0, 6.3, 3.6, 4.4, 4.7, 6.8,
                         0.3, 1.7, 3.6, 5, 4, 4.3, 2.9, 2.4, 3.7]
            max_grade = [0, 10.1, 5, 5.6, 6.5, 6.8,
                         2.4, 2.4, 4.6, 5, 5, 4.8, 3.9, 3.2, 5]
            ref_slope = [0, -7, -7, -5, -5, -8, -
                         8, -3, -5, -5, -5, -5, -5, -5, -5]
            max_slope = [0, -10, -10, -6.5, -6.5, -8, -8,
                         -2.9, -5.6, -5.1, -5.1, -5.1, -5.1, -5.1, -5.1]
            ax[1].step(x, ref_grade, c='#2ca02c',
                       label='Reference inclination')
            ax[1].step(x, max_grade, c='#d62728', label='Maximum inclination')
            ax[1].step(x, ref_slope, '--', c='#2ca02c',
                       label='Reference slope')
            ax[1].step(x, max_slope, '--', c='#d62728', label='Maximum slope')
            ax[1].legend()
            pyplot.show()

        print("\nTrack resistance calculated.")

    def saveToDatabase(self, wdir):
        """Save calculated data to database."""

        con = sql.connect(os.path.join(wdir, "condRealizations.db"))
        for each in self.condRealizations:
            each.to_sql(each['trackName'].iloc[0], con,
                        if_exists='replace', index=False)
        con.close()

        con = sql.connect(os.path.join(wdir, "avRealization.db"))
        self.avRealization.to_sql(
            "av", con, if_exists='replace', index=False)
        con.close()

        print("\nCalculated data saved.")


"""Calling simulation model to calculate."""
Model = Realizations()
main()
"""EOF"""
