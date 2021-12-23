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

from matplotlib import pyplot as plt

import os

# ??? What to do with the NaNs coming from the np.gradient at altitude?

# ??? What to do with 'a'?

"""Simulation input parameters"""

database_dir = "Database/"
working_dir = "Results/Test/"
graph_dir = "Graphs/"
destination_dir = "ToCopy/"
name_tag = ""
palette = plt.cm.tab10
palette = palette(range(palette.N))
plt.style.use('mplstyle.work')

"""PATH"""

working_path = working_dir


"""Main simulation"""


def main():
    """Calling simulation model."""

    Model.queryRealizations(working_path)
    Model.averageRealization()
    Model.filterRealization()
    Model.calcAltitude()
    Model.calcvMax()
    Model.calcResistance()
    Model.calcTraction(graph=True)
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
        print("\nRealizations loaded.\n")

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

    def calcAltitude(self):
        """Approximate altitude with piece-wise linear fit and calculate gradient."""

        spl = UnivariateSpline(self.avRealization.s,
                               self.avRealization.alt_filt, k=1)
        self.avRealization['alt_lin'] = spl(self.avRealization.s)

        print("Altitude piece-wise approximation done.")

        self.avRealization['alt_grad'] = np.gradient(
            self.avRealization.alt_lin, self.avRealization.s)

        if np.isnan(self.avRealization.alt_grad).any():
            print("Altitude gradient calculation contains NaNs!")
        else:
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

        for i in np.arange(1, len(v_i)-1):
            if v_i[i] == 0:
                v_i[i] = np.maximum(v_i[i-1], v_i[i+1])

        v.fill(np.nan)
        v[idxSteps] = v_i

        self.avRealization['v_max'] = v
        self.avRealization.v_max.fillna(method='ffill', inplace=True)

        print("\nCalculation of vmax done.")

    def calcResistance(self):
        """Calculate track and vehicle resistance."""

        M = 124
        g = 9.80665
        a, b, c = 2, 0, 0.022/100

        for each in self.condRealizations:
            each['veh_resistance'] = (a + b * each.v + c * each.v ** 2) * M * g
            each['track_resistance'] = self.avRealization.alt_grad * M * g

        print("\nVehicle and track resistances calculated.")

    def calcTraction(self, graph=False):
        """Calculate traction forces."""

        gamma = 0.2
        M = 124000

        for each in self.condRealizations:
            each['F_traction'] = (1 + gamma) * M * each.a + \
                each.veh_resistance + each.track_resistance

        if graph:
            for idx, each in enumerate(self.condRealizations):
                fig, ax = plt.subplots(3, 1)
                ax[0].plot(each.s, each.F_traction,
                           label=(str(self.query.dateTime.iloc[idx]) + " / " + str(self.query.receiverType.iloc[idx])))
                ax[1].plot(each.s, each.a*100)
                ax[1].plot(each.s, each.v)
                ax[2].plot(self.avRealization.s, self.avRealization.alt)
                ax[2].plot(self.avRealization.s, self.avRealization.alt_grad)
                ax[0].legend(ncol=3)

        print("\nTraction forces calculated.")

    def calcControlMatrix(self):
        """Calculate control matrix."""
        pass

    def saveToDatabase(self, wdir):
        """Save calculated data to database."""

        con = sql.connect(os.path.join(wdir, "condRealizations.db"))
        for each in self.condRealizations:
            each.to_sql(each['Track name'].iloc[0], con,
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
