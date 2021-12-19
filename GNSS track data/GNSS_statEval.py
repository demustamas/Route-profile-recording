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
from scipy.interpolate import interp1d

from matplotlib import pyplot as plt

import os

# TODO Implement altitude gradient based on linear approximation of the altitude

# TODO Introduce v_max detection and calculate the corresponding filtered v_max

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

        for idx, col in enumerate(cols):
            f = interp1d(self.avRealization.s, self.avRealization[col])
            x = np.linspace(
                self.avRealization.s.iloc[0], self.avRealization.s.iloc[-1], len(self.avRealization.s))
            resampled = f(x)
            b, a = butter(5, 0.01, output='ba')
            filtered = filtfilt(b, a, resampled)
            f = interp1d(x, filtered)
            backSampled = f(self.avRealization.s)
            self.avRealization[col + "_filt"] = backSampled
            if np.isnan(backSampled).any():
                print("Filtering failed!")
            else:
                print(f"Filtering successful for {col}.")

        print("Average values filtered.\n")

    def saveToDatabase(self, wdir):
        """Save calculated data to database."""

        con = sql.connect(os.path.join(wdir, "avRealization.db"))
        self.avRealization.to_sql(
            "av", con, if_exists='replace', index=False)
        con.close()

        print("\nCalculated data saved.")


"""Calling simulation model to calculate."""
Model = Realizations()
main()
"""EOF"""
"""EOF"""
"""EOF"""
