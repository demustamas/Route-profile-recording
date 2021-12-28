#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 20:10:23 2021

@author: demust
"""

import sqlite3 as sql
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

import os


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
    Model.calcVehicleResistance()
    Model.calcTraction(graph=False)
    Model.calcControlMatrix(graph=True)
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
        self.tractionMatrix = []
        self.brakeMatrix = []

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
        print("\nRealizations loaded.")

    def calcVehicleResistance(self):
        """Calculate vehicle resistance."""

        M = 124
        g = 9.80665
        a, b, c = 2, 0, 0.022/100

        for each in self.condRealizations:
            each['veh_resistance'] = (a + b * each.v + c * each.v ** 2) * M * g

        print("\nVehicle resistance calculated.")

    def calcTraction(self, graph=False):
        """Calculate traction forces."""

        gamma = 0.2
        M = 124

        for each in self.condRealizations:
            each['F_traction'] = (1 + gamma) * 1000 * M * \
                each.a + each.veh_resistance + each.track_resistance

        if graph:
            for idx, each in enumerate(self.condRealizations):
                fig, ax = plt.subplots(3, 1)
                ax[0].plot(each.s, each.F_traction,
                           label=(str(self.query.dateTime.iloc[idx]) + " / " + str(self.query.receiverType.iloc[idx])))
                ax[1].plot(each.s, each.a*100)
                ax[1].plot(each.s, each.v)
                ax[2].plot(each.s, self.avRealization.alt.iloc[np.searchsorted(
                    self.avRealization.s, each.s)])
                ax[2].plot(
                    each.s, self.avRealization.alt_grad.iloc[np.searchsorted(self.avRealization.s, each.s)])
                ax[0].legend(ncol=3)

        print("\nTraction forces calculated.")

    def calcControlMatrix(self, graph=False):
        """Calculate control matrix."""

        limit = 0

        for each in self.condRealizations:
            limit = np.maximum(limit, each.F_traction.abs().max())

        steps = np.linspace(0, limit, 8, endpoint=True)
        N = len(steps)

        for each in self.condRealizations:
            traction = each.F_traction.copy()
            brake = each.F_traction.copy()
            traction[traction < 0] = 0
            brake[brake > 0] = 0
            for idx in np.arange(len(steps[1:])):
                traction[traction.between(
                    steps[idx], steps[idx+1])] = idx
                brake[brake.abs().between(steps[idx],
                                          steps[idx+1])] = -idx
            each['traction'] = traction
            each['brake'] = brake

            self.tractionMatrix.append(np.zeros(N*N).reshape(N, N))
            self.brakeMatrix.append(np.zeros(N*N).reshape(N, N))

            traction = traction.to_numpy()
            brake = brake.to_numpy()
# !!! Betenni a fékezési control mátrixot
            row = np.where(traction[:-1] != traction[1:])[0]
            column = row + 1
            coord = list(zip(traction[row].astype(
                int), traction[column].astype(int)))
            counter = dict((i, coord.count(i)) for i in coord)

            for key, value in counter.items():
                self.tractionMatrix[-1][key] = value

        print("\nControl matrices generated.")

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
