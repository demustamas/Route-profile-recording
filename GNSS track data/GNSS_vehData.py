#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 20:10:23 2021

@author: demust
"""

import sqlite3 as sql
import pandas as pd
import numpy as np

from matplotlib import pyplot

import os


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

if os.path.exists(os.path.join(working_path, "controlMatrices.npz")):
    os.remove(os.path.join(working_path, "controlMatrices.npz"))

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
        self.controlMatrix = []
        self.controlMatrixSum = []
        self.controlMatrixNorm = []
        self.controlMatrixSumNorm = []
        self.controlDuration = []
        self.controlDurationSum = []

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
                fig, ax = pyplot.subplots(3, 1)
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
        N_steps = 9
        control = ['traction', 'brake']

        for each in self.condRealizations:
            limit = np.maximum(limit, each.F_traction.abs().max())
        steps = np.linspace(0, limit, N_steps, endpoint=True)
        N = len(steps)

        for _ in control:
            self.controlMatrixSum.append(np.zeros(N*N).reshape(N, N))
            self.controlMatrixSumNorm.append(np.zeros(N*N).reshape(N, N))
            self.controlDurationSum.append([np.array([]) for x in range(N)])

        for each in self.condRealizations:
            self.controlMatrix.append([])
            self.controlMatrixNorm.append([])
            self.controlDuration.append([])
            for ctrlIdx, ctrl in enumerate(control):
                control_func = each.F_traction.copy()
# !!! Így a control func kitörli az ellentétes vezérléseket és megtéveszt,
# hogy hol fut szabadon a jármű, a control_func-t nem szabad nullázni
# -steps-től +steps-ig kell osztályokba sorolni és azt kell kiértékelni
                if ctrl == 'traction':
                    control_func[control_func < 0] = 0
                if ctrl == 'brake':
                    control_func[control_func > 0] = 0
                for idx in range(len(steps[1:])):
                    control_func[control_func.abs().between(
                        steps[idx], steps[idx+1])] = idx
                each[ctrl] = control_func

                self.controlMatrix[-1].append(np.zeros(N*N).reshape(N, N))
                self.controlMatrixNorm[-1].append(np.zeros(N*N).reshape(N, N))
                self.controlDuration[-1].append([np.array([])
                                                for x in range(N)])

                control_func = control_func.to_numpy(dtype=int)
                row = np.where(control_func[:-1] != control_func[1:])[0]
                column = row + 1
                coord = list(
                    zip(control_func[row], control_func[column]))
                counter = dict((i, coord.count(i)) for i in coord)

                for key, value in counter.items():
                    self.controlMatrix[-1][ctrlIdx][key] = value
                    self.controlMatrixSum[ctrlIdx][key] += value

                for j in range(N):
                    if self.controlMatrix[-1][ctrlIdx][j].sum() != 0:
                        self.controlMatrixNorm[-1][ctrlIdx][j] = self.controlMatrix[-1][ctrlIdx][j] / \
                            self.controlMatrix[-1][ctrlIdx][j].sum()

                time_col = each.t.to_numpy()
                time = time_col[column[1:]-1] - time_col[column[:-1]]
                for idx in range(len(steps)):
                    self.controlDuration[-1][ctrlIdx][idx] = time[control_func[column[1:]-1] == idx]
                    self.controlDurationSum[ctrlIdx][idx] = np.append(self.controlDurationSum[ctrlIdx][idx],
                                                                      self.controlDuration[-1][ctrlIdx][idx])

        for idx in range(len(control)):
            for j in range(N):
                if self.controlMatrixSum[idx][j].sum() != 0:
                    self.controlMatrixSumNorm[idx][j] = self.controlMatrixSum[idx][j] / \
                        self.controlMatrixSum[idx][j].sum()

        print("\nControl matrices generated.")

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

        arrayFile = os.path.join(wdir, "controlMatrices.npz")
        control = ['traction_', 'brake_']
        dataset = {}
        for i, ctrl in enumerate(control):
            for idx, each in enumerate(self.condRealizations):
                dataset['controlMatrix_' + ctrl +
                        each['trackName'].iloc[0]] = self.controlMatrix[idx][i]
                dataset['controlMatrixNorm_' + ctrl +
                        each['trackName'].iloc[0]] = self.controlMatrixNorm[idx][i]
                dataset['controlDuration_' + ctrl +
                        each['trackName'].iloc[0]] = np.array(self.controlDuration[idx][i], dtype=object)
            dataset['controlMatrixSum_' + ctrl +
                    each['trackName'].iloc[0]] = self.controlMatrixSum[i]
            dataset['controlMatrixSumNorm_' + ctrl +
                    each['trackName'].iloc[0]] = self.controlMatrixSumNorm[i]
            dataset['controlDurationSum_' + ctrl +
                    each['trackName'].iloc[0]] = np.array(self.controlDurationSum[i], dtype=object)

        np.savez(arrayFile, **dataset)

        print("\nCalculated data saved.")


"""Calling simulation model to calculate."""
Model = Realizations()
main()
"""EOF"""
"""EOF"""
