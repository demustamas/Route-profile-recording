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
        self.controlMatrixNorm = []
        self.controlMatrixSum = {}
        self.controlMatrixSumNorm = {}
        self.controlDuration = []
        self.controlDurationSum = {}

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
        N = 9
        control = ['brake', 'traction']

        for each in self.condRealizations:
            limit = np.maximum(limit, each.F_traction.abs().max())
        steps = np.linspace(-limit, limit, 2*N, endpoint=True)

        for ctrl in control:
            self.controlMatrixSum[ctrl] = np.zeros(N*N).reshape(N, N)
            self.controlMatrixSumNorm[ctrl] = np.zeros(N*N).reshape(N, N)
            self.controlDurationSum[ctrl] = [np.array([]) for x in range(N)]

        M_sum = np.zeros(4*N*N).reshape(2*N, 2*N)
        t_sum = [np.array([]) for x in range(2*N-1)]
        for each in self.condRealizations:
            self.controlMatrix.append({})
            self.controlMatrixNorm.append({})
            self.controlDuration.append({})

            for ctrl in control:
                self.controlMatrix[-1][ctrl] = np.zeros(N*N).reshape(N, N)
                self.controlMatrixNorm[-1][ctrl] = np.zeros(N*N).reshape(N, N)
                self.controlDuration[-1][ctrl] = [np.array([])
                                                  for x in range(N)]

            control_func = np.zeros(len(each))
            for i in range(2*N-1):
                control_func[each.F_traction.between(
                    steps[i], steps[i+1])] = i

            control_func = control_func.astype(int)
            each['control'] = control_func

            row = np.where(control_func[:-1] != control_func[1:])[0]
            column = row + 1
            coord = list(
                zip(control_func[row], control_func[column]))
            counter = dict((i, coord.count(i)) for i in coord)

            M = np.zeros(4*N*N).reshape(2*N, 2*N)
            for key, value in counter.items():
                M[key] = value
                M_sum[key] += value
            M[0:N, 0:N] = np.flip(M[0:N, 0:N])

            time_col = each.t.to_numpy()
            time = time_col[column[1:]-1] - time_col[column[:-1]]
            t = [np.array([]) for x in range(2*N-1)]
            for i in range(2*N-1):
                t[i] = np.append(t[i], time[control_func[column[1:]-1] == i])
                t_sum[i] = np.append(t_sum[i], t[i])

            for ctrlIdx, ctrl in enumerate(control):
                self.controlMatrix[-1][ctrl] = M[ctrlIdx *
                                                 (N-1):ctrlIdx*(N-1)+N, ctrlIdx*(N-1):ctrlIdx*(N-1)+N]
                self.controlDuration[-1][ctrl] = t[N-1::2*ctrlIdx-1]
                for i in range(N):
                    if self.controlMatrix[-1][ctrl][i].sum() != 0:
                        self.controlMatrixNorm[-1][ctrl][i] = self.controlMatrix[-1][ctrl][i] / \
                            self.controlMatrix[-1][ctrl][i].sum()

        M_sum[0:N, 0:N] = np.flip(M_sum[0:N, 0:N])

        for ctrlIdx, ctrl in enumerate(control):
            self.controlMatrixSum[ctrl] = M_sum[ctrlIdx *
                                                (N-1):ctrlIdx*(N-1)+N, ctrlIdx*(N-1):ctrlIdx*(N-1)+N]
            self.controlDurationSum[ctrl] = t_sum[N-1::2*ctrlIdx-1]
            for i in range(N):
                if self.controlMatrixSum[ctrl][i].sum() != 0:
                    self.controlMatrixSumNorm[ctrl][i] = self.controlMatrixSum[ctrl][i] / \
                        self.controlMatrixSum[ctrl][i].sum()

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
        control = ['traction', 'brake']
        dataset = {}
        for ctrl in control:
            for idx, each in enumerate(self.condRealizations):
                dataset['controlMatrix_' + ctrl + '_' +
                        each['trackName'].iloc[0]] = self.controlMatrix[idx][ctrl]
                dataset['controlMatrixNorm_' + ctrl + '_' +
                        each['trackName'].iloc[0]] = self.controlMatrixNorm[idx][ctrl]
                dataset['controlDuration_' + ctrl + '_' +
                        each['trackName'].iloc[0]] = np.array(self.controlDuration[idx][ctrl], dtype=object)
            dataset['controlMatrixSum_' + ctrl + '_' +
                    each['trackName'].iloc[0]] = self.controlMatrixSum[ctrl]
            dataset['controlMatrixSumNorm_' + ctrl + '_' +
                    each['trackName'].iloc[0]] = self.controlMatrixSumNorm[ctrl]
            dataset['controlDurationSum_' + ctrl + '_' +
                    each['trackName'].iloc[0]] = np.array(self.controlDurationSum[ctrl], dtype=object)

        np.savez(arrayFile, **dataset)

        print("\nCalculated data saved.")


"""Calling simulation model to calculate."""
Model = Realizations()
main()
"""EOF"""
"""EOF"""
