#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 20:10:23 2021

@author: demust
"""

import sqlite3 as sql
import pandas as pd
import numpy as np

from scipy.interpolate import interp1d
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
pyplot.style.use('mplstyle.presentation')

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

        gamma = 0.1
        M = 124

        for each in self.condRealizations:
            each['F_traction'] = (1 + gamma) * 1000 * M * \
                each.a + each.veh_resistance + each.track_resistance

        # if graph:
        #     for idx, each in enumerate(self.condRealizations):
        #         fig, ax = pyplot.subplots(3, 1)
        #         ax[0].plot(each.s, each.F_traction,
        #                    label=(str(self.query.dateTime.iloc[idx]) + " / " + str(self.query.receiverType.iloc[idx])))
        #         ax[1].plot(each.s, each.a*100)
        #         ax[1].plot(each.s, each.v)
        #         ax[2].plot(each.s, self.avRealization.alt.iloc[np.searchsorted(
        #             self.avRealization.s, each.s)])
        #         ax[2].plot(
        #             each.s, self.avRealization.alt_grad.iloc[np.searchsorted(self.avRealization.s, each.s)])
        #         ax[0].legend(ncol=3)

        if graph:
            fix, ax = pyplot.subplots(1, 1, figsize=(4, 1.5))
            ax.plot(self.condRealizations[0].s,
                    self.condRealizations[0].F_traction / 1000)
            ax.set_xlabel('Distance travelled [km]')
            ax.set_ylabel('Traction / Braking force [kN]')

        print("\nTraction forces calculated.")

    def calcControlMatrix(self, graph=False):
        """Calculate control matrix."""
        N = 9
        control = ['brake', 'traction']

        v_input = np.array([-10, 48, 50, 60, 70, 80, 90, 100,
                            110, 120, 130, 140, 150, 160])
        F_input = np.array([200, 200, 186, 155, 131, 116,
                            103, 94, 86, 79, 73, 68, 64, 60], dtype='float')
        M = 100
        v = np.linspace(min(v_input), max(v_input), M)

        f = interp1d(v_input, F_input, kind='linear')
        F = f(v)

        F *= 1000

        steps = np.zeros((len(v), 2*N))
        for i in range(len(v)):
            steps[i] = np.linspace(-F[i], F[i], 2*N, endpoint=True)
            steps[i, 1] *= 1.1
            steps[i, -1] *= 1.1

        for ctrl in control:
            self.controlMatrixSum[ctrl] = np.zeros((N, N))
            self.controlMatrixSumNorm[ctrl] = np.zeros((N, N))
            self.controlDurationSum[ctrl] = [np.array([]) for x in range(N)]

        M_sum = np.zeros((2*N, 2*N))
        t_sum = [np.array([]) for x in range(2*N-1)]
        for each in self.condRealizations:
            self.controlMatrix.append({})
            self.controlMatrixNorm.append({})
            self.controlDuration.append({})

            for ctrl in control:
                self.controlMatrix[-1][ctrl] = np.zeros((N, N))
                self.controlMatrixNorm[-1][ctrl] = np.zeros((N, N))
                self.controlDuration[-1][ctrl] = [np.array([])
                                                  for x in range(N)]

            control_func = np.zeros(len(each))
            for i in range(len(v)-1):
                for j in range(2*N-1):
                    control_func[each.v.between(
                        v[i], v[i+1]) & each.F_traction.between(steps[i][j], steps[i][j+1])] = j

            control_func = control_func.astype(int)

            row = np.where(control_func[: -1] != control_func[1:])[0]
            column = row + 1
            coord = list(
                zip(control_func[row], control_func[column]))
            counter = dict((i, coord.count(i)) for i in coord)

            M = np.zeros(4*N*N).reshape(2*N, 2*N)
            for key, value in counter.items():
                M[key] = value
                M_sum[key] += value
            M[0: N, 0: N] = np.flip(M[0: N, 0: N])

            time_col = each.t.to_numpy()
            time = time_col[column[1:]-1] - time_col[column[:-1]]
            t = [np.array([]) for x in range(2*N-1)]
            for i in range(2*N-1):
                t[i] = np.append(t[i], time[control_func[column[1:]-1] == i])
                t_sum[i] = np.append(t_sum[i], t[i])

            for ctrlIdx, ctrl in enumerate(control):
                self.controlMatrix[-1][ctrl] = M[ctrlIdx *
                                                 (N-1): ctrlIdx*(N-1)+N, ctrlIdx*(N-1): ctrlIdx*(N-1)+N]
                self.controlDuration[-1][ctrl] = t[N-1:: 2*ctrlIdx-1]
                for i in range(N):
                    if self.controlMatrix[-1][ctrl][i].sum() != 0:
                        self.controlMatrixNorm[-1][ctrl][i] = self.controlMatrix[-1][ctrl][i] / \
                            self.controlMatrix[-1][ctrl][i].sum()

            control_func -= N-1
            each['control'] = control_func
        M_sum[0: N, 0: N] = np.flip(M_sum[0: N, 0: N])

        for ctrlIdx, ctrl in enumerate(control):
            self.controlMatrixSum[ctrl] = M_sum[ctrlIdx *
                                                (N-1): ctrlIdx*(N-1)+N, ctrlIdx*(N-1): ctrlIdx*(N-1)+N]
            self.controlDurationSum[ctrl] = t_sum[N-1:: 2*ctrlIdx-1]
            for i in range(N):
                if self.controlMatrixSum[ctrl][i].sum() != 0:
                    self.controlMatrixSumNorm[ctrl][i] = self.controlMatrixSum[ctrl][i] / \
                        self.controlMatrixSum[ctrl][i].sum()

        if graph:
            # fig, ax = pyplot.subplots(3, 1, figsize=(4, 4))
            # ax[0].plot(self.condRealizations[0].s,
            #            self.condRealizations[0].F_traction / 1000)
            # ax[1].plot(self.condRealizations[0].s,
            #            self.condRealizations[0].v, c='#2ca02c')
            # ax[2].plot(self.condRealizations[0].s,
            #            self.condRealizations[0].control, c='#d62728')
            # ax[2].set_xlabel('Distance travelled [km]')
            # ax[0].set_ylabel('Traction / Braking force [kN]')
            # ax[1].set_ylabel('Vehicle speed [km/h]')
            # ax[2].set_ylabel('Control position')

            fig, ax = pyplot.subplots(3, 1)
            N = 2
            ax[0].plot(self.condRealizations[N].s,
                       self.condRealizations[N].F_traction / 1000)
            F_min = np.zeros(len(self.condRealizations[N].s))
            for i in range(len(v)-1):
                F_min[self.condRealizations[N].v.between(
                    v[i], v[i+1])] = steps[i][N] / 1000
            ax[0].plot(self.condRealizations[N].s, -F_min, '--',
                       c='#d62728', label='Maximum traction force')
            ax[0].plot(self.condRealizations[N].s, F_min, '--',
                       c='#d62728', label='Maximum brake force')
            ax[1].plot(self.condRealizations[N].s,
                       self.condRealizations[N].v, c='#2ca02c')
            ax[2].plot(self.condRealizations[N].s,
                       self.condRealizations[N].control, c='#d62728')
            ax[2].set_xlabel('Distance [km]')
            ax[0].set_ylabel('Traction / Brake force [kN]')
            ax[1].set_ylabel('Vehicle speed [km/h]')
            ax[2].set_ylabel('Control position')
            ax[0].set_yticks([-160, -80, 0, 80, 160])
            ax[1].set_yticks([0, 40, 80, 120])
            ax[2].set_yticks([-8, -4, 0, 4, 8])

            fig = pyplot.figure()
            ax_3d = fig.add_subplot(111, projection='3d')
            data = pd.concat(self.condRealizations, ignore_index=True)
            data.F_traction /= 1000

            N_bin = 120
            H, xedges, yedges = np.histogram2d(
                data.v, data.F_traction, bins=N_bin, density=False)
            xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
            xpos = xpos.ravel()
            ypos = ypos.ravel()
            zpos = np.zeros(N_bin ** 2)

            dx = np.ones(N_bin ** 2) * (np.ptp(xedges)) / N_bin
            dy = np.ones(N_bin ** 2) * (np.ptp(yedges)) / N_bin
            dz = H.ravel()

            dc = pyplot.cm.Blues((dz / dz.max()) ** (1/5))

            ax_3d.bar3d(xpos, ypos, zpos, dx, dy, dz, color=dc)
            ax_3d.set_xlabel('v [km/h]')
            ax_3d.set_ylabel('Traction / Brake force [kN]')
            ax_3d.set_zlabel('Frequency')

            fig, ax = pyplot.subplots(2, 1)
            ax[0].hist(data.F_traction, bins=N_bin, density=True)
            ax_0 = ax[0].twinx()
            ax_0.hist(data.F_traction, bins=N_bin, density=True,
                      cumulative=True, histtype='step', color='#d62728')
            ax[1].hist(data.v, bins=N_bin, density=True)
            ax_1 = ax[1].twinx()
            ax_1.hist(data.v, bins=N_bin, density=True,
                      cumulative=True, histtype='step', color='#d62728')
            # fig = pyplot.figure()
            # D = 4
            # gs = fig.add_gridspec(D, 2*D)
            # ax = fig.add_subplot(gs[1:, D:-1])
            # ax_F = fig.add_subplot(gs[1:, -1], sharey=ax)
            # ax_v = fig.add_subplot(gs[0, D:-1], sharex=ax)
            # ax_Fc = ax_F.twiny()
            # ax_vc = ax_v.twinx()

            # ax.scatter(data.v, data.F_traction, marker=".",
            #            linewidth=0.1, edgecolors="none", alpha=0.5)
            # ax.set_xlabel('v [km/h]', loc='right')
            # ax.set_ylabel('Traction / Brake force [kN]')
            # ax_F.hist(data.F_traction, bins=N_bin,
            #           orientation='horizontal',  density=True)
            # ax_Fc.hist(data.F_traction, bins=N_bin, orientation='horizontal',
            #            density=True, cumulative=True, histtype='step')
            # ax_F.yaxis.set_tick_params(left=False, labelleft=False)
            # ax_Fc.tick_params('x')
            # ax_v.hist(data.v, bins=N_bin,  density=True)
            # ax_v.xaxis.set_tick_params(bottom=False, labelbottom=False)
            # ax_vc.hist(data.v, bins=N_bin,
            #            density=True, cumulative=True, histtype='step')
            # ax_vc.tick_params('y')

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
