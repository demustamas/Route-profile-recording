#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 20:10:23 2021

@author: demust
"""

import sqlite3 as sql
import pandas as pd
import numpy as np

from scipy import signal, stats
from scipy.interpolate import interp1d

from matplotlib import pyplot

import os
import shutil


# TODO During aggregation calculate all offset values and start the aggregation
# with the most precise one (a.k.a. smallest std) - this would increase the
# overall DF precision / accuracy.

"""Simulation input parameters"""

database_dir = "Database/"
database_name = "GNSS recordings.db"
working_dir = "Results/Test/"

sql_query = """SELECT *
    FROM listOfRecordings
    WHERE recType = 'route'
    AND fromStation = 'Füzesabony'
    AND toStation = 'Keleti'
    AND trainType = 'IR'
    AND (trainConfig = 'FLIRT' OR trainConfig = 'FLIRT+FLIRT')
    AND receiverType = 'U-blox M8N'
    """

"""SELECT *
    FROM listOfRecordings
    WHERE recType = 'static'
    AND dateTime = '2022/01/31 15:40:42'
    """

"""SELECT *
    FROM listOfRecordings
    WHERE recType = 'route'
    AND fromStation = 'Füzesabony'
    AND toStation = 'Keleti'
    AND trainType = 'IR'
    AND (trainConfig = 'FLIRT' OR trainConfig = 'FLIRT+FLIRT')
    AND receiverType = 'U-blox M8N'
    """

"""SELECT *
    FROM listOfRecordings
    WHERE recType = 'dynamic'
    AND dateTime = '2021/12/23 12:00:08'
    AND fromStation = 'Füzesabony'
    AND toStation = 'Keleti'
    AND trainConfig = ''
    AND trainType = ''
    AND receiverType = ''
    """

pyplot.style.use('mplstyle.work')
np.set_printoptions(precision=3)

"""PATH"""
database_path = os.path.join(database_dir, database_name)
working_path = working_dir

if os.path.exists(working_path):
    shutil.rmtree(working_path)
os.makedirs(working_path)

"""Main simulation"""


def main():
    """Calling simulation model."""

    Model.queryRealizations(database_path, sql_query)
    Model.conditionRealization(graph=False)
    Model.aggregateRealization(fit_curves=True, graph=False)
    Model.saveToDatabase(working_path)


"""Simulation model"""


class Realizations:
    """Class definition for storing GNSS data and calculating track parameters."""

    def __init__(self):
        self.query = pd.DataFrame()
        self.rawRealizations = []
        self.condRealizations = []
        self.sumRealization = pd.DataFrame(
            columns=["lon", "lat", "alt", "v", "s", "a"])
        self.avRealization = pd.DataFrame(
            columns=["lon", "lat", "alt", "alt_std",
                     "v", "v_std", "s", "a", "a_std"]
        )

    def queryRealizations(self, dbPath, sqlQuery):
        con = sql.connect(dbPath)
        self.query = pd.read_sql(sqlQuery, con)
        print("\nFollowing realizations selected:")
        pd.set_option('display.max_columns', None)
        print(self.query, "\n")
        for each in self.query.fileName:
            self.rawRealizations.append(
                pd.read_sql(f"SELECT * FROM \"{each}\"", con))
            print(
                f"{len(self.rawRealizations[-1].index):10} datapoints found in {each}.")
        con.close()
        print("\nRealizations loaded.\n")

    def conditionRealization(self, graph=False):
        """Condition raw GNSS data to allow further processing."""

        """Copy raw dataset to conditioned dataset."""
        for each in self.rawRealizations:
            self.condRealizations.append(each.copy())

        """Remove nan values."""
        for each in self.condRealizations:
            each.dropna(subset=["v"], inplace=True)
            each.dropna(subset=["a"], inplace=True)
            each.reset_index(drop=True, inplace=True)
            each.v = each.v.astype(float)
            each.a = each.a.astype(float)

        """Remove points with high xDOP."""
        dops = ["hdop", "vdop", "pdop"]
        for each in self.condRealizations:
            for dop in dops:
                each.drop(each.index[each[dop] > 20], inplace=True)
            each.reset_index(drop=True, inplace=True)

        """Remove points with low number of satellites."""
        for each in self.condRealizations:
            if each.nSAT.any() > 0:
                each.drop(each.index[each.nSAT < 6], inplace=True)
                each.reset_index(drop=True, inplace=True)

        """Remove outliers."""
        cols = ["lon", "lat", "alt", "v", "a"]
        N_iter = 1
        for each in self.condRealizations:
            for col in cols:
                for _ in range(N_iter):
                    each_centered = (
                        each[col]
                        - each[col].rolling(50, center=True,
                                            min_periods=1).mean()
                    )
                    z = np.abs(stats.zscore(each_centered))
                    each.drop(each[col].index[z > 3], inplace=True)
            each.reset_index(drop=True, inplace=True)

        """Smooth data."""
        cols = ["lon", "lat", "alt", "v", "a"]
        window = [11, 11, 21, 21, 151]
        polyorder = [2, 2, 2, 2, 3]
        for each in self.condRealizations:
            for idx, col in enumerate(cols):
                each[col] = signal.savgol_filter(
                    each[col], window[idx], polyorder[idx])

        """Print out conditioned data."""
        for idx, each in enumerate(self.condRealizations):
            N_deleted = len(self.rawRealizations[idx]) - len(each)
            print(f"Removed {N_deleted:5} points from {each.iloc[0,1]}")

        if graph:
            cols = ['alt', 'v']
            for idx, each in enumerate(self.rawRealizations):
                fig, ax = pyplot.subplots(2, 2)
                fig.suptitle(each['trackName'].iloc[0])
                for col_idx, col in enumerate(cols):
                    ax[col_idx, 0].plot(each.s, each[col],
                                        color='green', label=col)
                    ax[col_idx, 1].plot(self.condRealizations[idx].s,
                                        self.condRealizations[idx][col], color='blue', label=col)
                    ax[col_idx, 0].legend()
                    ax[col_idx, 1].legend()

        print("\nData conditioning performed.\n")

    def aggregateRealization(self, fit_curves=True, graph=False):
        """Aggregate GNSS data into one dataframe."""

        """Copy first dataset."""
        for each in ["lon", "lat", "alt", "v", "s", "a"]:
            self.sumRealization[each] = self.condRealizations[0][each]
        if fit_curves:
            print(
                f"{self.condRealizations[0]['trackName'].iloc[0]} aggregated.")
        else:
            print("Curves not fitted!")

        """Merge additional GNSS data sets."""
        df = self.sumRealization.copy()

        cols = ["alt", "v"]

        for idx in range(1, len(self.condRealizations)):
            """Set distance offset based on cross-correlation."""

            if fit_curves:
                interp_df = pd.DataFrame(columns=["x"] + cols)
                interp_cond = pd.DataFrame(columns=["x"] + cols)
                correlation = pd.DataFrame(
                    columns=cols + [each + "_lag" for each in cols])

                f_df = []
                f_cond = []

                dist_offset = np.arange(len(cols), dtype=float)

                interp_df.x = np.arange(df.s.iloc[0], df.s.iloc[-1], 0.001)
                interp_cond.x = np.arange(
                    self.condRealizations[idx].s.iloc[0],
                    self.condRealizations[idx].s.iloc[-1],
                    0.001,
                )

                for i, col in enumerate(cols):
                    f_df.append(interp1d(df.s, df[col]))
                    f_cond.append(
                        interp1d(
                            self.condRealizations[idx].s, self.condRealizations[idx][col]
                        )
                    )

                    interp_df[col] = f_df[i](interp_df.x)
                    interp_cond[col] = f_cond[i](interp_cond.x)

                    interp_df[col] -= interp_df[col].mean()
                    interp_cond[col] -= interp_cond[col].mean()

                    correlation[col] = signal.correlate(
                        interp_df[col], interp_cond[col], mode="full"
                    )

                    correlation[col + "_lag"] = signal.correlation_lags(
                        interp_df[col].size, interp_cond[col].size, mode="full"
                    )

                    dist_offset[i] = (
                        correlation[col +
                                    "_lag"][np.argmax(correlation[col])] / 1000
                    )

                offset = np.mean(dist_offset)
                print(f"\tOffset vector:  {dist_offset}")
                print(f"\tRelative error: {(dist_offset-offset)/offset}")
                print(f"\tMean offset:    {offset}")

                self.condRealizations[idx].s += offset - \
                    self.condRealizations[idx].s.iloc[0]
                print(
                    f"{self.condRealizations[idx]['trackName'].iloc[0]} aggregated.")

            """Merge dataset."""
            df = df.append(
                self.condRealizations[idx][df.columns.values.tolist()])
            df.sort_values(by=["s"], inplace=True, ignore_index=True)

        self.sumRealization = df

        if graph:
            for col in cols:
                fig, ax = pyplot.subplots(1, 1)
                for idx, each in enumerate(self.condRealizations):
                    ax.plot(each.s, each[col],
                            label=(str(self.query.dateTime.iloc[idx]) + " / " + str(self.query.receiverType.iloc[idx])))
                    ax.legend(ncol=3)

        print("\nData aggregation completed.\n")

    def saveToDatabase(self, wdir):
        """Save calculated data to database."""

        con = sql.connect(os.path.join(wdir, "rawRealizations.db"))
        for each in self.rawRealizations:
            each.to_sql(each['trackName'].iloc[0], con,
                        if_exists='replace', index=False)
        con.close()

        con = sql.connect(os.path.join(wdir, "condRealizations.db"))
        for each in self.condRealizations:
            each.to_sql(each['trackName'].iloc[0], con,
                        if_exists='replace', index=False)
        con.close()

        con = sql.connect(os.path.join(wdir, "avRealization.db"))
        self.avRealization.to_sql(
            "av", con, if_exists='replace', index=False)
        con.close()

        con = sql.connect(os.path.join(wdir, "sumRealization.db"))
        self.sumRealization.to_sql(
            "sum", con, if_exists='replace', index=False)
        con.close()

        con = sql.connect(os.path.join(wdir, "query.db"))
        self.query.to_sql("query", con, if_exists='replace', index=False)
        con.close()

        print("Calculated data saved.")


"""Calling simulation model to calculate."""
Model = Realizations()
main()
"""EOF"""
