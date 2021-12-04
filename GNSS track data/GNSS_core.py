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

"""TODOs"""

"""Simulation input parameters"""

database_dir = "Database/"
database_name = "GNSS recordings.db"
working_dir = "Dynamic/"

sql_query = """SELECT * 
    FROM listOfRecordings 
    WHERE recType = 'dynamic'
    AND fromStation = 'A'
    AND toStation = 'B'
    """

"""SELECT * 
    FROM listOfRecordings 
    WHERE recType = 'dynamic'
    AND fromStation = ''
    AND toStation = ''
    AND trainConfig = ''
    AND trainType = ''
    AND receiverType = ''
    """

"""PATH"""
database_path = os.path.join(database_dir, database_name)
workingdir_path = os.path.join(database_dir, working_dir)

if not os.path.exists(workingdir_path):
    os.makedirs(workingdir_path)

"""Main simulation"""


def main():
    """Calling simulation model."""

    Model.queryRealizations(database_path, sql_query)
    Model.conditionRealization()
    Model.aggregateRealization(fit_curves=False)
    Model.averageRealization()
    Model.saveToDatabase(workingdir_path)


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
        print(self.query)
        for each in self.query.fileName:
            self.rawRealizations.append(
                pd.read_sql(f"SELECT * FROM \"{each}\"", con))
        con.close()
        print("\nRealizations loaded.\n")

    def conditionRealization(self):
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
                for _ in np.arange(N_iter):
                    each_centered = (
                        each[col]
                        - each[col].rolling(100, center=True,
                                            min_periods=1).mean()
                    )
                    z = np.abs(stats.zscore(each_centered))
                    each.drop(each[col].index[z > 3], inplace=True)
            each.reset_index(drop=True, inplace=True)

        """Smooth data."""
        cols = ["lon", "lat", "alt", "v"]
        for each in self.condRealizations:
            for col in cols:
                each[col] = signal.savgol_filter(each[col], 51, 2)

        cols = ["a"]
        for each in self.condRealizations:
            for col in cols:
                each[col] = signal.savgol_filter(each[col], 51, 2)

        """Print out conditioning data."""
        for idx, each in enumerate(self.condRealizations):
            N_deleted = len(self.rawRealizations[idx]) - len(each)
            print(f"Removed {N_deleted} points from {each.iloc[0,1]}")

        cols = ['alt', 'v']
        for idx, each in enumerate(self.rawRealizations):
            fig, ax = pyplot.subplots(2, 2, dpi=400, figsize=(16, 9))
            fig.suptitle(each['Track name'].iloc[0])
            for col_idx, col in enumerate(cols):
                ax[col_idx, 0].plot(each.s, each[col], color='green')
                ax[col_idx, 1].plot(self.condRealizations[idx].s,
                                    self.condRealizations[idx][col], color='blue')

        print("\nData conditioning performed.\n")

    def aggregateRealization(self, fit_curves=True):
        """Aggregate GNSS data into one dataframe."""

        """Copy first dataset."""
        for each in ["lon", "lat", "alt", "v", "s", "a"]:
            self.sumRealization[each] = self.condRealizations[0][each]
        if fit_curves:
            print(
                f"{self.condRealizations[0]['Track name'].iloc[0]} aggregated.")
        else:
            print("Curves not fitted!")

        """Merge additional GNSS data sets."""
        df = self.sumRealization.copy()

        cols = ["lat", "lon", "alt", "v"]
        N_rolling = 16

        for idx in np.arange(1, len(self.condRealizations)):
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

                    interp_df_mean = (
                        interp_df[col]
                        .rolling(2 * N_rolling, center=True, min_periods=1)
                        .mean()
                    )
                    interp_cond_mean = (
                        interp_cond[col]
                        .rolling(2 * N_rolling, center=True, min_periods=1)
                        .mean()
                    )

                    interp_df[col] -= interp_df_mean
                    interp_cond[col] -= interp_cond_mean

                    correlation[col] = signal.correlate(
                        interp_df[col], interp_cond[col], mode="valid"
                    )
                    correlation[col + "_lag"] = signal.correlation_lags(
                        interp_df[col].size, interp_cond[col].size, mode="valid"
                    )

                    dist_offset[i] = (
                        correlation[col +
                                    "_lag"][np.argmax(correlation[col])] / 1000
                    )

                print(dist_offset)
                dist_mean = np.mean(dist_offset)
                dist_stdev = np.std(dist_offset)
                dist_offset = [
                    each for each in dist_offset if abs(each - dist_mean) <= dist_stdev
                ]

                offset = np.mean(dist_offset)
                print(dist_offset)

                self.condRealizations[idx].s += offset
                print(
                    f"{self.condRealizations[idx]['Track name'].iloc[0]} aggregated.")

            """Merge dataset."""
            df = df.append(
                self.condRealizations[idx][df.columns.values.tolist()])
            df.sort_values(by=["s"], inplace=True, ignore_index=True)

        self.sumRealization = df

        for col in cols:
            fig, ax = pyplot.subplots(dpi=400, figsize=(16, 9))
            for idx, each in enumerate(self.condRealizations):
                ax.plot(each.s, each[col],
                        label=self.query.receiverType.iloc[idx])
                ax.legend()

        print("\nData aggregation completed.\n")

    def averageRealization(self):
        """Clean sum GNSS data."""
        self.avRealization.s = self.sumRealization.s

        """Calculate rolling median of GNSS data based on distance."""
        cols = ["alt", "v", "lat", "lon", "a"]
        N = [50, 50, 50, 50, 50]

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

        print("Rolling means calculated.")

        print("\nMean and standard deviation values calculated.")

    def saveToDatabase(self, wdir):
        """Save calculated data to database."""

        con = sql.connect(os.path.join(wdir, "rawRealizations.db"))
        for each in self.rawRealizations:
            each.to_sql(each['Track name'].iloc[0], con,
                        if_exists='replace', index=False)
        con.close()

        con = sql.connect(os.path.join(wdir, "condRealizations.db"))
        for each in self.condRealizations:
            each.to_sql(each['Track name'].iloc[0], con,
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

        print("\nCalculated data saved.")


"""Calling simulation model to calculate."""
Model = Realizations()
main()
"""EOF"""
