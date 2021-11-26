#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 20:10:23 2021

@author: demust
"""

import gpxpy
import folium
import pandas as pd
import numpy as np

from geopy.distance import distance as gdist
from geopy.point import Point
from scipy import signal, stats
from scipy.interpolate import interp1d
from bokeh import plotting as plt
from bokeh import layouts as lyt
from bokeh.models import (
    ColumnDataSource,
    LabelSet,
    ZoomInTool,
    ZoomOutTool,
)
from bokeh.io import export_png

from matplotlib import pyplot
from decorators import time_it, time_its

import sys
import os

"""TODOs"""
# static diagram
# dynamic diagram

"""Simulation input parameters"""

working_dir = "Article/"
station_dir = "Stations/"
graph_dir = "Graphs/"
station_file = "FA_SU_IR.csv"


"""PATH"""

GNSS_files = [
    f for f in os.listdir(working_dir) if os.path.isfile(os.path.join(working_dir, f))
]
GNSS_files.sort(key=lambda f: os.path.splitext(f)[1], reverse=True)
graph_dir = os.path.join(working_dir, graph_dir)
graph_map = os.path.join(graph_dir, "map.html")
graph_GNSS_data = os.path.join(graph_dir, "GNSS_data.html")
graph_GNSS_sum_data = os.path.join(graph_dir, "GNSS_sum_data.html")

if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)

"""Main simulation"""


def main():
    """Calling simulation model."""
    print("\nFiles found:")
    [print(x) for x in GNSS_files]

    Model.generateStations(working_dir, station_dir, station_file)
    Model.loadRealization(working_dir, GNSS_files)
    Model.conditionRealization()
    Model.aggregateRealization()
    Model.averageRealization()
    Model.graph(
        GNSS_files,
        graph_map,
        graph_GNSS_data,
        graph_GNSS_sum_data,
    )

    [print(x) for x in time_its]


"""Simulation model"""


class Realizations:
    """Class definition for storing GNSS data and calculating track parameters."""

    @time_it
    def __init__(self):
        self.stations = pd.DataFrame(columns=["s", "station"])
        self.rawRealizations = []
        self.condRealizations = []
        self.sumRealization = pd.DataFrame(columns=["lon", "lat", "alt", "v", "s", "a"])
        self.avRealization = pd.DataFrame(
            columns=["lon", "lat", "alt", "alt_std", "v", "v_std", "s", "a", "a_std"]
        )

    @time_it
    def generateStations(self, wdir, sdir, filename):
        try:
            file = os.path.join(wdir, sdir, filename)
            print("\nStation file:")
            print(station_file, "\n")
            self.stations = pd.read_csv(file)
        except FileNotFoundError:
            print(f"{filename} Station data: File not found!")
            print("Stations not generated!\n")

    @time_it
    def loadRealization(self, wdir, fileList):
        """Load and calculate GNSS data from file."""
        for file_idx, file_instance in enumerate(fileList):
            if file_instance.endswith("UBX.CSV") or file_instance.endswith("UBX.csv"):
                try:
                    filename = os.path.join(wdir, file_instance)
                    df = pd.read_csv(filename)
                except FileNotFoundError:
                    print(f"{file_instance} Track data: File not found!")
                    sys.exit()
                self.rawRealizations.append(
                    pd.DataFrame(
                        columns=[
                            "Track index",
                            "Track name",
                            "Segment index",
                            "time",
                            "lon",
                            "lat",
                            "alt",
                            "hdop",
                            "vdop",
                            "pdop",
                            "nSAT",
                            "hAcc",
                            "vAcc",
                            "sAcc",
                            "v",
                            "t",
                            "s",
                            "a",
                        ]
                    )
                )
                t0 = df.Hour.iloc[0] * 3600 + df.Minute.iloc[0] * 60 + df.Second.iloc[0]

                self.rawRealizations[-1].time = (
                    df.Year.astype(str)
                    + df.Month.astype(str)
                    + df.Day.astype(str)
                    + df.Hour.astype(str)
                    + df.Minute.astype(str)
                    + df.Second.astype(str)
                )
                self.rawRealizations[-1].lon = df.Lon / 1.0e7
                self.rawRealizations[-1].lat = df.Lat / 1.0e7
                self.rawRealizations[-1].alt = df.Alt2 / 1.0e3
                self.rawRealizations[-1].hdop = 0.0
                self.rawRealizations[-1].vdop = 0.0
                self.rawRealizations[-1].pdop = df.PDOP / 1.0e2
                self.rawRealizations[-1].nSAT = df.nSAT
                self.rawRealizations[-1].hAcc = df.hAcc / 1.0e3
                self.rawRealizations[-1].vAcc = df.vAcc / 1.0e3
                self.rawRealizations[-1].sAcc = df.speedAcc / 1.0e3
                self.rawRealizations[-1].v = df.speed / 1.0e3
                self.rawRealizations[-1].t = (
                    df.Hour * 3600 + df.Minute * 60 + df.Second - t0
                )
                self.rawRealizations[-1].s = 0.0
                self.rawRealizations[-1].a = 0.0
                self.rawRealizations[-1]["Track index"] = 1
                self.rawRealizations[-1]["Track name"] = file_instance
                self.rawRealizations[-1]["Segment index"] = 1
                df_t = self.rawRealizations[-1].t.copy().astype(float)
                df_s = self.rawRealizations[-1].s.copy().astype(float)

                t_offset = 0.0
                for i in np.arange(1, len(df_t)):
                    if df_t[i] == t_offset:
                        df_t[i] = df_t[i - 1] + 0.2
                    else:
                        t_offset = df_t[i]

                    node1 = Point(
                        df.Lat.iloc[i] / 1.0e7,
                        df.Lon.iloc[i] / 1.0e7,
                    )
                    node2 = Point(
                        df.Lat.iloc[i - 1] / 1.0e7,
                        df.Lon.iloc[i - 1] / 1.0e7,
                    )
                    df_s[i] = np.sqrt(
                        (gdist(node1, node2).m) ** 2
                        + ((df.Alt2.iloc[i] - df.Alt2.iloc[i - 1]) / 1.0e3) ** 2
                    )

                df_s = np.cumsum(df_s)

                self.rawRealizations[-1].t = df_t
                self.rawRealizations[-1].s = df_s
                self.rawRealizations[-1].a = np.gradient(
                    self.rawRealizations[-1].v, self.rawRealizations[-1].t
                )

                point_no = len(self.rawRealizations[-1].index)
                print(f"Loaded {point_no} points from file {filename}")

            if file_instance.endswith("PMTK.CSV") or file_instance.endswith("PMTK.csv"):
                try:
                    filename = os.path.join(wdir, file_instance)
                    df = pd.read_csv(filename)
                except FileNotFoundError:
                    print(f"{file_instance} Track data: File not found!")
                    sys.exit()
                self.rawRealizations.append(
                    pd.DataFrame(
                        columns=[
                            "Track index",
                            "Track name",
                            "Segment index",
                            "time",
                            "lon",
                            "lat",
                            "alt",
                            "hdop",
                            "vdop",
                            "pdop",
                            "nSAT",
                            "hAcc",
                            "vAcc",
                            "sAcc",
                            "v",
                            "t",
                            "s",
                            "a",
                        ]
                    )
                )
                t0 = df.Hour.iloc[0] * 3600 + df.Minute.iloc[0] * 60 + df.Second.iloc[0]

                self.rawRealizations[-1].time = (
                    df.Year.astype(str)
                    + df.Month.astype(str)
                    + df.Day.astype(str)
                    + df.Hour.astype(str)
                    + df.Minute.astype(str)
                    + df.Second.astype(str)
                )
                self.rawRealizations[-1].lon = df.Lon
                self.rawRealizations[-1].lat = df.Lat
                self.rawRealizations[-1].alt = df.Alt
                self.rawRealizations[-1].hdop = df.hDOP / 100
                self.rawRealizations[-1].vdop = 0.0
                self.rawRealizations[-1].pdop = 0.0
                self.rawRealizations[-1].hAcc = 0.0
                self.rawRealizations[-1].vAcc = 0.0
                self.rawRealizations[-1].sAcc = 0.0
                self.rawRealizations[-1].nSAT = df.nSAT
                self.rawRealizations[-1].v = df.Speed
                self.rawRealizations[-1].t = (
                    df.Hour * 3600 + df.Minute * 60 + df.Second - t0
                )
                self.rawRealizations[-1].s = 0.0
                self.rawRealizations[-1].a = 0.0
                self.rawRealizations[-1]["Track index"] = 1
                self.rawRealizations[-1]["Track name"] = file_instance
                self.rawRealizations[-1]["Segment index"] = 1

                df_t = self.rawRealizations[-1].t.copy().astype(float)
                df_s = self.rawRealizations[-1].s.copy().astype(float)

                t_offset = 0.0
                for i in np.arange(1, len(df_t)):
                    if df_t[i] == t_offset:
                        df_t[i] = df_t[i - 1] + 0.1
                    else:
                        t_offset = df_t[i]

                    node1 = Point(
                        df.Lat.iloc[i],
                        df.Lon.iloc[i],
                    )
                    node2 = Point(
                        df.Lat.iloc[i - 1],
                        df.Lon.iloc[i - 1],
                    )
                    df_s[i] = np.sqrt(
                        (gdist(node1, node2).m) ** 2
                        + ((df.Alt.iloc[i] - df.Alt.iloc[i - 1])) ** 2
                    )

                df_s = np.cumsum(df_s)

                self.rawRealizations[-1].t = df_t
                self.rawRealizations[-1].s = df_s
                self.rawRealizations[-1].a = np.gradient(
                    self.rawRealizations[-1].v, self.rawRealizations[-1].t
                )

                point_no = len(self.rawRealizations[-1].index)
                print(f"Loaded {point_no} points from file {filename}")

            if file_instance.endswith(".gpx"):
                try:
                    filename = os.path.join(wdir, file_instance)
                    gpx_file = open(filename, "r")
                except FileNotFoundError:
                    print(f"{filename} Track data: File not found!")
                    sys.exit()
                gpx = gpxpy.parse(gpx_file)

                for track_idx, track in enumerate(gpx.tracks):
                    self.rawRealizations.append(
                        pd.DataFrame(
                            columns=[
                                "Track index",
                                "Track name",
                                "Segment index",
                                "time",
                                "lon",
                                "lat",
                                "alt",
                                "hdop",
                                "vdop",
                                "pdop",
                                "nSAT",
                                "hAcc",
                                "vAcc",
                                "sAcc",
                                "v",
                                "t",
                                "s",
                                "a",
                            ]
                        )
                    )
                    track.name = file_instance

                    for seg_idx, segment in enumerate(track.segments):
                        for point_idx, point in enumerate(segment.points):
                            t = point.time_difference(
                                gpx.tracks[track_idx].segments[seg_idx].points[0]
                            )
                            if point_idx > 0:
                                s = (
                                    point.distance_3d(
                                        gpx.tracks[track_idx]
                                        .segments[seg_idx]
                                        .points[point_idx - 1]
                                    )
                                    + self.rawRealizations[-1].s[point_idx - 1]
                                )
                                if point.speed == None:
                                    point.speed = point.speed_between(
                                        gpx.tracks[track_idx]
                                        .segments[seg_idx]
                                        .points[point_idx - 1]
                                    )
                                    if point.speed == None:
                                        point.speed = (
                                            gpx.tracks[track_idx]
                                            .segments[seg_idx]
                                            .points[point_idx - 1]
                                            .speed
                                        )

                            else:
                                s = 0.0
                                t = 0.0
                                point.v = 0.0
                                if point.speed == None:
                                    point.speed = 0.0

                            self.rawRealizations[-1] = self.rawRealizations[-1].append(
                                {
                                    "Track index": track_idx,
                                    "Track name": track.name,
                                    "Segment index": seg_idx,
                                    "time": point.time,
                                    "lon": point.longitude,
                                    "lat": point.latitude,
                                    "alt": point.elevation,
                                    "hdop": point.horizontal_dilution,
                                    "vdop": point.vertical_dilution,
                                    "pdop": 0.0,
                                    "nSAT": 0,
                                    "hAcc": 0.0,
                                    "vAcc": 0.0,
                                    "sAcc": 0.0,
                                    "v": point.speed,
                                    "t": t,
                                    "s": s,
                                    "a": 0.0,
                                },
                                ignore_index=True,
                            )

                        self.rawRealizations[-1].a = np.gradient(
                            self.rawRealizations[-1].v, self.rawRealizations[-1].t
                        )

                        point_no = len(segment.points)
                        print(
                            f"Loaded {point_no} points from segment {seg_idx}, track {track_idx}, file {filename}"
                        )
                gpx_file.close()

        for track in self.rawRealizations:
            track.s /= 1000
            track.v *= 3.6

        print("\nGNSS data loaded.\n")

    @time_it
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
                        - each[col].rolling(100, center=True, min_periods=1).mean()
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
        print("\nData conditioning performed.\n")

    @time_it
    def aggregateRealization(self):
        """Aggregate GNSS data into one dataframe."""

        """Copy first dataset."""
        for each in ["lon", "lat", "alt", "v", "s", "a"]:
            self.sumRealization[each] = self.condRealizations[0][each]
        print(f"{GNSS_files[0]} aggregated.")

        """Merge additional GNSS data sets."""
        df = self.sumRealization.copy()

        cols = ["lat", "lon", "alt", "v"]
        N_rolling = 16

        for idx in np.arange(1, len(self.condRealizations)):
            """Set distance offset based on cross-correlation."""

            interp_df = pd.DataFrame(columns=["x"] + cols)
            interp_cond = pd.DataFrame(columns=["x"] + cols)
            correlation = pd.DataFrame(columns=cols + [each + "_lag" for each in cols])

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
                    correlation[col + "_lag"][np.argmax(correlation[col])] / 1000
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

            """Merge dataset."""
            df = df.append(self.condRealizations[idx][df.columns.values.tolist()])
            df.sort_values(by=["s"], inplace=True, ignore_index=True)

            print(f"{GNSS_files[idx]} aggregated.")

        self.sumRealization = df

        for col in cols:
            fig, ax = pyplot.subplots()
            for idx, each in enumerate(self.condRealizations):
                ax.plot(each.s, each[col], label=idx)
                ax.legend()

        print("\nData aggregation completed.\n")

    @time_it
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

        print("\nMean and standard deviation values calculated.\n")

    @time_it
    def graph(
        self,
        filelist,
        gmap,
        gGNSS,
        gGNSSsum,
        conditionedData=True,
    ):
        """Create graphs for visualizing GNSS data."""
        linecolor = [
            "red",
            "blue",
            "green",
            "orange",
            "magenta",
            "brown",
            "purple",
            "greenyellow",
            "lightskyblue",
            "lightred",
        ]

        legendlabel = ["iPhone 11 Pro", "Gadget 1", "Gadget 2"]

        latitude, longitude, length = 0, 0, 0
        for track_idx, track in enumerate(self.condRealizations):
            latitude += np.sum(track.lat)
            longitude += np.sum(track.lon)
            length += len(track)
        latitude /= length
        longitude /= length

        """Static graph."""
        for each_idx, each in enumerate(self.rawRealizations):
            mean_x = each.lat.mean()
            mean_y = each.lon.mean()
            mean_z = each.alt.mean()
            dist_x = np.zeros(each.lat.size)
            dist_y = np.zeros(each.lon.size)
            dist_z = each.alt - mean_z
            for i in each.lat.index:
                dist_x[i] = gdist(
                    [each.lat.iloc[i], each.lon.iloc[i]],
                    [mean_x, each.lon.iloc[i]],
                ).m
                dist_y[i] = gdist(
                    [each.lat.iloc[i], each.lon.iloc[i]],
                    [each.lat.iloc[i], mean_y],
                ).m
                if each.lat.iloc[i] < mean_x:
                    dist_x[i] *= -1
                if each.lon.iloc[i] < mean_y:
                    dist_y[i] *= -1
            mean_x = dist_x.mean()
            mean_y = dist_y.mean()
            mean_z = dist_z.mean()
            std_x = dist_x.std()
            std_y = dist_y.std()
            std_z = dist_z.std()
            R_2D_2DRMS = 2 * np.sqrt(std_x ** 2 + std_y ** 2)
            R_3D_2DRMS = 2 * np.sqrt(std_x ** 2 + std_y ** 2 + std_z ** 2)
            fig, ax = pyplot.subplots(
                1,
                2,
                dpi=400,
                figsize=(16, 9),
                gridspec_kw={"width_ratios": [5, 1]},
                sharey=True,
            )
            ax[0].scatter(
                dist_x,
                dist_y,
                c="blue",
                s=1,
                marker=".",
                label="Recorded positions",
            )
            circle_2D = pyplot.Circle(
                (mean_x, mean_y),
                radius=R_2D_2DRMS,
                color="red",
                fill=False,
                label="2DRMS (2D): " + "{0:1.2f}".format(R_2D_2DRMS) + " m",
            )
            circle_3D = pyplot.Circle(
                (mean_x, mean_y),
                radius=R_3D_2DRMS,
                color="green",
                fill=False,
                label="2DRMS (3D): " + "{0:1.2f}".format(R_3D_2DRMS) + " m",
            )
            ax[0].add_patch(circle_2D)
            ax[0].add_patch(circle_3D)
            ax[0].set(
                xlabel="Longitudinal distance [m]",
                ylabel="Lateral distance [m]",
                aspect="equal",
                adjustable="datalim",
            )
            ax[0].legend()
            ax[1].scatter(
                each.index,
                dist_z,
                c="blue",
                s=1,
                marker=".",
                label="Recorded altitude",
            )
            ax[1].axhline(R_3D_2DRMS, color="green")
            ax[1].axhline(-R_3D_2DRMS, color="green")
            ax[1].set(ylabel="Vertical distance [m]", xticks=[])
            ax[1].yaxis.set_tick_params(labelleft=True)
            pyplot.savefig(
                os.path.join(graph_dir, "raw_static_" + str(each_idx) + ".png"),
                dpi=400,
            )
        for each_idx, each in enumerate(self.condRealizations):
            mean_x = each.lat.mean()
            mean_y = each.lon.mean()
            mean_z = each.alt.mean()
            dist_x = np.zeros(each.lat.size)
            dist_y = np.zeros(each.lon.size)
            dist_z = each.alt - mean_z
            for i in each.lat.index:
                dist_x[i] = gdist(
                    [each.lat.iloc[i], each.lon.iloc[i]],
                    [mean_x, each.lon.iloc[i]],
                ).m
                dist_y[i] = gdist(
                    [each.lat.iloc[i], each.lon.iloc[i]],
                    [each.lat.iloc[i], mean_y],
                ).m
                if each.lat.iloc[i] < mean_x:
                    dist_x[i] *= -1
                if each.lon.iloc[i] < mean_y:
                    dist_y[i] *= -1
            mean_x = dist_x.mean()
            mean_y = dist_y.mean()
            mean_z = dist_z.mean()
            std_x = dist_x.std()
            std_y = dist_y.std()
            std_z = dist_z.std()
            R_2D_2DRMS = 2 * np.sqrt(std_x ** 2 + std_y ** 2)
            R_3D_2DRMS = 2 * np.sqrt(std_x ** 2 + std_y ** 2 + std_z ** 2)
            fig, ax = pyplot.subplots(
                1,
                2,
                dpi=400,
                figsize=(16, 9),
                gridspec_kw={"width_ratios": [5, 1]},
                sharey=True,
            )
            ax[0].scatter(
                dist_x,
                dist_y,
                c="blue",
                s=1,
                marker=".",
                label="Recorded positions",
            )
            circle_2D = pyplot.Circle(
                (mean_x, mean_y),
                radius=R_2D_2DRMS,
                color="red",
                fill=False,
                label="2DRMS (2D): " + "{0:1.2f}".format(R_2D_2DRMS) + " m",
            )
            circle_3D = pyplot.Circle(
                (mean_x, mean_y),
                radius=R_3D_2DRMS,
                color="green",
                fill=False,
                label="2DRMS (3D): " + "{0:1.2f}".format(R_3D_2DRMS) + " m",
            )
            ax[0].add_patch(circle_2D)
            ax[0].add_patch(circle_3D)
            ax[0].set(
                xlabel="Longitudinal distance [m]",
                ylabel="Lateral distance [m]",
                aspect="equal",
                adjustable="datalim",
            )
            ax[0].legend()
            ax[1].scatter(
                each.index,
                dist_z,
                c="blue",
                s=1,
                marker=".",
                label="Recorded altitude",
            )
            ax[1].axhline(R_3D_2DRMS, color="green")
            ax[1].axhline(-R_3D_2DRMS, color="green")
            ax[1].set(ylabel="Vertical distance [m]", xticks=[])
            ax[1].yaxis.set_tick_params(labelleft=True)
            pyplot.savefig(
                os.path.join(graph_dir, "cond_static_" + str(each_idx) + ".png"),
                dpi=400,
            )
        print("Static graph plotted.")

        """Plot single rides on map."""
        myMap = folium.Map(location=[latitude, longitude], tiles="CartoDB positron")
        for track_idx, track in enumerate(self.rawRealizations):
            points = list(zip(track.lat, track.lon))
            folium.PolyLine(
                points, color=linecolor[track_idx], weight=2.5, opacity=1
            ).add_to(myMap)
        myMap.save(os.path.join(graph_dir, "map_raw.html"))

        myMap = folium.Map(location=[latitude, longitude], tiles="CartoDB positron")
        for track_idx, track in enumerate(self.condRealizations):
            points = list(zip(track.lat, track.lon))
            folium.PolyLine(
                points, color=linecolor[track_idx], weight=2.5, opacity=1
            ).add_to(myMap)
        myMap.save(os.path.join(graph_dir, "map_cond.html"))

        print("Track map plotted.")

        """Plot altitude."""
        fig = plt.figure(plot_width=700, plot_height=400)
        for each_idx, each in enumerate(self.rawRealizations):
            fig.line(
                each.s,
                each.alt,
                line_color=linecolor[each_idx],
                legend_label=legendlabel[each_idx],
            )
            fig.xaxis[0].axis_label = "s [km]"
            fig.yaxis[0].axis_label = "Altitude [m]"
            fig.toolbar_location = None
        export_png(fig, filename=os.path.join(graph_dir, "raw_alt.png"))

        fig = plt.figure(plot_width=700, plot_height=400)
        for each_idx, each in enumerate(self.condRealizations):
            fig.line(
                each.s,
                each.alt,
                line_color=linecolor[each_idx],
                legend_label=legendlabel[each_idx],
            )
            fig.xaxis[0].axis_label = "s [km]"
            fig.yaxis[0].axis_label = "Altitude [m]"
            fig.toolbar_location = None
        export_png(fig, filename=os.path.join(graph_dir, "cond_alt.png"))
        print("Altitude plotted.")

        """Plot speed."""
        fig = plt.figure(plot_width=700, plot_height=400)
        for each_idx, each in enumerate(self.rawRealizations):
            fig.line(
                each.s,
                each.v,
                line_color=linecolor[each_idx],
                legend_label=legendlabel[each_idx],
            )
            fig.xaxis[0].axis_label = "s [km]"
            fig.yaxis[0].axis_label = "Speed [km/h]"
            fig.toolbar_location = None
        export_png(fig, filename=os.path.join(graph_dir, "raw_speed.png"))

        fig = plt.figure(plot_width=700, plot_height=400)
        for each_idx, each in enumerate(self.condRealizations):
            fig.line(
                each.s,
                each.v,
                line_color=linecolor[each_idx],
                legend_label=legendlabel[each_idx],
            )
            fig.xaxis[0].axis_label = "s [km]"
            fig.yaxis[0].axis_label = "Speed [km/h]"
            fig.toolbar_location = None
        export_png(fig, filename=os.path.join(graph_dir, "cond_speed.png"))
        print("Speed plotted.")

        """Plot xDOP and nSAT."""
        cols = ["pdop", "hdop", "vdop", "nSAT"]
        legendlabel = ["PDOP", "HDOP", "VDOP", "nSAT"]
        for each_idx, each in enumerate(self.rawRealizations):
            fig = plt.figure(plot_width=350, plot_height=200)
            for col_idx, col in enumerate(cols):
                fig.line(
                    each.s,
                    each[col],
                    line_color=linecolor[col_idx],
                    legend_label=legendlabel[col_idx],
                )
                fig.xaxis[0].axis_label = "s [km]"
                fig.toolbar_location = None
            export_png(
                fig,
                filename=os.path.join(graph_dir, "raw_dop_" + str(each_idx) + ".png"),
            )

        for each_idx, each in enumerate(self.condRealizations):
            fig = plt.figure(plot_width=350, plot_height=200)
            for col_idx, col in enumerate(cols):
                fig.line(
                    each.s,
                    each[col],
                    line_color=linecolor[col_idx],
                    legend_label=legendlabel[col_idx],
                )
                fig.xaxis[0].axis_label = "s [km]"
                fig.toolbar_location = None
            export_png(
                fig,
                filename=os.path.join(graph_dir, "cond_dop_" + str(each_idx) + ".png"),
            )
        print("xDOP plotted.")

        """Plot conditioned single rides graphs."""
        if conditionedData:
            k = 0
            fig = []
            for each_idx, each in enumerate(self.condRealizations):
                for j, y in enumerate(["alt", "v", "a"]):
                    for track in [
                        self.rawRealizations[each_idx],
                        self.condRealizations[each_idx],
                    ]:
                        fig.append(
                            plt.figure(
                                title=track.loc[0, "Track name"],
                                title_location="left",
                                plot_height=250,
                            )
                        )
                        fig[k].line(track.s, track[y], line_color=linecolor[j])
                        fig[k].xaxis[0].axis_label = track.s.name
                        fig[k].yaxis[0].axis_label = track[y].name
                        k += 1

                for track in [
                    self.rawRealizations[each_idx],
                    self.condRealizations[each_idx],
                ]:
                    fig.append(
                        plt.figure(
                            title=track.loc[0, "Track name"],
                            title_location="left",
                            plot_height=250,
                        )
                    )
                    for j, y in enumerate(["hdop", "vdop", "pdop", "nSAT"]):
                        fig[k].line(
                            track.s,
                            track[y],
                            line_color=linecolor[j + 3],
                            legend_label=y,
                        )
                    fig[k].xaxis[0].axis_label = track.s.name
                    fig[k].yaxis[0].axis_label = "xDOP, nSAT"
                    fig[k].legend.location = "top_left"
                    k += 1

            plt.output_file(gGNSS)
            plt.save(lyt.gridplot(fig, ncols=2, sizing_mode="scale_width"))
            print("Conditioned data plotted.")

        """Plot sum ride data."""
        k = 0
        fig_sum = []
        for j, y in enumerate(["alt", "v", "a"]):
            fig_sum.append(plt.figure(plot_height=250))
            fig_sum[k].line(
                self.avRealization.s,
                self.avRealization[y],
                line_color=linecolor[j],
                line_dash="dotted",
            )
            for each in self.condRealizations:
                fig_sum[k].line(
                    each.s, each[y], line_alpha=0.2, line_color=linecolor[j]
                )

            fig_sum[k].varea(
                self.avRealization.s,
                self.avRealization[y] - 3 * self.avRealization[y + "_std"],
                self.avRealization[y] + 3 * self.avRealization[y + "_std"],
                fill_color=linecolor[j],
                fill_alpha=0.1,
            )

            source = ColumnDataSource(data=self.stations)
            labels = LabelSet(
                x="s",
                y=0,
                text="station",
                text_font_size="12px",
                text_font_style="bold",
                angle=np.pi / 2,
                source=source,
                render_mode="canvas",
            )
            fig_sum[k].add_layout(labels)
            fig_sum[k].add_tools(ZoomInTool(), ZoomOutTool())
            fig_sum[k].xaxis[0].axis_label = self.sumRealization.s.name
            fig_sum[k].yaxis[0].axis_label = self.sumRealization[y].name
            k += 1
        plt.output_file(gGNSSsum)
        plt.save(lyt.gridplot(fig_sum, ncols=1, sizing_mode="scale_width"))
        print("Track parameters plotted.")
        print("\nGraph completed.\n")


"""Calling simulation model to calculate."""
Model = Realizations()
main()
"""EOF"""
