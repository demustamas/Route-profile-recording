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
from bokeh.io import export_png


import sys
import os

"""TODOs"""
# Parameter calculation

"""Simulation input parameters"""

working_dir = "Dynamic_2/"
graph_dir = "Graphs/"


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
    print("\n")
    Model.loadRealization(working_dir, GNSS_files)
    Model.conditionRealization()
    Model.calcParams()
    Model.graph(
        GNSS_files,
        graph_map,
        graph_GNSS_data,
        graph_GNSS_sum_data,
    )


"""Simulation model"""


class Realizations:
    """Class definition for storing GNSS data and calculating track parameters."""

    def __init__(self):
        self.rawRealizations = []
        self.condRealizations = []
        self.interpRealizations = []

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
                t0 = df.Hour.iloc[0] * 3600 + \
                    df.Minute.iloc[0] * 60 + df.Second.iloc[0]

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
                        + ((df.Alt2.iloc[i] -
                           df.Alt2.iloc[i - 1]) / 1.0e3) ** 2
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
                t0 = df.Hour.iloc[0] * 3600 + \
                    df.Minute.iloc[0] * 60 + df.Second.iloc[0]

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
        print("\nData conditioning performed.\n")

    def calcParams(self):
        """Calculate dynamic characteristics."""
        v_lim = [60, 70, 80, 90, 100, 110, 120]
        v_mean = np.zeros(3)
        v_std = np.zeros(3)
        print("Dynamic parameters - RAW data:\n")
        for i in range(len(v_lim)):
            for each_idx, each in enumerate(self.rawRealizations):
                v_mean[each_idx] = each.v[(v_lim[i] < each.v) & (
                    each.v < v_lim[i] + 10)].mean()
                v_std[each_idx] = each.v[(v_lim[i] < each.v) & (
                    each.v < v_lim[i] + 10)].std()
            print("\t{0:3d}\t{1:7.2f} {2:7.2f}\t{3:7.2f} {4:7.2f}\t{5:7.2f} {6:7.2f}".format(
                v_lim[i]+10, v_mean[0], v_std[0], v_mean[1], v_std[1], v_mean[2], v_std[2]))
        print("\nDynamic parameters - COND data:\n")
        for i in range(len(v_lim)):
            for each_idx, each in enumerate(self.condRealizations):
                v_mean[each_idx] = each.v[(v_lim[i] < each.v) & (
                    each.v < v_lim[i] + 10)].mean()
                v_std[each_idx] = each.v[(v_lim[i] < each.v) & (
                    each.v < v_lim[i] + 10)].std()
            print("\t{0:3d}\t{1:7.2f} {2:7.2f}\t{3:7.2f} {4:7.2f}\t{5:7.2f} {6:7.2f}".format(
                v_lim[i]+10, v_mean[0], v_std[0], v_mean[1], v_std[1], v_mean[2], v_std[2]))
        print("\nDynamic characteristics calculated.\n")

        cols = [
            "lon",
            "lat",
            "alt",
            "hdop",
            "vdop",
            "pdop",
            "nSAT",
            "v",
            "t",
            "s",
            "a",
        ]
        t_start = max(self.condRealizations[i].t.iloc[0] for i in range(3))
        t_stop = min(self.condRealizations[i].t.iloc[-1] for i in range(3))
        for each in self.condRealizations:
            self.interpRealizations.append(
                pd.DataFrame(
                    columns=cols
                )
            )
            x_new = np.arange(t_start, t_stop,
                              0.1)
            for col in cols:
                f = interp1d(each.t, each[col])
                self.interpRealizations[-1][col] = f(x_new)

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

        """Plot single rides on map."""
        myMap = folium.Map(
            location=[latitude, longitude], tiles="CartoDB positron")
        for track_idx, track in enumerate(self.rawRealizations):
            points = list(zip(track.lat, track.lon))
            folium.PolyLine(
                points, color=linecolor[track_idx], weight=2.5, opacity=1
            ).add_to(myMap)
        myMap.save(os.path.join(graph_dir, "map_raw.html"))

        myMap = folium.Map(
            location=[latitude, longitude], tiles="CartoDB positron")
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

        """Plot altitude deviation."""
        fig = plt.figure(plot_width=700, plot_height=400)
        # for each_idx, each in enumerate(self.interpRealizations):
        each = self.interpRealizations
        fig.line(
            each[0].alt,
            each[1].alt,
            # line_color=linecolor[each_idx],
            # legend_label=legendlabel[each_idx],
        )
        fig.xaxis[0].axis_label = "s [km]"
        fig.yaxis[0].axis_label = "Altitude difference [m]"
        fig.toolbar_location = None
        # export_png(fig, filename=os.path.join(graph_dir, "cond_alt_diff"+ str(each_idx) +".png"))
        print("Altitude deviation plotted.")

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
                filename=os.path.join(
                    graph_dir, "raw_dop_" + str(each_idx) + ".png"),
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
                filename=os.path.join(
                    graph_dir, "cond_dop_" + str(each_idx) + ".png"),
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
                        fig[k].line(track.s, track[y],
                                    line_color=linecolor[j])
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
        print("\nGraph completed.\n")


"""Calling simulation model to calculate."""
Model = Realizations()
main()
"""EOF"""
