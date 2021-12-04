#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 20:10:23 2021

@author: demust
"""

import sqlite3 as sql
import folium
import pandas as pd
import numpy as np

from geopy.distance import distance as gdist
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

import os

"""TODOs"""
# static diagram
# dynamic diagram

"""Simulation input parameters"""

database_dir = "Database/"
working_dir = "Dynamic/"
station_dir = "Stations/"
graph_dir = "Graphs/"


"""PATH"""

working_path = os.path.join(database_dir, working_dir)
graph_path = os.path.join(database_dir, working_dir, graph_dir)
station_path = os.path.join(
    database_dir, working_dir, station_dir + "stations.csv")


if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)

"""Main simulation"""


def main():
    """Calling simulation model."""

    Model.generateStations(station_path)
    Model.queryRealizations(working_path)
    Model.graph()


"""Simulation model"""


class Realizations:
    """Class definition for storing GNSS data and calculating track parameters."""

    def __init__(self):
        self.stations = pd.DataFrame(columns=["s", "station"])
        self.query = pd.DataFrame()
        self.rawRealizations = []
        self.condRealizations = []
        self.sumRealization = pd.DataFrame()
        self.avRealization = pd.DataFrame()

    def generateStations(self, stat_path):
        try:
            print("\nStation file:")
            print(stat_path, "\n")
            self.stations = pd.read_csv(stat_path)
        except FileNotFoundError:
            print(f"{stat_path} File not found!")
            print("Stations not generated!\n")

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

    def graph(self):
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
                os.path.join(graph_dir, "raw_static_" +
                             str(each_idx) + ".png"),
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
                os.path.join(graph_dir, "cond_static_" +
                             str(each_idx) + ".png"),
                dpi=400,
            )
        print("Static graph plotted.")

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
