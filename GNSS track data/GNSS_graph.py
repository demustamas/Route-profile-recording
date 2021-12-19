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

from matplotlib import pyplot
from matplotlib import colors as color_func
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import os
from shutil import copy

# TODOs

"""Simulation input parameters"""

database_dir = "Database/"
working_dir = "Results/Test/"
station_dir = "Stations/"
graph_dir = "Graphs/"
destination_dir = "ToCopy/"
name_tag = ""
palette = pyplot.cm.tab10
palette = palette(range(palette.N))
pyplot.style.use('mplstyle.work')

"""PATH"""

working_path = working_dir
graph_path = os.path.join(working_dir, graph_dir)
station_path = os.path.join(working_dir, station_dir + "stations.csv")
destination_path = destination_dir

if not os.path.exists(graph_path):
    os.makedirs(graph_path)
else:
    [os.remove(os.path.join(graph_path, f)) for f in os.listdir(
        graph_path) if os.path.isfile(os.path.join(graph_path, f))]

"""Main simulation"""


def main():
    """Calling simulation model."""

    # Model.generateStations(station_path)
    Model.queryRealizations(working_path)
    # Model.staticGraph(graph_path)
    # Model.altitudeGraph(graph_path)
    # Model.speedGraph(graph_path)
    # Model.accuracyGraph(graph_path)
    Model.mapGraph(graph_path)
    Model.characteristicsGraph(graph_path)
    Model.statisticGraph(graph_path)
    # Model.copyFiles(graph_path, destination_path, name_tag)


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
            print("Stations not generated!")

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

    def staticGraph(self, graphPath):
        """Create static graphs."""

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

            fig, ax = pyplot.subplots(1, 1)
            ax.plot(
                dist_x,
                dist_y,
                color=palette[0],
                marker=".",
                markersize=2,
                linestyle="",
                label="Recorded positions",
            )
            circle_2D = pyplot.Circle(
                (mean_x, mean_y),
                radius=R_2D_2DRMS,
                color=palette[1],
                fill=False,
                label="2DRMS (2D)",
            )
            circle_3D = pyplot.Circle(
                (mean_x, mean_y),
                radius=R_3D_2DRMS,
                color=palette[2],
                fill=False,
                label="2DRMS (3D)",
            )
            print("Raw Data - 2DRMS (2D): " +
                  "{0:1.2f}".format(R_2D_2DRMS) + " m")
            print("Raw Data - 2DRMS (3D): " +
                  "{0:1.2f}".format(R_3D_2DRMS) + " m")
            ax.add_patch(circle_2D)
            ax.add_patch(circle_3D)
            ax.set(
                xlabel="Longitudinal distance [m]",
                ylabel="Lateral distance [m]",
                aspect="equal",
                adjustable="datalim",
            )
            ax_divider = make_axes_locatable(ax)
            ax_div = ax_divider.append_axes("right", size="7%", pad="20%")

            ax_div.plot(
                each.index,
                dist_z,
                color=palette[0],
                marker=".",
                markersize=2,
                linestyle="",
            )
            ax_div.axhline(R_3D_2DRMS, color="green")
            ax_div.axhline(-R_3D_2DRMS, color="green")
            ax_div.set(ylabel="Vertical distance [m]", xticks=[])
            ax_div.tick_params(left=False, right=True,
                               labelleft=False, labelright=True)
            pyplot.savefig(
                os.path.join(
                    graphPath, f"raw_static_{self.query.receiverType.iloc[each_idx]}.png"),
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
            fig, ax = pyplot.subplots(1, 1)
            ax.plot(
                dist_x,
                dist_y,
                color=palette[0],
                marker=".",
                markersize=2,
                linestyle="",
                label="Recorded positions",
            )
            circle_2D = pyplot.Circle(
                (mean_x, mean_y),
                radius=R_2D_2DRMS,
                color=palette[1],
                fill=False,
                label="2DRMS (2D)",
            )
            circle_3D = pyplot.Circle(
                (mean_x, mean_y),
                radius=R_3D_2DRMS,
                color=palette[2],
                fill=False,
                label="2DRMS (3D)",
            )
            print("Conditioned Data - 2DRMS (2D): " +
                  "{0:1.2f}".format(R_2D_2DRMS) + " m")
            print("Conditioned Data - 2DRMS (3D): " +
                  "{0:1.2f}".format(R_3D_2DRMS) + " m")
            ax.add_patch(circle_2D)
            ax.add_patch(circle_3D)
            ax.set(
                xlabel="Longitudinal distance [m]",
                ylabel="Lateral distance [m]",
                aspect="equal",
                adjustable="datalim",
            )
            ax_divider = make_axes_locatable(ax)
            ax_div = ax_divider.append_axes("right", size="7%", pad="20%")

            ax_div.plot(
                each.index,
                dist_z,
                color=palette[0],
                marker=".",
                markersize=2,
                linestyle="",
            )
            ax_div.axhline(R_3D_2DRMS, color="green")
            ax_div.axhline(-R_3D_2DRMS, color="green")
            ax_div.set(ylabel="Vertical distance [m]", xticks=[])
            ax_div.tick_params(left=False, right=True,
                               labelleft=False, labelright=True)
            pyplot.savefig(
                os.path.join(
                    graphPath, f"cond_static_{self.query.receiverType.iloc[each_idx]}.png")
            )
        print("Static graph plotted.")

    def altitudeGraph(self, graphPath):
        """Create altitude graph."""

        colors = iter(palette)
        fig, ax = pyplot.subplots(1, 1)
        for each_idx, each in enumerate(self.rawRealizations):
            ax.plot(
                each.s,
                each.alt,
                color=next(colors),
                label=self.query.receiverType[each_idx],
            )
            ax.legend(ncol=3)
            ax.set(
                xlabel="s [km]",
                ylabel="Altitude [m]",
            )
            pyplot.savefig(os.path.join(graphPath, "raw_alt.png"))

        colors = iter(palette)
        fig, ax = pyplot.subplots(1, 1)
        for each_idx, each in enumerate(self.condRealizations):
            ax.plot(
                each.s,
                each.alt,
                color=next(colors),
                label=self.query.receiverType[each_idx],
            )
            ax.legend(ncol=3)
            ax.set(
                xlabel="s [km]",
                ylabel="Altitude [m]",
            )
            pyplot.savefig(os.path.join(graphPath, "cond_alt.png"))

        print("Altitude plotted.")

    def speedGraph(self, graphPath):
        """Plot speed."""

        colors = iter(palette)
        fig, ax = pyplot.subplots(1, 1)
        v_tresh = np.arange(60, 140, 10, )
        for each_idx, each in enumerate(self.rawRealizations):
            for i in range(len(v_tresh)-1):
                v_step = each.v[(each.v > v_tresh[i])
                                & (each.v < v_tresh[i+1])].mean()
                sigma_step = each.v[(each.v > v_tresh[i])
                                    & (each.v < v_tresh[i+1])].std()
                print(
                    f"{v_tresh[i]:5d}{v_tresh[i+1]:5d}{v_step:9.2f}{sigma_step:9.4f}\t\t{self.query.receiverType[each_idx]}")

            ax.plot(
                each.s,
                each.v,
                color=next(colors),
                label=self.query.receiverType[each_idx],
            )
            ax.legend(ncol=3)
            ax.set(
                xlabel="s [km]",
                ylabel="Speed [km/h]",
            )
            pyplot.savefig(os.path.join(graphPath, "raw_speed.png"))

        colors = iter(palette)
        fig, ax = pyplot.subplots(1, 1)
        for each_idx, each in enumerate(self.condRealizations):
            for i in range(len(v_tresh)-1):
                v_step = each.v[(each.v > v_tresh[i])
                                & (each.v < v_tresh[i+1])].mean()
                sigma_step = each.v[(each.v > v_tresh[i])
                                    & (each.v < v_tresh[i+1])].std()
                print(
                    f"{v_tresh[i]:5d}{v_tresh[i+1]:5d}{v_step:9.2f}{sigma_step:9.4f}\t\t{self.query.receiverType[each_idx]}")

            ax.plot(
                each.s,
                each.v,
                color=next(colors),
                label=self.query.receiverType[each_idx],
            )
            ax.legend(ncol=3)
            ax.set(
                xlabel="s [km]",
                ylabel="Speed [km/h]",
            )
            pyplot.savefig(os.path.join(graphPath, "cond_speed.png"))
        print("Speed plotted.")

    def accuracyGraph(self, graphPath):
        """Plot xDOP and nSAT."""
        cols = ["pdop", "hdop", "vdop", "nSAT"]

        for each_idx, each in enumerate(self.rawRealizations):
            colors = iter(palette)
            fig, ax = pyplot.subplots(1, 1)
            for col_idx, col in enumerate(cols):
                ax.plot(each.s, each[col], color=next(colors), label=col)
                ax.set(
                    xlabel="s [km]",
                )
                ax.legend(ncol=len(cols))
            pyplot.savefig(os.path.join(
                graphPath, f"raw_dop_{self.query.receiverType[each_idx]}.png"))

        for each_idx, each in enumerate(self.condRealizations):
            colors = iter(palette)
            fig, ax = pyplot.subplots(1, 1)
            for col_idx, col in enumerate(cols):
                ax.plot(each.s, each[col], color=next(colors), label=col)
                ax.set(
                    xlabel="s [km]",
                )
                ax.legend(ncol=len(cols))
            pyplot.savefig(os.path.join(
                graphPath, f"cond_dop_{self.query.receiverType[each_idx]}.png"))

        print("xDOP plotted.")

    def mapGraph(self, graphPath):
        """Put single rides on map."""

        latitude, longitude, length = 0, 0, 0
        for track_idx, track in enumerate(self.condRealizations):
            latitude += np.sum(track.lat)
            longitude += np.sum(track.lon)
            length += len(track)
        latitude /= length
        longitude /= length

        colors = iter([color_func.rgb2hex(each) for each in palette])
        myMap = folium.Map(
            location=[latitude, longitude], tiles="CartoDB positron")
        for track_idx, track in enumerate(self.rawRealizations):
            points = list(zip(track.lat, track.lon))
            folium.PolyLine(
                points, color=next(colors), weight=2.5, opacity=1
            ).add_to(myMap)
        myMap.save(os.path.join(graphPath, "map_raw.html"))

        colors = iter([color_func.rgb2hex(each) for each in palette])
        myMap = folium.Map(
            location=[latitude, longitude], tiles="CartoDB positron")
        for track_idx, track in enumerate(self.condRealizations):
            points = list(zip(track.lat, track.lon))
            folium.PolyLine(
                points, color=next(colors), weight=2.5, opacity=1
            ).add_to(myMap)
        myMap.save(os.path.join(graphPath, "map_cond.html"))

        print("Track map plotted.")

    def characteristicsGraph(self, graphPath):
        """Plot characteristics of each recorded route."""

        k = 0
        fig = []
        cols = ["alt", "v", "a"]
        linecolor = ['seagreen', 'royalblue', 'tomato']
        for idx in range(len(self.query.index)):
            for j, col in enumerate(cols):
                for track in [
                    self.rawRealizations[idx],
                    self.condRealizations[idx],
                ]:
                    fig.append(
                        plt.figure(
                            title=track.loc[0, "Track name"],
                            title_location="left",
                            plot_height=250,
                        )
                    )
                    fig[k].line(track.s, track[col],
                                line_color=linecolor[j])
                    fig[k].xaxis[0].axis_label = track.s.name
                    fig[k].yaxis[0].axis_label = track[col].name
                    k += 1

            for track in [
                self.rawRealizations[idx],
                self.condRealizations[idx],
            ]:
                fig.append(
                    plt.figure(
                        title=track.loc[0, "Track name"],
                        title_location="left",
                        plot_height=250,
                    )
                )
                colors = iter([color_func.rgb2hex(each) for each in palette])
                for j, y in enumerate(["hdop", "vdop", "pdop", "nSAT"]):
                    fig[k].line(
                        track.s,
                        track[y],
                        line_color=next(colors),
                        legend_label=y,
                    )
                fig[k].xaxis[0].axis_label = track.s.name
                fig[k].yaxis[0].axis_label = "xDOP, nSAT"
                fig[k].legend.location = "top_left"
                k += 1

        plt.output_file(os.path.join(
            graphPath, "route_characteristics.html"))
        plt.save(lyt.gridplot(fig, ncols=2, sizing_mode="scale_width"))
        print("Route characteristics plotted.")

    def statisticGraph(self, graphPath):
        """Plot sum ride data."""
        k = 0
        fig_sum = []
        cols = ["alt", "v", "a"]
        linecolor = ['seagreen', 'royalblue', 'tomato']
        for j, col in enumerate(cols):
            fig_sum.append(plt.figure(plot_height=250))
            fig_sum[k].line(
                self.avRealization.s,
                self.avRealization[col],
                line_color=linecolor[j],
                line_dash="dotted",
            )
            for each in self.condRealizations:
                fig_sum[k].line(
                    each.s, each[col], line_alpha=0.2, line_color=linecolor[j]
                )

            fig_sum[k].varea(
                self.avRealization.s,
                self.avRealization[col] - 3 * self.avRealization[col + "_std"],
                self.avRealization[col] + 3 * self.avRealization[col + "_std"],
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
            fig_sum[k].yaxis[0].axis_label = self.sumRealization[col].name
            k += 1
        plt.output_file(os.path.join(
            graphPath, "route_statistics.html"))
        plt.save(lyt.gridplot(fig_sum, ncols=1, sizing_mode="scale_width"))
        print("Route statistics plotted.")

    def copyFiles(self, graphPath, destPath, tag):
        if not os.path.exists(destPath):
            os.makedirs(destPath)
        [copy(graphPath + f, destPath + os.path.splitext(f)[0] + tag + os.path.splitext(f)[1])
         for f in os.listdir(graphPath) if os.path.isfile(os.path.join(graphPath, f))]


"""Calling simulation model to calculate."""
Model = Realizations()
main()
"""EOF"""
