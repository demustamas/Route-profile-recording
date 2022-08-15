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
import scipy.stats as ss

from geopy.distance import distance as gdist
from bokeh import plotting as bkh
from bokeh import layouts as lyt
from bokeh.models import (
    ColumnDataSource,
    LinearAxis,
    Range1d,
    LabelSet,
    ZoomInTool,
    ZoomOutTool,
)

from matplotlib import pyplot
from matplotlib import colors as mpl_color
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import os
import shutil


"""Simulation input parameters"""

database_dir = "Database/"
working_dir = "Results/Test/"
station_dir = "Stations/"
graph_dir = "Graphs/"
destination_dir = "ToCopy/"
name_tag = "_20220101"
palette = pyplot.cm.tab10
palette = palette(range(palette.N))
pyplot.style.use('mplstyle.work')

"""PATH"""

working_path = working_dir
graph_path = os.path.join(working_dir, graph_dir)
station_path = os.path.join(working_dir, station_dir + "stations.csv")
destination_path = destination_dir

if os.path.exists(graph_path):
    shutil.rmtree(graph_path)
os.makedirs(graph_path)


"""Main simulation"""


def main():
    """Calling simulation model."""

    # Model.generateStations(station_path)
    Model.queryRealizations(working_path)
    # Model.staticGraph(graph_path)
    # Model.altitudeGraph(graph_path)
    # Model.speedGraph(graph_path)
    # Model.accuracyGraph(graph_path)
    # Model.mapGraph(graph_path)
    # Model.characteristicsGraph(graph_path)
    # Model.trackGraph(graph_path)
    Model.vehGraph(graph_path)
    Model.controlMatrixGraph(graph_path)
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
        self.controlMatrix = []
        self.controlMatrixNorm = []
        self.controlMatrixSum = {}
        self.controlMatrixSumNorm = {}
        self.controlDuration = []
        self.controlDurationSum = {}

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

        npzFile = np.load(wPath + "controlMatrices.npz", allow_pickle=True)
        arrayName = []
        for file in npzFile.files:
            arrayName.append([])
            strings = file.split('_')
            arrayName[-1].append(strings[0])
            arrayName[-1].append(strings[1])
            arrayName[-1].append("_".join(strings[2:]))

        N_realizations = int((len(npzFile.files) - 6) / 2 / 3)
        control = ['traction', 'brake']
        for i in range(N_realizations):
            self.controlMatrix.append({})
            self.controlMatrixNorm.append({})
            self.controlDuration.append({})
            for ctrl in control:
                self.controlMatrix[-1][ctrl] = np.array([])
                self.controlMatrixNorm[-1][ctrl] = np.array([])
                self.controlDuration[-1][ctrl] = []
        for ctrl in control:
            self.controlMatrixSum[ctrl] = np.array([])
            self.controlMatrixSumNorm[ctrl] = np.array([])
            self.controlDurationSum[ctrl] = []

        for each in arrayName:
            try:
                idx = self.query.index[self.query.fileName == each[2]][0]
            except ValueError:
                print(
                    "Control function load failure! (Filename not found in query list.)")
            if each[0] == 'controlMatrix':
                self.controlMatrix[idx][each[1]] = npzFile["_".join(
                    each)]
            elif each[0] == 'controlMatrixNorm':
                self.controlMatrixNorm[idx][each[1]] = npzFile["_".join(
                    each)]
            elif each[0] == 'controlMatrixSum':
                self.controlMatrixSum[each[1]] = npzFile["_".join(each)]
            elif each[0] == 'controlMatrixSumNorm':
                self.controlMatrixSumNorm[each[1]] = npzFile["_".join(each)]
            elif each[0] == 'controlDuration':
                self.controlDuration[idx][each[1]] = npzFile["_".join(
                    each)]
            elif each[0] == 'controlDurationSum':
                self.controlDurationSum[each[1]] = npzFile["_".join(
                    each)]
            else:
                print("Wrong keyword found during loading npz file.")

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

        colors = iter([mpl_color.rgb2hex(each) for each in palette])
        myMap = folium.Map(
            location=[latitude, longitude], tiles="CartoDB positron")
        for track_idx, track in enumerate(self.rawRealizations):
            points = list(zip(track.lat, track.lon))
            folium.PolyLine(
                points, color=next(colors), weight=2.5, opacity=1
            ).add_to(myMap)
        myMap.save(os.path.join(graphPath, "map_raw.html"))

        colors = iter([mpl_color.rgb2hex(each) for each in palette])
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
        labels = ["Altitude [m]", "v [km/h]", "a [m/s2]"]
        linecolor = ['seagreen', 'royalblue', 'tomato']
        for idx in range(len(self.query.index)):
            for j, col in enumerate(cols):
                for track in [
                    self.rawRealizations[idx],
                    self.condRealizations[idx],
                ]:
                    fig.append(
                        bkh.figure(
                            title=track.loc[0, "trackName"],
                            title_location="left",
                            plot_height=250,
                        )
                    )
                    fig[k].line(track.s, track[col],
                                line_color=linecolor[j], legend_label=labels[j])
                    fig[k].xaxis[0].axis_label = "s [km]"
                    fig[k].yaxis[0].axis_label = labels[j]
                    k += 1

            for track in [
                self.rawRealizations[idx],
                self.condRealizations[idx],
            ]:
                fig.append(
                    bkh.figure(
                        title=track.loc[0, "trackName"],
                        title_location="left",
                        plot_height=250,
                    )
                )
                colors = iter([mpl_color.rgb2hex(each) for each in palette])
                for j, y in enumerate(["hdop", "vdop", "pdop", "nSAT"]):
                    fig[k].line(
                        track.s,
                        track[y],
                        line_color=next(colors),
                        legend_label=y,
                    )
                fig[k].xaxis[0].axis_label = "s [km]"
                fig[k].yaxis[0].axis_label = "xDOP, nSAT"
                fig[k].legend.location = "top_left"
                k += 1

        bkh.output_file(os.path.join(
            graphPath, "route_characteristics.html"))
        bkh.save(lyt.gridplot(fig, ncols=2, sizing_mode="stretch_width"))
        print("Route characteristics plotted.")

    def trackGraph(self, graphPath):
        """Plot track data."""
        k = 0
        fig_sum = []
        cols = ["alt", "v"]
        yaxis_label = ["Altitude [m]", "v [km/h]"]
        colorRealizations = ['seagreen', 'royalblue', 'lightsalmon']
        colorAverage = ['green', 'blue', 'red']

        for j, col in enumerate(cols):
            fig_sum.append(bkh.figure(plot_height=250))
            if col == "alt":
                fig_sum[k].line(
                    self.avRealization.s,
                    self.avRealization[col + "_filt"],
                    line_color=colorAverage[j],
                    line_width=2,
                    legend_label="Average altitude [m]",
                )
                fig_sum[k].line(
                    self.avRealization.s,
                    self.avRealization.alt_lin,
                    line_color='orange',
                    legend_label='Linear approximation',
                )
                fig_sum[k].extra_y_ranges = {"secondary": Range1d(
                    start=min(self.avRealization.alt_grad)-1, end=max(self.avRealization.alt_grad)+1)}
                fig_sum[k].add_layout(LinearAxis(
                    y_range_name="secondary", axis_label="Track gradient [1/1000]"), 'left')
                fig_sum[k].line(
                    self.avRealization.s,
                    self.avRealization.alt_grad,
                    line_color='orange',
                    legend_label='Track gradient [1/1000]',
                    y_range_name="secondary"
                )
            if col == "v":
                fig_sum[k].line(
                    self.avRealization.s,
                    self.avRealization[col + "_max"],
                    line_color=colorAverage[j],
                    line_width=2,
                    legend_label="v [km/h]",
                )
            for each in self.condRealizations:
                fig_sum[k].line(
                    each.s, each[col], line_alpha=0.2, line_color=colorRealizations[j], legend_label=col + " realizations",
                )

            fig_sum[k].varea(
                self.avRealization.s,
                self.avRealization[col] - 3 *
                self.avRealization[col + "_std"],
                self.avRealization[col] + 3 *
                self.avRealization[col + "_std"],
                fill_color=colorRealizations[j],
                fill_alpha=0.1,
                legend_label=col + " realizations",
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
            fig_sum[k].xaxis[0].axis_label = "s [km]"
            fig_sum[k].yaxis[0].axis_label = yaxis_label[j]
            k += 1
        bkh.output_file(os.path.join(
            graphPath, "track_data.html"))
        bkh.save(lyt.gridplot(fig_sum, ncols=1, sizing_mode="scale_width"))
        print("Track data plotted.")

    def vehGraph(self, graphPath):
        """Plot vehicle data."""
        k = 0
        fig_sum = []
        color = ['crimson', 'royalblue', 'seagreen']

        for each in self.condRealizations:
            fig_sum.append(bkh.figure(
                plot_height=200,
                title=each.loc[0, "trackName"],
                title_location="left",))
            data = each[['s', 'track_resistance',
                        'veh_resistance', 'F_traction']].copy()
            data[['track_resistance', 'veh_resistance', 'F_traction']] /= 1000
            fig_sum[k].line(data.s, data.F_traction,
                            line_color=color[1], legend_label='Traction and braking forces [kN]')
            fig_sum[k].varea_stack(
                ['veh_resistance', 'track_resistance'], x='s', color=['mediumslateblue', 'orange'], fill_alpha=0.2, legend_label=['Vehicle resistance [kN]', 'Track resistance [kN]'], source=data)
            fig_sum[k].yaxis[0].axis_label = "Traction / Braking forces, Track / Vehicle resistances [kN]"
            k += 1

            fig_sum.append(bkh.figure(
                plot_height=80,
                title=each.loc[0, "trackName"],
                title_location="left",))
            fig_sum[k].line(each.s, each.control,
                            line_color=color[0], legend_label='Control function')
            fig_sum[k].yaxis[0].axis_label = "Control positions"
            k += 1

        for each in fig_sum:
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
            each.add_layout(labels)
            each.add_tools(ZoomInTool(), ZoomOutTool())
            each.xaxis[0].axis_label = "s [km]"

        bkh.output_file(os.path.join(
            graphPath, "veh_data.html"))
        bkh.save(lyt.gridplot(fig_sum, ncols=1, sizing_mode="scale_width"))

        data = pd.concat(self.condRealizations, ignore_index=True)
        data.F_traction /= 1000

        fig = pyplot.figure()
        D = 4
        gs = fig.add_gridspec(D, 2*D)
        ax_3d = fig.add_subplot(gs[1:, :D], projection='3d')
        ax = fig.add_subplot(gs[1:, D:-1])
        ax_F = fig.add_subplot(gs[1:, -1], sharey=ax)
        ax_v = fig.add_subplot(gs[0, D:-1], sharex=ax)
        ax_text = fig.add_subplot(gs[0, :D])
        ax_Fc = ax_F.twiny()
        ax_vc = ax_v.twinx()

        N_bin = 120
        H, xedges, yedges = np.histogram2d(
            data.v, data.F_traction, bins=N_bin, density=True)
        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros(N_bin ** 2)

        dx = np.ones(N_bin ** 2) * (np.ptp(xedges)) / N_bin
        dy = np.ones(N_bin ** 2) * (np.ptp(yedges)) / N_bin
        dz = H.ravel()

        dc = pyplot.cm.Blues((dz / dz.max()) ** (1/5))

        ax_3d.bar3d(xpos, ypos, zpos, dx, dy, dz, color=dc)
        ax_3d.set_title('Probability density function')
        ax_3d.set_xlabel('v [km/h]')
        ax_3d.set_ylabel('Traction / Brake force [kN]')

        ax.scatter(data.v, data.F_traction, marker=".", c=color[1],
                   linewidth=0.5, edgecolors="none", alpha=0.5)
        ax.set_xlabel('v [km/h]', loc='right')
        ax.set_ylabel('Traction / Brake force [kN]')
        ax_F.hist(data.F_traction, bins=N_bin,
                  orientation='horizontal', color=color[1], density=True)
        ax_Fc.hist(data.F_traction, bins=N_bin, orientation='horizontal',
                   color=color[0], density=True, cumulative=True, histtype='step')
        ax_F.yaxis.set_tick_params(left=False, labelleft=False)
        ax_Fc.tick_params('x', colors=color[0])
        ax_v.hist(data.v, bins=N_bin, color=color[1], density=True)
        ax_v.xaxis.set_tick_params(bottom=False, labelbottom=False)
        ax_vc.hist(data.v, bins=N_bin,
                   color=color[0], density=True, cumulative=True, histtype='step')
        ax_vc.tick_params('y', colors=color[0])

        routes = 'Recordings:\n  ' + self.query.fileName.str.cat(sep='\n  ')
        ax_text.text(0.02, 0.7, routes, transform=ax_text.transAxes)
        ax_text.grid(False)
        ax_text.xaxis.set_tick_params(bottom=False, labelbottom=False)
        ax_text.yaxis.set_tick_params(left=False, labelleft=False)

        pyplot.savefig(
            os.path.join(
                graphPath, "F_v_distribution.png")
        )

        print("Vehicle data plotted.")

    def controlMatrixGraph(self, graphPath):
        """Create control matrix representations."""

        fig, ax = pyplot.subplots(1, 2)
        for idx, each in enumerate(self.controlMatrixSumNorm.items()):
            ax[idx].imshow(each[1], cmap='YlGn')
            ax[idx].set_xticks(
                np.arange(-0.5, len(each[1])-0.5, 1), minor=True)
            ax[idx].set_yticks(
                np.arange(-0.5, len(each[1])-0.5, 1), minor=True)
            ax[idx].set_xticks(np.arange(0, len(each[1]), 1))
            ax[idx].set_yticks(np.arange(0, len(each[1]), 1))
            ax[idx].xaxis.set_tick_params(
                top=True, bottom=False, labeltop=True, labelbottom=False, which='both')
            ax[idx].yaxis.grid(False, which='major')
            ax[idx].yaxis.grid(True, which='minor')
            ax[idx].xaxis.grid(False, which='major')
            ax[idx].xaxis.grid(True, which='minor')
            for (j, i), label in np.ndenumerate(each[1]):
                if label != 0:
                    ax[idx].text(i, j, f"{label:1.3f}",
                                 ha='center', va='center')
            ax[idx].set_title(f"Control matrix for {each[0]}")
        pyplot.savefig(
            os.path.join(
                graphPath, "control_matrix.png")
        )
        print("Control matrix graphs generated.")

        control = ['traction', 'brake']
        N = int(np.ceil(np.sqrt(len(self.controlDurationSum['brake']))))
        fig = pyplot.figure()
        gs = fig.add_gridspec(1, len(control))
        for ctrlIdx, ctrl in enumerate(control):
            gsub = gs[ctrlIdx].subgridspec(N, N)
            ax = gsub.subplots()
            ax = ax.flat
            for idx, each in enumerate(self.controlDurationSum[ctrl]):
                if each.size > 1:
                    ax[idx].hist(each, bins=20, density=True)
                    params = ss.expon.fit(each)
                    print(
                        f"\tExponential distribution fitted - {ctrl} - {idx}: {params[0]:1.3f}\t\t{params[1]:1.3f}")
                    rx = np.linspace(min(each), max(each), 100)
                    rp = ss.expon.pdf(rx, *params)
                    ax[idx].plot(rx, rp)
                    ax[idx].text(0.9, 0.9, f"λ = {params[1]:1.3f}",
                                 size=12, transform=ax[idx].transAxes, ha='right', va='top')
                    ax[idx].set_xlim(left=0)
                    ax[idx].set_xlabel("t[s]", loc='right')
                else:
                    print(
                        f"\tExponential distribution not fitted - {ctrl} - {idx}.")
                ax[idx].title.set_text(f"{ctrl} - {idx}")
        pyplot.savefig(
            os.path.join(
                graphPath, "control_duration.png")
        )
        print("Control duration histograms plotted.")

    def copyFiles(self, graphPath, destPath, tag):
        """Copy graph files to given directory."""

        if not os.path.exists(destPath):
            os.makedirs(destPath)
        [shutil.copy(graphPath + f, destPath + os.path.splitext(f)[0] + tag + os.path.splitext(f)[1])
         for f in os.listdir(graphPath) if os.path.isfile(os.path.join(graphPath, f))]


"""Calling simulation model to calculate."""
Model = Realizations()
main()
"""EOF"""
