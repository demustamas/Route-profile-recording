#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 20:10:23 2021

@author: demust
"""

import sqlite3 as sql
import gpxpy
import pandas as pd
import numpy as np

from geopy.distance import distance as gdist
from geopy.point import Point

import sys
import os


"""PATH"""

database_dir = "Database/"
database_name = "GNSS recordings.db"
recordings_dir = "GNSS recordings/"
list_of_recordings = "listOfRecordings.csv"

if not os.path.exists(database_dir):
    os.makedirs(database_dir)

database_path = os.path.join(database_dir, database_name)
recordings_path = os.path.join(database_dir, recordings_dir)

GNSS_files = [
    f for f in os.listdir(os.path.join(database_dir, recordings_dir)) if os.path.isfile(os.path.join(database_dir, recordings_dir, f))
]
GNSS_files.sort()

"""Main simulation class"""


def main():
    """Calling simulation model."""

    Model.initDatabase(database_path, GNSS_files, list_of_recordings)
    Model.loadRealization(recordings_path)
    Model.saveToDatabase(database_path)


"""Simulation model"""


class Realizations:
    """Class definition for storing GNSS data."""

    def __init__(self):
        self.rawRealizations = []
        self.newRecordings = pd.DataFrame()

    def initDatabase(self, dbPath, fileList, listOfRecs):
        """Check input data interity, open connection to database and check database integrity. Generate list of new files."""
        try:
            df_recs = pd.read_csv(listOfRecs)
            self.newRecordings = self.newRecordings.reindex(
                columns=df_recs.columns)
        except FileNotFoundError:
            print(f"{listOfRecs} initDatabase: File not found!")
            sys.exit()

        print("\n")
        for each in fileList:
            entryCounter = df_recs.fileName.str.contains(each).sum()
            if entryCounter == 0:
                print(f"No entry for {each}")
            if entryCounter == 1:
                self.newRecordings = self.newRecordings.append(
                    df_recs[df_recs.fileName == each])
            if entryCounter > 1:
                print(f"Duplicate entry for {each}")

        for each in df_recs.fileName:
            if not any(each in x for x in fileList):
                print(f"No file found for {each}")

        con = sql.connect(dbPath)
        try:
            db_recs = pd.read_sql("""SELECT * FROM listOfRecordings""", con)
        except pd.io.sql.DatabaseError:
            con.execute("""CREATE TABLE listOfRecordings (
                id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                fileName TEXT,
                recType TEXT,
                dateTime DATETIME,
                fromStation TEXT,
                toStation TEXT,
                trainConfig TEXT,
                trainType TEXT,
                receiverType TEXT);""")
            con.commit()
            db_recs = pd.read_sql("""SELECT * FROM listOfRecordings""", con)

        con.close()
        pd.set_option('display.max_columns', None)
        print("\nDatabase content:")
        print(db_recs)

        print(
            f"\nChecking integrity of database with {listOfRecs}")
        db_recs.drop(columns='id', inplace=True)
        for each in db_recs.fileName:
            posf = df_recs.loc[df_recs.fileName == each]
            posd = db_recs.loc[db_recs.fileName == each]

            if (posf.iloc[0] != posd.iloc[0]).any():
                print(f"Inconsistency found: {each}")

        for each in db_recs.fileName:
            if any(each in x for x in self.newRecordings.fileName):
                self.newRecordings.drop(
                    self.newRecordings.loc[self.newRecordings.fileName == each].index, inplace=True)

        print("\nNew recordings found:")
        print(self.newRecordings)
        if not self.newRecordings.empty:
            ans = input("\nContinue with database update? (y/n): ")
            if ans == 'y' or ans == '':
                print("\nUpdating database.\n")
            else:
                print("\nExiting.\n")
                sys.exit()
        else:
            print("\nDatabase is up to date.\n")
            sys.exit()

    def loadRealization(self, recPath):
        """Load and calculate GNSS data from file."""
        for file_idx, file_instance in enumerate(self.newRecordings.fileName):
            file = os.path.join(recPath, file_instance)
            if file_instance.endswith("UBX.CSV") or file_instance.endswith("UBX.csv") or file_instance.startswith("UBX"):
                try:
                    df = pd.read_csv(file)
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
                print(f"Loaded {point_no} points from file {file_instance}")

            if file_instance.endswith("PMTK.CSV") or file_instance.endswith("PMTK.csv") or file_instance.startswith("PMTK"):
                try:
                    df = pd.read_csv(file)
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
                print(f"Loaded {point_no} points from file {file_instance}")

            if file_instance.endswith(".gpx"):
                try:
                    gpx_file = open(file, "r")
                except FileNotFoundError:
                    print(f"{file_instance} Track data: File not found!")
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
                            f"Loaded {point_no} points from segment {seg_idx}, track {track_idx}, file {file_instance}"
                        )
                gpx_file.close()

        for track in self.rawRealizations:
            track.s /= 1000
            track.v *= 3.6

        print("\nGNSS data loaded.\n")

    def saveToDatabase(self, db_path):
        con = sql.connect(db_path)
        for each in self.rawRealizations:
            print(f"Uploading {each['Track name'].iloc[0]} to database.")
            each.to_sql(each['Track name'].iloc[0], con,
                        if_exists='replace', index=False)
        print("Uploading new entries to database recordings list.")
        self.newRecordings.to_sql(
            'listOfRecordings', con, if_exists='append', index=False)
        con.close()


"""Calling simulation model to calculate."""
Model = Realizations()
main()
"""EOF"""
