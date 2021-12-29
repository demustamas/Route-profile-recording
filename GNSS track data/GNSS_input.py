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
    Model.calcRawRealization()
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
        print("Database content:")
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

        if not self.newRecordings.empty:
            print("\nNew recordings found:")
            print(self.newRecordings)
            ans = input("\nContinue with database update? (y/n): ")
            if ans == 'y' or ans == 'Y' or ans == '':
                print("\nUpdating database.\n")
            else:
                print("\nExiting.\n")
                sys.exit()
        else:
            print("\nDatabase is up to date.\n")
            sys.exit()

    def loadRealization(self, recPath):
        """Load and calculate GNSS data from file."""
        print("Loading GNSS data:")
        for file_idx, file_instance in enumerate(self.newRecordings.fileName):
            file = os.path.join(recPath, file_instance)
            self.rawRealizations.append(
                pd.DataFrame(
                    columns=[
                        "trackName",
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
                )
            )

            if file_instance.endswith("UBX.CSV") or file_instance.endswith("UBX.csv") or file_instance.startswith("UBX"):
                try:
                    fileType = "UBX"
                    samplingFreq = 5
                    df = pd.read_csv(file)
                except FileNotFoundError:
                    print(f"{file_instance} Track data: File not found!")
                    sys.exit()
            elif file_instance.endswith("PMTK.CSV") or file_instance.endswith("PMTK.csv") or file_instance.startswith("PMTK"):
                try:
                    fileType = "PMTK"
                    samplingFreq = 10
                    df = pd.read_csv(file)
                except FileNotFoundError:
                    print(f"{file_instance} Track data: File not found!")
                    sys.exit()
            elif file_instance.endswith(".gpx"):
                try:
                    fileType = "GPX"
                    samplingFreq = 1
                    gpx_file = open(file, "r")
                    gpx = gpxpy.parse(gpx_file)
                except FileNotFoundError:
                    print(f"{file_instance} Track data: File not found!")
                    sys.exit()
            else:
                print(f"Unknown file format {file_instance}!")
                sys.exit()

            if fileType == "UBX":
                t0 = df.Hour.iloc[0] * 3600 + \
                    df.Minute.iloc[0] * 60 + df.Second.iloc[0]
                self.rawRealizations[-1].lon = df.Lon / 1.0e7
                self.rawRealizations[-1].lat = df.Lat / 1.0e7
                self.rawRealizations[-1].alt = df.Alt2 / 1.0e3
                self.rawRealizations[-1].hdop = 0.0
                self.rawRealizations[-1].vdop = 0.0
                self.rawRealizations[-1].pdop = df.PDOP / 1.0e2
                self.rawRealizations[-1].nSAT = df.nSAT
                self.rawRealizations[-1].v = df.speed / 1.0e3
                self.rawRealizations[-1].t = (
                    df.Hour * 3600 + df.Minute * 60 + df.Second - t0
                )
            elif fileType == "PMTK":
                t0 = df.Hour.iloc[0] * 3600 + \
                    df.Minute.iloc[0] * 60 + df.Second.iloc[0]
                self.rawRealizations[-1].lon = df.Lon
                self.rawRealizations[-1].lat = df.Lat
                self.rawRealizations[-1].alt = df.Alt
                self.rawRealizations[-1].hdop = df.hDOP / 100
                self.rawRealizations[-1].vdop = 0.0
                self.rawRealizations[-1].pdop = 0.0
                self.rawRealizations[-1].nSAT = df.nSAT
                self.rawRealizations[-1].v = df.Speed
                self.rawRealizations[-1].t = (
                    df.Hour * 3600 + df.Minute * 60 + df.Second - t0
                )
            elif fileType == "GPX":
                segment = gpx.tracks[0].segments[0]
                self.rawRealizations[-1].lon = [x.longitude for x in segment.points]
                self.rawRealizations[-1].lat = [x.latitude for x in segment.points]
                self.rawRealizations[-1].alt = [x.elevation for x in segment.points]
                self.rawRealizations[-1].hdop = [
                    x.horizontal_dilution for x in segment.points]
                self.rawRealizations[-1].vdop = [
                    x.vertical_dilution for x in segment.points]
                self.rawRealizations[-1].pdop = 0.0
                self.rawRealizations[-1].nSAT = 0.0
                self.rawRealizations[-1].t = [x.time_difference(
                    segment.points[0]) for x in segment.points]
                self.rawRealizations[-1].v = [x.speed for x in segment.points]
                gpx_file.close()

            self.rawRealizations[-1].s = 0.0
            self.rawRealizations[-1].a = np.nan
            self.rawRealizations[-1]["trackName"] = file_instance

            df_t = self.rawRealizations[-1].t.copy().astype(float).to_numpy()
            for i in range(samplingFreq-1):
                cond = df_t[1:] == df_t[:-1]
                cond = np.insert(cond, 0, False)
                df_t[cond] += 1 / samplingFreq
            self.rawRealizations[-1].t = df_t

            point_no = len(self.rawRealizations[-1].index)
            print(f"\t\tLoaded {point_no} points from file {file_instance}")

    def calcRawRealization(self):
        """Calculate basic data for rawRealizations."""
        print("Calculating missing raw data:")
        for each in self.rawRealizations:
            df_lat = each.lat.copy().astype(float).to_numpy()
            df_lon = each.lon.copy().astype(float).to_numpy()
            df_alt = each.alt.copy().astype(float).to_numpy()
            node1 = list(map(Point, zip(df_lat[1:], df_lon[1:])))
            node2 = list(map(Point, zip(df_lat[:-1], df_lon[:-1])))
            dist = [x.m for x in list(map(gdist, node1, node2))]
            dist = np.sqrt(np.array(dist) ** 2 +
                           (df_alt[1:] - df_alt[:-1]) ** 2)
            dist = np.insert(dist, 0, 0.0)
            each.s = np.cumsum(dist)

            cond = each.t.shift() != each.t

            if each.v.isna().any():
                vel = each.v.copy()
                vel.loc[cond] = np.gradient(each.s.loc[cond], each.t.loc[cond])
                vel.fillna(method='ffill', inplace=True)
                each.v = vel
            acc = each.a.copy()
            acc.loc[cond] = np.gradient(each.v.loc[cond], each.t.loc[cond])
            acc.fillna(method='ffill', inplace=True)
            each.a = acc
            each.s /= 1000
            each.v *= 3.6

            print(each.head())
        print("Missing raw data calculation done.\n")

    def saveToDatabase(self, db_path):
        con = sql.connect(db_path)
        print("Uploading raw realizations to database:")
        for each in self.rawRealizations:
            print(f"\t\tUploading {each['trackName'].iloc[0]} to database.")
            each.to_sql(each['trackName'].iloc[0], con,
                        if_exists='replace', index=False)
        print("Uploading new entries to database recordings list.")
        self.newRecordings.to_sql(
            'listOfRecordings', con, if_exists='append', index=False)
        con.close()


"""Calling simulation model to calculate."""
Model = Realizations()
main()
"""EOF"""
