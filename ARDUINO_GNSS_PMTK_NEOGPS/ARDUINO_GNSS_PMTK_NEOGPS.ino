#include <GPSport.h>
#include <SPI.h>
#include <SD.h>
#include <NMEAGPS.h>

#define PMTK_RESET "PMTK104*37"
#define PMTK_SET_BAUD_57600 "PMTK251,57600*2C"
#define PMTK_SET_NMEA_UPDATE_1HZ "PMTK220,1000*1F"
#define PMTK_SET_NMEA_UPDATE_5HZ "PMTK220,200*2C"
#define PMTK_SET_NMEA_UPDATE_10HZ "PMTK220,100*2F"
#define PMTK_SET_NMEA_OUTPUT_GLLONLY "PMTK314,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0*29"
#define PMTK_SET_NMEA_OUTPUT_RMCGGA "PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0*28"
#define PMTK_SET_NMEA_OUTPUT_RMCGGAGSA "PMTK314,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0*29"
#define PMTK_SET_NMEA_OUTPUT_ALLDATA "PMTK314,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0*28"
#define PMTK_SET_NMEA_OUTPUT_OFF "PMTK314,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0*28"
#define PMTK_Q_RELEASE "PMTK605*31"
#define PMTK_Q_DGPS_MODE "PMTK401*37"
#define PGCMD_ANTENNA_ON "PGCMD,33,1*6C"
#define PGCMD_ANTENNA_OFF "PGCMD,33,0*6D"

File sdFile;

NMEAGPS gps;
gps_fix fix;

long fileNum = 1000;
char fileName[13];

class dataStructure
{
  public:
    String names[14] = {"Year", "Month", "Day", "Hour", "Minute", "Second",
                        "nSAT",
                        "Lon", "Lat", "Alt",
                        "Speed", "hDOP", "vDOP", "pDOP"
                       };

    long year = 1996;
    byte month = 0;
    byte day = 0;
    byte hour = 0;
    byte minute = 0;
    byte second = 0;
    byte nSat = 0;
    double lon = 0;
    double lat = 0;
    double alt = 0;
    double gSpeed = 0;
    byte hdop = 0;
    byte vdop = 0;
    byte pdop = 0;

    void printItems() {
      String data = "";
      data += year;
      data += "\t";
      data += month;
      data += "\t";
      data += day;
      data += "\t";
      data += hour;
      data += "\t";
      data += minute;
      data += "\t";
      data += second;
      data += "\t";
      data += nSat;
      data += "\t";
      data += String(lon, 6);
      data += "\t";
      data += String(lat, 6);
      data += "\t";
      data += String(alt, 1);
      data += "\t";
      data += String(gSpeed, 2);
      data += "\t";
      data += hdop;
      data += "\t";
      data += vdop;
      data += "\t";
      data += pdop;
      Serial.println(data);
    }

    void saveItems() {
      String data = "";
      data += year;
      data += ',';
      data += month;
      data += ',';
      data += day;
      data += ',';
      data += hour;
      data += ',';
      data += minute;
      data += ',';
      data += second;
      data += ',';
      data += nSat;
      data += ',';
      data += String(lon, 6);
      data += ',';
      data += String(lat, 6);
      data += ',';
      data += String(alt, 1);
      data += ',';
      data += String(gSpeed, 2);
      data += ',';
      data += hdop;
      data += ',';
      data += vdop;
      data += ',';
      data += pdop;
      sdFile.println(data);
      sdFile.flush();
    }
} dataSet;

void setupGNSS();
void setupSDCard();
void incFileNum ();

void setup() {
  Serial.begin(57600);
  Serial.println();
  Serial.println(F("************************************************************************************************************* CONFIGURATION *************************************************************************************************************"));
  setupGNSS();
  setupSDCard();
  Serial.println(F("*********************************************************************************************************** DATA TRANSMISSION ***********************************************************************************************************"));
  for (byte i = 0; i < sizeof(dataSet.names) / sizeof(dataSet.names[0]); i++) {
    Serial.print(dataSet.names[i]);
    if (i == 7 or i == 8) Serial.print(F("\t"));
    Serial.print(F("\t"));
  }
  Serial.println();
}

void loop() {
  while (gps.available(gpsPort)) {
    fix = gps.read();
    if (fix.valid.location) {
      dataSet.year = fix.dateTime.year;
      dataSet.month = fix.dateTime.month;
      dataSet.day = fix.dateTime.date;
      dataSet.hour = fix.dateTime.hours;
      dataSet.minute = fix.dateTime.minutes;
      dataSet.second = fix.dateTime.seconds;
      dataSet.nSat = fix.satellites;
      dataSet.lon = fix.longitude();
      dataSet.lat = fix.latitude();
      dataSet.alt = fix.altitude();
      dataSet.gSpeed = fix.speed_kph();
      dataSet.hdop = fix.hdop;
      dataSet.vdop = fix.vdop;
      dataSet.pdop = fix.pdop;

      dataSet.printItems();
      dataSet.saveItems();
    }
  }
}

void setupGNSS () {
  Serial.println(F("Setting up GNSS receiver"));
  gpsPort.begin(9600);
  gps.send_P(&gpsPort, F(PMTK_SET_BAUD_57600));
  gpsPort.begin(57600);
  gps.send_P(&gpsPort, F(PMTK_SET_NMEA_UPDATE_10HZ));
  gps.send_P(&gpsPort, F(PMTK_SET_NMEA_OUTPUT_RMCGGAGSA));
}

void setupSDCard() {
  Serial.println(F("Setting up SD card"));
  if (!SD.begin(10)) {
    Serial.println(F("SD card setup failed!"));
  }
  else {
    incFileNum();
    while (SD.exists(fileName)) {
      incFileNum();
    }
    sdFile = SD.open(fileName, FILE_WRITE);
    Serial.println(fileName);
    if (!sdFile) {
      Serial.println(F("File open error!"));
    }
    else {
      Serial.print(F("File opened successfully: "));
      Serial.println(sdFile.name());
      for (byte i = 0; i < sizeof(dataSet.names) / sizeof(dataSet.names[0]); i++) {
        sdFile.print(dataSet.names[i]);
        sdFile.print(",");
      }
      sdFile.println();
      sdFile.flush();
    }
  }
}

void incFileNum() {
  String s = "pmtk" + String(++fileNum) + ".csv";
  s.toCharArray(fileName, 13);
}
