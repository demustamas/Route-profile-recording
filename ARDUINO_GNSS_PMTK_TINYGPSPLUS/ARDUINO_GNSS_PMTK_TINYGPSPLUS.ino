#include <NeoSWSerial.h>
#include <SPI.h>
#include <SD.h>
#include <TinyGPSPlus.h>

#define PMTK_RESET "$PMTK104*37"
#define PMTK_SET_BAUD_38400 "$PMTK251,38400*27"
#define PMTK_SET_BAUD_57600 "$PMTK251,57600*2C"
#define PMTK_SET_NMEA_UPDATE_1HZ "$PMTK220,1000*1F"
#define PMTK_SET_NMEA_UPDATE_5HZ "$PMTK220,200*2C"
#define PMTK_SET_NMEA_UPDATE_10HZ "$PMTK220,100*2F"
#define PMTK_API_SET_FIX_CTL_5HZ "$PMTK300,200,0,0,0,0*2F"
#define PMTK_SET_NMEA_OUTPUT_GLLONLY "$PMTK314,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0*29"
#define PMTK_SET_NMEA_OUTPUT_RMCGGA "$PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0*28"
#define PMTK_SET_NMEA_OUTPUT_RMCGGAGSA "$PMTK314,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0*29"
#define PMTK_SET_NMEA_OUTPUT_ALLDATA "$PMTK314,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0*28"
#define PMTK_SET_NMEA_OUTPUT_OFF "$PMTK314,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0*28"
#define PMTK_Q_RELEASE "$PMTK605*31"
#define PMTK_Q_DGPS_MODE "$PMTK401*37"
#define PGCMD_ANTENNA_ON "$PGCMD,33,1*6C"
#define PGCMD_ANTENNA_OFF "$PGCMD,33,0*6D"

NeoSWSerial gpsPort(8, 7);
File sdFile;
TinyGPSPlus gps;

long fileNum = 1000;
char fileName[13];

class dataSet
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
    int hdop = 0;
    int vdop = 0;
    int pdop = 0;

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
} fix;

void setupGNSS();
void sendMSG(const char*);
void setupSDCard();
void incFileNum ();

void setup() {
  Serial.begin(57600);
  Serial.println();
  Serial.println(F("************************************************************************************************************* CONFIGURATION *************************************************************************************************************"));
  setupGNSS();
  setupSDCard();
  Serial.println(F("*********************************************************************************************************** DATA TRANSMISSION ***********************************************************************************************************"));
  for (byte i = 0; i < sizeof(fix.names) / sizeof(fix.names[0]); i++) {
    Serial.print(fix.names[i]);
    if (i == 7 or i == 8) Serial.print(F("\t"));
    Serial.print(F("\t"));
  }
  Serial.println();
}

void loop() {
  while (gpsPort.available() > 0)
  gps.encode(gpsPort.read());
  if (gps.location.isUpdated()){
    fix.year = gps.date.year();
    fix.month = gps.date.month();
    fix.day = gps.date.day();
    fix.hour = gps.time.hour();
    fix.minute = gps.time.minute();
    fix.second = gps.time.second();
    fix.nSat = gps.satellites.value();
    fix.lon = gps.location.lng();
    fix.lat = gps.location.lat();
    fix.alt = gps.altitude.meters();
    fix.gSpeed = gps.speed.mps();
    fix.hdop = gps.hdop.value();
    fix.printItems();
    fix.saveItems();
  }
}

void setupGNSS () {
  char c;
  Serial.println(F("Setting up GNSS receiver"));
  gpsPort.begin(9600);
  gpsPort.print(PMTK_SET_BAUD_38400);
  gpsPort.begin(38400);
  gpsPort.println(PMTK_SET_NMEA_OUTPUT_RMCGGAGSA);
  gpsPort.println(PMTK_SET_NMEA_UPDATE_10HZ);
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
      for (byte i = 0; i < sizeof(fix.names) / sizeof(fix.names[0]); i++) {
        sdFile.print(fix.names[i]);
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
