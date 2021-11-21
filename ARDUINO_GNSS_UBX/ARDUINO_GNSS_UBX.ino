#include <AltSoftSerial.h>
#include <SPI.h>
#include <SD.h>

AltSoftSerial gnss;

File sdFile;
long fileNum = 10000;
char fileName[13];
double dataBuffer = 0;
bool replyFound = false;
bool checkSum = true;
byte pos = 0;
byte idxData = 10;
byte idxMessage = 0;
byte CK_A = 0;
byte CK_B = 0;

byte UBX_getGNSSInformation[8] = {0xB5, 0x62, 0x0A, 0x28, 0x00, 0x00,
                                  0x00, 0x00                // Checksum placeholder
                                 };

byte UBX_getGNSSConfig[8] = {0xB5, 0x62, 0x06, 0x3E, 0x00, 0x00,
                             0x00, 0x00                // Checksum placeholder
                            };

byte UBX_setMessageRate[14] = {0xB5, 0x62, 0x06, 0x08, 0x06, 0x00,
                               0xC8, 0x00,                  // measRate: 5 Hz
                               0x01, 0x00,                  // navRate
                               0x01, 0x00,                  // timeRef - GPS time
                               0x00, 0x00                   // Checksum placeholder
                              };

byte UBX_setConfig[28] = {0xB5, 0x62, 0x06, 0x00, 0x14, 0x00,
                          0x01,                     // portID
                          0x00,                     // reserved
                          0x00, 0x00,               // txReady
                          0xC0, 0x08, 0x00, 0x00,   // UART mode
                          0x00, 0xE1, 0x00, 0x00,   // Baudrate to 57600
                          0x01, 0x00,               // inProtoMask - only UBX
                          0x01, 0x00,               // outProtoMask - only UBX
                          0x00, 0x00,               // Flags
                          0x00, 0x00,               // Reserved
                          0x00, 0x00                // Checksum placeholder
                         };

byte UBX_set_NAV_PVT[11] = {0xB5, 0x62, 0x06, 0x01, 0x03, 0x00,
                            0x01,                     // Message class
                            0x07,                     // Message ID
                            0x01,                     // Message rate
                            0x00, 0x00                // Checksum placeholder
                           };

byte UBX_disable_NAV_PVT[11] = {0xB5, 0x62, 0x06, 0x01, 0x03, 0x00,
                                0x01,                     // Message class
                                0x07,                     // Message ID
                                0x00,                     // Message rate
                                0x00, 0x00                // Checksum placeholder
                               };

class dataSet
{
  public:
    byte messageLength = 100;

    String names[16] = {"Year", "Month", "Day", "Hour", "Minute", "Second",
                        "nSAT",
                        "Lon", "Lat", "Alt1", "Alt2", "hAcc", "vAcc",
                        "speed", "speedAcc", "PDOP"
                       };

    byte positions[16] = {4, 6, 7, 8, 9, 10,
                          23,
                          24, 28, 32, 36, 40, 44,
                          60, 68, 76
                         };

    byte lengths[16] = {2, 1, 1, 1, 1, 1,
                        1,
                        4, 4, 4, 4, 4, 4,
                        4, 4, 2
                       };

    long year = 0;
    byte month = 0;
    byte day = 0;
    byte hour = 0;
    byte minute = 0;
    byte second = 0;
    byte nSat = 0;
    double lon = 0;
    double lat = 0;
    double altEllipsoid = 0;
    double altMSL = 0;
    double hAcc = 0;
    double vAcc = 0;
    double gSpeed = 0;
    double sAcc = 0;
    long PDOP = 0;

    void setItem(byte index, double value) {
      switch (index) {
        case 0:
          year = long(round(value));
          break;
        case 1:
          month = byte(round(value));
          break;
        case 2:
          day = byte(round(value));
          break;
        case 3:
          hour = byte(round(value));
          break;
        case 4:
          minute = byte(round(value));
          break;
        case 5:
          second = byte(round(value));
          break;
        case 6:
          nSat = byte(round(value));
          break;
        case 7:
          lon = value;
          break;
        case 8:
          lat = value;
          break;
        case 9:
          altEllipsoid = value;
          break;
        case 10:
          altMSL = value;
          break;
        case 11:
          hAcc = value;
          break;
        case 12:
          vAcc = value;
          break;
        case 13:
          gSpeed = value;
          break;
        case 14:
          sAcc = value;
          break;
        case 15:
          PDOP = long(round(value));
          break;
      }
    }

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
      data += lon;
      data += "\t";
      data += lat;
      data += "\t";
      data += altEllipsoid;
      data += "\t";
      data += altMSL;
      data += "\t";
      data += hAcc;
      data += "\t";
      data += vAcc;
      data += "\t";
      data += gSpeed;
      data += "\t";
      data += sAcc;
      data += "\t";
      data += PDOP;
      data += "\t";
      Serial.print(data);
    }

    void saveItems() {
      String data = "";
      if (nSat >= 3) {
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
        data += lon;
        data += ',';
        data += lat;
        data += ',';
        data += altEllipsoid;
        data += ',';
        data += altMSL;
        data += ',';
        data += hAcc;
        data += ',';
        data += vAcc;
        data += ',';
        data += gSpeed;
        data += ',';
        data += sAcc;
        data += ',';
        data += PDOP;
        sdFile.println(data);
        sdFile.flush();
        Serial.print(F("Data saved!"));
      }
      else {
        Serial.print(F("Data dropped! (nSAT<3)"));
      }
    }
} NAV_PVT;

void setupGNSS();
void setupSDCard();
void sendMSG (byte [], int);
void incFileNum ();
void calcCheckSum (byte [], int);
byte hex2dec (char);

void setup() {
  Serial.begin(57600);
  Serial.println();
  Serial.println(F("***************************************************************************************************************** SETUP *****************************************************************************************************************"));
  Serial.println(F("Serial communication OK"));
  delay(500);

  setupGNSS();
  setupSDCard();

  while (gnss.available()) {
    gnss.read();
  }
  Serial.println(F("Communication buffer flushed."));

  replyFound = false;
  pos = 0;
  Serial.println(F("Serial communication with GNSS receiver initialized."));

  Serial.println();
  Serial.println(F("*********************************************************************************************************** DATA TRANSMISSION ***********************************************************************************************************"));
  for (byte i = 0; i < sizeof(NAV_PVT.names) / sizeof(NAV_PVT.names[0]); i++) {
    Serial.print(NAV_PVT.names[i]);
    if (NAV_PVT.lengths[i] > 2) {
      Serial.print(F("\t"));
    }
    Serial.print(F("\t"));
  }
  Serial.println();
}

void loop() {
  char rbBuffer[3];
  while (gnss.available() > 0 and pos < NAV_PVT.messageLength) {
    byte rb = gnss.read();
    if (rb == 0xB5 or replyFound == true) {
      replyFound = true;
      sprintf(rbBuffer, "%02X", rb);

      if (pos > 1 and pos < NAV_PVT.messageLength - 2) {
        CK_A +=  rb;
        CK_B +=  CK_A;
      }

      if (pos == NAV_PVT.positions[idxMessage] + 6) {
        idxData = 0;
      }
      if (idxData < NAV_PVT.lengths[idxMessage]) {
        for (byte i = 0; i < sizeof(rbBuffer) / sizeof(rbBuffer[0]); i++) {
          dataBuffer += hex2dec(rbBuffer[i]) * pow(16, 1 - i) * pow(16, 2 * idxData);
        }
        idxData++;
      }

      if (idxData == NAV_PVT.lengths[idxMessage]) {
        NAV_PVT.setItem(idxMessage, dataBuffer);
        if (idxMessage < sizeof(NAV_PVT.positions) / sizeof(NAV_PVT.positions[0]) - 1) {
          idxMessage++;
        }
        dataBuffer = 0;
        idxData = 10;
      }

      if (pos == NAV_PVT.messageLength - 2) {
        if (rb != CK_A) {
          checkSum = false;
        }
      }

      if (pos == NAV_PVT.messageLength - 1) {
        if (rb != CK_B) {
          checkSum = false;
        }
      }

      if (pos == NAV_PVT.messageLength - 1) {
        NAV_PVT.printItems();
        Serial.print(F("\t\t\t\t"));
        sprintf(rbBuffer, "%02X", CK_A);
        Serial.print(rbBuffer);
        Serial.print(F(" "));
        sprintf(rbBuffer, "%02X", CK_B);
        Serial.print(rbBuffer);
        if (checkSum) {
          Serial.print(F(" CheckSum OK!"));
          Serial.print(F("\t"));
          NAV_PVT.saveItems();
        }
        else {
          Serial.print(F(" CheckSum error!\t Data dropped!"));
        }
        Serial.println();
        replyFound = false;
        checkSum = true;
        pos = 0;
        idxMessage = 0;
        CK_A = 0;
        CK_B = 0;
      }
      else {
        pos++;
      }
    }
  }
}

void setupGNSS() {
  Serial.println(F("Setting up GNSS"));
  gnss.begin(9600);
  sendMSG(UBX_setConfig, sizeof(UBX_setConfig));
  gnss.begin(57600);
  sendMSG(UBX_disable_NAV_PVT, sizeof(UBX_disable_NAV_PVT));
  sendMSG(UBX_setMessageRate, sizeof(UBX_setMessageRate));
  sendMSG(UBX_getGNSSInformation, sizeof(UBX_getGNSSInformation));
  sendMSG(UBX_getGNSSConfig, sizeof(UBX_getGNSSConfig));
  sendMSG(UBX_set_NAV_PVT, sizeof(UBX_set_NAV_PVT));
  delay(500);
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
    if (!sdFile) {
      Serial.println(F("File open error!"));
    }
    else {
      Serial.print(F("File opened succesfully: "));
      Serial.println(sdFile.name());
      for (byte i = 0; i < sizeof(NAV_PVT.names) / sizeof(NAV_PVT.names[0]); i++) {
        sdFile.print(NAV_PVT.names[i]);
        sdFile.print(",");
      }
      sdFile.println();
      sdFile.flush();
    }
  }
}

void sendMSG(byte msg[], int msgSize) {
  char rbBuffer[3];
  long replyLength = 8;
  replyFound = false;
  pos = 0;
  calcCheckSum(msg, msgSize);
  Serial.print(F("    Sending message:    "));
  for (byte i = 0; i < msgSize; i++) {
    gnss.write(msg[i]);
    sprintf(rbBuffer, "%02X", msg[i]);
    Serial.print(rbBuffer);
    Serial.print(F(" "));
  }
  Serial.println();
  Serial.print(F("    Fetching reply:     "));
  delay(500);
  while (gnss.available() > 0 and pos < replyLength) {
    byte rb = gnss.read();
    if (rb == 0xB5 or replyFound == true) {
      sprintf(rbBuffer, "%02X", rb);
      Serial.print(rbBuffer);
      Serial.print(F(" "));
      if (pos > 1 and pos < replyLength - 2) {
        CK_A +=  rb;
        CK_B +=  CK_A;
      }

      if (pos == 4) {
        for (byte i = 0; i < sizeof(rbBuffer) / sizeof(rbBuffer[0]); i++) {
          replyLength += hex2dec(rbBuffer[i]) * round(pow(16, 1 - i));
        }
      }
      if (pos == 5) {
        for (byte i = 0; i < sizeof(rbBuffer) / sizeof(rbBuffer[0]); i++) {
          replyLength += hex2dec(rbBuffer[i]) * round(pow(16, 1 - i) * 256);
        }
      }

      if (pos == replyLength - 2) {
        if (rb != CK_A) {
          checkSum = false;
        }
      }

      if (pos == replyLength - 1) {
        Serial.print(F("\t\t\t\t\t\t"));
        sprintf(rbBuffer, "%02X", CK_A);
        Serial.print(rbBuffer);
        Serial.print(F(" "));
        sprintf(rbBuffer, "%02X", CK_B);
        Serial.print(rbBuffer);
        if (rb == CK_B and checkSum) {
          Serial.print(F(" CheckSum OK!"));
        }
        else {
          Serial.print(F(" CheckSum error!"));
        }
      }

      pos++;
      replyFound = true;
    }
  }

  CK_A = 0;
  CK_B = 0;
  checkSum = true;
  Serial.println();
}

void incFileNum() {
  String s = "ubx" + String(++fileNum) + ".csv";
  s.toCharArray(fileName, 13);
}

void calcCheckSum(byte arr[], int arrSize) {
  CK_A = 0;
  CK_B = 0;
  for (byte i = 2; i < arrSize - 2; i++) {
    CK_A +=  arr[i];
    CK_B +=  CK_A;
  }
  arr[arrSize - 2] = CK_A;
  arr[arrSize - 1] = CK_B;
  CK_A = 0;
  CK_B = 0;
}

byte hex2dec(char value) {
  switch (value) {
    case '0':
      return 0;
      break;
    case '1':
      return 1;
      break;
    case '2':
      return 2;
      break;
    case '3':
      return 3;
      break;
    case '4':
      return 4;
      break;
    case '5':
      return 5;
      break;
    case '6':
      return 6;
      break;
    case '7':
      return 7;
      break;
    case '8':
      return 8;
      break;
    case '9':
      return 9;
      break;
    case 'A':
      return 10;
      break;
    case 'B':
      return 11;
      break;
    case 'C':
      return 12;
      break;
    case 'D':
      return 13;
      break;
    case 'E':
      return 14;
      break;
    case 'F':
      return 15;
      break;
  }
}
