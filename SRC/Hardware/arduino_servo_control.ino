#include <Servo.h>

Servo servo8; // Category 1: <40mm
Servo servo7; // Category 2: 40-50mm
Servo servo9; // Category 3: 50-60mm
Servo servo6; // Category 4: >60mm

// Home positions
const int HOME8 = 90;
const int HOME7 = 90;
const int HOME9 = 190;
const int HOME6 = 80;

// Active positions
const int ACTIVE8 = 170;
const int ACTIVE7 = 15;
const int ACTIVE9 = 280; // see note below
const int ACTIVE6 = 0;

void setup() {
  Serial.begin(9600);

  servo8.attach(8);
  servo7.attach(7);
  servo9.attach(9);
  servo6.attach(6);

  servo8.write(HOME8);
  servo7.write(HOME7);
  servo9.write(HOME9);
  servo6.write(HOME6);

  delay(2000);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    switch (command) {
      case '1':
        servo8.write(ACTIVE8);
        delay(1000);
        servo8.write(HOME8);
        break;
      case '2':
        servo7.write(ACTIVE7);
        delay(1000);
        servo7.write(HOME7);
        break;
      case '3':
        servo9.write(ACTIVE9);
        delay(1000);
        servo9.write(HOME9);
        break;
      case '4':
        servo6.write(ACTIVE6);
        delay(1000);
        servo6.write(HOME6);
        break;
    }
  }
}
