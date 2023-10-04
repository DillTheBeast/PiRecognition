void setup() {
  Serial.begin(9600);
  pinMode(13, OUTPUT);  // Pin 13 = LED
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '1') {
      digitalWrite(13, HIGH);   // Turn on LED if the known face is detected
    } else if (c == '0') {
      digitalWrite(13, LOW);    // Turn off LED otherwise
    }
  }
}