/*
 * NAWIS AI — ESP32 LED Status Controller
 * Controls 5 LEDs + buzzer based on serial commands from Python backend
 * 
 * Commands (single character via Serial at 115200 baud):
 *   I = Idle        → Green LED steady
 *   L = Listening   → Blue/White LED fast blink
 *   T = Thinking    → Yellow LED blink
 *   S = Speaking    → Green LED + short beep
 *   E = Escalate    → Red LED blink + urgent beep
 */

#define PIN_GREEN  12   // Idle / Speaking
#define PIN_BLUE   13   // Listening
#define PIN_YELLOW 14   // Thinking
#define PIN_WHITE  15   // Listening (secondary)
#define PIN_RED    16   // Escalate / Error
#define PIN_BUZZER 26   // Buzzer

char currentState = 'I';
unsigned long lastBlink = 0;
bool blinkOn = false;

void setup() {
  Serial.begin(115200);
  pinMode(PIN_GREEN,  OUTPUT);
  pinMode(PIN_BLUE,   OUTPUT);
  pinMode(PIN_YELLOW, OUTPUT);
  pinMode(PIN_WHITE,  OUTPUT);
  pinMode(PIN_RED,    OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  setIdle();
  Serial.println("NAWIS AI ESP32 Ready");
}

void clearAll() {
  digitalWrite(PIN_GREEN,  LOW);
  digitalWrite(PIN_BLUE,   LOW);
  digitalWrite(PIN_YELLOW, LOW);
  digitalWrite(PIN_WHITE,  LOW);
  digitalWrite(PIN_RED,    LOW);
  noTone(PIN_BUZZER);
}

void setIdle() {
  clearAll();
  digitalWrite(PIN_GREEN, HIGH);
}

void setSpeaking() {
  clearAll();
  digitalWrite(PIN_GREEN, HIGH);
  tone(PIN_BUZZER, 880, 80);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == '\n' || cmd == '\r') return;
    currentState = cmd;
    clearAll();

    if (cmd == 'I') {
      setIdle();
    } else if (cmd == 'S') {
      setSpeaking();
    }
    Serial.print("State: ");
    Serial.println(cmd);
  }

  unsigned long now = millis();

  if (currentState == 'L') {
    if (now - lastBlink > 150) {
      lastBlink = now;
      blinkOn = !blinkOn;
      digitalWrite(PIN_BLUE,  blinkOn ? HIGH : LOW);
      digitalWrite(PIN_WHITE, blinkOn ? HIGH : LOW);
    }
  } else if (currentState == 'T') {
    if (now - lastBlink > 400) {
      lastBlink = now;
      blinkOn = !blinkOn;
      digitalWrite(PIN_YELLOW, blinkOn ? HIGH : LOW);
    }
  } else if (currentState == 'E') {
    if (now - lastBlink > 300) {
      lastBlink = now;
      blinkOn = !blinkOn;
      digitalWrite(PIN_RED, blinkOn ? HIGH : LOW);
      if (blinkOn) tone(PIN_BUZZER, 440, 200);
    }
  }
}
