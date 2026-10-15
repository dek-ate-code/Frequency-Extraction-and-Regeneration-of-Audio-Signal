#include <math.h>
#define DAC_PIN 25
#define SAMPLE_RATE 8000
#define PI 3.14159265
#define FADE_TIME 20000   // smoother fade
// 3 sec, 2 sec durations
unsigned long durations[2] = {3000000, 2000000};
float freqs[10];
int freqCount = 0;
bool play = false;
unsigned long lastSample = 0;
unsigned long noteStart = 0;
int currentNote = 0;
int nextNote = 1;
float phaseA = 0;
float phaseB = 0;
void setup() {
  Serial.begin(115200);
  Serial.println("Enter frequencies separated by comma:");
}
void loop() {
  // ===== Serial input =====
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    parseFrequencies(input);
    play = true;
    currentNote = 0;
    nextNote = 1 % freqCount;
    phaseA = 0;
    phaseB = 0;
    noteStart = micros();
  }
  if (!play || freqCount == 0) return;
  unsigned long elapsed = micros() - noteStart;
  // ===== switch note (3 sec / 2 sec) =====
  if (elapsed >= durations[currentNote % 2]) {
    currentNote = nextNote;
    nextNote = (currentNote + 1) % freqCount;
    phaseA = phaseB;
    noteStart = micros();
    elapsed = 0;
  }
  // ===== audio engine =====
  if (micros() - lastSample >= 1000000 / SAMPLE_RATE) {
    lastSample = micros();
    phaseA += 2 * PI * freqs[currentNote] / SAMPLE_RATE;
    if (phaseA > 2 * PI) phaseA -= 2 * PI;
    phaseB += 2 * PI * freqs[nextNote] / SAMPLE_RATE;
    if (phaseB > 2 * PI) phaseB -= 2 * PI;
    float fade = 0;
    if (elapsed > durations[currentNote % 2] - FADE_TIME) {
      fade = (elapsed - (durations[currentNote % 2] - FADE_TIME)) / (float)FADE_TIME;
      if (fade > 1) fade = 1;
    }
    float sample =
      sin(phaseA) * (1 - fade) +
      sin(phaseB) * fade
    // reduce volume (clean sound)
    int dacValue = 127 + sample * 90;
    dacValue = constrain(dacValue, 0, 255);
    dacWrite(DAC_PIN, dacValue);
  }
}
void parseFrequencies(String input) {
  freqCount = 0;
  int start = 0;
  for (int i = 0; i <= input.length(); i++) {
    if (input[i] == ',' || i == input.length()) {
      String part = input.substring(start, i);
      freqs[freqCount++] = part.toFloat();
      start = i + 1;
    }
  }
  Serial.println("Loaded:");
  for (int i = 0; i < freqCount; i++) {
    Serial.println(freqs[i]);
  }
}
