#ifndef MFCC_H
#define MFCC_H

#include <stdint.h>

// --- MFCC Configuration ---
// Parameters MUST match generate_tables.py exactly.
//   Sample Rate : 16000 Hz
//   N_FFT       : 512  (32ms window)
//   HOP_LENGTH  : 160  (10ms hop)
//   N_MELS      : 40   (Mel bands)
//   N_MFCC      : 40   (output features)

// --- INITIALIZATION ---
// Call this ONCE when the device boots up.
// Allocates memory for the KissFFT engine.
void mfcc_init();

// --- CLEANUP ---
// Call this on shutdown to free the KissFFT memory.
void mfcc_free();

// --- EXECUTION ---
// Call this once per audio frame (every HOP_LENGTH = 160 samples = 10ms).
//
// Inputs:
//   audio_data : Pointer to N_FFT (512) float samples (one windowed frame).
//   mfcc_out   : Pointer to N_MFCC (40) floats where the result is written.
//
// Note: The caller is responsible for feeding overlapping frames.
//       At 16kHz with hop=160, call this ~100 times per second.
void mfcc_compute(const float* audio_data, float* mfcc_out);

#endif // MFCC_H