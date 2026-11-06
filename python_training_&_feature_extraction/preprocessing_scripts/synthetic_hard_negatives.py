import os
import random
import librosa
import numpy as np
import soundfile as sf
import asyncio
import edge_tts
from audiomentations import Compose, PitchShift, TimeStretch

# ==============================    
# CONFIGURATION
# ==============================
HARD_NEGATIVES = [
    "Hey Maya", "Hey Jaya", "Hey Chaya", "Hey Siya", "Hey Arya",
    "Say Saya", "Play Saya", "Pay Saya", "They saw ya",
    "Hey Soya", "Hey Surya", "Hey Sarah", "High Saya",
    "He saw ya", "Taste Papaya", "Say Hiya", "Hey wire",
    "Hey", "Saya", "Hey Sa", "Ay Saya", "Heeeeey"
]

# UPDATED: Pointed directly to your training negative folder
OUTPUT_FOLDER = "../dataset/train/negative"
NOISE_FILE = "freesound_community-029974_inside-the-robot-70923.wav"

# 🟢 ADDED BOTH THE MALE AND FEMALE NEURAL VOICES
VOICE_MODELS = ["en-IN-NeerjaNeural", "en-IN-PrabhatNeural"] 

TARGET_SR = 16000
TARGET_DURATION = 2.0
TARGET_LENGTH = int(TARGET_SR * TARGET_DURATION)
TOTAL_SAMPLES_NEEDED = 400 

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

voice_augmenter = Compose([
    PitchShift(min_semitones=-2.0, max_semitones=2.0, p=0.8),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.8),
])

async def generate_base_audio(text, voice, output_filename):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_filename)

async def main():
    print("🚀 Initializing Dual-Voice Neural Hard Negative Pipeline...\n")
    
    try:
        hw_noise, _ = librosa.load(NOISE_FILE, sr=TARGET_SR)
        max_n = np.max(np.abs(hw_noise))
        if max_n > 0: hw_noise = hw_noise / max_n
    except FileNotFoundError:
        print(f"[ERROR] Could not find '{NOISE_FILE}'. Please ensure it is in the same directory.")
        return

    # 1. GENERATE BASE AUDIO FOR BOTH GENDERS
    base_audio_pool = []
    print("🎙️ Synthesizing Male and Female Indian-English voices...")
    
    for phrase in HARD_NEGATIVES:
        for voice in VOICE_MODELS:
            temp_file = "temp_neural.mp3"
            
            # Fetch the specific phrase in the specific voice
            await generate_base_audio(phrase, voice, temp_file)
            
            audio, _ = librosa.load(temp_file, sr=TARGET_SR)
            base_audio_pool.append(audio)
            os.remove(temp_file)
            
            gender = "Female" if "Neerja" in voice else "Male"
            print(f"   -> Downloaded {gender} voice for: '{phrase}'")

    # 2. MULTIPLY AND AUGMENT
    print(f"\n🧬 Warping and mixing {TOTAL_SAMPLES_NEEDED} Pi-Safe Hard Negatives...")
    
    for i in range(TOTAL_SAMPLES_NEEDED):
        # Now it randomly grabs from a pool of both male and female base recordings!
        base_audio = random.choice(base_audio_pool)
        
        v_aug = voice_augmenter(samples=base_audio, sample_rate=TARGET_SR)
        
        if len(v_aug) < TARGET_LENGTH:
            pad = TARGET_LENGTH - len(v_aug)
            v_aug = np.pad(v_aug, (pad//2, pad - pad//2), mode='constant')
        else:
            v_aug = v_aug[:TARGET_LENGTH]
            
        if len(hw_noise) > TARGET_LENGTH:
            start = random.randint(0, len(hw_noise) - TARGET_LENGTH)
            n_slice = hw_noise[start : start + TARGET_LENGTH]
        else:
            n_slice = np.pad(hw_noise, (0, TARGET_LENGTH - len(hw_noise)), mode='wrap')

        snr = random.uniform(0.1, 0.4)
        mixed = v_aug + (n_slice * snr)
        mixed = np.clip(mixed, -1.0, 1.0)
        
        out_path = os.path.join(OUTPUT_FOLDER, f"synth_hard_neg_{i:03d}.wav")
        sf.write(out_path, mixed, TARGET_SR, subtype='PCM_16')

    print(f"\n✅ SUCCESS! {TOTAL_SAMPLES_NEEDED} Dual-Voice Hard Negatives injected into '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    asyncio.run(main())