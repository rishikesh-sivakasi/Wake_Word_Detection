# import os
# import glob
# import argparse
# import librosa
# import soundfile as sf
# import numpy as np

# def process_audio(input_path, output_path, target_sr=16000, target_duration=2.0):
#     """
#     Loads an audio file, converts it to mono, resamples to target_sr,
#     and crops or pads it to exactly target_duration (2 seconds).
#     - If shorter: pads with silence equally on both sides.
#     - If longer: finds the active speech part and centers the 2-second crop around it.
#     """
#     # Load audio: librosa automatically converts to mono and resamples if sr is provided
#     audio, _ = librosa.load(input_path, sr=target_sr, mono=True)
    
#     target_length = int(target_sr * target_duration)
    
#     if len(audio) == target_length:
#         final_audio = audio
#     elif len(audio) < target_length:
#         # Pad front and back equally (center pad)
#         pad_total = target_length - len(audio)
#         pad_front = pad_total // 2
#         pad_back = pad_total - pad_front
#         final_audio = np.pad(audio, (pad_front, pad_back), mode='constant')
#     else:
#         # Audio is longer than target.
#         # Find the non-silent region to center the spoken word.
#         # top_db=30 is a reasonable threshold to distinguish speech from background
#         yt, index = librosa.effects.trim(audio, top_db=30)
        
#         # Center of the spoken word
#         center = (index[0] + index[1]) // 2
        
#         start = center - (target_length // 2)
#         end = start + target_length
        
#         # Ensure we don't go out of bounds
#         if start < 0:
#             start = 0
#             end = target_length
#         elif end > len(audio):
#             end = len(audio)
#             start = end - target_length
            
#         final_audio = audio[start:end]
        
#     # Save the processed audio
#     sf.write(output_path, final_audio, target_sr, subtype='PCM_16')

# def main():
#     parser = argparse.ArgumentParser(description="Format audio files to 16kHz, mono, 2-sec centered clips")
#     parser.add_argument("--input_dir", required=True, help="Directory containing input .wav files")
#     parser.add_argument("--output_dir", required=True, help="Directory to save formatted .wav files")

#     parser.add_argument("/home/arz-1029/pt_task/python_training_&_feature_extraction/dataset/pre_clean")
#     parser.add_argument("/home/arz-1029/pt_task/python_training_&_feature_extraction/dataset/hey_saya_samples")
    
#     args = parser.parse_args()
    
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     wav_files = glob.glob(os.path.join(args.input_dir, "*.wav"))
#     if not wav_files:
#         print(f"No .wav files found in {args.input_dir}")
#         return
        
#     print(f"Found {len(wav_files)} files. Formatting...")
    
#     for i, wav_path in enumerate(wav_files):
#         filename = os.path.basename(wav_path)
#         out_path = os.path.join(args.output_dir, filename)
        
#         try:
#             process_audio(wav_path, out_path)
#             print(f"[{i+1}/{len(wav_files)}] Processed: {filename}")
#         except Exception as e:
#             print(f"[{i+1}/{len(wav_files)}] Error processing {filename}: {e}")
            
#     print("All files processed successfully!")

# if __name__ == "__main__":
#     main()
import os
from pydub import AudioSegment

def process_audio_with_background(
    speech_path, 
    bg_noise_path, 
    output_path, 
    target_duration_ms=2000, 
    target_sample_rate=16000,
    bg_gain_db=-3.0  # Adjust volume of background noise relative to speech (e.g., -3dB to reduce it if too loud)
):
    """
    Pads speech audio to target_duration_ms, then overlays a continuous 
    background noise WAV file across the entire duration.
    """
    # 1. Load speech audio and normalize format (16kHz mono)
    speech = AudioSegment.from_file(speech_path).set_channels(1).set_frame_rate(target_sample_rate)
    
    # 2. Pad or trim speech to 2 seconds
    speech_len = len(speech)
    if speech_len > target_duration_ms:
        speech = speech[:target_duration_ms]
    elif speech_len < target_duration_ms:
        padding_needed = target_duration_ms - speech_len
        front_pad = padding_needed // 2
        back_pad = padding_needed - front_pad
        
        silence_front = AudioSegment.silent(duration=front_pad, frame_rate=target_sample_rate)
        silence_back = AudioSegment.silent(duration=back_pad, frame_rate=target_sample_rate)
        speech = silence_front + speech + silence_back

    # 3. Load background noise audio and format it
    bg_noise = AudioSegment.from_file(bg_noise_path).set_channels(1).set_frame_rate(target_sample_rate)
    
    # Adjust background noise volume if needed
    bg_noise = bg_noise + bg_gain_db
    
    # Make sure background noise is long enough for the 2-second clip
    if len(bg_noise) < target_duration_ms:
        # Loop background noise if it's shorter than 2 seconds
        loops = (target_duration_ms // len(bg_noise)) + 1
        bg_noise = (bg_noise * loops)[:target_duration_ms]
    else:
        # Trim background noise to exactly 2 seconds
        bg_noise = bg_noise[:target_duration_ms]
        
    # 4. Overlay the speech onto the continuous background noise track
    final_audio = bg_noise.overlay(speech)
    
    # Ensure final clip is exactly target_duration_ms
    final_audio = final_audio[:target_duration_ms]
    
    # 5. Export
    final_audio.export(output_path, format="wav")
    print(f"Processed: {os.path.basename(speech_path)} -> Duration: {len(final_audio)}ms")

def batch_process_folder(input_dir, bg_noise_path, output_dir, bg_gain_db=-3.0):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".wav"):
            in_file = os.path.join(input_dir, filename)
            out_file = os.path.join(output_dir, filename)
            process_audio_with_background(in_file, bg_noise_path, out_file, bg_gain_db=bg_gain_db)

if __name__ == "__main__":

    INPUT_FOLDER = "/home/arz-1029/pt_task/python_training_&_feature_extraction/dataset/pre_clean"
    OUTPUT_FOLDER = "/home/arz-1029/pt_task/python_training_&_feature_extraction/dataset/hey_saya_samples"
    BG_NOISE_FILE = "/home/arz-1029/pt_task/python_training_&_feature_extraction/preprocessing_scripts/pi_noise_11.wav"
    
    batch_process_folder(INPUT_FOLDER, BG_NOISE_FILE, OUTPUT_FOLDER, bg_gain_db=-3.0)
