import os
import glob
import random
from pydub import AudioSegment

def get_noise_slice(noise_segments, duration_ms):
    if duration_ms <= 0:
        return AudioSegment.empty()
    if not noise_segments:
        return AudioSegment.silent(duration=duration_ms, frame_rate=16000).set_channels(1)
    
    # Pick a random noise segment and take a random slice
    noise = random.choice(noise_segments)
    if len(noise) < duration_ms:
        times = (duration_ms // len(noise)) + 1
        noise = noise * times
    
    max_start = len(noise) - duration_ms
    start = random.randint(0, max_start)
    return noise[start : start + duration_ms]

def process_audio(input_dir, output_dir, noise_dir="room_noise"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wav_files = glob.glob(os.path.join(input_dir, "*.wav"))
    target_duration = 1000  # 1 second in milliseconds

    if not wav_files:
        print(f"No .wav files found in {input_dir}")
        return

    # Load noise segments if noise_dir exists
    noise_segments = []
    if noise_dir and os.path.exists(noise_dir):
        noise_files = glob.glob(os.path.join(noise_dir, "*.wav"))
        for nf in noise_files:
            try:
                seg = AudioSegment.from_file(nf).set_frame_rate(16000).set_channels(1)
                if len(seg) > 0:
                    noise_segments.append(seg)
            except Exception as e:
                print(f"Warning: Could not load noise file {nf}: {e}")

    if noise_segments:
        print(f"Loaded {len(noise_segments)} background noise files for padding.")
    else:
        print("No background noise files found; falling back to silent padding.")

    for file_path in wav_files:
        filename = os.path.basename(file_path).replace(" ", "_").replace("(", "").replace(")", "")
        output_path = os.path.join(output_dir, filename)
        
        # Load audio and enforce 16kHz Mono
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        duration = len(audio)
        if duration > target_duration:
            # Trim from center if longer than 1 sec
            start = (duration - target_duration) // 2
            audio = audio[start : start + target_duration]
        elif duration < target_duration:
            # Pad equally at front and back with room noise to center the speech
            padding_needed = target_duration - duration
            front_pad = padding_needed // 2
            back_pad = padding_needed - front_pad
            
            # Create mono 16kHz room noise segments
            noise_front = get_noise_slice(noise_segments, front_pad)
            noise_back = get_noise_slice(noise_segments, back_pad)
            
            audio = noise_front + audio + noise_back

        # Ensure exact 16000 samples (32000 bytes for 16-bit PCM)
        raw_pcm = audio.raw_data
        target_bytes = 16000 * 2
        if len(raw_pcm) > target_bytes:
            raw_pcm = raw_pcm[:target_bytes]
        elif len(raw_pcm) < target_bytes:
            raw_pcm = raw_pcm + b'\x00' * (target_bytes - len(raw_pcm))
        audio = AudioSegment(data=raw_pcm, sample_width=2, frame_rate=16000, channels=1)
            
        # Export as 16-bit PCM WAV
        audio.export(output_path, format="wav", parameters=["-acodec", "pcm_s16le"])
        print(f"Processed: {filename} -> Final Duration: {len(audio)} ms")

    print("Done processing all files!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess and normalize audio files to 16kHz Mono 1-second WAVs for Multilingual KWS."
    )
    parser.add_argument(
        "--input", "-i",
        default="hey_saya",
        help="Input directory containing raw .wav files (default: 'hey_saya')",
    )
    parser.add_argument(
        "--output", "-o",
        default="hey_saya_processed",
        help="Output directory for processed 16kHz 1-second .wav files (default: 'hey_saya_processed')",
    )
    parser.add_argument(
        "--noise", "-n",
        default="room_noise",
        help="Directory with background/room noise .wav files for padding (default: 'room_noise')",
    )
    args = parser.parse_args()

    process_audio(args.input, args.output, noise_dir=args.noise)
