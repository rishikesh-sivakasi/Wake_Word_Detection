import time
import numpy as np
import pyaudio
from openwakeword.model import Model

MODEL_PATH = "/home/arz-1029/openWakeWord/my_custom_model_saya/hey_saya.onnx"
CHUNK = 1280
THRESHOLD = 0.65
COOLDOWN = 1.5

oww = Model(wakeword_models=[MODEL_PATH],
            enable_speex_noise_suppression=True,
            inference_framework="onnx")
key = list(oww.models.keys())[0]

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=CHUNK)

last_fire = 0
print(f"Universal detector active for '{key}'. Anyone can speak...")

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frame = np.frombuffer(data, dtype=np.int16)

        pred = oww.predict(frame)
        score = pred.get(key, 0.0)

        now = time.time()
        if (now - last_fire) > COOLDOWN:
            if score >= THRESHOLD:
                print(f"\n🔥 [TRIGGERED] 'hey saya' detected! (Score: {score:.3f})")
                last_fire = now
                oww.reset()
            elif score > 0.15:
                bars = int(score * 25)
                print(f"\rConfidence: {score:.2f} | {'█' * bars:<25}", end="")

except KeyboardInterrupt:
    print("\nExiting...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()