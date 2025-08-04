import pyaudio
import numpy as np
import soundfile as sf
import threading
import time
import queue
from datetime import datetime

# === Paramètres ===
SAMPLE_RATE = 48000
FRAME_SIZE = 1024
CHANNELS = 1
DURATION_SECONDS = 30
RUNNING = True

OUTPUT_ORIGINAL = "original.wav"
OUTPUT_AMPLIFIED = "amplified.wav"

# === Fonctions d’analyse/amplification ===
def compute_rms(signal):
    return np.sqrt(np.mean(signal ** 2))

def get_amplification_factor(rms, min_rms=0.01, max_rms=0.1):
    if rms < min_rms:
        return 4.0
    elif rms > max_rms:
        return 1.0
    else:
        return 5.0 - 3.0 * ((rms - min_rms) / (max_rms - min_rms))

# === Initialisation PyAudio ===
pa = pyaudio.PyAudio()

# === Files de traitement ===
audio_queue = queue.Queue()
latencies = []
original_chunks = []
amplified_chunks = []

# === Thread de capture ===
def record_and_process():
    global RUNNING
    print(" Streaming audio en temps réel avec amplification dynamique...")

    stream = pa.open(format=pyaudio.paFloat32,
                     channels=CHANNELS,
                     rate=SAMPLE_RATE,
                     input=True,
                     frames_per_buffer=FRAME_SIZE)

    while RUNNING:
        start_time = time.time()

        # Lecture d'un chunk
        data = stream.read(FRAME_SIZE, exception_on_overflow=False)
        audio_np = np.frombuffer(data, dtype=np.float32)

        # RMS & amplification
        rms = compute_rms(audio_np)
        amplification = get_amplification_factor(rms)
        amplified = np.clip(audio_np * amplification, -1.0, 1.0)

        # Stockage
        original_chunks.append(audio_np.copy())
        amplified_chunks.append(amplified.copy())

        # Statistiques
        latency = (time.time() - start_time) * 1000
        latencies.append(latency)
        print(f" Chunk traité | RMS={rms:.4f} | Amplification={amplification:.2f} | Latence={latency:.2f} ms")

        time.sleep(0.001)

    stream.stop_stream()
    stream.close()

# === Lancement ===
t1 = threading.Thread(target=record_and_process)
start_time = time.time()
t1.start()

# Durée d'enregistrement
try:
    while time.time() - start_time < DURATION_SECONDS:
        time.sleep(0.1)
    print("\n Durée atteinte, arrêt du streaming.")
except KeyboardInterrupt:
    print("\n Arrêt manuel détecté.")

RUNNING = False
t1.join()
pa.terminate()

# === Sauvegarde des fichiers ===
original = np.concatenate(original_chunks)
amplified = np.concatenate(amplified_chunks)

sf.write(OUTPUT_ORIGINAL, original, SAMPLE_RATE)
sf.write(OUTPUT_AMPLIFIED, amplified, SAMPLE_RATE)

print(f"\n Fichier original sauvegardé : {OUTPUT_ORIGINAL}")
print(f" Fichier amplifié sauvegardé : {OUTPUT_AMPLIFIED}")
print(f" Latence moyenne : {np.mean(latencies):.2f} ms sur {len(latencies)} chunks")