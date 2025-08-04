import soundcard as sc
import soundfile as sf
import numpy as np
import threading
import time
import queue
from datetime import datetime


SAMPLE_RATE = 48000
FRAME_SIZE = 1024
CHANNELS = 1
DURATION_SECONDS = 50
RUNNING = True

OUTPUT_ORIGINAL = "original.wav"
OUTPUT_AMPLIFIED = "amplified.wav"


def compute_rms(signal):
    return np.sqrt(np.mean(signal ** 2))

def get_amplification_factor(rms, min_rms=0.01, max_rms=0.1):
    if rms < min_rms:
        return 7.0
    elif rms > max_rms:
        return 1.0
    else:
        return 5.0 - 3.0 * ((rms - min_rms) / (max_rms - min_rms))


mic = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)


audio_queue = queue.Queue()
latencies = []
original_chunks = []
amplified_chunks = []


def record_and_process():
    global RUNNING
    with mic.recorder(samplerate=SAMPLE_RATE, channels=CHANNELS) as recorder:
        print(" Streaming audio en temps réel avec amplification dynamique...")
        while RUNNING:
            start_time = time.time()

            
            data = recorder.record(numframes=FRAME_SIZE)
            mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]

            
            rms = compute_rms(mono)
            amplification = get_amplification_factor(rms)
            amplified = np.clip(mono * amplification, -1.0, 1.0)

            
            original_chunks.append(mono.copy())
            amplified_chunks.append(amplified.copy())

            
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            print(f" Chunk traité | RMS={rms:.4f} | Amplification={amplification:.2f} | Latence={latency:.2f} ms")

            time.sleep(0.001)


t1 = threading.Thread(target=record_and_process)
start_time = time.time()
t1.start()


try:
    while time.time() - start_time < DURATION_SECONDS:
        time.sleep(0.1)
    print("\n Durée atteinte, arrêt du streaming.")
except KeyboardInterrupt:
    print("\n Arrêt manuel détecté.")

RUNNING = False
t1.join()


original = np.concatenate(original_chunks)
amplified = np.concatenate(amplified_chunks)

sf.write(OUTPUT_ORIGINAL, original, SAMPLE_RATE)
sf.write(OUTPUT_AMPLIFIED, amplified, SAMPLE_RATE)

print(f"\ Fichier original sauvegardé : {OUTPUT_ORIGINAL}")
print(f" Fichier amplifié sauvegardé : {OUTPUT_AMPLIFIED}")
print(f" Latence moyenne : {np.mean(latencies):.2f} ms sur {len(latencies)} chunks")
