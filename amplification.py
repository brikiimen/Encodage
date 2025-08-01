import soundcard as sc
import soundfile as sf
import numpy as np
import time


SAMPLE_RATE = 48000       
RECORD_SEC = 20            
AMPLIFICATION_FACTOR = 2.0  
OUTPUT_ORIGINAL = "original.wav"
OUTPUT_AMPLIFIED = "amplified.wav"


mic = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)

with mic.recorder(samplerate=SAMPLE_RATE) as recorder:
    print(" Enregistrement en cours...")
    start_time = time.time()
    data = recorder.record(numframes=int(SAMPLE_RATE * RECORD_SEC))
    end_time = time.time()
    print(" Enregistrement terminé.")


sf.write(OUTPUT_ORIGINAL, data[:, 0], SAMPLE_RATE)  


amp_start_time = time.time()
amplified = data[:, 0] * AMPLIFICATION_FACTOR
amplified = np.clip(amplified, -1.0, 1.0)  
sf.write(OUTPUT_AMPLIFIED, amplified, SAMPLE_RATE)
amp_end_time = time.time()


capture_latency = end_time - start_time
amplification_latency = amp_end_time - amp_start_time

print(f" Fichier original sauvegardé : {OUTPUT_ORIGINAL}")
print(f" Fichier amplifié sauvegardé : {OUTPUT_AMPLIFIED}")
print(f" Latence d'enregistrement : {capture_latency:.3f} secondes")
print(f" Latence d'amplification  : {amplification_latency:.3f} secondes")
