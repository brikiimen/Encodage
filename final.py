import soundcard as sc
import soundfile as sf
import torch
import torchaudio
import time
import numpy as np
import base64  


OUTPUT_ORIGINAL = "original_full.wav"
OUTPUT_DECODED = "decoded_full.wav"
OUTPUT_ULAW_RAW = "output_ulaw.pcm"
OUTPUT_ULAW_BASE64 = "output_ulaw_base64.txt"

SAMPLE_RATE = 48000       
TARGET_SAMPLE_RATE = 8000 
RECORD_SECONDS = 20
CHUNK_DURATION = 0.1      
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
NUM_CHUNKS = int(RECORD_SECONDS / CHUNK_DURATION)


mic = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
print(" Début de l'enregistrement (loopback)...")

original_chunks = []
decoded_chunks = []
latencies = []


with mic.recorder(samplerate=SAMPLE_RATE) as recorder, open(OUTPUT_ULAW_RAW, 'wb') as ulaw_file:
    for i in range(NUM_CHUNKS):

        chunk = recorder.record(numframes=CHUNK_SIZE)
        chunk = torch.tensor(chunk[:, 0]).unsqueeze(0)  # Mono

        if SAMPLE_RATE != TARGET_SAMPLE_RATE:
            chunk = torchaudio.functional.resample(chunk, SAMPLE_RATE, TARGET_SAMPLE_RATE)


        original_chunks.append(chunk)

        start_time = time.time()
        encoded = torchaudio.functional.mu_law_encoding(chunk, quantization_channels=256)
        decoded = torchaudio.functional.mu_law_decoding(encoded, quantization_channels=256)
        end_time = time.time()

        latency = end_time - start_time
        latencies.append(latency)
        decoded_chunks.append(decoded)

        encoded_bytes = encoded.squeeze().numpy().astype(np.uint8).tobytes()
        ulaw_file.write(encoded_bytes)

        print(f" Chunk {i+1}/{NUM_CHUNKS} traité | Latence : {latency:.4f}s")

print(" Fin de l'enregistrement.")

original_audio = torch.cat(original_chunks, dim=1)
decoded_audio = torch.cat(decoded_chunks, dim=1)

torchaudio.save(OUTPUT_ORIGINAL, original_audio, TARGET_SAMPLE_RATE)
torchaudio.save(OUTPUT_DECODED, decoded_audio, TARGET_SAMPLE_RATE)

total_latency = sum(latencies)
avg_latency = total_latency / len(latencies)
print(f"\n Latence totale : {total_latency:.4f}s")
print(f" Latence moyenne par chunk : {avg_latency:.4f}s")
print(f" Fichiers générés : {OUTPUT_ORIGINAL}, {OUTPUT_DECODED}, {OUTPUT_ULAW_RAW}")

with open(OUTPUT_ULAW_RAW, 'rb') as f:
    ulaw_data = f.read()
    ulaw_base64 = base64.b64encode(ulaw_data).decode('utf-8')

with open(OUTPUT_ULAW_BASE64, 'w') as f:
    f.write(ulaw_base64)

print(f"Fichier µ-law encodé en base64 généré : {OUTPUT_ULAW_BASE64}")
