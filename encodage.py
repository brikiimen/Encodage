import soundcard as sc
import numpy as np
import threading
import time
import queue
import audioop
import base64
from datetime import datetime

SAMPLE_RATE = 8000     
FRAME_SIZE = 1024      
CHANNELS = 1
DURATION_SECONDS = 60  
RUNNING = True         


mic = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)


latencies = []           
encode_latencies = []   
timestamps_queue = queue.Queue()

audio_queue = queue.Queue()

ULAW_OUTPUT = "solution.ulaw"
B64_OUTPUT = "solution_base64.txt"
open(ULAW_OUTPUT, "wb").close()
open(B64_OUTPUT, "w").close()


def record_stream():
    global RUNNING
    with mic.recorder(samplerate=SAMPLE_RATE, channels=CHANNELS) as recorder:
        print(" Enregistrement temps réel (µ-law + Base64)...")
        while RUNNING:
            start = time.time()

            data = recorder.record(numframes=FRAME_SIZE)
            mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
            pcm_bytes = mono.astype(np.float32).tobytes()

            latency = (time.time() - start) * 1000
            latencies.append(latency)
            print(f" Chunk capturé en {latency:.2f} ms")

            timestamps_queue.put(time.time())
            audio_queue.put(pcm_bytes)


def encode_mulaw():
    while RUNNING or not audio_queue.empty():
        try:
            pcm_chunk = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        encode_start = time.time()

        int16_array = np.frombuffer(pcm_chunk, dtype=np.float32)
        int16_array = np.clip(int16_array, -1, 1)
        int16_array = (int16_array * 32767).astype(np.int16)
        mu_law_bytes = audioop.lin2ulaw(int16_array.tobytes(), 2)

        
        with open(ULAW_OUTPUT, "ab") as f:
            f.write(mu_law_bytes)

        b64_str = base64.b64encode(mu_law_bytes).decode('utf-8')
        timestamp = datetime.now().isoformat(timespec='milliseconds')

        with open(B64_OUTPUT, "a") as f:
            f.write(f"[{timestamp}] {b64_str}\n")

        
        if not timestamps_queue.empty():
            sent_time = timestamps_queue.get()
            encode_latency = (time.time() - sent_time) * 1000
            encode_latencies.append(encode_latency)
            print(f"µ-law ({len(mu_law_bytes)} octets) encodé en Base64 | Latence : {encode_latency:.2f} ms")


t1 = threading.Thread(target=record_stream)
t2 = threading.Thread(target=encode_mulaw)
start_time = time.time()
t1.start()
t2.start()


try:
    while time.time() - start_time < DURATION_SECONDS:
        time.sleep(0.1)
    print("\nTemps écoulé. Arrêt automatique.")
except KeyboardInterrupt:
    print(" Arrêt manuel détecté.")


RUNNING = False
t1.join()
t2.join()


total_time = time.time() - start_time
print(f"\nEnregistrement terminé. Durée totale : {total_time:.2f} sec")

if latencies:
    avg_latency = sum(latencies) / len(latencies)
    print(f"Latence moyenne audio : {avg_latency:.2f} ms")

if encode_latencies:
    avg_encode = sum(encode_latencies) / len(encode_latencies)
    print(f"Latence moyenne µ-law+Base64 : {avg_encode:.2f} ms")
