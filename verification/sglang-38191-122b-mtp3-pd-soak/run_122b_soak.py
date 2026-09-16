import json, subprocess, threading, time, random
MODEL = "<pfs>/models/Qwen3.5-122B-A10B"
URL = "http://127.0.0.1:30003/v1/chat/completions"
PROMPTS = [
    "Explain the greenhouse effect and its impact on global warming.",
    "Write a short essay about the importance of biodiversity.",
    "Describe how a modern smartphone works, layer by layer.",
    "Explain the difference between supervised and unsupervised learning.",
    "Summarize the causes and consequences of World War II.",
    "Write a creative story about a time traveler who visits ancient Rome.",
    "Explain how quantum computing differs from classical computing.",
    "Describe the water cycle and why it matters for agriculture.",
    "Explain the principles of supply and demand in economics.",
    "Write a detailed explanation of how vaccines work.",
]
RESULTS = {"ok": 0, "fail": 0, "latencies": [], "errors": []}
LOCK = threading.Lock()
STOP = threading.Event()


def worker():
    while not STOP.is_set():
        p = random.choice(PROMPTS)
        body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": p}], "max_tokens": 120, "temperature": 0.7})
        t0 = time.time()
        try:
            r = subprocess.run(["curl", "-s", URL, "-H", "Content-Type: application/json", "-d", body, "--max-time", "180"],
                                capture_output=True, text=True, timeout=200)
            ok = r.returncode == 0 and "object" in r.stdout
            dt = time.time() - t0
            with LOCK:
                if ok:
                    RESULTS["ok"] += 1
                    RESULTS["latencies"].append(dt)
                else:
                    RESULTS["fail"] += 1
                    RESULTS["errors"].append(r.stdout[:80])
        except Exception as e:
            with LOCK:
                RESULTS["fail"] += 1
                RESULTS["errors"].append(repr(e)[:80])
        time.sleep(random.uniform(0.05, 0.3))


def phase(concurrency, minutes):
    STOP.clear()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    end = time.time() + minutes * 60
    while time.time() < end:
        time.sleep(30)
        with LOCK:
            lat = sum(RESULTS["latencies"]) / max(1, len(RESULTS["latencies"]))
            print("  [c=%d] ok=%d fail=%d avg_lat=%.1fs" % (concurrency, RESULTS["ok"], RESULTS["fail"], lat), flush=True)
    STOP.set()
    for t in threads:
        t.join()


print("=== 122B MTP3 PD SOAK (FULL, 1h) START ===", flush=True)
for c, mins in [(4, 20), (16, 20), (32, 20)]:
    print("--- phase c=%d %dmin ---" % (c, mins), flush=True)
    phase(c, mins)
print("=== SOAK DONE ===", flush=True)
summary = {"ok": RESULTS["ok"], "fail": RESULTS["fail"], "errors_sample": RESULTS["errors"][:5]}
json.dump(summary, open("/root/soak_summary.json", "w"))
print("SUMMARY:", json.dumps(summary), flush=True)
