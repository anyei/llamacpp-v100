#!/usr/bin/env python3
"""Traffic driver for LLAMA_EXPERT_PROFILE collection (TASKS #148 addendum 2).
Usage: profile-driver.py <base_url> <bucket A|B> <out.jsonl>
Sends the bucket's prompts sequentially via /v1/chat/completions at temp 0.8,
max_tokens 320, records completion tokens + the full text for the coherence read.
"""
import json
import sys
import time
import urllib.request

BASE, BUCKET, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

PRINTING = ("The printing press was introduced to Europe by Johannes Gutenberg around 1440 "
            "in Mainz. Adapting the screw press used for wine and olives, he combined it with "
            "movable metal type cast from a lead alloy, oil-based ink and a hand mould that let "
            "a worker cast type quickly and uniformly. His 42-line Bible, finished around 1455, "
            "demonstrated that printed books could rival manuscripts in quality. Within fifty "
            "years presses operated in more than two hundred European cities and had produced "
            "an estimated twenty million volumes. The falling cost of books widened literacy, "
            "standardised vernacular languages, and gave reformers and scientists a way to "
            "spread arguments faster than authorities could suppress them. Historians often "
            "date the start of the modern information age to this shift.")

CHANGELOG = ("v2.4.0: added streaming responses for the chat endpoint; the default request "
             "timeout rose from 30s to 120s; the /metrics endpoint now exposes per-model "
             "latency histograms; fixed a crash when a client disconnects mid-stream; "
             "deprecated the legacy /v0/complete route (removal planned for v3.0); Python 3.8 "
             "support dropped; new config key max_concurrent_requests defaults to 4; docker "
             "images are now built for arm64 as well as amd64.")

NOTES = ("Meeting 2026-09-04, project Falcon. Maria reported the ingestion pipeline is 2 days "
         "late because the vendor API changed; she will send the vendor a bug report by Friday. "
         "Tom will rewrite the parser to tolerate the new field names, target next Wednesday. "
         "Priya asked for a load test before launch; Tom agreed to hand her a staging build on "
         "Thursday. Budget review moved to the 18th; Maria owns the slide deck. Everyone should "
         "update their tickets before Monday standup.")

BUCKETS = {
    # code + factual chat, thinking OFF (real code/prose tokens, not planning)
    "A": (False, [
        "Write a Python function that parses an ISO-8601 date string and returns a Unix timestamp. Include error handling and a couple of usage examples.",
        "Implement a thread-safe LRU cache in C++17 with get/put and a maximum size. Show the full class.",
        "Write a bash script that finds the 10 largest files under a directory given as the first argument and prints them with human-readable sizes.",
        "Write a SQL query that returns the top 5 customers by total order value in the last 90 days, including each customer's order count. Assume tables customers(id, name) and orders(id, customer_id, total, created_at).",
        "Write a React function component that fetches a list of users from /api/users and renders them in a paginated table, 20 rows per page.",
        "Explain the difference between TCP and UDP and when you would choose each one.",
        "What causes the seasons on Earth? Explain it for a curious 12-year-old.",
        "Write a short professional email declining a meeting invitation and proposing two alternative times next week.",
        "Give me a 5-day itinerary for a first trip to Kyoto in autumn, one paragraph per day.",
        "Explain how a hash map handles collisions, with a concrete example.",
    ]),
    # reasoning / math + summarization, thinking ON
    "B": (True, [
        "A train leaves city A at 9:00 traveling at 80 km/h toward city B. Another train leaves city B, 300 km away, at 10:00 toward A at 100 km/h. When and where do they meet?",
        "If 3 workers can build 2 walls in 4 days, how many days do 5 workers need to build 5 walls? Show your reasoning.",
        "Prove that the square root of 2 is irrational.",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain the common mistake people make.",
        "Summarize the following text in three sentences:\n\n" + PRINTING,
        "Summarize the key points of this changelog for a release announcement, grouped by user impact:\n\n" + CHANGELOG,
        "Find all real x such that x^4 - 5x^2 + 4 = 0.",
        "A farmer has chickens and cows. There are 30 heads and 74 legs in total. How many chickens and how many cows are there?",
        "You have a 3-liter jug and a 5-liter jug and unlimited water. Measure exactly 4 liters. Explain step by step.",
        "From the following meeting notes, list every action item with its owner and due date:\n\n" + NOTES,
    ]),
}

think, prompts = BUCKETS[BUCKET]
total = 0
t0 = time.time()
with open(OUT, "a") as out:
    for i, p in enumerate(prompts):
        body = {
            "model": "flash-next-profile",
            "messages": [{"role": "user", "content": p}],
            "temperature": 0.8,
            "max_tokens": 320,
            "chat_template_kwargs": {"enable_thinking": think},
        }
        req = urllib.request.Request(BASE + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t1 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                j = json.load(r)
        except Exception as e:  # noqa: BLE001
            print(f"[{BUCKET}{i}] ERROR {e}", flush=True)
            continue
        dt = time.time() - t1
        msg = j["choices"][0]["message"]
        text = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        ct = j.get("usage", {}).get("completion_tokens", 0)
        total += ct
        rec = {"bucket": BUCKET, "i": i, "prompt": p[:80], "completion_tokens": ct,
               "seconds": round(dt, 1), "tps": round(ct / dt, 1) if dt else 0,
               "content": text, "reasoning": reasoning}
        out.write(json.dumps(rec) + "\n")
        out.flush()
        preview = (reasoning or text).replace("\n", " ")[:90]
        print(f"[{BUCKET}{i}] {ct} tok {dt:.0f}s {ct/dt:.1f} t/s | {preview}", flush=True)
print(f"BUCKET {BUCKET} DONE: {total} completion tokens in {time.time()-t0:.0f}s", flush=True)
