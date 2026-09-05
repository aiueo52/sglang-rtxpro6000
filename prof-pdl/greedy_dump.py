#!/usr/bin/env python3
"""Greedy (temperature=0) generation on the fnbench workload prompts; dump full text.
Usage: greedy_dump.py <out.json> [max_tokens=400]"""
import json, sys, urllib.request, hashlib
EP = "http://127.0.0.1:8001/v1/chat/completions"
WL = "/home/user/tools/flash-next-bench/workloads"
out_path = sys.argv[1]
mt = int(sys.argv[2]) if len(sys.argv) > 2 else 400
res = {}
for name in ("code-edit", "prose-en", "agent-loop", "prose-ja"):
    prompt = open(f"{WL}/{name}.txt").read()
    body = {"model": "flash-next", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mt, "temperature": 0, "top_p": 1, "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(EP, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    msg = r["choices"][0]["message"]
    txt = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
    res[name] = {"text": txt, "n": r["usage"]["completion_tokens"],
                 "sha": hashlib.sha256(txt.encode()).hexdigest()[:16]}
    print(f"{name:11s} n={res[name]['n']:4d} sha={res[name]['sha']} {txt[:60]!r}")
json.dump(res, open(out_path, "w"))
print("wrote", out_path)
