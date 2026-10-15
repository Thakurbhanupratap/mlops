"""Send sample requests to the running API so the dashboards have data.

Usage: python scripts/load_test.py [base_url] [n_requests]
"""
import json
import random
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200


def sample():
    return dict(
        age=random.randint(35, 75), sex=random.randint(0, 1), cp=random.randint(1, 4),
        trestbps=random.randint(100, 180), chol=random.randint(150, 350),
        fbs=random.randint(0, 1), restecg=random.randint(0, 2),
        thalach=random.randint(100, 200), exang=random.randint(0, 1),
        oldpeak=round(random.random() * 4, 1), slope=random.randint(1, 3),
        ca=random.randint(0, 3), thal=random.choice([3, 6, 7]),
    )


for i in range(N):
    body = json.dumps(sample()).encode()
    if i % 20 == 0:  # occasional bad request to show error metrics
        body = b'{"age": 1}'
    req = urllib.request.Request(f"{BASE}/predict", body,
                                 {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req).read()
    except urllib.error.HTTPError:
        pass
print(f"sent {N} requests to {BASE}")
