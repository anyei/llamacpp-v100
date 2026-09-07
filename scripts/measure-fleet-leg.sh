#!/usr/bin/env bash
# Fleet leg measurement (gate discipline: dev-workflow section 5 + coherence
# gate). 6x 100-token greedy completions, cache_prompt false, prints per-run
# decode t/s, draft acceptance when speculation is on, and the FULL text of
# run 1 - READ it before trusting any number.
# Usage: [PORT=8098] [RUNS=6] [NPRED=100] ./scripts/measure-fleet-leg.sh <leg-name>
set -euo pipefail
PORT=${PORT:-8098}
RUNS=${RUNS:-6}
NPRED=${NPRED:-100}
API_KEY=${API_KEY:-anyei}
LEG=${1:-leg}
PROMPT="Why is the sky blue? Explain the physics in detail."

for i in $(seq 1 "$RUNS"); do
  resp=$(curl -s -m 900 -H "Authorization: Bearer $API_KEY" \
    http://127.0.0.1:$PORT/completion -d "{
      \"prompt\": \"$PROMPT\",
      \"n_predict\": $NPRED, \"temperature\": 0, \"cache_prompt\": false
    }")
  python3 - "$LEG" "$i" <<'EOF' "$resp"
import json, sys
leg, i = sys.argv[1], sys.argv[2]
d = json.loads(sys.argv[3])
t = d.get('timings', {})
tps = t.get('predicted_per_second')
acc = t.get('draft_n_accepted'); dn = t.get('draft_n')
extra = f"  draft {acc}/{dn} ({acc/dn*100:.0f}%)" if acc and dn else ""
print(f"{leg} run {i}: {tps:.2f} t/s{extra}" if tps else f"{leg} run {i}: NO TIMINGS: {json.dumps(d)[:200]}")
if i == '1':
    print(f"--- {leg} run 1 text (coherence gate - READ THIS):")
    print(d.get('content', '<NO CONTENT>'))
    print("---")
EOF
done
