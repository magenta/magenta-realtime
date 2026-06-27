#!/usr/bin/env bash
# Sweep the jax-vs-nnx streaming latency matrix on a single GPU (one process at
# a time). Each row runs in its own process so device memory is released between
# configs. Results are appended as JSON lines to $OUT and echoed as a table.
#
# Usage: scripts/latency_nnx_vs_jax.sh [results.jsonl]
set -u

PY=.venv/bin/python
OUT="${1:-/tmp/latency_results.jsonl}"
: > "$OUT"

# Filter the noisy XLA allocator / rematerialization chatter from stderr.
FILTER='bfc_allocator|hlo_remat|Current allocation|TF_GPU_ALLOCATOR|memory fragmentation|cuda_timer|^W[0-9]|^E[0-9]|WARNING|INFO|UserWarning|warnings.warn'

# Columns: backend size param_dtype compute_dtype
# (The nnx backend's streaming step is the functional jax.jit state-threading
# path; see magenta_rt/nnx/system.py.)
ROWS=(
  "jax mrt2_small fp32 fp32"
  "nnx mrt2_small fp32 fp32"
  "jax mrt2_small fp32 bf16"
  "nnx mrt2_small fp32 bf16"
  "jax mrt2_base  fp32 bf16"
  "nnx mrt2_base  bf16 bf16"
)

echo "=== latency sweep (warmup 50 / measure 25 x4 reps), out=$OUT ==="
for row in "${ROWS[@]}"; do
  read -r backend size pdt cdt <<< "$row"
  echo "--- running: $backend $size param=$pdt compute=$cdt ---"
  "$PY" scripts/latency_nnx_vs_jax.py \
    --backend "$backend" --size "$size" \
    --param-dtype "$pdt" --compute-dtype "$cdt" \
    --warmup-frames 50 --measure-frames 25 --reps 4 \
    --out "$OUT" 2>&1 | grep -viE "$FILTER" | tail -4 \
    || echo "  !! $backend $size param=$pdt compute=$cdt FAILED (see above)"
done

echo ""
echo "=== summary ==="
"$PY" - "$OUT" <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
hdr = f"{'backend':>4} {'size':<11} {'param':>5} {'compute':>7} {'ms/step':>8} {'steps/s':>8} {'RTF':>6}"
print(hdr); print('-' * len(hdr))
for r in rows:
    print(f"{r['backend']:>4} {r['size']:<11} {r['param_dtype']:>5} {r['compute_dtype']:>7} "
          f"{r['ms_per_step']:8.2f} {r['steps_per_s']:8.2f} {r['rtf']:6.3f}")
PYEOF
