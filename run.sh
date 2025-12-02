#!/bin/bash
# Weaver two-phase curriculum graph construction
# Timeouts: ~20 min harvest (0.33h) / ~31 min weave (0.51h) demo

cargo run --release -- uncc_cs2-pretext-project \
  --chapters "source/*/toctree.ptx" \
  --max-concurrent-chapters 13 \
  --harvest-timeout-hours 0.33 \
  --weave-timeout-hours 0.51
