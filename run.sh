#!/bin/bash
# Weaver two-phase curriculum graph construction
# Timeouts: 1 hour per phase for demo

cargo run --release -- uncc_cs2-pretext-project \
  --chapters "source/*/toctree.ptx" \
  --max-concurrent-chapters 4 \
  --harvest-timeout-hours 0.24 \
  --weave-timeout-hours 0.24
