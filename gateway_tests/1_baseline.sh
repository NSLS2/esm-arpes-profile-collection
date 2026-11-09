#!/bin/bash
# Step 1 (read-only): current FE slit values and normal read time.
# Usage: bash gateway_tests/1_baseline.sh
S='FE:C21A-OP{Slt:12-Ax:X}'

caget -n "${S}size" "${S}t2.C" "${S}DMOV"
echo "--- normal read time"
time caget "${S}size"
echo "Write down size as ORIG. Pick NEW = ORIG - 0.1 (smaller gap)."
