#!/bin/bash
# Step 3 (MOVES the FE slit): plain put, no callback, then read the same PV.
# Usage: bash gateway_tests/3_plain_put.sh ORIG
S='FE:C21A-OP{Slt:12-Ax:X}'
ORIG=${1:?usage: $0 ORIG}

caput "${S}size" "$ORIG" &
sleep 0.2
echo "--- read while moving"
time caget -w 30 "${S}size"
wait
