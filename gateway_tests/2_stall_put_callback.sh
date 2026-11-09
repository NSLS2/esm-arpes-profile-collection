#!/bin/bash
# Step 2 (MOVES the FE slit): put with callback, as ophyd does now, then read the same PV.
# Usage: bash gateway_tests/2_stall_put_callback.sh NEW
S='FE:C21A-OP{Slt:12-Ax:X}'
NEW=${1:?usage: $0 NEW}

# Ignore Ctrl-C: killing caput before the put finishes is the gateway crash condition.
trap '' INT

caput -c -w 60 "${S}size" "$NEW" &
sleep 0.2
echo "--- read with caget's default 1 s timeout"
time caget "${S}size"
echo "--- read again, waiting up to 30 s"
time caget -w 30 "${S}size"
wait
echo "Put finished."
