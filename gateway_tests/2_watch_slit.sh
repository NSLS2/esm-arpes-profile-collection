#!/bin/bash
# Steps 2 and 3, terminal A (read-only): watch the slit move. Ctrl-C to stop.
# Usage: bash gateway_tests/2_watch_slit.sh
S='FE:C21A-OP{Slt:12-Ax:X}'

camonitor -n "${S}t2.C" "${S}DMOV"
