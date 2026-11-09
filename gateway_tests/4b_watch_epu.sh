#!/bin/bash
# Step 4b, terminal A (read-only): watch the EPU during the bsui move. Ctrl-C to stop.
# Usage: bash gateway_tests/4b_watch_epu.sh 57|105
case "$1" in
  57)  E='SR:C21-ID:G1A{EPU:1' ;;
  105) E='SR:C21-ID:G1B{EPU:2' ;;
  *)   echo "usage: $0 57|105"; exit 1 ;;
esac

camonitor -n "${E}}Cmd:Start-Cmd" "${E}}Sts:Moving-Sts" "${E}-Ax:Gap}Mtr.DMOV" "${E}-Ax:Gap}Mtr.RBV"
