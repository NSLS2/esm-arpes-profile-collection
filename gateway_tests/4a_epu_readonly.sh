#!/bin/bash
# Step 4a (read-only): EPU state and Start-Cmd details.
# Usage: bash gateway_tests/4a_epu_readonly.sh 57|105
case "$1" in
  57)  E='SR:C21-ID:G1A{EPU:1' ;;
  105) E='SR:C21-ID:G1B{EPU:2' ;;
  *)   echo "usage: $0 57|105"; exit 1 ;;
esac

echo "--- both gaps (the other EPU must be above 219)"
caget 'SR:C21-ID:G1A{EPU:1-Ax:Gap}Mtr.RBV'
caget 'SR:C21-ID:G1B{EPU:2-Ax:Gap}Mtr.RBV'
echo "--- Start-Cmd as a number, and its labels"
caget -n "${E}}Cmd:Start-Cmd"
caget "${E}}Cmd:Start-Cmd.ZNAM"
caget "${E}}Cmd:Start-Cmd.ONAM"
echo "--- motor record mode, Moving-Sts, DMOV with alarm status (stop if DISABLE)"
caget "${E}-Ax:Gap}Mtr.SPMG"
caget "${E}}Sts:Moving-Sts"
caget -n "${E}}Sts:Moving-Sts"
caget -a "${E}-Ax:Gap}Mtr.DMOV"
