#!/usr/bin/env bash

echo "PBSMT-R"
easse evaluate -t turk -a -m sari < data/system_outputs/turk/lower/PBMT-R.tok.low
echo "====================================================================="

echo "Hybrid"
easse evaluate -t turk -a -m sari < data/system_outputs/turk/lower/Hybrid.tok.low
echo "====================================================================="

echo "SBMT-SARI"
easse evaluate -t turk -a -m sari < data/system_outputs/turk/lower/SBMT-SARI.tok.low
echo "====================================================================="

echo "NTS-SARI"
easse evaluate -t turk -a -m sari < data/system_outputs/turk/lower/NTS-SARI.tok.low
echo "====================================================================="

echo "Dress-Ls"
easse evaluate -t turk -a -m sari < data/system_outputs/turk/lower/Dress-Ls.tok.low
echo "====================================================================="

echo "DMASS-DCSS"
easse evaluate -t turk -a -m sari < data/system_outputs/turk/lower/DMASS-DCSS.tok.low
echo "====================================================================="
