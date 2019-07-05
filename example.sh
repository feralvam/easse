#!/bin/bash
easse evaluate -t turk -m 'bleu,sari,fkgl' -q < data/system_outputs/turk/lower/DMASS-DCSS.tok.low
easse report -t turk -m 'bleu,sari,fkgl' < data/system_outputs/turk/lower/DMASS-DCSS.tok.low
