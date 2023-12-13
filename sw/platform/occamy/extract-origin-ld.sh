#!/usr/bin/env bash

RISCV_OBJDUMP=$1
ELF=$2
ORIGIN_LD=$3

# NOTE: is the first address in characters 1-8 the correct one?
RELOC_ADDR=$($RISCV_OBJDUMP -t "$ELF" | grep snitch_main | cut -c1-8)
echo "Writing device object relocation address 0x${RELOC_ADDR} to ${ORIGIN_LD}"
echo "L3_ORIGIN = 0x${RELOC_ADDR};" > "${ORIGIN_LD}"