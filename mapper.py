#!/usr/bin/env python3
import sys

for line in sys.stdin:
    parts = line.strip().split()
    if len(parts) == 2:
        ip, time = parts
        print(f"{ip}\t{time}")
