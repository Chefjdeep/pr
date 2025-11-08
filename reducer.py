#!/usr/bin/env python3
import sys
from collections import defaultdict

ip_time = defaultdict(int)

for line in sys.stdin:
    ip, time = line.strip().split('\t')
    ip_time[ip] += int(time)

for ip, total in ip_time.items():
    print(f"{ip}\t{total}")
