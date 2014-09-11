#!/usr/bin/env python

import sys
f1 = open(sys.argv[1])
f2 = open(sys.argv[2])
res = 0
for l1, l2 in zip(f1, f2):
    v1 = float(l1.strip())
    v2 = float(l2.strip())
    rel_diff = abs(v1 - v2) / abs(v1)
    if rel_diff > 10 ** -4:
        print "Difference: {} vs {}".format(v1, v2)
        res = 1
sys.exit(res)
