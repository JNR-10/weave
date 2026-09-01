"""Cluster Profiler — what the hardware can currently do.

Measures each Mac's available unified memory, per-layer execution speed, and interconnect
latency and bandwidth (JACCL over Thunderbolt 5, or RING over TCP/Ethernet).

Deliberately measured rather than inferred from the machine's model identifier: a nominally
fast Mac under background load or thermal throttling is not a fast Mac right now, and that
gap is a large part of what this project is investigating.
"""
