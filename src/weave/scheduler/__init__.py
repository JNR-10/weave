"""Adaptive Placement Scheduler — the project's primary contribution.

Consumes the profiler's view of the cluster, the estimator's prediction for the incoming
request, and telemetry from previous executions, then decides two things:

    1. how many Macs should serve this workload (possibly one — distributing a model that
       already fits can cost more in communication than it returns), and
    2. how the model's layers are divided among the chosen machines.

Placement is re-evaluated at safe boundaries — before a new request, between batches, or
between conversation turns — rather than mid-token, which would require moving weights and
KV state while generation is in flight.

Baselines live here alongside the adaptive policy so they share the same interface and can be
compared directly: equal split, memory-proportional split, and static profiled split.
"""
