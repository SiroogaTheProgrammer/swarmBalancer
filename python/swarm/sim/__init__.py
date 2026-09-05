"""Swarm simulator.

Discrete-event simulation of a swarm of AI devices that share the work of
processing a sensor stream:

* :mod:`swarm.sim.core`       - event loop, ``Node`` base class, metrics
* :mod:`swarm.sim.channel`    - shared radio channel (bandwidth, latency, loss)
* :mod:`swarm.sim.membership` - heartbeats, failure detection, leader election
* :mod:`swarm.sim.strategies` - the workload-distribution strategies under test
* :mod:`swarm.sim.run`        - CLI: run one strategy or compare them
"""
