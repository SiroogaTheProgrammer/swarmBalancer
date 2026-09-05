"""Device profiles: the hardware each swarm member is assumed to run on.

Used by the simulator (to turn MACs into seconds and to size radio links)
and by the stress harness (to enforce RAM caps and deadlines on the host).
"""

from .profiles import DEVICE_PROFILES, DeviceProfile, get_profile

__all__ = ["DEVICE_PROFILES", "DeviceProfile", "get_profile"]
