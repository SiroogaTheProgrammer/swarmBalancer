"""Hardware profiles for swarm members.

A profile is a *model* of a device, not a measurement: it lets the simulator
and the stress harness answer "how long would this brain take on that board,
and does it fit in RAM" deterministically on the development PC. Numbers are
order-of-magnitude figures for the class of hardware named; tune them against
real boards as you get them (``compute_seconds`` is the single place that
turns MACs into time).

``macs_per_cycle`` is *effective* throughput of the int8/f32 GEMM kernel on
that core (SIMD width x pipelines x achieved utilisation), which is why a
Cortex-M4 gets 0.5 and a Cortex-A72 with NEON gets 8.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    description: str
    cpu_mhz: float
    macs_per_cycle: float          # effective MACs per clock for the brain's GEMM kernel
    ram_bytes: int                 # RAM available to the brain (weights + activations + scratch)
    radio_bps: float               # link throughput to the swarm
    radio_latency_s: float         # one-way propagation + MAC delay
    energy_per_mac_nj: float = 0.0  # optional, for power budgeting
    energy_per_bit_nj: float = 0.0
    tags: tuple[str, ...] = field(default_factory=tuple)

    def compute_seconds(self, macs: int, overhead_macs: int = 0) -> float:
        """Wall time to execute ``macs`` multiply-accumulates on this device."""
        return (macs + overhead_macs) / (self.cpu_mhz * 1e6 * self.macs_per_cycle)

    def tx_seconds(self, payload_bytes: int) -> float:
        """Time to serialise ``payload_bytes`` onto the radio (excluding latency)."""
        return payload_bytes * 8.0 / self.radio_bps

    def energy_joules(self, macs: int, bytes_sent: int) -> float:
        return (macs * self.energy_per_mac_nj + bytes_sent * 8 * self.energy_per_bit_nj) * 1e-9


DEVICE_PROFILES: dict[str, DeviceProfile] = {
    p.name: p
    for p in [
        DeviceProfile(
            "mcu-m4",
            "Cortex-M4F class microcontroller (e.g. STM32F4) - the smallest drone brain",
            cpu_mhz=168, macs_per_cycle=0.5, ram_bytes=192 * 1024,
            radio_bps=250e3, radio_latency_s=0.004, energy_per_mac_nj=0.5, energy_per_bit_nj=100,
            tags=("drone", "tiny"),
        ),
        DeviceProfile(
            "mcu-m7",
            "Cortex-M7 class MCU (e.g. STM32H7) - typical flight controller with spare cycles",
            cpu_mhz=480, macs_per_cycle=1.0, ram_bytes=1024 * 1024,
            radio_bps=1e6, radio_latency_s=0.003, energy_per_mac_nj=0.3, energy_per_bit_nj=60,
            tags=("drone",),
        ),
        DeviceProfile(
            "drone-a53",
            "Quad Cortex-A53 companion computer (Raspberry Pi 3 class) on a small drone",
            cpu_mhz=1200, macs_per_cycle=4.0, ram_bytes=64 * 1024 * 1024,
            radio_bps=6e6, radio_latency_s=0.002, energy_per_mac_nj=0.1, energy_per_bit_nj=20,
            tags=("drone",),
        ),
        DeviceProfile(
            "drone-a72",
            "Cortex-A72 companion computer (Raspberry Pi 4 class) - the 'command drone'",
            cpu_mhz=1500, macs_per_cycle=8.0, ram_bytes=512 * 1024 * 1024,
            radio_bps=20e6, radio_latency_s=0.002, energy_per_mac_nj=0.08, energy_per_bit_nj=15,
            tags=("drone", "leader"),
        ),
        DeviceProfile(
            "cubesat-obc",
            "Radiation-tolerant CubeSat on-board computer (Cortex-M/R or LEON class)",
            cpu_mhz=100, macs_per_cycle=0.5, ram_bytes=256 * 1024,
            radio_bps=9600, radio_latency_s=0.010, energy_per_mac_nj=1.0, energy_per_bit_nj=5000,
            tags=("satellite", "tiny"),
        ),
        DeviceProfile(
            "sat-payload",
            "Small-sat payload processor (Cortex-A class, e.g. Zynq PS) doing on-orbit image recognition",
            cpu_mhz=667, macs_per_cycle=4.0, ram_bytes=128 * 1024 * 1024,
            radio_bps=1e6, radio_latency_s=0.010, energy_per_mac_nj=0.2, energy_per_bit_nj=1000,
            tags=("satellite",),
        ),
        DeviceProfile(
            "host",
            "This development PC (no throttling) - used to measure real kernel speed",
            cpu_mhz=3000, macs_per_cycle=32.0, ram_bytes=8 * 1024 * 1024 * 1024,
            radio_bps=1e9, radio_latency_s=0.0001,
            tags=("host",),
        ),
    ]
}


def get_profile(name: str) -> DeviceProfile:
    try:
        return DEVICE_PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown device profile {name!r}; known: {sorted(DEVICE_PROFILES)}") from None
