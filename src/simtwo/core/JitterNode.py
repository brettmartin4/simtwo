import numpy as np
from sequence.topology.node import Node


class _PerlinNoise1D:
    def __init__(self, rng):
        p = np.asarray(rng.permutation(256), dtype=int)
        self.permutation = np.concatenate([p, p])

    @staticmethod
    def _fade(t: float) -> float:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    @staticmethod
    def _grad(hash_value: int, x: float) -> float:
        return x if (hash_value & 1) == 0 else -x

    def noise(self, x: float) -> float:
        xi = int(np.floor(x)) & 255
        xf = float(x - np.floor(x))
        u = self._fade(xf)
        g0 = self._grad(int(self.permutation[xi]), xf)
        g1 = self._grad(int(self.permutation[xi + 1]), xf - 1.0)
        return float(2.0 * self._lerp(g0, g1, u))


class RxNodeWithJitter(Node):
    def __init__(self, name, timeline, rng, jitter_std_ps=50_000, jitter_method: str = "default", perlin_step: float = 0.05, perlin_amplitude_ps: float | None = None):  # 50,000 ps = 50 ns (TODO: verify correct jitter range later)
        super().__init__(name, timeline)
        self.rng = rng
        self.jitter_std_ps = float(jitter_std_ps)

        # for perlin noise:
        self.jitter_method = self._normalize_jitter_method(jitter_method)
        self.perlin_step = float(perlin_step)
        self.perlin_amplitude_ps = float(perlin_amplitude_ps if perlin_amplitude_ps is not None else jitter_std_ps)
        self._perlin = None
        self._perlin_x = 0.0
        self._perlin_sample = 0
        if self.jitter_method == "perlin":
            self._perlin = _PerlinNoise1D(rng)
            self._perlin_x = float(rng.uniform(0.0, 256.0))

        self.t_rx_ps = None # raw arrival time
        self.t_done_ps = None # arrival + jitter

    @staticmethod
    def _normalize_jitter_method(jitter_method: str) -> str:
        method = str(jitter_method or "default").strip().lower()
        if method in {"default", "normal", "gaussian"}:
            return "default"
        if method in {"perlin", "perlin1d", "1d_perlin", "perlin_1d"}:
            return "perlin"
        raise ValueError(f"Unsupported jitter_method '{jitter_method}'. Use 'default' or 'perlin'.")

    def _sample_default_jitter_ps(self) -> int:
        extra = self.rng.normal(loc=0.0, scale=self.jitter_std_ps)
        return int(max(0.0, extra))

    def _sample_perlin_jitter_ps(self) -> int:
        x = self._perlin_x + self._perlin_sample * self.perlin_step
        self._perlin_sample += 1
        if self._perlin is None:
            self._perlin = _PerlinNoise1D(self.rng)
        noise_value = self._perlin.noise(x)
        scaled = 0.5 * (noise_value + 1.0) * self.perlin_amplitude_ps
        return int(max(0.0, scaled))

    def _sample_jitter_ps(self) -> int:
        if self.jitter_method == "perlin":
            return self._sample_perlin_jitter_ps()
        return self._sample_default_jitter_ps()

    def receive_qubit(self, src: str, qubit):
        # This time is just "when the photon arrived at the node"
        # Can convert to time sync error later
        self.t_rx_ps = self.timeline.now()

        # Add non negative jitter to emulate timetag behavior
        #extra = self.rng.normal(loc=0.0, scale=self.jitter_std_ps)
        #extra_ps = int(max(0.0, extra))
        extra_ps = self._sample_jitter_ps()

        self.t_done_ps = self.t_rx_ps + extra_ps

        print(f"[{self.name}] arrived at {self.t_rx_ps:,} ps; + jitter {extra_ps:,} ps -> reported {self.t_done_ps:,} ps")
        #self.timeline.stop()
        