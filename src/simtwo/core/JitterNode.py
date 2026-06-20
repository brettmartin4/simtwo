import numpy as np
from sequence.topology.node import Node


class _PerlinNoise1D:
    """Generate deterministic one-dimensional Perlin noise samples based on "Improving Noise" (https://doi.org/10.1145/566654.566636)."""

    def __init__(self, rng):
        """Initialize the _PerlinNoise1D instance.
        
        Args:
            rng: Numpy random generator used to create the permutation table. (For reproducability)
        """
        p = np.asarray(rng.permutation(256), dtype=int)
        self.permutation = np.concatenate([p, p])

    @staticmethod
    def _fade(t: float) -> float:
        """Smooth an interpolation coordinate using Perlins fade curve.
        
        Args:
            t (float): Fractional coordinate between two adjacent lattice points.
        
        Returns:
            float: Smoothed interpolation weight in the range used by the Perlin blend step.
        """
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        """Linearly interpolate between two scalar values.
        
        Args:
            a (float): Value at the left endpoint.
            b (float): Value at the right endpoint.
            t (float): Interpolation weight, should ideally be between 0 and 1.
        
        Returns:
            float: Interpolated scalar value.
        """
        return a + t * (b - a)

    @staticmethod
    def _grad(hash_value: int, x: float) -> float:
        """Choose a one-dimensional gradient contribution for a lattice point.
        
        Args:
            hash_value (int): Permutation table value used to choose the gradient sign.
            x (float): Offset from the lattice point.
        
        Returns:
            float: Signed gradient contribution for the offset.
        """
        return x if (hash_value & 1) == 0 else -x

    def noise(self, x: float) -> float:
        """Evaluate the one-dimensional Perlin noise function.
        
        Args:
            x (float): Continuous coordinate at which to sample the noise function.
        
        Returns:
            float: Smooth pseudo random value near the range [-1, 1].
        """
        xi = int(np.floor(x)) & 255
        xf = float(x - np.floor(x))
        u = self._fade(xf)
        g0 = self._grad(int(self.permutation[xi]), xf)
        g1 = self._grad(int(self.permutation[xi + 1]), xf - 1.0)
        return float(2.0 * self._lerp(g0, g1, u))


class RxNodeWithJitter(Node):
    """Sequencee receiver node that records arrivals with configurable jitter.
    
    The node stores both the true arrival time reported by the timeline and a jittered reported time.
    The default jitter mode draws independent non-negative Gaussian samples (When 1D Perlin noise isn't used).
    The Perlin mode produces more correlated samples that vary more smoothly from photon t photon.
    """

    def __init__(self, name, timeline, rng, jitter_std_ps=50_000, jitter_method: str = "default", perlin_step: float = 0.05, perlin_amplitude_ps: float | None = None):  # 50,000 ps = 50 ns (TODO: verify correct jitter range later)
        """Create a receiver node that applies synthetic timetag jitter.
        
        Args:
            name: Name passed to the underlying sequence node.
            timeline: sequence timeline that provides photon arrival times in picoseconds.
            rng: Numpy random generator used for Gaussian or Perlin jitter sampling.
            jitter_std_ps: Standard deviation, in picoseconds, used by the default Gaussian jitter sampler. 
                This value is also used as the default Perlin amplitude when perlin_amplitude_ps is not provided.
            jitter_method (str): Jitter source to use. Supported values are "default" and "perlin"; 
                common aliases such as "gaussian" and "perlin_1d" are normalized automatically.
            perlin_step (float): Distance to advance along the Perlin noise curve for each sample.
                Smaller values produce smoother sample-to-sample drift.
            perlin_amplitude_ps (float): Maximum scale, in picoseconds, for the Perlin jitter samples. When omitted, ``jitter_std_ps`` is used.
        """
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
        """Normalize user facing jitter method names.
        
        Args:
            jitter_method (str): User-provided method name or alias.
        
        Returns:
            str: Method name--either "default" or "perlin".
        
        Raises:
            ValueError: If the method name is not supported.
        """
        method = str(jitter_method or "default").strip().lower()
        if method in {"default", "normal", "gaussian"}:
            return "default"
        if method in {"perlin", "perlin1d", "1d_perlin", "perlin_1d"}:
            return "perlin"
        raise ValueError(f"Unsupported jitter_method '{jitter_method}'. Use 'default' or 'perlin'.")

    def _sample_default_jitter_ps(self) -> int:
        """Draw one non negative Gaussian jitter sample.
        
        Returns:
            int: Jitter delay in picoseconds, clipped at zero and converted to an integer because sequence timeline times are represented in integer picoseconds.
        """
        extra = self.rng.normal(loc=0.0, scale=self.jitter_std_ps)
        return int(max(0.0, extra))

    def _sample_perlin_jitter_ps(self) -> int:
        """Draw one correlated Perlin jitter sample.
        
        Returns:
            int: Jitter delay in picoseconds. Consecutive calls advance along the same Perlin noise curve, so the returned values drift smoothly rather than jumping independently.
        """
        x = self._perlin_x + self._perlin_sample * self.perlin_step
        self._perlin_sample += 1
        if self._perlin is None:
            self._perlin = _PerlinNoise1D(self.rng)
        noise_value = self._perlin.noise(x)
        scaled = 0.5 * (noise_value + 1.0) * self.perlin_amplitude_ps
        return int(max(0.0, scaled))

    def _sample_jitter_ps(self) -> int:
        """Sample jitter using the currently configured jitter method.
        
        Returns:
            Jitter delay in picoseconds from either the default sampler or the Perlin sampler.
        """
        if self.jitter_method == "perlin":
            return self._sample_perlin_jitter_ps()
        return self._sample_default_jitter_ps()

    def receive_qubit(self, src: str, qubit):
        """Record a qubit arrival and apply receiver side timing jitter.
        
        Args:
            src (str): Name of the sending node as supplied by sequence.
            qubit: Received photon/qubit object. The current implementation does not inspect the object--it only records the arrival time.
        """

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
        