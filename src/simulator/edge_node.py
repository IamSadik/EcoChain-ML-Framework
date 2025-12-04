class EdgeNode:
    def __init__(self, node_id, power_profile, cpu_freqs, energy_source_type, stake=100): # <-- ADD STAKE
        self.id = node_id
        self.power_profile = power_profile 
        self.cpu_freqs = cpu_freqs 
        self.source_type = energy_source_type
        self.stake = stake # Reputation/Stake for PoS consensus
        
        # State variables
        self.current_load = 0.0
        self.current_frequency = cpu_freqs[-1]

    def get_max_ops(self):
        """
        Returns estimated FLOPs capacity (in FLOPS/second) at max frequency.
        
        The performance of an edge device is roughly linear with frequency.
        We'll use a conservative factor: 1 MHz ~= 0.5 MFLOPs/sec (a common proxy 
        for low-power edge CPUs running ML inference).
        """
        max_freq_mhz = max(self.cpu_freqs)
        # 1 MHz = 1e6 Hz. Let's assume an efficiency factor of 0.5 (MFLOPs/MHz)
        ops_per_second = max_freq_mhz * 0.5 * 1e6 
        return ops_per_second