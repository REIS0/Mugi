import numpy as np


class Sine:
    def generate_wave(freq: float, size: int, samplerate=44100) -> np.ndarray:
        """
        Generate a sinewave according to the frequency, number
        of samples and samplerate.
        Samplerate is 44100 by default.
        """
        t_sample = 1 / samplerate
        time = 0
        wave = np.empty(size, dtype=float)
        for t in range(size):
            wave[t] = np.sin(np.pi * 2 * time * freq)
            time += t_sample
        return wave
