import numpy as np


class Sine:
    def generate_wave(freq: float, samplerate: int) -> np.array:
        """
        Generate a 1s sinewave according to the samplerate and frequency.
        """
        t_sample = 1 / samplerate
        time = 0
        wave = np.empty(samplerate, dtype=float)
        for t in range(samplerate):
            wave[t] = np.sin(np.pi * 2 * time * freq)
            time += t_sample
        return wave
