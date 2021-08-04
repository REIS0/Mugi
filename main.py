from src.algorithms.simple_ea import SimpleEA
from src.utils.sine import Sine

ALPHA = 0.5
INDV_SIZE = 1024
FIT = 0.2


def main() -> None:
    target = []
    for i in range(5):
        sine = Sine.generate_wave(i * 100, INDV_SIZE)
        target.append(sine)

    simple_ea = SimpleEA(INDV_SIZE, FIT, ALPHA)

    for t in range(len(target)):
        print(f"----- TARGET {t+1} -----")
        simple_ea.run(target[t])


if __name__ == "__main__":
    main()
