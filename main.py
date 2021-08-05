from src.algorithms.simple_ea import SimpleEA
import soundfile as sf
import glob

ALPHA = 0.5
INDV_SIZE = 1024
FIT = 0.2


def main() -> None:
    target = []

    for file in glob.glob("files/*.flac"):
        waveform, _ = sf.read(file)
        target.append(waveform)

    simple_ea = SimpleEA(INDV_SIZE, FIT, ALPHA)

    for t in range(len(target)):
        print(f"----- TARGET {t+1} -----")
        simple_ea.run(target[t])


if __name__ == "__main__":
    main()
