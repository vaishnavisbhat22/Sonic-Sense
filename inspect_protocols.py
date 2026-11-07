from pathlib import Path

folder = Path(r"E:\SonicSense\extractASVspoof5_protocols")

for file in folder.glob("*.tsv"):
    print("\n" + "=" * 70)
    print("FILE:", file.name)
    print("=" * 70)

    with open(file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 5:
                break
            print(line.strip())