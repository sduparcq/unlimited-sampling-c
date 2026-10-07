import csv
from pathlib import Path


class CSVWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.file = self.path.open("w", newline="")
        self.writer = None

    def write(self, row: dict):
        if self.writer is None:
            self.writer = csv.DictWriter(
                self.file,
                fieldnames=row.keys(),
            )
            self.writer.writeheader()

        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
