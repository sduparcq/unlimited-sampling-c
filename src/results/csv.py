import csv
from pathlib import Path


class CSVWriter:
    def __init__(
        self,
        path: str | Path,
        fieldnames: list[str],
    ):
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.file = self.path.open(
            "w",
            newline="",
        )

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=fieldnames,
        )

        self.writer.writeheader()

    def write(self, row: dict):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()
