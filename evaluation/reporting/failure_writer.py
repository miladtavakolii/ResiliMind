from pathlib import Path
import csv


class FailureCSVWriter:
    """
    Writes evaluation failures into CSV format
    for manual analysis.
    """


    def write(
        self,
        failures,
        output: Path,
    ) -> None:

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    "case_id",
                    "category",
                    "details",
                ]
            )


            for failure in failures:

                writer.writerow(
                    [
                        failure.case_id,
                        failure.category,
                        failure.details,
                    ]
                )
