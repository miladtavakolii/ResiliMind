from pathlib import Path
import json


class FinalReportGenerator:
    """
    Generates final benchmark summary report.
    """


    def generate(
        self,
        summary: dict,
        failures: list,
        output: Path,
    ) -> None:

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        report = {
            "summary": summary,

            "error_statistics": {

                "total_failures":
                    len(failures),

                "failure_types":
                    self._count_failures(
                        failures
                    )
            }
        }


        output.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


    def _count_failures(
        self,
        failures,
    ):

        counter = {}

        for failure in failures:

            counter[
                failure.category
            ] = (
                counter.get(
                    failure.category,
                    0,
                )
                + 1
            )

        return counter
