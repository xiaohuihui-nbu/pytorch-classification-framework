"""Serial prediction requests over stdin; results are atomically published per job."""

import json
import logging
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from .predictor import PredictionSession
from .worker import watch_parent, write_result


def main():
    watch_parent()
    session = PredictionSession()
    for line in sys.stdin:
        request = json.loads(line)
        spec = json.loads(Path(request["spec"]).read_text(encoding="utf-8"))
        target = Path(spec["directory"]) / "result.json"
        with Path(request["log"]).open("w", encoding="utf-8") as log:
            handler = logging.StreamHandler(log)
            logger = logging.getLogger()
            logger.addHandler(handler)
            try:
                with redirect_stdout(log), redirect_stderr(log):
                    try:
                        result = session.execute(spec)
                    except Exception as exc:
                        result = {"error": str(exc)}
                        traceback.print_exc()
            finally:
                logger.removeHandler(handler)
                handler.close()
        write_result(target, result)
        print("done", flush=True)


if __name__ == "__main__":
    main()
