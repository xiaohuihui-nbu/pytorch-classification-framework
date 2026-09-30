"""Export Web contracts without opening the live workspace or starting workers."""

import json
import tempfile
from pathlib import Path

from clsweb.app import create_app
from clsweb.settings import WebSettings


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="cls-schema-") as directory:
        schema = create_app(Path(directory)).openapi()
    for path, value in (
        (root / "docs/web.openapi.json", schema),
        (root / "docs/web.schema.json", WebSettings.model_json_schema()),
    ):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
