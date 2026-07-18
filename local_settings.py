"""Optional local-only overrides, loaded from a gitignored local_settings.json
in the project root. Falls back to today's exact shared-repo defaults when
the file doesn't exist, so a fresh clone behaves identically to before this
mechanism existed. See local_settings.example.json for the shape.
"""
import json
from pathlib import Path
from typing import List

from pydantic import BaseModel

_PATH = Path(__file__).parent / "local_settings.json"


class LocalSettings(BaseModel):
    default_inspiration_domains: List[str] = []  # [] == unrestricted search (today's default)
    enable_tail_pipeline: bool = True
    enable_calendar_sync: bool = True


def load_local_settings(path: Path = _PATH) -> LocalSettings:
    if not path.exists():
        return LocalSettings()
    data = json.loads(path.read_text(encoding="utf-8"))
    return LocalSettings(**data)
