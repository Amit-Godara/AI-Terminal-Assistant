"""Command history manager module."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class HistoryManager:
    """Manages persistent command history."""

    def __init__(self, history_file_path: Path) -> None:
        self.history_file_path = history_file_path
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.history_file_path.exists():
            self.history_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def log_entry(self, entry: Dict[str, Any]) -> None:
        """Appends an execution record to the history."""
        try:
            records = self.get_history()
            entry["timestamp"] = datetime.now().isoformat()
            records.append(entry)
            with open(self.history_file_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
        except Exception:
            pass

    def get_history(self) -> List[Dict[str, Any]]:
        """Retrieves past command records safely, recovering from file corruption."""
        try:
            if not self.history_file_path.exists():
                return []
            with open(self.history_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []
