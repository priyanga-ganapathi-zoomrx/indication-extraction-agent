"""Storage protocol and implementations for file operations.

Provides a unified interface for storage operations that can work with
local filesystem or cloud storage (GCS, S3, etc.).
"""

import json
from pathlib import Path
from typing import Protocol


class StorageClient(Protocol):
    """Protocol for storage operations (GCS or local filesystem)."""
    
    def download_json(self, path: str) -> dict:
        """Download JSON from storage."""
        ...
    
    def upload_json(self, path: str, data: dict) -> None:
        """Upload JSON to storage."""
        ...
    
    def exists(self, path: str) -> bool:
        """Check if path exists in storage."""
        ...


class LocalStorageClient:
    """Local filesystem storage client for development/testing."""
    
    def __init__(self, base_dir: str = "output"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, path: str) -> Path:
        return self.base_dir / path
    
    def download_json(self, path: str) -> dict:
        """Download JSON from storage."""
        file_path = self._get_path(path)
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def upload_json(self, path: str, data: dict) -> None:
        """Upload JSON to storage."""
        file_path = self._get_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def exists(self, path: str) -> bool:
        """Check if path exists in storage."""
        return self._get_path(path).exists()

