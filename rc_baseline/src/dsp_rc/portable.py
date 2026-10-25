"""Relocate audited paths without changing raw data or the original CSV."""
import csv
from pathlib import Path
ORIGINAL_ROOT = Path('/Users/oliver/Desktop/DSP')
def load_audit_index(root):
    root = Path(root).resolve()
    with (root/'audit_outputs/2026-10-16/file_index.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        original = Path(row['path'])
        relative = original.relative_to(ORIGINAL_ROOT) if original.is_absolute() else original
        relocated = (root/relative).resolve()
        if not relocated.is_relative_to(root):
            raise ValueError('Audit path escapes project root')
        result[str(relocated)] = row
    return result
