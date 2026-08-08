"""Structured summary of a loaded dataset, used by the inspection script
and tests. Deliberately dependency-light (no pandas types in the public
dataclass) so it stays easy to serialize to JSON/markdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    missing_count: int
    is_categorical: bool


@dataclass
class DatasetInfo:
    """Machine-readable summary produced by inspecting a raw or processed dataset."""

    source_files: list[str]
    num_rows: int
    num_columns: int
    columns: list[ColumnProfile]
    duplicate_rows: int
    infinite_value_count: int
    label_column: str
    class_distribution: dict[str, int] = field(default_factory=dict)
    memory_usage_mb: float = 0.0

    @property
    def numerical_columns(self) -> list[str]:
        return [c.name for c in self.columns if not c.is_categorical and c.name != self.label_column]

    @property
    def categorical_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.is_categorical]

    def to_dict(self) -> dict:
        return {
            "source_files": self.source_files,
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "duplicate_rows": self.duplicate_rows,
            "infinite_value_count": self.infinite_value_count,
            "label_column": self.label_column,
            "class_distribution": self.class_distribution,
            "memory_usage_mb": round(self.memory_usage_mb, 3),
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "missing_count": c.missing_count,
                    "is_categorical": c.is_categorical,
                }
                for c in self.columns
            ],
        }
