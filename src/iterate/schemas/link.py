"""The link plan: how a folder's images connect to their labels.

Built by the rules ladder, by explicit flags, or by the Linker; applied by one
path whichever built it. Extra fields are refused, every choice is a closed enum
and the parts must agree, so a plan a model emits can only take a shape that
`apply` can execute.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Shape = Literal["csv_of_paths", "class_folders", "table_join"]
Source = Literal["rules", "flags", "agent"]
KeyMethod = Literal["path", "basename", "stem", "stem_int"]
Task = Literal["classification", "regression"]
Split = Literal["ours", "folders", "column"]


class LinkPlan(BaseModel):
    """One way of reading a folder as a labelled image dataset."""

    model_config = ConfigDict(extra="forbid")

    shape: Shape
    source: Source = "rules"
    labels_file: str | None = None
    key_column: str | None = None
    key_to_file: KeyMethod | None = None
    target_column: str | None = None
    onehot_columns: list[str] = Field(default_factory=list)
    task: Task
    split: Split = "ours"
    split_column: str | None = None
    coverage: float = Field(ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _parts_agree(self) -> Self:
        if self.shape == "class_folders":
            if self.split == "column":
                raise ValueError("class folders carry no split column")
            return self
        if not (self.labels_file and self.key_column and self.key_to_file):
            raise ValueError("a table plan needs labels_file, key_column and key_to_file")
        if bool(self.target_column) == bool(self.onehot_columns):
            raise ValueError("a table plan needs a target column or one-hot columns, not both")
        if (self.split == "column") != (self.split_column is not None):
            raise ValueError("split 'column' needs split_column, and only that split does")
        return self


__all__ = ["KeyMethod", "LinkPlan", "Shape", "Source", "Split", "Task"]
