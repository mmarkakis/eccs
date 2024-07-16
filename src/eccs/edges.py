import pandas as pd
from typing import TypeAlias

Edge: TypeAlias = tuple[str, str]
Path: TypeAlias = list[Edge]


class EdgeEditType:
    """
    Class to represent possible edit types to a directed edge.
    """

    ADD: str = "Add"
    REMOVE: str = "Remove"
    FLIP: str = "Flip"


class EdgeEdit:
    """
    Class to represent a causal graph edit.
    """

    def __init__(self, src: str, dst: str, edit_type: EdgeEditType):
        self.src = src
        self.dst = dst
        self.edit_type = edit_type

    @property
    def edge(self):
        return (self.src, self.dst)

    def __str__(self):
        return f"{self.edit_type} edge from {self.src} to {self.dst}"

    def __eq__(self, other):
        return (
            self.edit_type == other.edit_type
            and self.src == other.src
            and self.dst == other.dst
        )

    def __hash__(self):
        return hash((self.src, self.dst, self.edit_type))

    def __repr__(self):
        return f"EdgeEdit({self.src},{self.dst},{self.edit_type})"

    def to_dict(self):
        return {
            "edit_type": self.edit_type,
            "src": self.src,
            "dst": self.dst,
        }

    def __iter__(self):
        return iter((self.src, self.dst, self.edit_type))

    @staticmethod
    def from_dict(d):
        return EdgeEdit(d["src"], d["dst"], d["edit_type"])

    @staticmethod
    def to_df(edits):
        return pd.DataFrame(
            [edit.to_dict() for edit in edits],
            columns=["Source", "Destination", "Edit Type"],
        )
