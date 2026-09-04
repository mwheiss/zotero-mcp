"""Shared comparison semantics for API and SQLite metadata search paths."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .utils import _normalize_for_search

OPERATORS: frozenset[str] = frozenset(
    {
        "is",
        "isNot",
        "contains",
        "doesNotContain",
        "beginsWith",
        "endsWith",
        "isGreaterThan",
        "isLessThan",
        "isBefore",
        "isAfter",
    }
)
NEGATED: frozenset[str] = frozenset({"isNot", "doesNotContain"})
RANGE_OPERATIONS: frozenset[str] = frozenset(
    {"isGreaterThan", "isLessThan", "isBefore", "isAfter"}
)
FIELD_ALIASES: dict[str, str] = {
    "author": "creator",
    "authors": "creator",
    "creator": "creator",
    "creators": "creator",
    "tag": "tag",
    "tags": "tag",
    "collection": "collection",
    "collections": "collection",
    "itemtype": "itemType",
    "dateadded": "dateAdded",
    "datemodified": "dateModified",
    "doi": "DOI",
}


def canonical_field(field: str) -> str:
    """Resolve a caller-supplied condition field to its canonical spelling."""
    return FIELD_ALIASES.get(field.lower(), field)


def normalize(text: str | None) -> str:
    """Fold text identically in both search backends."""
    return _normalize_for_search(text or "").casefold()


def _as_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def compare(candidate: str, expected: str, operation: str) -> bool:
    """Evaluate one supported operation against one candidate value."""
    left = normalize(candidate)
    right = normalize(expected)
    if operation == "is":
        return left == right
    if operation == "isNot":
        return left != right
    if operation == "contains":
        return right in left
    if operation == "doesNotContain":
        return right not in left
    if operation == "beginsWith":
        return left.startswith(right)
    if operation == "endsWith":
        return left.endswith(right)

    left_number = _as_float(left)
    right_number = _as_float(right)
    if (
        operation in RANGE_OPERATIONS
        and left_number is not None
        and right_number is not None
    ):
        if operation in {"isGreaterThan", "isAfter"}:
            return left_number > right_number
        return left_number < right_number
    if operation in {"isGreaterThan", "isAfter"}:
        return left > right
    if operation in {"isLessThan", "isBefore"}:
        return left < right
    return False


def matches(
    values: Sequence[str] | Iterable[str], expected: str, operation: str
) -> bool:
    """Evaluate a condition against a possibly multivalued field.

    Missing fields satisfy no operation, including negative operations. This
    prevents a negative creator or tag condition from sweeping in fieldless
    items.
    """
    candidates = list(values)
    if not candidates:
        return False
    comparisons = [compare(value, expected, operation) for value in candidates]
    return all(comparisons) if operation in NEGATED else any(comparisons)
