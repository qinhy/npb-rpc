from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

MISSING = object()
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
TIME_RE = re.compile(r"\d{6}\.\d+JST")


# -----------------------------------------------------------------------------
# Filesystem documents
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Attachment:
    path: Path
    relative_path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def suffix(self) -> str:
        return self.path.suffix

    @property
    def size(self) -> int:
        return self.path.stat().st_size


@dataclass
class Document:
    id: str
    parent: str | None
    json_file: Path | None
    attachments: list[Attachment]

    def load(self) -> dict[str, Any]:
        if self.json_file is None:
            data: dict[str, Any] = {}
        else:
            with self.json_file.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                raise ValueError(f"JSON document must contain an object: {self.json_file}")
            data = dict(loaded)

        data["_id"] = self.id
        data["parent"] = self.parent
        if self.attachments:
            data["_attachments"] = {
                a.name: {
                    "path": a.relative_path.as_posix(),
                    "suffix": a.suffix,
                    "size": a.size,
                }
                for a in self.attachments
            }
        return data


def relative_to_id(path: Path) -> str:
    return ":".join(path.parts)


def id_parent(doc_id: str) -> str | None:
    return doc_id.rsplit(":", 1)[0] if ":" in doc_id else None


def document_from_id(root: str | Path, doc_id: str) -> Document | None:
    """Resolve one document directly, without scanning the DB."""
    root = Path(root).resolve()
    resource = root.joinpath(*doc_id.split(":"))

    if resource.is_dir():
        sidecar = resource.parent / f"{resource.name}.json"
        return Document(
            id=doc_id,
            parent=id_parent(doc_id),
            json_file=sidecar if sidecar.is_file() else None,
            attachments=[
                Attachment(p, p.relative_to(root))
                for p in sorted(resource.iterdir(), key=lambda x: x.name)
                if p.is_file() and p.suffix.lower() != ".json"
            ],
        )

    json_file = resource.with_suffix(".json")
    if json_file.is_file():
        return Document(doc_id, id_parent(doc_id), json_file, [])
    return None


def rglob_documents(
    root: str | Path,
    *,
    scan_root: str | Path | None = None,
) -> Iterator[Document]:
    """Materialize virtual documents from root, optionally below one subtree."""
    root = Path(root).resolve()
    scan_root = root if scan_root is None else Path(scan_root).resolve()

    if not root.is_dir():
        raise NotADirectoryError(root)
    if not scan_root.exists():
        return
    if not scan_root.is_dir():
        raise NotADirectoryError(scan_root)

    try:
        scan_root.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"scan_root must be below root: {scan_root}") from exc

    paths = list(scan_root.rglob("*"))
    if scan_root != root:
        paths.append(scan_root)
    paths.sort(key=lambda p: p.as_posix())

    directories = [p for p in paths if p.is_dir()]
    json_files = {p for p in paths if p.is_file() and p.suffix.lower() == ".json"}
    consumed: set[Path] = set()
    by_id: dict[str, Document] = {}

    for directory in directories:
        doc_id = relative_to_id(directory.relative_to(root))
        sidecar = directory.parent / f"{directory.name}.json"
        json_file = sidecar if sidecar.is_file() else None
        if json_file is not None:
            consumed.add(json_file)

        doc = Document(
            id=doc_id,
            parent=id_parent(doc_id),
            json_file=json_file,
            attachments=[
                Attachment(p, p.relative_to(root))
                for p in sorted(directory.iterdir(), key=lambda x: x.name)
                if p.is_file() and p.suffix.lower() != ".json"
            ],
        )
        if doc_id in by_id:
            raise ValueError(f"Multiple filesystem resources map to _id={doc_id!r}")
        by_id[doc_id] = doc

    for json_file in sorted(json_files - consumed, key=lambda p: p.as_posix()):
        doc_id = relative_to_id(json_file.relative_to(root).with_suffix(""))
        if doc_id in by_id:
            raise ValueError(f"Multiple filesystem resources map to _id={doc_id!r}")
        by_id[doc_id] = Document(doc_id, id_parent(doc_id), json_file, [])

    for doc_id in sorted(by_id):
        yield by_id[doc_id]


# -----------------------------------------------------------------------------
# _id scope extraction
# -----------------------------------------------------------------------------

def _date_part(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    date = value[:10]
    return date if DATE_RE.fullmatch(date) and (len(value) == 10 or value[10] == ":") else None


def _record_part(value: Any) -> tuple[str, str, str] | None:
    """Parse YYYY-MM-DD:field:HHMMSS.nnnJST[:...]."""
    if not isinstance(value, str):
        return None
    parts = value.split(":", 3)
    if len(parts) < 3:
        return None
    date, field, timestamp = parts[:3]
    if not DATE_RE.fullmatch(date) or not field or not TIME_RE.fullmatch(timestamp):
        return None
    return date, field, timestamp


def _id_date_scope(
    selector: dict[str, Any],
) -> tuple[set[str] | None, str | None, str | None] | None:
    """Return (exact_dates, lower_date, upper_date), conservatively inclusive."""
    condition = selector.get("_id", MISSING)
    if condition is MISSING:
        return None

    if isinstance(condition, str):
        date = _date_part(condition)
        return ({date}, None, None) if date else None
    if not isinstance(condition, dict):
        return None

    for op in ("$eq", "$beginsWith"):
        if op in condition:
            date = _date_part(condition[op])
            if date:
                return ({date}, None, None)

    values = condition.get("$in")
    if isinstance(values, (list, tuple, set)):
        dates = {_date_part(v) for v in values}
        if None in dates:
            return None
        return (dates, None, None)  # type: ignore[arg-type]

    lower = upper = None
    for op in ("$gte", "$gt"):
        if op in condition:
            lower = _date_part(condition[op])
            if lower is None:
                return None
            break
    for op in ("$lte", "$lt"):
        if op in condition:
            upper = _date_part(condition[op])
            if upper is None:
                return None
            break
    return None if lower is None and upper is None else (None, lower, upper)


def _id_record_scope(
    selector: dict[str, Any],
) -> tuple[str, str, str, str] | None:
    """
    Return (date, field, lower_time, upper_time) for a safe record-level range.

    Only same-date + same-field two-sided ranges are narrowed this way.
    Wider ranges fall back to date scanning so query semantics stay exact.
    """
    condition = selector.get("_id")
    if not isinstance(condition, dict):
        return None

    lower = upper = None
    for op in ("$gte", "$gt"):
        if op in condition:
            lower = _record_part(condition[op])
            if lower is None:
                return None
            break
    for op in ("$lte", "$lt"):
        if op in condition:
            upper = _record_part(condition[op])
            if upper is None:
                return None
            break

    if lower is None or upper is None:
        return None
    if lower[:2] != upper[:2]:
        return None

    date, field, lower_time = lower
    return date, field, lower_time, upper[2]


# -----------------------------------------------------------------------------
# Small Mango-like query engine
# -----------------------------------------------------------------------------

def get_field(doc: dict[str, Any], field: str) -> Any:
    value: Any = doc
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return MISSING
        value = value[part]
    return value


def match_condition(value: Any, condition: Any) -> bool:
    if not isinstance(condition, dict) or not any(str(k).startswith("$") for k in condition):
        return value is not MISSING and value == condition

    for op, expected in condition.items():
        if op == "$eq":
            ok = value is not MISSING and value == expected
        elif op == "$ne":
            ok = value is MISSING or value != expected
        elif op == "$gt":
            ok = value is not MISSING and value > expected
        elif op == "$gte":
            ok = value is not MISSING and value >= expected
        elif op == "$lt":
            ok = value is not MISSING and value < expected
        elif op == "$lte":
            ok = value is not MISSING and value <= expected
        elif op == "$in":
            ok = value is not MISSING and value in expected
        elif op == "$exists":
            ok = (value is not MISSING) == bool(expected)
        elif op == "$beginsWith":
            ok = isinstance(value, str) and isinstance(expected, str) and value.startswith(expected)
        elif op == "$regex":
            ok = isinstance(value, str) and isinstance(expected, str) and re.search(expected, value) is not None
        elif op == "$elemMatch":
            ok = isinstance(value, list) and any(
                matches(item, expected) if isinstance(item, dict) else match_condition(item, expected)
                for item in value
            )
        else:
            raise ValueError(f"Unsupported query operator: {op}")

        if not ok:
            return False
    return True


def matches(doc: dict[str, Any], selector: dict[str, Any]) -> bool:
    for key, condition in selector.items():
        if key == "$and":
            ok = all(matches(doc, item) for item in condition)
        elif key == "$or":
            ok = any(matches(doc, item) for item in condition)
        elif key == "$not":
            ok = not matches(doc, condition)
        else:
            ok = match_condition(get_field(doc, key), condition)
        if not ok:
            return False
    return True


def project(doc: dict[str, Any], fields: list[str] | None) -> dict[str, Any]:
    if fields is None:
        return doc
    return {field: value for field in fields if (value := get_field(doc, field)) is not MISSING}


class MemoryDB:
    def __init__(self, docs: Iterable[dict[str, Any]] = ()) -> None:
        self._docs: dict[str, dict[str, Any]] = {}
        self.put_many(docs)

    def put(self, doc: dict[str, Any]) -> None:
        doc_id = doc.get("_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError("Document requires a non-empty string '_id'")
        self._docs[doc_id] = doc

    def put_many(self, docs: Iterable[dict[str, Any]]) -> None:
        for doc in docs:
            self.put(doc)

    def get(self, doc_id: str) -> dict[str, Any] | None:
        return self._docs.get(doc_id)

    def find(
        self,
        selector: dict[str, Any],
        *,
        fields: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        result = []
        for doc in self._docs.values():
            if matches(doc, selector):
                result.append(project(doc, fields))
                if limit is not None and len(result) >= limit:
                    break
        return result


# -----------------------------------------------------------------------------
# Filesystem DB facade
# -----------------------------------------------------------------------------

class FileSystemDB:
    """
    Filesystem = source of truth.

    Scan strategy:
      1. same-day/same-field record range -> only matching record directories
      2. otherwise date range            -> only matching date directories
      3. otherwise                       -> whole DB
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._db: MemoryDB | None = None

    def _record_scan_roots(self, selector: dict[str, Any]) -> list[Path] | None:
        scope = _id_record_scope(selector)
        if scope is None:
            return None

        date, field, lower_time, upper_time = scope
        field_dir = self.root / date / field
        if not field_dir.is_dir():
            return []

        # Boundary records are intentionally included. MemoryDB applies the
        # exact $gt/$gte/$lt/$lte condition afterwards.
        return sorted(
            (
                p
                for p in field_dir.iterdir()
                if p.is_dir()
                and TIME_RE.fullmatch(p.name)
                and lower_time <= p.name <= upper_time
            ),
            key=lambda p: p.name,
        )

    def _date_scan_roots(self, selector: dict[str, Any]) -> list[Path] | None:
        scope = _id_date_scope(selector)
        if scope is None:
            return None

        exact, lower, upper = scope
        return sorted(
            (
                p
                for p in self.root.iterdir()
                if p.is_dir()
                and DATE_RE.fullmatch(p.name)
                and (exact is None or p.name in exact)
                and (lower is None or p.name >= lower)
                and (upper is None or p.name <= upper)
            ),
            key=lambda p: p.name,
        )

    def _scan_roots(self, selector: dict[str, Any]) -> list[Path] | None:
        roots = self._record_scan_roots(selector)
        return roots if roots is not None else self._date_scan_roots(selector)

    def _memory_db(self, selector: dict[str, Any]) -> MemoryDB:
        scan_roots = self._scan_roots(selector)
        documents = (
            rglob_documents(self.root)
            if scan_roots is None
            else (
                doc
                for scan_root in scan_roots
                for doc in rglob_documents(self.root, scan_root=scan_root)
            )
        )
        return MemoryDB(doc.load() for doc in documents)

    def get(self, doc_id: str) -> dict[str, Any] | None:
        if self._db is not None and (cached := self._db.get(doc_id)) is not None:
            return cached
        document = document_from_id(self.root, doc_id)
        return None if document is None else document.load()

    def find(
        self,
        selector: dict[str, Any],
        *,
        fields: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self._db = self._memory_db(selector)
        return self._db.find(selector, fields=fields, limit=limit)
