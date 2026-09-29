from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


MISSING = object()


# ============================================================================
# Virtual filesystem documents
# ============================================================================

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
    """
    Virtual document backed by the real filesystem.

    Rules:

    1) Every directory below root is a document.

       .../pcd/
         -> _id    = "...:pcd"
         -> parent = "..."

    2) A sibling JSON named after a directory is that directory's body.

       .../pcd/rgbd_hand.json
       .../pcd/rgbd_hand/
           full.pcd

         -> _id    = "...:pcd:rgbd_hand"
         -> parent = "...:pcd"
         -> JSON body = rgbd_hand.json
         -> full.pcd = attachment

    3) A standalone JSON without a same-named directory is a leaf document.

       .../yolo/cam_a/rgb.json
         -> _id    = "...:yolo:cam_a:rgb"
         -> parent = "...:yolo:cam_a"
    """

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
                raise ValueError(
                    f"JSON document must contain an object: {self.json_file}"
                )

            data = dict(loaded)

        # DB-owned fields override file contents.
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
    if ":" not in doc_id:
        return None
    return doc_id.rsplit(":", 1)[0]



def document_from_id(root: str | Path, doc_id: str) -> Document | None:
    """Resolve one virtual document without scanning the whole database."""
    root = Path(root).resolve()
    resource = root.joinpath(*doc_id.split(":"))

    if resource.is_dir():
        sidecar_json = resource.parent / f"{resource.name}.json"
        json_file = sidecar_json if sidecar_json.is_file() else None

        attachments = [
            Attachment(
                path=child,
                relative_path=child.relative_to(root),
            )
            for child in sorted(resource.iterdir(), key=lambda p: p.name)
            if child.is_file() and child.suffix.lower() != ".json"
        ]

        return Document(
            id=doc_id,
            parent=id_parent(doc_id),
            json_file=json_file,
            attachments=attachments,
        )

    json_file = resource.with_suffix(".json")
    if json_file.is_file():
        return Document(
            id=doc_id,
            parent=id_parent(doc_id),
            json_file=json_file,
            attachments=[],
        )

    return None


def _id_date_scope(
    selector: dict[str, Any],
) -> tuple[set[str] | None, str | None, str | None] | None:
    """
    Extract a conservative YYYY-MM-DD scope from a direct _id condition.

    Returns (exact_dates, lower_date, upper_date).

    Date bounds are intentionally inclusive. MemoryDB still evaluates the full
    _id condition, so loading one extra boundary date is safe.
    """
    condition = selector.get("_id", MISSING)

    if condition is MISSING:
        return None

    def date_part(value: Any) -> str | None:
        if not isinstance(value, str) or len(value) < 10:
            return None

        candidate = value[:10]

        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate):
            return None

        if len(value) > 10 and value[10] != ":":
            return None

        return candidate

    if isinstance(condition, str):
        d = date_part(condition)
        return ({d}, None, None) if d is not None else None

    if not isinstance(condition, dict):
        return None

    for op in ("$eq", "$beginsWith"):
        if op in condition:
            d = date_part(condition[op])
            if d is not None:
                return ({d}, None, None)

    if "$in" in condition and isinstance(condition["$in"], (list, tuple, set)):
        dates: set[str] = set()
        for value in condition["$in"]:
            d = date_part(value)
            if d is None:
                return None
            dates.add(d)
        return (dates, None, None)

    lower = None
    upper = None

    for op in ("$gte", "$gt"):
        if op in condition:
            lower = date_part(condition[op])
            if lower is None:
                return None
            break

    for op in ("$lte", "$lt"):
        if op in condition:
            upper = date_part(condition[op])
            if upper is None:
                return None
            break

    if lower is None and upper is None:
        return None

    return (None, lower, upper)


def rglob_documents(
    root: str | Path,
    *,
    scan_root: str | Path | None = None,
) -> Iterator[Document]:
    """
    Scan an ordinary filesystem as a tree of virtual documents.

    - Every directory below root becomes a document.
    - <directory>.json is the optional body of that directory document.
    - Direct non-JSON children of a directory are attachments.
    - Unconsumed JSON files become standalone leaf documents.
    """
    root = Path(root).resolve()
    scan_root = root if scan_root is None else Path(scan_root).resolve()

    if not root.exists():
        raise FileNotFoundError(root)

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

    # Keep document IDs relative to the DB root, while walking only one subtree.
    paths = list(scan_root.rglob("*"))
    if scan_root != root:
        paths.append(scan_root)
    paths = sorted(paths, key=lambda p: p.as_posix())
    directories = [p for p in paths if p.is_dir()]
    files = [p for p in paths if p.is_file()]

    json_files = {p for p in files if p.suffix.lower() == ".json"}
    consumed_json: set[Path] = set()

    documents: list[Document] = []

    # 1. Every real directory becomes a virtual document.
    for directory in directories:
        rel_dir = directory.relative_to(root)
        doc_id = relative_to_id(rel_dir)

        sidecar_json = directory.parent / f"{directory.name}.json"
        json_file = sidecar_json if sidecar_json.is_file() else None

        if json_file is not None:
            consumed_json.add(json_file)

        attachments = [
            Attachment(
                path=child,
                relative_path=child.relative_to(root),
            )
            for child in sorted(directory.iterdir(), key=lambda p: p.name)
            if child.is_file() and child.suffix.lower() != ".json"
        ]

        documents.append(
            Document(
                id=doc_id,
                parent=id_parent(doc_id),
                json_file=json_file,
                attachments=attachments,
            )
        )

    # 2. Remaining JSON files become leaf documents.
    for json_file in sorted(json_files - consumed_json, key=lambda p: p.as_posix()):
        rel_resource = json_file.relative_to(root).with_suffix("")
        doc_id = relative_to_id(rel_resource)

        documents.append(
            Document(
                id=doc_id,
                parent=id_parent(doc_id),
                json_file=json_file,
                attachments=[],
            )
        )

    # Defensive de-duplication.
    by_id: dict[str, Document] = {}

    for doc in documents:
        if doc.id in by_id:
            raise ValueError(
                f"Multiple filesystem resources map to _id={doc.id!r}"
            )
        by_id[doc.id] = doc

    for doc_id in sorted(by_id):
        yield by_id[doc_id]


# ============================================================================
# Small Mango-like in-memory query engine
# ============================================================================

def get_field(doc: dict[str, Any], field: str) -> Any:
    value: Any = doc

    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return MISSING
        value = value[part]

    return value


def match_condition(value: Any, condition: Any) -> bool:
    # Implicit equality.
    if not isinstance(condition, dict) or not any(
        str(k).startswith("$") for k in condition
    ):
        return value is not MISSING and value == condition

    for op, expected in condition.items():
        if op == "$eq":
            if value is MISSING or value != expected:
                return False

        elif op == "$ne":
            if value is not MISSING and value == expected:
                return False

        elif op == "$gt":
            if value is MISSING or not (value > expected):
                return False

        elif op == "$gte":
            if value is MISSING or not (value >= expected):
                return False

        elif op == "$lt":
            if value is MISSING or not (value < expected):
                return False

        elif op == "$lte":
            if value is MISSING or not (value <= expected):
                return False

        elif op == "$in":
            if value is MISSING or value not in expected:
                return False

        elif op == "$exists":
            if (value is not MISSING) != bool(expected):
                return False

        elif op == "$beginsWith":
            if (
                value is MISSING
                or not isinstance(value, str)
                or not isinstance(expected, str)
                or not value.startswith(expected)
            ):
                return False

        elif op == "$regex":
            if (
                value is MISSING
                or not isinstance(value, str)
                or not isinstance(expected, str)
                or re.search(expected, value) is None
            ):
                return False

        elif op == "$elemMatch":
            if value is MISSING or not isinstance(value, list):
                return False

            matched = False

            for item in value:
                if isinstance(item, dict):
                    if matches(item, expected):
                        matched = True
                        break
                else:
                    if match_condition(item, expected):
                        matched = True
                        break

            if not matched:
                return False

        else:
            raise ValueError(f"Unsupported query operator: {op}")

    return True


def matches(doc: dict[str, Any], selector: dict[str, Any]) -> bool:
    for key, condition in selector.items():
        if key == "$and":
            if not all(matches(doc, item) for item in condition):
                return False

        elif key == "$or":
            if not any(matches(doc, item) for item in condition):
                return False

        elif key == "$not":
            if matches(doc, condition):
                return False

        else:
            if not match_condition(get_field(doc, key), condition):
                return False

    return True


def project(
    doc: dict[str, Any],
    fields: list[str] | None,
) -> dict[str, Any]:
    if fields is None:
        return doc

    result = {}

    for field in fields:
        value = get_field(doc, field)

        if value is not MISSING:
            result[field] = value

    return result


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
            if not matches(doc, selector):
                continue

            result.append(
                project(doc, fields)
            )

            if limit is not None and len(result) >= limit:
                break

        return result


# ============================================================================
# Filesystem DB facade
# ============================================================================

class FileSystemDB:
    """
    Pure-Python V0.

    Filesystem = source of truth.
    Query = materialize only the relevant date subtree(s) -> MemoryDB -> find().
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._db = None

    def _date_scan_roots(
        self,
        selector: dict[str, Any],
    ) -> list[Path] | None:
        """
        Return matching top-level YYYY-MM-DD directories.

        None means the selector cannot be safely narrowed by date, so the
        caller must fall back to scanning the whole root.
        """
        scope = _id_date_scope(selector)

        if scope is None:
            return None

        exact_dates, lower_date, upper_date = scope
        roots: list[Path] = []

        for child in sorted(self.root.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue

            name = child.name

            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", name):
                continue

            if exact_dates is not None and name not in exact_dates:
                continue

            if lower_date is not None and name < lower_date:
                continue

            if upper_date is not None and name > upper_date:
                continue

            roots.append(child)

        return roots

    def _memory_db(self, selector: dict[str, Any]) -> MemoryDB:
        scan_roots = self._date_scan_roots(selector)

        if scan_roots is None:
            documents = rglob_documents(self.root)
        else:
            documents = (
                doc
                for scan_root in scan_roots
                for doc in rglob_documents(self.root, scan_root=scan_root)
            )

        return MemoryDB(doc.load() for doc in documents)

    def get(self, doc_id: str) -> dict[str, Any] | None:
        # Reuse the most recently loaded date scope when possible.
        if self._db is not None:
            cached = self._db.get(doc_id)
            if cached is not None:
                return cached

        # Date-scoped find() no longer guarantees that every DB document is
        # cached, so resolve an individual miss directly from the filesystem.
        document = document_from_id(self.root, doc_id)
        return None if document is None else document.load()

    def find(
        self,
        selector: dict[str, Any],
        *,
        fields: list[str] | None = None,
        limit: int | None = None,
    ):
        self._db = self._memory_db(selector)
        return self._db.find(
            selector,
            fields=fields,
            limit=limit,
        )


# ============================================================================
# Demo
# ============================================================================

if __name__ == "__main__":
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser(
        description="Scan a filesystem as virtual NoSQL documents."
    )
    parser.add_argument("root", help="Root directory to scan")
    args = parser.parse_args()

    for document in rglob_documents(args.root):
        print(f"\n[{document.id}]")
        pprint(document.load())

    db = FileSystemDB(args.root)
    query = {
        "_id": {
            "$gte": "2026-09-29:field_all:090000.000000000JST",
            "$lt":  "2026-09-29:field_all:170000.000000000JST",
            "$regex": ":yolo:",
        },
        "detections": {
            "$elemMatch": {
                "class_name": "suitcase",
                "confidence": {"$gte": 0.01}
            }
        }
    }
    yolos = db.find(query,fields=["_id"])
    print(f"matched: {len(yolos)}")

    query = {
        "_id": {
            "$gte": "2026-09-29:field_all:090000.000000000JST",
            "$lt":  "2026-09-29:field_all:170000.000000000JST",
            "$regex": ":pcd:",
        },
        "detections": {
            "$elemMatch": {
                "class_name": "suitcase",
                "confidence": {"$gte": 0.01}
            }
        }
    }
    pcds = db.find(query,fields=["_id","_attachments"])
    print(f"matched: {len(pcds)}")

    gnss = []
    for yolo in yolos:
        gn_id = yolo["_id"].split(":yolo:")[0]+":gnss:baselink"
        gn = db.get(gn_id)

        if gn is not None:
            gnss.append(gn)
        # capture = db.ancestor(yolo, levels=3)
        # if capture is None: continue
        # pcd_id = f"{capture['_id']}:pcd"
        # pcds.extend(
        #     db.children(pcd_id)
        # )

    # for doc in result:
    #     pprint(doc)

