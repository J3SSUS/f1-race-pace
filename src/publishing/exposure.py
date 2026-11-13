"""The control of exposure: look at the files that would be published and say which rule each one breaks.

A `Finding` carries the file and the rule, never the text that matched: a leaked key must not be copied into a log by
the very check that is looking for it. Every file is read as text only when its type is a text type; the control does
not guess about types it does not know, it flags them.
"""

import base64
import binascii
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from common.supabase_tables import PRIVATE_TABLES
from publishing.snapshot import check_columns

RUN_ID = re.compile(r"\b[0-9a-f]{32}\b")  # an MLflow run id
SECRET_KEY = re.compile(r"sb_secret_[\w-]+")  # a newer Supabase secret key, which has no role to decode
JWT = re.compile(r"eyJ[\w-]+\.([\w-]+)\.[\w-]+")  # a JSON Web Token; group 1 is its payload
PRIVATE_TABLE = re.compile(r"\b(?:" + "|".join(PRIVATE_TABLES) + r")\b")
ALLOWED_TYPES = {".html", ".css", ".js", ".json", ".svg", ".woff2", ".txt", ".md"}  # plus the file `.gitkeep`
BINARY_TYPES = {".woff2"}  # allowed, but not text: never searched for strings
SNAPSHOT_NAME = "snapshot.json"

# The order the rules are reported in.
RULES = ("file-type", "service-key", "service-key-name", "private-table", "mlflow-run-id", "snapshot-columns")


class PublishedFilesError(RuntimeError):
    """The list of published files could not be made. The message names the failure, never git's own output."""


@dataclass(frozen=True)
class Finding:
    path: str  # relative to the repository root, with forward slashes
    rule: str  # one of RULES


def published_files(root: Path) -> list[Path]:
    """The files under `site/` that git would publish: tracked files and new files that are not ignored.

    Paths are relative to `root` and sorted. A tracked file already deleted from the disk is left out (the next
    commit removes it, it publishes nothing). It fails closed: if `git` cannot run, or no file is left to check, it
    raises, because an empty list would read as "nothing found".
    """
    command = ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "site"]
    try:
        listing = subprocess.run(command, capture_output=True, check=True).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise PublishedFilesError(f"git ls-files failed: {type(error).__name__}") from None
    names = {name for name in listing.decode("utf-8", "surrogateescape").split("\0") if name}
    files = sorted(Path(name) for name in names if (root / name).is_file())
    if not files:
        raise PublishedFilesError("no file under site/ is listed as published")
    return files


def jwt_roles(text: str) -> list[str | None]:
    """The `role` of every JWT in `text`; None when a token's payload cannot be decoded or has no role (fail closed)."""
    roles: list[str | None] = []
    for match in JWT.finditer(text):
        payload = match.group(1)
        try:
            claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        except (binascii.Error, ValueError):  # bad base64, bad UTF-8 and bad JSON are all ValueErrors
            roles.append(None)
            continue
        roles.append(claims.get("role") if isinstance(claims, dict) else None)
    return roles


def text_rules(text: str) -> set[str]:
    """The rules a text breaks, apart from the file type and the snapshot columns."""
    broken = set()
    if any(role != "anon" for role in jwt_roles(text)) or SECRET_KEY.search(text) or "service_role" in text:
        broken.add("service-key")
    if "SUPABASE_SERVICE_KEY" in text:
        broken.add("service-key-name")
    if PRIVATE_TABLE.search(text):
        broken.add("private-table")
    if RUN_ID.search(text):
        broken.add("mlflow-run-id")
    return broken


def snapshot_columns_broken(text: str) -> bool:
    """True when a `snapshot.json` is not valid JSON or has a key or column the page does not read."""
    try:
        snapshot = json.loads(text)
    except ValueError:
        return True
    return not isinstance(snapshot, dict) or bool(check_columns(snapshot))


def check_files(root: Path, files: list[Path]) -> list[Finding]:
    """The findings over `files` (paths relative to `root`, or absolute under it), sorted by file and rule.

    A file that cannot be read raises instead of passing: an unchecked file is not a clean one.
    """
    findings: list[Finding] = []
    for file in files:
        path = root / file
        relative = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        if suffix not in ALLOWED_TYPES and path.name != ".gitkeep":
            findings.append(Finding(relative, "file-type"))
            continue
        if suffix in BINARY_TYPES or path.name == ".gitkeep":
            continue
        text = path.read_bytes().decode("utf-8", "replace")
        broken = text_rules(text)
        if path.name == SNAPSHOT_NAME and snapshot_columns_broken(text):
            broken.add("snapshot-columns")
        findings += [Finding(relative, rule) for rule in RULES if rule in broken]
    return sorted(findings, key=lambda f: (f.path, RULES.index(f.rule)))
