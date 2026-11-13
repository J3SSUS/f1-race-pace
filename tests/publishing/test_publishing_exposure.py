import base64
import json
from pathlib import Path

import pytest

from publishing.exposure import Finding, check_files

GOLDEN = Path(__file__).resolve().parents[1] / "site" / "golden"
HEX32 = "0123456789abcdef0123456789abcdef"


def b64(data: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()


def jwt(role):
    return f"{b64({'alg': 'HS256', 'typ': 'JWT'})}.{b64({'role': role, 'ref': 'abc'})}.signature123"


ANON_JWT = jwt("anon")
SERVICE_JWT = jwt("service_role")
PUBLISHABLE = "sb_publishable_" + "x" * 30
SECRET = "sb_secret_" + "y" * 30


def put(root, name, content):
    path = root / "site" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return path.relative_to(root)


def rules(root, name, content):
    return [f.rule for f in check_files(root, [put(root, name, content)])]


@pytest.mark.parametrize(
    "content, rule",
    [
        (f"const KEY = '{SERVICE_JWT}';", "service-key"),
        (f"const KEY = '{jwt('authenticated')}';", "service-key"),
        (f"const KEY = '{SECRET}';", "service-key"),
        ("// uses the service_role key", "service-key"),
        ("process.env.SUPABASE_SERVICE_KEY", "service-key-name"),
        ("fetch('/rest/v1/drift_reports')", "private-table"),
        ("fetch('/rest/v1/feature_set_items')", "private-table"),
        (f"Sampling run: {HEX32}", "mlflow-run-id"),
    ],
)
def test_each_rule_fires_on_a_file_made_for_it(tmp_path, content, rule):
    assert rules(tmp_path, "page.js", content) == [rule]


def test_a_jwt_that_cannot_be_decoded_fails_closed(tmp_path):
    not_json = "eyJhbGciOiJIUzI1NiJ9.AAAAAAAA.sig"  # a payload with valid characters that is not JSON
    assert rules(tmp_path, "page.js", not_json) == ["service-key"]
    assert rules(tmp_path, "page.js", f"eyJhbGciOiJIUzI1NiJ9.{b64({'no': 'role'})}.sig") == ["service-key"]


def test_a_file_type_that_is_not_allowed_is_flagged_and_not_read(tmp_path):
    for name in ("data.parquet", "model.pkl", ".env", "notes", "dump.sql", "archive.zip"):
        assert rules(tmp_path, name, f"{SECRET} {HEX32}") == ["file-type"], name


def test_a_snapshot_with_a_column_that_is_not_allowed_is_flagged(tmp_path):
    good = json.loads((GOLDEN / "expected_snapshot.json").read_text())
    assert rules(tmp_path, "snapshot.json", json.dumps(good)) == []
    bad = json.loads(json.dumps(good))
    bad["results"][0]["model_version_id"] = 4
    assert rules(tmp_path, "snapshot.json", json.dumps(bad)) == ["snapshot-columns"]
    assert rules(tmp_path, "snapshot.json", "{not json") == ["snapshot-columns"]
    assert rules(tmp_path, "snapshot.json", "[1, 2]") == ["snapshot-columns"]


def test_a_json_that_is_not_the_snapshot_is_not_checked_for_columns(tmp_path):
    assert rules(tmp_path, "charts/other.json", '{"anything": 1}') == []


def test_a_finding_carries_the_file_and_the_rule_and_never_the_matched_text(tmp_path):
    content = f"{SERVICE_JWT} {SECRET} {HEX32} drift_reports SUPABASE_SERVICE_KEY"
    findings = check_files(tmp_path, [put(tmp_path, "page.js", content)])
    assert [f.rule for f in findings] == ["service-key", "service-key-name", "private-table", "mlflow-run-id"]
    assert all(f.path == "site/page.js" for f in findings)
    shown = " ".join(repr(f) for f in findings) + " ".join(str(vars(f)) for f in findings)
    for secret in (SERVICE_JWT, SECRET, HEX32, SERVICE_JWT.split(".")[1]):
        assert secret not in shown


def test_the_anon_key_in_either_format_the_allowed_types_and_gitkeep_pass(tmp_path):
    files = [
        put(tmp_path, "a.js", f"const A = '{ANON_JWT}'; const B = '{PUBLISHABLE}';"),
        put(tmp_path, "index.html", "<html></html>"),
        put(tmp_path, "site.css", "body{}"),
        put(tmp_path, "charts/plot.svg", "<svg/>"),
        put(tmp_path, "README.txt", "hello"),
        put(tmp_path, "NOTES.md", "# notes"),
        put(tmp_path, "charts/.gitkeep", ""),
    ]
    assert check_files(tmp_path, files) == []


def test_a_woff2_is_allowed_and_not_read_as_text(tmp_path):
    font = b"wOF2\x00\x01" + SERVICE_JWT.encode() + HEX32.encode() + b"service_role" + b"\xff\xfe\x00"
    assert check_files(tmp_path, [put(tmp_path, "fonts/a.woff2", font)]) == []


def test_findings_are_sorted_by_file_then_rule_and_absolute_paths_work(tmp_path):
    b = put(tmp_path, "b.js", "drift_reports")
    a = put(tmp_path, "a.js", f"{HEX32} service_role")
    found = check_files(tmp_path, [b, tmp_path / a])
    assert found == [
        Finding("site/a.js", "service-key"),
        Finding("site/a.js", "mlflow-run-id"),
        Finding("site/b.js", "private-table"),
    ]


def test_a_file_that_cannot_be_read_raises_instead_of_passing(tmp_path):
    (tmp_path / "site").mkdir()
    with pytest.raises(FileNotFoundError):
        check_files(tmp_path, [tmp_path / "site" / "missing.js"])


def test_a_file_outside_the_root_raises(tmp_path):
    outside = tmp_path.parent / "outside.js"
    outside.write_text("x")
    with pytest.raises(ValueError):
        check_files(tmp_path, [outside])
