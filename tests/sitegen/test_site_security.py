"""What the built site must never contain: the service key, a request to anyone but Supabase, a private table."""

import base64
import hashlib
import json
import re

import pytest

from sitegen.build import SOURCE_DIR, build_site
from sitegen.inputs import ROOT, read_inputs

SUPABASE_URL = "https://abc.supabase.co"


def jwt(role):
    """A Supabase-style key: three base64url parts, the middle one naming the role."""

    def part(data):
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'HS256', 'typ': 'JWT'})}.{part({'role': role, 'ref': 'abc'})}.{part({'sig': role})[:20]}"


ANON_KEY = jwt("anon")
SERVICE_KEY = jwt("service_role")
# Supabase's newer keys are not JWTs: a publishable (anon) key and a secret (service) key with no role to decode.
PUBLISHABLE_KEY = "sb_publishable_" + "x" * 30
SECRET_KEY = "sb_secret_" + "y" * 30
SECRET_KEY_FORMAT = re.compile(r"sb_secret_[\w-]+")
KEYS = {"jwt": (ANON_KEY, SERVICE_KEY), "new": (PUBLISHABLE_KEY, SECRET_KEY)}  # (anon, service) in each format
PAGES = ("index.html", "model.html", "site.css", "site.js")  # what a browser loads
# The namespace of the SVG elements the script creates: an identifier, not a request.
ALLOWED_URLS = {SUPABASE_URL, "http://www.w3.org/2000/svg"}


@pytest.fixture(params=sorted(KEYS))
def keys(request):
    """The (anon, service) keys of the project, in each of the two formats Supabase issues."""
    return KEYS[request.param]


@pytest.fixture
def built(root, tmp_path, monkeypatch, keys):
    """The site built from the real sources, with the service key in the environment and in `.env`."""
    anon, service = keys
    monkeypatch.setenv("SUPABASE_URL", SUPABASE_URL)
    monkeypatch.setenv("SUPABASE_ANON_KEY", anon)
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", service)
    (root / ".env").write_text(f"SUPABASE_SERVICE_KEY={service}\n")
    out = tmp_path / "out"
    build_site(read_inputs(root), out)
    return out


def files(directory):
    return [p for p in sorted(directory.rglob("*")) if p.is_file()]


def jwt_roles(text):
    roles = []
    for match in re.finditer(r"eyJ[\w-]+\.([\w-]+)\.[\w-]+", text):
        payload = match.group(1)
        roles.append(json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))).get("role"))
    return roles


def test_the_service_key_is_in_no_file_of_the_built_site(built, keys):
    _, service = keys
    needles = [service] + (["".join(service.split(".")[1:2])] if service.count(".") == 2 else [])  # a JWT's payload too
    assert files(built)
    for path in files(built):
        for needle in needles:
            assert needle.encode() not in path.read_bytes(), path.name


def test_the_name_of_the_service_key_variable_is_in_no_file_of_the_built_site(built):
    for path in files(built):
        assert b"SUPABASE_SERVICE_KEY" not in path.read_bytes(), path.name
        assert b"service_role" not in path.read_bytes(), path.name
        assert not SECRET_KEY_FORMAT.search(path.read_bytes().decode("utf-8", "ignore")), path.name


def test_the_only_key_in_the_site_is_the_anon_key(built, keys):
    anon, _ = keys
    found = {role for name in PAGES for role in jwt_roles((built / name).read_text())}
    assert found <= {"anon"}  # empty for the newer format, which is not a JWT
    assert anon in (built / "site.js").read_text()
    assert not any(SECRET_KEY_FORMAT.search((built / name).read_text()) for name in PAGES)


def test_no_page_references_a_url_other_than_supabase_or_the_svg_namespace(built):
    for name in PAGES:
        urls = set(re.findall(r"""(?:https?:)?//[^\s"'<>)`]+""", (built / name).read_text()))
        stray = {u for u in urls if not any(u == allowed or u.startswith(allowed + "/") for allowed in ALLOWED_URLS)}
        assert not stray, f"{name}: {sorted(stray)}"


def test_the_pages_load_only_files_next_to_them(built):
    html = (built / "index.html").read_text() + (built / "model.html").read_text()
    loaded = re.findall(r"""<(?:script|link|img|source|iframe)\b[^>]*?\b(?:src|href)="([^"]+)""", html)
    assert loaded
    for url in loaded:
        assert not re.match(r"(?:[a-z]+:)?//", url), url
        # the charts are written by the explain step into site/ of the repository, not by the build
        assert (built / url).is_file() or (ROOT / "site" / url).is_file(), url
    assert not re.search(r"@import|<iframe|<form", html + (built / "site.css").read_text())


def test_the_script_reads_only_the_public_tables(built):
    js = (built / "site.js").read_text()
    read = set(re.findall(r"""read\(\s*"([a-z_]+)"\s*,""", js))
    assert read == {"champion_history", "model_versions", "races", "metrics", "prediction_results"}
    for private in (
        "prediction_runs",
        "feature_values",
        "feature_sets",
        "feature_set_items",
        "drift_reports",
        "drift_feature_results",
    ):
        assert private not in js, private


def test_the_build_changes_none_of_its_inputs(root, tmp_path):
    inputs = [root / "MODEL_CARD.md", root / "site" / "charts" / "explain.json", root / "config" / "team_colors.json"]
    inputs += [p for p in files(SOURCE_DIR)]

    def digest():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}

    before = digest()
    build_site(read_inputs(root, {"SUPABASE_URL": SUPABASE_URL, "SUPABASE_ANON_KEY": ANON_KEY}), tmp_path / "out")
    assert digest() == before
    assert not list(root.glob("*.html"))  # nothing was written beside the inputs


def test_the_committed_site_holds_no_service_key():
    """The real `site/` of the repository, if it has been built: the only key in it is an anon key."""
    site = ROOT / "site"
    if not (site / "site.js").is_file():
        pytest.skip("site/ has not been built")
    for name in PAGES:
        text = (site / name).read_text()
        assert "service_role" not in text, name
        assert "SUPABASE_SERVICE_KEY" not in text, name
        assert not SECRET_KEY_FORMAT.search(text), f"{name} has a key in the format of a Supabase secret key"
        assert set(jwt_roles(text)) <= {"anon"}, name
    js = (site / "site.js").read_text()
    # The page needs a key to read Supabase: an anon JWT or a publishable key. Without this the check could pass empty.
    assert "sb_publishable_" in js or "anon" in {role for role in jwt_roles(js)}
