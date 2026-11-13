import subprocess
from pathlib import Path

import pytest

from publishing.exposure import PublishedFilesError, check_files, published_files


def git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def write(root, name, content="x"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    write(tmp_path, "site/index.html")
    write(tmp_path, "site/site.css")
    write(tmp_path, "README.md")
    write(tmp_path, "src/code.py")
    git(tmp_path, "add", "site/index.html", "README.md", "src/code.py")  # site.css stays new, not added
    return tmp_path


def test_it_lists_tracked_files_and_new_files_that_are_not_ignored_under_site(repo):
    assert published_files(repo) == [Path("site/index.html"), Path("site/site.css")]


def test_it_skips_ignored_files_and_files_outside_site(repo):
    write(repo, ".gitignore", "site/charts/explain.json\n*.log\n")
    write(repo, "site/charts/explain.json", "private")
    write(repo, "site/debug.log", "private")
    write(repo, "notes.txt")
    assert published_files(repo) == [Path("site/index.html"), Path("site/site.css")]


def test_a_file_that_is_tracked_and_ignored_later_is_still_listed(repo):
    write(repo, "site/charts/explain.json", "private")
    git(repo, "add", "site/charts/explain.json")
    write(repo, ".gitignore", "site/charts/explain.json\n")
    assert Path("site/charts/explain.json") in published_files(repo)
    git(repo, "rm", "--cached", "-q", "site/charts/explain.json")  # what step 12 does
    assert Path("site/charts/explain.json") not in published_files(repo)
    assert (repo / "site/charts/explain.json").exists()  # still on the disk


def test_a_tracked_file_deleted_from_the_disk_is_left_out(repo):
    git(repo, "add", "site/site.css")
    (repo / "site/site.css").unlink()
    assert published_files(repo) == [Path("site/index.html")]


def test_names_with_spaces_and_accents_come_back_intact(repo):
    write(repo, "site/charts/são paulo.svg", "<svg/>")
    assert Path("site/charts/são paulo.svg") in published_files(repo)


def test_it_raises_when_git_fails(tmp_path):
    write(tmp_path, "site/index.html")  # a folder that is not a repository
    with pytest.raises(PublishedFilesError, match="git ls-files failed: CalledProcessError"):
        published_files(tmp_path)


def test_it_raises_when_git_cannot_run(repo, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git: /secret/path")

    monkeypatch.setattr("publishing.exposure.subprocess.run", missing)
    with pytest.raises(PublishedFilesError) as error:
        published_files(repo)
    assert str(error.value) == "git ls-files failed: FileNotFoundError"
    assert "/secret/path" not in str(error.value)


def test_it_raises_instead_of_returning_an_empty_list(tmp_path):
    git(tmp_path, "init", "-q")
    write(tmp_path, "README.md")
    with pytest.raises(PublishedFilesError, match="no file under site/"):
        published_files(tmp_path)  # no site/ at all
    write(tmp_path, "site/a.log")
    write(tmp_path, ".gitignore", "*.log\n")
    with pytest.raises(PublishedFilesError, match="no file under site/"):
        published_files(tmp_path)  # everything under site/ is ignored


def test_the_listing_feeds_the_control(repo):
    write(repo, "site/leak.js", "SUPABASE_SERVICE_KEY")
    findings = check_files(repo, published_files(repo))
    assert [(f.path, f.rule) for f in findings] == [("site/leak.js", "service-key-name")]
