import pytest

from common.supabase_io import BATCH_SIZE, get_client, upsert_rows

SECRET = "service-key-that-must-not-leak"
ENV = {"SUPABASE_URL": "https://example.supabase.co", "SUPABASE_SERVICE_KEY": SECRET}


class FakeClient:
    """Records every upsert; stores rows with a growing id, like a table with a serial key."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on
        self.next_id = 1

    def table(self, name):
        self.name = name
        return self

    def upsert(self, rows, on_conflict):
        self.calls.append((self.name, list(rows), on_conflict))
        self.pending = rows
        return self

    def execute(self):
        if self.fail_on == len(self.calls):
            raise ConnectionError(f"boom with key {SECRET}")
        data = []
        for row in self.pending:
            data.append({"id": self.next_id, **row})
            self.next_id += 1
        return type("Response", (), {"data": data})()


def test_batches_of_at_most_1000_rows():
    client = FakeClient()
    rows = [{"code": str(i)} for i in range(2 * BATCH_SIZE + 1)]
    stored = upsert_rows(client, "drivers", rows, "code")
    assert [len(batch) for _, batch, _ in client.calls] == [1000, 1000, 1]
    assert len(stored) == len(rows)


def test_passes_table_and_on_conflict_and_returns_ids():
    client = FakeClient()
    stored = upsert_rows(client, "races", [{"year": 2024, "round": 8}, {"year": 2024, "round": 9}], "year,round")
    assert client.calls == [("races", [{"year": 2024, "round": 8}, {"year": 2024, "round": 9}], "year,round")]
    assert [row["id"] for row in stored] == [1, 2]
    assert stored[0]["round"] == 8


def test_no_rows_sends_nothing():
    client = FakeClient()
    assert upsert_rows(client, "teams", [], "name") == []
    assert client.calls == []


def test_error_names_table_and_key_columns_but_not_the_key():
    with pytest.raises(RuntimeError) as error:
        upsert_rows(FakeClient(fail_on=1), "race_entries", [{"race_id": 1}], "race_id,driver_id")
    message = str(error.value)
    assert "race_entries" in message and "race_id,driver_id" in message
    assert SECRET not in message and SECRET not in repr(error.value.__cause__)


def test_get_client_builds_with_url_and_key():
    seen = []
    client = get_client(ENV, create=lambda url, key: seen.append((url, key)) or "client")
    assert client == "client"
    assert seen == [("https://example.supabase.co", SECRET)]


@pytest.mark.parametrize("missing", ["SUPABASE_URL", "SUPABASE_SERVICE_KEY"])
def test_missing_variable_gives_a_clear_error(missing):
    env = {k: v for k, v in ENV.items() if k != missing}
    with pytest.raises(RuntimeError, match=missing) as error:
        get_client(env, create=lambda *a: pytest.fail("must not create a client"))
    assert SECRET not in str(error.value)


def test_get_client_reads_the_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(f"SUPABASE_URL={ENV['SUPABASE_URL']}\nSUPABASE_SERVICE_KEY={SECRET}\n")
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    seen = []
    get_client(env_file=env_file, create=lambda url, key: seen.append((url, key)))
    assert seen == [(ENV["SUPABASE_URL"], SECRET)]
