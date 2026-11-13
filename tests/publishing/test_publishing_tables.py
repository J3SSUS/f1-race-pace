from postgrest.exceptions import APIError

from common.supabase_tables import PRIVATE_TABLES, PUBLIC_TABLES, can_select


class FakeTable:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def select(self, columns):
        self.calls.append(("select", columns))
        return self

    def limit(self, n):
        self.calls.append(("limit", n))
        return self

    def execute(self):
        if self.error:
            raise self.error
        return object()


class FakeClient:
    def __init__(self, error=None):
        self.error = error
        self.tables = []

    def table(self, name):
        self.tables.append(name)
        self.last = FakeTable(self.error)
        return self.last


def permission_error():
    return APIError({"message": "permission denied", "code": "42501", "hint": None, "details": None})


def test_can_select_is_true_when_the_read_works():
    client = FakeClient()
    assert can_select(client, "races") is True
    assert client.tables == ["races"]
    assert client.last.calls == [("select", "*"), ("limit", 1)]


def test_can_select_is_false_on_a_permission_error():
    assert can_select(FakeClient(error=permission_error()), "drift_reports") is False


def test_the_lists_do_not_overlap_and_hold_the_expected_tables():
    assert not set(PUBLIC_TABLES) & set(PRIVATE_TABLES)
    assert len(PRIVATE_TABLES) == 6
    assert "prediction_results" in PUBLIC_TABLES
    assert {"feature_sets", "feature_values", "drift_reports"} <= set(PRIVATE_TABLES)
