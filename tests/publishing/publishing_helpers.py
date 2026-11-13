"""Fakes shared by the tests of `src/publishing/`."""


class FakeQuery:
    def __init__(self, client, table):
        self.client = client
        self.table = table
        self.calls = []

    def select(self, columns):
        self.calls.append(("select", columns))
        return self

    def eq(self, column, value):
        self.calls.append(("eq", column, value))
        return self

    def is_(self, column, value):
        self.calls.append(("is", column, value))
        return self

    def order(self, column):
        self.calls.append(("order", column))
        return self

    def range(self, start, end):
        self.calls.append(("range", start, end))
        return self

    def execute(self):
        self.client.queries.append((self.table, self.calls))
        if self.client.error:
            raise self.client.error
        start, end = self.calls[-1][1], self.calls[-1][2]
        return type("Response", (), {"data": self.client.tables[self.table][start : end + 1]})()


class FakeClient:
    """Serves each table from a list, honouring `range`, and records every query it receives."""

    def __init__(self, tables, error=None):
        self.tables = tables
        self.error = error
        self.queries = []

    def table(self, name):
        return FakeQuery(self, name)


def golden_client(raw, **kwargs):
    tables = {
        "champion_history": [r for r in raw["championHistory"] if r["demoted_at"] is None],
        "model_versions": raw["modelVersions"],
        "races": raw["races"],
        "metrics": raw["metrics"],
        "prediction_results": raw["results"],
    }
    return FakeClient(tables, **kwargs)
