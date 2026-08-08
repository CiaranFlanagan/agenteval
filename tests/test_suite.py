import pytest

from agenteval.runner import Suite


def write(tmp_path, body):
    path = tmp_path / "suite.yaml"
    path.write_text(body)
    return path


def test_loads_cases(tmp_path):
    suite = Suite.load(
        write(
            tmp_path,
            """
name: demo
cases:
  - id: one
    input: {question: "why?"}
    expected: "because"
  - id: two
    expected: "other"
""",
        )
    )

    assert suite.name == "demo"
    assert len(suite.cases) == 2
    assert suite.expected == {"one": "because", "two": "other"}
    assert suite.cases[1].input == {}


def test_name_defaults_to_filename(tmp_path):
    suite = Suite.load(write(tmp_path, "cases:\n  - id: a\n    expected: b\n"))
    assert suite.name == "suite"


@pytest.mark.parametrize(
    "body,message",
    [
        ("cases:\n  - expected: b\n", "missing"),
        ("cases:\n  - id: a\n", "missing"),
        ("cases:\n  - id: a\n    expected: b\n  - id: a\n    expected: c\n", "duplicate"),
        ("name: nope\n", "must be a mapping"),
        ("[]\n", "must be a mapping"),
    ],
)
def test_rejects_malformed_suites(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        Suite.load(write(tmp_path, body))
