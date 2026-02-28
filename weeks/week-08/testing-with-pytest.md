# Testing with pytest

> pytest is Python's de facto testing framework — its plain `assert` statements, automatic test discovery, and revolutionary fixture system make writing and maintaining tests dramatically simpler than the stdlib's unittest.

## Table of Contents
- [Core Concepts](#core-concepts)
- [Code Examples](#code-examples)
- [Common Pitfalls](#common-pitfalls)
- [Key Takeaways](#key-takeaways)
- [Exercises](#exercises)

## Core Concepts

### Why pytest Over unittest

#### What

Python ships with `unittest` in the standard library, which follows the xUnit pattern from Java: you create classes that inherit from `unittest.TestCase`, define methods named `test_*`, and use assertion methods like `self.assertEqual(a, b)`. pytest takes a radically different approach: plain functions, plain `assert` statements, and a dependency injection system called fixtures.

#### How

The contrast is stark:

```python
# unittest style — verbose, class-based, custom assertion methods
import unittest

class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = Calculator()

    def test_add(self):
        self.assertEqual(self.calc.add(2, 3), 5)

    def test_divide_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            self.calc.divide(1, 0)


# pytest style — minimal, function-based, plain assert
import pytest

def test_add():
    calc = Calculator()
    assert calc.add(2, 3) == 5

def test_divide_by_zero():
    calc = Calculator()
    with pytest.raises(ZeroDivisionError):
        calc.divide(1, 0)
```

pytest can also run unittest-style tests, so migration is incremental — you do not have to rewrite existing tests.

#### Why It Matters

Plain `assert` is the key insight. With unittest, `self.assertEqual(a, b)` gives you a generic failure message. With pytest, `assert a == b` triggers pytest's assertion introspection engine, which rewrites the AST to show you exactly what `a` and `b` were. You get better error messages with less code. The removal of mandatory classes means tests are lighter — you write a function, not a method on a class. This reduces friction, which means more tests get written.

### Test Discovery

#### What

pytest automatically finds and runs your tests without any configuration. It walks your project directory, finds files matching certain patterns, and collects test functions and classes within those files. Understanding the discovery rules means you never have to manually register tests.

#### How

pytest's discovery rules:

1. **Files**: Searches for files named `test_*.py` or `*_test.py`
2. **Functions**: Collects functions named `test_*`
3. **Classes**: Collects classes named `Test*` (no `TestCase` inheritance needed)
4. **Methods**: Within `Test*` classes, collects methods named `test_*`

```bash
# Run all tests (discovers automatically)
pytest

# Run tests in a specific directory
pytest tests/

# Run tests in a specific file
pytest tests/test_users.py

# Run a specific test function
pytest tests/test_users.py::test_create_user

# Run a specific test method on a class
pytest tests/test_users.py::TestUserService::test_create_user

# Run tests matching a keyword expression
pytest -k "create and not delete"

# Verbose output (show each test name)
pytest -v

# Show print() output (not captured)
pytest -s

# Stop on first failure
pytest -x

# Run last failed tests first
pytest --lf
```

#### Why It Matters

Convention over configuration eliminates boilerplate. You do not need test runners, test suites, or registration — just name your files and functions correctly and pytest finds them. The `-k` flag is particularly powerful for large test suites: `pytest -k "api and not slow"` runs only API tests that are not marked as slow, without needing to remember file paths. The `--lf` flag is essential during development — it reruns only the tests that failed last time, giving you a tight feedback loop.

### Assert Statements and Introspection

#### What

pytest uses Python's built-in `assert` statement instead of custom assertion methods. When an assertion fails, pytest's assertion rewriting engine intercepts the failure and produces a detailed comparison showing the actual values on both sides of the expression. This is called assertion introspection.

#### How

```python
def test_string_comparison() -> None:
    result = "hello world"
    assert result == "hello World"
    # Output:
    # AssertionError: assert 'hello world' == 'hello World'
    #   - hello World
    #   + hello world
    #          ^  (shows exact character difference)

def test_list_comparison() -> None:
    actual = [1, 2, 3, 4, 5]
    expected = [1, 2, 3, 4, 6]
    assert actual == expected
    # Output:
    # AssertionError: assert [1, 2, 3, 4, 5] == [1, 2, 3, 4, 6]
    #   At index 4 diff: 5 != 6

def test_dict_comparison() -> None:
    actual = {"name": "Alice", "role": "user", "active": True}
    expected = {"name": "Alice", "role": "admin", "active": True}
    assert actual == expected
    # Output shows exact key where values differ:
    # E   {'active': True, 'name': 'Alice', 'role': 'user'}
    # E   {'active': True, 'name': 'Alice', 'role': 'admin'}

def test_exception_raised() -> None:
    import pytest
    with pytest.raises(ValueError, match="must be positive"):
        validate_age(-5)
    # match= accepts a regex pattern checked against the exception message

def test_approximate_float() -> None:
    result = 0.1 + 0.2
    assert result == pytest.approx(0.3)
    # pytest.approx handles floating-point comparison
    # Default tolerance: 1e-6 relative
```

#### Why It Matters

Assertion introspection is what makes plain `assert` viable. Without it, `assert a == b` would just say `AssertionError` with no context. With it, you see exactly what `a` and `b` were, where they differ, and for containers, which specific element is wrong. This is better than `assertEqual` because it works with any expression — `assert len(items) > 5`, `assert "error" in message`, `assert all(x > 0 for x in values)` — and pytest shows the intermediate values for all of them.

### Fixtures — Dependency Injection for Tests

#### What

Fixtures are pytest's mechanism for providing test dependencies. Instead of setUp/tearDown methods, you declare what a test needs by adding parameters to the test function. pytest inspects the parameter names, finds matching fixture functions, calls them, and injects the results. This is dependency injection — the same pattern used by FastAPI's `Depends()` and Spring's `@Autowired`.

#### How

```python
import pytest
from myapp.database import Database, User


@pytest.fixture
def db() -> Database:
    """Provide a clean database connection for each test."""
    database = Database(":memory:")
    database.create_tables()
    return database


@pytest.fixture
def sample_user(db: Database) -> User:
    """Provide a pre-created user (depends on db fixture)."""
    user = User(name="Alice", email="alice@example.com")
    db.insert(user)
    return user


# Tests declare dependencies by parameter name — pytest injects them
def test_insert_user(db: Database) -> None:
    user = User(name="Bob", email="bob@example.com")
    db.insert(user)
    assert db.count("users") == 1


def test_find_user(db: Database, sample_user: User) -> None:
    # Both db and sample_user are injected automatically
    found = db.find_by_email("alice@example.com")
    assert found is not None
    assert found.name == "Alice"
```

The fixture dependency graph is resolved automatically:

```mermaid
graph TD
    A[test_find_user] -->|requests| B[sample_user fixture]
    A -->|requests| C[db fixture]
    B -->|depends on| C
    C -->|creates| D[Database ':memory:']
    B -->|creates| E[User 'Alice']
```

When `test_find_user` runs, pytest sees it needs `db` and `sample_user`. It calls `db()` first (because `sample_user` depends on it), then passes the database to `sample_user()`, and finally passes both to the test function. All of this is automatic — you just declare what you need.

#### Why It Matters

Fixtures solve the fundamental problem of test setup: how do you share setup logic across tests without coupling them together? With setUp/tearDown, every test in a class shares the same setup — you cannot easily compose different setups for different tests. With fixtures, each test declares exactly what it needs. Two tests in the same file can use completely different fixtures. Fixtures can depend on other fixtures, forming a graph that pytest resolves automatically. This composability is what makes large test suites maintainable.

### Fixture Scope

#### What

By default, fixtures run once per test function (scope `"function"`). But some fixtures are expensive to create — a database connection, a Docker container, a large test dataset. Fixture scope controls how often the fixture is created and destroyed: per function, per class, per module, or per session.

#### How

```python
import pytest
from myapp.database import Database


@pytest.fixture(scope="function")  # Default — fresh for every test
def db() -> Database:
    """New database for each test. Slow but fully isolated."""
    database = Database(":memory:")
    database.create_tables()
    return database


@pytest.fixture(scope="module")
def shared_db() -> Database:
    """One database per test file. Tests in this file share it."""
    database = Database(":memory:")
    database.create_tables()
    return database


@pytest.fixture(scope="session")
def app_config() -> dict[str, str]:
    """One config for the entire test session. Created once, never destroyed."""
    return {
        "db_url": "sqlite:///:memory:",
        "log_level": "DEBUG",
        "api_key": "test-key-12345",
    }
```

Scope hierarchy (from narrowest to widest):

| Scope | Lifetime | Use Case |
|-------|----------|----------|
| `function` | One test | Mutable state that must be isolated |
| `class` | All tests in a class | Shared setup within a test class |
| `module` | All tests in a file | Expensive setup shared per file |
| `session` | Entire test run | Global resources (config, connections) |

A critical rule: a fixture can only depend on fixtures of the same or wider scope. A `function`-scoped fixture can use a `session`-scoped fixture, but not the other way around.

#### Why It Matters

Scope is the lever for balancing test speed against test isolation. Session-scoped database fixtures make test suites 10x faster because you create the schema once instead of per test. But they also mean tests can interfere with each other through shared state. The right strategy is usually: session scope for immutable resources (config, schema creation), module scope for moderately expensive setup, and function scope for any mutable state that tests modify.

### Yield Fixtures — Setup and Teardown

#### What

Yield fixtures use Python's `yield` statement to split a fixture into setup (before yield) and teardown (after yield). Everything before `yield` runs before the test, the yielded value is injected into the test, and everything after `yield` runs after the test completes — even if the test fails or raises an exception.

#### How

```python
import pytest
from pathlib import Path
from myapp.database import Database


@pytest.fixture
def db() -> Database:
    """Database with automatic cleanup after each test."""
    database = Database(":memory:")
    database.create_tables()
    yield database                   # <-- test runs here
    database.close()                 # Teardown: always runs


@pytest.fixture
def temp_config(tmp_path: Path) -> Path:
    """Create a temporary config file, clean up after test."""
    config_file = tmp_path / "config.json"
    config_file.write_text('{"debug": true}')
    yield config_file                # <-- test runs here
    # tmp_path is cleaned up by pytest automatically,
    # but you could add explicit cleanup here if needed


@pytest.fixture(scope="session")
def docker_postgres():
    """Start a PostgreSQL container for the test session."""
    import subprocess
    container_id = subprocess.check_output([
        "docker", "run", "-d", "-p", "5433:5432",
        "-e", "POSTGRES_PASSWORD=test",
        "postgres:16-alpine"
    ]).decode().strip()

    # Wait for database to be ready
    import time
    time.sleep(2)

    yield f"postgresql://postgres:test@localhost:5433/postgres"

    # Teardown: stop and remove the container
    subprocess.run(["docker", "rm", "-f", container_id], check=True)
```

#### Why It Matters

Yield fixtures are Python's context manager pattern applied to testing. The `yield` is conceptually identical to `__enter__` returning a value, and the code after `yield` is `__exit__`. This guarantees cleanup regardless of test outcome — database connections get closed, temporary files get deleted, Docker containers get stopped. Without yield fixtures, teardown requires a separate `finalizer` function registered with `request.addfinalizer()`, which is harder to read and maintain.

### conftest.py — Shared Fixture Definitions

#### What

`conftest.py` is a special file that pytest loads automatically. Fixtures defined in `conftest.py` are available to all tests in the same directory and all subdirectories, without any import statement. You can have multiple `conftest.py` files at different levels of your test directory tree, each providing fixtures relevant to that scope.

#### How

```
project/
├── conftest.py              # Session-wide fixtures (db, config)
├── tests/
│   ├── conftest.py          # Shared test fixtures (sample data)
│   ├── unit/
│   │   ├── conftest.py      # Unit test-specific fixtures (mocks)
│   │   ├── test_models.py
│   │   └── test_services.py
│   ├── integration/
│   │   ├── conftest.py      # Integration fixtures (real db)
│   │   ├── test_api.py
│   │   └── test_database.py
│   └── e2e/
│       ├── conftest.py      # E2E fixtures (test client, auth)
│       └── test_workflows.py
```

```python
# tests/conftest.py — available to ALL tests
import pytest
from myapp.config import AppConfig


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    """Application config for the test environment."""
    return AppConfig(
        database_url="sqlite:///:memory:",
        debug=True,
        secret_key="test-secret",
    )


# tests/unit/conftest.py — available only to tests/unit/
import pytest
from unittest.mock import MagicMock
from myapp.database import DatabaseService


@pytest.fixture
def mock_db() -> MagicMock:
    """Mock database for unit tests (no real DB needed)."""
    return MagicMock(spec=DatabaseService)


# tests/integration/conftest.py — available only to tests/integration/
import pytest
from myapp.database import Database


@pytest.fixture(scope="module")
def real_db(app_config) -> Database:
    """Real database connection for integration tests."""
    db = Database(app_config.database_url)
    db.create_tables()
    yield db
    db.drop_tables()
    db.close()
```

Key rules for `conftest.py`:
- Never import from `conftest.py` — pytest discovers and loads it automatically
- Fixtures in `conftest.py` override those from parent directories (shadowing)
- Place fixtures at the narrowest scope that makes sense — do not put unit test mocks in the root conftest

#### Why It Matters

`conftest.py` is the mechanism for sharing fixtures without imports. This matters because it keeps test files focused on assertions rather than setup boilerplate. It also creates a natural hierarchy: root conftest has project-wide fixtures (config, auth), while subdirectory conftest files have specialized fixtures (mocks for unit tests, real connections for integration tests). This hierarchy mirrors the test organization itself and scales to large projects with hundreds of test files.

### Parametrize — Running Tests with Multiple Inputs

#### What

`@pytest.mark.parametrize` runs the same test function multiple times with different arguments. Instead of writing five nearly identical test functions that differ only in input/output, you write one function and provide a table of test cases. pytest generates a separate test for each set of parameters, with clear IDs in the output.

#### How

```python
import pytest


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        (1, "1"),
        (0, "0"),
        (-5, "-5"),
        (3.14, "3.14"),
        (True, "True"),
    ],
)
def test_str_conversion(input_value: object, expected: str) -> None:
    assert str(input_value) == expected


# With IDs for readable output
@pytest.mark.parametrize(
    ("email", "is_valid"),
    [
        ("user@example.com", True),
        ("user@domain.co.uk", True),
        ("", False),
        ("no-at-sign", False),
        ("@no-local-part.com", False),
        ("user@", False),
    ],
    ids=[
        "standard_email",
        "uk_domain",
        "empty_string",
        "missing_at",
        "missing_local",
        "missing_domain",
    ],
)
def test_email_validation(email: str, is_valid: bool) -> None:
    assert validate_email(email) == is_valid


# Stacking parametrize decorators creates a cross-product
@pytest.mark.parametrize("x", [1, 2, 3])
@pytest.mark.parametrize("y", [10, 20])
def test_multiplication_commutative(x: int, y: int) -> None:
    # Runs 6 times: (1,10), (1,20), (2,10), (2,20), (3,10), (3,20)
    assert x * y == y * x
```

Output with `-v`:
```
test_validation.py::test_email_validation[standard_email] PASSED
test_validation.py::test_email_validation[uk_domain] PASSED
test_validation.py::test_email_validation[empty_string] PASSED
test_validation.py::test_email_validation[missing_at] PASSED
test_validation.py::test_email_validation[missing_local] PASSED
test_validation.py::test_email_validation[missing_domain] PASSED
```

#### Why It Matters

Parametrize is the answer to the most common testing antipattern: copy-pasted tests that differ only in input values. Without parametrize, developers either write one test with multiple asserts (which stops at the first failure, hiding other failures) or copy-paste test functions (which creates maintenance burden). Parametrize gives you the best of both worlds: each parameter set is a separate test (so failures are independent), but the logic is written once (so maintenance is trivial).

### Markers — Categorizing and Filtering Tests

#### What

Markers are labels you attach to tests using `@pytest.mark.<name>`. They let you categorize tests and selectively run subsets. pytest includes built-in markers (`skip`, `skipif`, `xfail`, `parametrize`) and lets you define custom markers for your project.

#### How

```python
import pytest
import sys


# Built-in markers
@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature() -> None:
    ...


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Unix-specific test",
)
def test_unix_permissions() -> None:
    ...


@pytest.mark.xfail(reason="Known bug #1234, fix in progress")
def test_edge_case_handling() -> None:
    # xfail = expected failure — test runs but failure is not an error
    result = process_edge_case(None)
    assert result == "handled"


# Custom markers — register in pyproject.toml to avoid warnings
# [tool.pytest.ini_options]
# markers = [
#     "slow: marks tests as slow (deselect with '-m not slow')",
#     "integration: marks integration tests requiring external services",
#     "smoke: critical path tests that should always pass",
# ]

@pytest.mark.slow
def test_large_dataset_processing() -> None:
    data = generate_dataset(size=1_000_000)
    result = process(data)
    assert len(result) > 0


@pytest.mark.integration
def test_external_api_call() -> None:
    response = call_external_api("/health")
    assert response.status_code == 200


@pytest.mark.smoke
def test_app_starts() -> None:
    app = create_app()
    assert app is not None
```

Running with markers:

```bash
# Run only smoke tests
pytest -m smoke

# Run everything except slow tests
pytest -m "not slow"

# Run integration OR smoke tests
pytest -m "integration or smoke"

# Run slow integration tests
pytest -m "slow and integration"
```

#### Why It Matters

Markers create a vocabulary for your test suite. In CI, you might run `pytest -m "not slow"` on every push but run the full suite nightly. During development, `pytest -m smoke` gives you a 5-second confidence check before committing. The `xfail` marker is particularly valuable for documenting known bugs — the test exists, runs, and is expected to fail. When someone fixes the bug, the test unexpectedly passes, and pytest flags it as `XPASS` so you know to remove the marker.

### Test Organization — Directory Structure and Naming

#### What

How you organize test files determines how easy they are to find, run selectively, and maintain. The Python testing community has converged on a standard structure that separates tests by type (unit, integration, e2e) and mirrors the source code layout.

#### How

The recommended structure:

```
project/
├── pyproject.toml
├── src/
│   └── myapp/
│       ├── __init__.py
│       ├── models.py
│       ├── services.py
│       ├── api/
│       │   ├── __init__.py
│       │   └── routes.py
│       └── database/
│           ├── __init__.py
│           └── repository.py
├── tests/
│   ├── conftest.py           # Shared fixtures
│   ├── unit/
│   │   ├── conftest.py       # Unit-specific fixtures (mocks)
│   │   ├── test_models.py    # Tests for src/myapp/models.py
│   │   ├── test_services.py  # Tests for src/myapp/services.py
│   │   └── api/
│   │       └── test_routes.py
│   ├── integration/
│   │   ├── conftest.py       # Integration fixtures (real deps)
│   │   ├── test_api.py
│   │   └── test_database.py
│   └── e2e/
│       ├── conftest.py
│       └── test_workflows.py
```

Naming conventions:

| Element | Convention | Example |
|---------|-----------|---------|
| Test file | `test_<module>.py` | `test_models.py` |
| Test function | `test_<behavior>` | `test_create_user_with_valid_email` |
| Test class | `Test<Unit>` | `TestUserService` |
| Fixture | `<descriptive_name>` | `sample_user`, `db_session` |
| conftest | `conftest.py` (exact) | Always `conftest.py` |

Test function names should describe the behavior being tested, not the implementation. `test_create_user_returns_user_object` is better than `test_create_user_calls_db_insert`.

#### Why It Matters

Structure is what makes a test suite with 500 tests manageable. When tests mirror the source layout, you can instantly find the tests for any module. The unit/integration/e2e split lets you run fast tests (unit) frequently and slow tests (e2e) less often. Descriptive test names serve as documentation — when a test fails, the name tells you exactly what broke without reading the code.

### Mocking with unittest.mock

#### What

Mocking replaces real objects with controlled substitutes during testing. When you test a function that calls an external API, reads a database, or sends an email, you do not want those side effects in your tests. The `unittest.mock` module (part of the standard library) provides `patch`, `MagicMock`, and `Mock` for replacing dependencies with fakes that you control.

#### How

The core tools:

```python
from unittest.mock import MagicMock, Mock, patch


# MagicMock — a flexible fake that accepts any attribute access or method call
mock_db = MagicMock()
mock_db.query("SELECT * FROM users")          # Does not error
mock_db.query.return_value = [{"id": 1}]      # Control return values
mock_db.query("SELECT * FROM users")           # Returns [{"id": 1}]
mock_db.query.assert_called_once_with("SELECT * FROM users")  # Verify calls


# Mock with spec — restricted to the real object's interface
from myapp.database import DatabaseService
mock_db = MagicMock(spec=DatabaseService)
mock_db.query("SELECT 1")           # OK if DatabaseService has .query()
mock_db.nonexistent_method()         # AttributeError! (spec enforcement)


# patch — temporarily replace an object in a specific module
# Replaces the object for the duration of the test, then restores it
```

The `patch` function is the most important tool. It replaces a name in a specific module's namespace:

```python
from unittest.mock import patch, MagicMock
from myapp.services import UserService


# As a decorator
@patch("myapp.services.database")  # Replace 'database' in the services module
def test_get_user(mock_database: MagicMock) -> None:
    mock_database.find_by_id.return_value = {"id": 1, "name": "Alice"}

    service = UserService()
    user = service.get_user(1)

    assert user["name"] == "Alice"
    mock_database.find_by_id.assert_called_once_with(1)


# As a context manager
def test_send_notification() -> None:
    with patch("myapp.services.email_client") as mock_email:
        mock_email.send.return_value = True

        service = UserService()
        result = service.notify_user(1, "Hello")

        assert result is True
        mock_email.send.assert_called_once()
```

#### Why It Matters

Mocking is what makes unit tests "unit" — it isolates the code under test from its dependencies. Without mocking, testing a function that sends HTTP requests would require a running server. Testing database code would require a real database. Testing email code would send real emails. Mocking replaces these dependencies with controlled fakes, making tests fast, deterministic, and independent of external state. The `spec` parameter is critical for safety — it ensures your mock matches the real object's interface, so tests fail if the interface changes.

### patch — Where to Patch

#### What

The most common mocking mistake is patching in the wrong location. `patch` replaces a name in a specific module's namespace, not at the definition site. You must patch where the name is looked up, not where it is defined.

#### How

```python
# myapp/email.py
def send_email(to: str, subject: str, body: str) -> bool:
    """Send an email via SMTP."""
    ...

# myapp/services.py
from myapp.email import send_email  # <-- This creates a reference

class UserService:
    def welcome_user(self, email: str) -> bool:
        return send_email(email, "Welcome!", "Thanks for joining.")
```

```python
# WRONG — patching where send_email is defined
@patch("myapp.email.send_email")  # Patches the original, but services.py
                                    # already has its own reference
def test_welcome(mock_send):
    # This might not work because services.py imported send_email
    # at import time and holds a direct reference to the function
    ...

# RIGHT — patching where send_email is looked up
@patch("myapp.services.send_email")  # Patches the reference in services.py
def test_welcome(mock_send):
    mock_send.return_value = True
    service = UserService()
    result = service.welcome_user("alice@example.com")
    assert result is True
    mock_send.assert_called_once_with(
        "alice@example.com", "Welcome!", "Thanks for joining."
    )
```

The rule: **patch where the thing is used, not where it is defined.** If `services.py` does `from myapp.email import send_email`, then `send_email` is a name in the `myapp.services` namespace. That is where you patch it.

#### Why It Matters

This is the single most confusing aspect of mocking in Python. The confusion arises because `from X import Y` creates a new reference — it copies the name `Y` into the importing module's namespace. Patching the original module's `Y` does not affect the copy. Understanding Python's name resolution model (module namespaces, attribute lookups) is essential for effective mocking. When in doubt, check where the `import` statement is and patch that module's namespace.

### MagicMock In Depth

#### What

`MagicMock` is a mock object that supports Python's magic methods (`__getitem__`, `__len__`, `__iter__`, etc.) out of the box. It automatically creates new MagicMock objects for any attribute access or method call, making it infinitely flexible. `Mock` is the simpler version without magic method support.

#### How

```python
from unittest.mock import MagicMock


# MagicMock accepts any attribute or method call
mock = MagicMock()
mock.foo.bar.baz()                # No error — creates nested mocks
mock["key"]                        # No error — __getitem__ works
len(mock)                          # Returns 0 — __len__ works
list(mock)                         # Returns [] — __iter__ works


# Configuring return values
mock = MagicMock()
mock.get_user.return_value = {"name": "Alice"}
mock.get_user(42)                  # Returns {"name": "Alice"}
mock.get_user(99)                  # Also returns {"name": "Alice"}


# Configuring side effects
mock = MagicMock()
# Raise an exception
mock.connect.side_effect = ConnectionError("timeout")
# mock.connect()  # Raises ConnectionError

# Return different values on successive calls
mock.get_next.side_effect = [1, 2, 3]
mock.get_next()                    # Returns 1
mock.get_next()                    # Returns 2
mock.get_next()                    # Returns 3

# Use a function as side effect
mock.transform.side_effect = lambda x: x * 2
mock.transform(5)                  # Returns 10


# Assertion methods
mock = MagicMock()
mock.save({"name": "Alice"})

mock.save.assert_called_once()
mock.save.assert_called_with({"name": "Alice"})
mock.save.assert_called_once_with({"name": "Alice"})

# Check call count
assert mock.save.call_count == 1

# Check all calls
mock.save({"name": "Bob"})
assert mock.save.call_count == 2
mock.save.assert_any_call({"name": "Alice"})
mock.save.assert_any_call({"name": "Bob"})


# spec — restrict mock to real object's interface
class UserRepository:
    def find_by_id(self, user_id: int) -> dict:
        ...
    def save(self, user: dict) -> None:
        ...

mock_repo = MagicMock(spec=UserRepository)
mock_repo.find_by_id(1)           # OK
mock_repo.save({"name": "Alice"}) # OK
# mock_repo.delete(1)             # AttributeError! Not in UserRepository
```

#### Why It Matters

`MagicMock`'s flexibility is both its strength and its danger. It accepts any call without complaining, which means a typo in your test (`mock.find_by_idd` instead of `mock.find_by_id`) silently succeeds. The `spec` parameter is your safeguard — it restricts the mock to only allow attributes and methods that exist on the real class. Always use `spec` when mocking a known interface. Without `spec`, your tests can pass even when the code under test is calling methods that do not exist.

### Fixture Dependency Injection Flow

The following diagram shows how pytest resolves and injects fixtures into a test function:

```mermaid
graph TD
    A["pytest collects test_create_order()"] --> B["Inspect function signature"]
    B --> C["Parameters: db, sample_user, sample_product"]
    C --> D{"Resolve fixture graph"}
    D --> E["db fixture (no dependencies)"]
    D --> F["sample_user fixture (depends on: db)"]
    D --> G["sample_product fixture (depends on: db)"]

    E --> H["Create Database(':memory:')"]
    H --> F
    H --> G
    F --> I["Insert User('Alice') into db"]
    G --> J["Insert Product('Widget') into db"]

    I --> K["Inject all fixtures into test"]
    J --> K
    H --> K

    K --> L["test_create_order(db, sample_user, sample_product)"]
    L --> M["Test executes assertions"]
    M --> N["Teardown fixtures in reverse order"]
```

This graph shows the key insight: pytest builds a dependency graph from the parameter names, topologically sorts it, calls each fixture in the correct order, and injects the results. Teardown happens in reverse order — the last fixture created is the first one torn down.

## Code Examples

### Example 1: Complete Test Suite for a User Service

```python
# src/myapp/models.py
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class User:
    """Domain model for a user."""

    name: str
    email: str
    role: str = "user"
    is_active: bool = True
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def promote_to_admin(self) -> None:
        """Promote this user to admin role."""
        self.role = "admin"

    def deactivate(self) -> None:
        """Mark this user as inactive."""
        self.is_active = False


# src/myapp/services.py
from myapp.models import User


class UserService:
    """Business logic for user operations."""

    def __init__(self, repository, email_client) -> None:
        self.repository = repository
        self.email_client = email_client

    def create_user(self, name: str, email: str) -> User:
        """Create a new user and send welcome email."""
        if self.repository.find_by_email(email):
            raise ValueError(f"User with email {email} already exists")

        user = User(name=name, email=email)
        self.repository.save(user)
        self.email_client.send(
            to=email,
            subject="Welcome!",
            body=f"Hi {name}, welcome aboard.",
        )
        return user

    def deactivate_user(self, email: str) -> User:
        """Deactivate a user by email."""
        user = self.repository.find_by_email(email)
        if user is None:
            raise ValueError(f"No user found with email {email}")
        user.deactivate()
        self.repository.save(user)
        return user


# tests/conftest.py
import pytest
from unittest.mock import MagicMock
from myapp.models import User


@pytest.fixture
def mock_repository() -> MagicMock:
    """Mock repository that returns None by default for lookups."""
    repo = MagicMock()
    repo.find_by_email.return_value = None
    return repo


@pytest.fixture
def mock_email_client() -> MagicMock:
    """Mock email client that silently accepts sends."""
    client = MagicMock()
    client.send.return_value = True
    return client


@pytest.fixture
def user_service(
    mock_repository: MagicMock,
    mock_email_client: MagicMock,
) -> "UserService":
    """UserService wired with mock dependencies."""
    from myapp.services import UserService
    return UserService(
        repository=mock_repository,
        email_client=mock_email_client,
    )


@pytest.fixture
def sample_user() -> User:
    """A pre-built user for tests that need one."""
    return User(name="Alice", email="alice@example.com", role="admin")


# tests/unit/test_services.py
import pytest
from unittest.mock import MagicMock
from myapp.models import User
from myapp.services import UserService


class TestCreateUser:
    """Tests for UserService.create_user()."""

    def test_creates_user_with_correct_attributes(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ) -> None:
        user = user_service.create_user("Bob", "bob@example.com")

        assert user.name == "Bob"
        assert user.email == "bob@example.com"
        assert user.role == "user"
        assert user.is_active is True

    def test_saves_user_to_repository(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ) -> None:
        user_service.create_user("Bob", "bob@example.com")

        mock_repository.save.assert_called_once()
        saved_user = mock_repository.save.call_args[0][0]
        assert saved_user.email == "bob@example.com"

    def test_sends_welcome_email(
        self,
        user_service: UserService,
        mock_email_client: MagicMock,
    ) -> None:
        user_service.create_user("Bob", "bob@example.com")

        mock_email_client.send.assert_called_once_with(
            to="bob@example.com",
            subject="Welcome!",
            body="Hi Bob, welcome aboard.",
        )

    def test_rejects_duplicate_email(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ) -> None:
        # Simulate existing user
        mock_repository.find_by_email.return_value = User(
            name="Existing",
            email="bob@example.com",
        )

        with pytest.raises(ValueError, match="already exists"):
            user_service.create_user("Bob", "bob@example.com")

        # Ensure no user was saved and no email was sent
        mock_repository.save.assert_not_called()


class TestDeactivateUser:
    """Tests for UserService.deactivate_user()."""

    def test_deactivates_existing_user(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ) -> None:
        active_user = User(name="Alice", email="alice@example.com")
        mock_repository.find_by_email.return_value = active_user

        result = user_service.deactivate_user("alice@example.com")

        assert result.is_active is False
        mock_repository.save.assert_called_once()

    def test_raises_for_nonexistent_user(
        self,
        user_service: UserService,
        mock_repository: MagicMock,
    ) -> None:
        mock_repository.find_by_email.return_value = None

        with pytest.raises(ValueError, match="No user found"):
            user_service.deactivate_user("ghost@example.com")
```

### Example 2: Parametrized Tests for Data Validation

```python
# src/myapp/validation.py
import re
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    error: str | None = None


def validate_email(email: str) -> ValidationResult:
    """Validate an email address format."""
    if not email:
        return ValidationResult(False, "Email cannot be empty")

    # Simplified pattern — production code should use a library
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if not re.match(pattern, email):
        return ValidationResult(False, f"Invalid email format: {email}")

    return ValidationResult(True)


def validate_password(password: str) -> ValidationResult:
    """Validate password meets strength requirements."""
    if len(password) < 8:
        return ValidationResult(False, "Password must be at least 8 characters")
    if not re.search(r"[A-Z]", password):
        return ValidationResult(False, "Password must contain an uppercase letter")
    if not re.search(r"[a-z]", password):
        return ValidationResult(False, "Password must contain a lowercase letter")
    if not re.search(r"\d", password):
        return ValidationResult(False, "Password must contain a digit")
    return ValidationResult(True)


# tests/unit/test_validation.py
import pytest
from myapp.validation import validate_email, validate_password, ValidationResult


class TestValidateEmail:
    """Parametrized tests for email validation."""

    @pytest.mark.parametrize(
        ("email", "expected_valid"),
        [
            ("user@example.com", True),
            ("first.last@company.co.uk", True),
            ("user+tag@gmail.com", True),
            ("user123@test.io", True),
            ("a@b.co", True),
        ],
        ids=[
            "standard",
            "subdomain",
            "plus_addressing",
            "numeric_local",
            "minimal_valid",
        ],
    )
    def test_valid_emails(self, email: str, expected_valid: bool) -> None:
        result = validate_email(email)
        assert result.is_valid is expected_valid
        assert result.error is None

    @pytest.mark.parametrize(
        ("email", "expected_error"),
        [
            ("", "cannot be empty"),
            ("no-at-sign", "Invalid email format"),
            ("@missing-local.com", "Invalid email format"),
            ("user@", "Invalid email format"),
            ("user@.com", "Invalid email format"),
            ("user@domain", "Invalid email format"),
        ],
        ids=[
            "empty",
            "no_at",
            "no_local_part",
            "no_domain",
            "dot_start_domain",
            "no_tld",
        ],
    )
    def test_invalid_emails(self, email: str, expected_error: str) -> None:
        result = validate_email(email)
        assert result.is_valid is False
        assert expected_error in result.error


class TestValidatePassword:
    """Parametrized tests for password validation."""

    @pytest.mark.parametrize(
        "password",
        [
            "Abcdef1!",
            "MyP4ssword",
            "C0mplexPa$$",
            "Str0ngEnough",
        ],
    )
    def test_valid_passwords(self, password: str) -> None:
        result = validate_password(password)
        assert result.is_valid is True

    @pytest.mark.parametrize(
        ("password", "expected_error"),
        [
            ("short1A", "at least 8 characters"),
            ("alllowercase1", "uppercase letter"),
            ("ALLUPPERCASE1", "lowercase letter"),
            ("NoDigitsHere", "digit"),
        ],
        ids=[
            "too_short",
            "no_uppercase",
            "no_lowercase",
            "no_digit",
        ],
    )
    def test_invalid_passwords(
        self,
        password: str,
        expected_error: str,
    ) -> None:
        result = validate_password(password)
        assert result.is_valid is False
        assert expected_error in result.error
```

### Example 3: Yield Fixtures with Real Cleanup

```python
# tests/integration/conftest.py
import pytest
import sqlite3
from pathlib import Path
from collections.abc import Generator


@pytest.fixture(scope="module")
def db_connection(tmp_path_factory: pytest.TempPathFactory) -> Generator[
    sqlite3.Connection,
    None,
    None,
]:
    """Create a SQLite database with schema, clean up after all tests in module."""
    db_path = tmp_path_factory.mktemp("data") / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Setup: create schema
    conn.executescript("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            role TEXT DEFAULT 'user',
            is_active BOOLEAN DEFAULT 1
        );

        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            total REAL NOT NULL,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()

    yield conn  # <-- all tests in this module run here

    # Teardown: close connection, file is cleaned up by tmp_path
    conn.close()


@pytest.fixture
def db(db_connection: sqlite3.Connection) -> Generator[
    sqlite3.Connection,
    None,
    None,
]:
    """Provide a clean database for each test (rollback after each test)."""
    # Start a savepoint so we can rollback test data
    db_connection.execute("SAVEPOINT test_savepoint")

    yield db_connection

    # Rollback any changes made during the test
    db_connection.execute("ROLLBACK TO SAVEPOINT test_savepoint")


@pytest.fixture
def seed_users(db: sqlite3.Connection) -> list[dict[str, str]]:
    """Insert sample users and return them."""
    users = [
        {"name": "Alice", "email": "alice@example.com", "role": "admin"},
        {"name": "Bob", "email": "bob@example.com", "role": "user"},
        {"name": "Carol", "email": "carol@example.com", "role": "user"},
    ]
    for user in users:
        db.execute(
            "INSERT INTO users (name, email, role) VALUES (?, ?, ?)",
            (user["name"], user["email"], user["role"]),
        )
    db.commit()
    return users


# tests/integration/test_database.py
import sqlite3


def test_insert_and_retrieve_user(db: sqlite3.Connection) -> None:
    """Test basic insert and select operations."""
    db.execute(
        "INSERT INTO users (name, email) VALUES (?, ?)",
        ("Dave", "dave@example.com"),
    )
    db.commit()

    cursor = db.execute(
        "SELECT name, email, role FROM users WHERE email = ?",
        ("dave@example.com",),
    )
    user = cursor.fetchone()

    assert user is not None
    assert user["name"] == "Dave"
    assert user["role"] == "user"  # Default value


def test_seed_data_available(
    db: sqlite3.Connection,
    seed_users: list[dict[str, str]],
) -> None:
    """Test that seed fixture provides expected data."""
    cursor = db.execute("SELECT COUNT(*) as count FROM users")
    assert cursor.fetchone()["count"] == 3


def test_unique_email_constraint(
    db: sqlite3.Connection,
    seed_users: list[dict[str, str]],
) -> None:
    """Test that duplicate emails are rejected."""
    import pytest

    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            ("Duplicate", "alice@example.com"),  # Already exists
        )


def test_isolation_between_tests(db: sqlite3.Connection) -> None:
    """Verify test isolation — seed_users from other tests are rolled back."""
    cursor = db.execute("SELECT COUNT(*) as count FROM users")
    assert cursor.fetchone()["count"] == 0  # Clean slate
```

### Example 4: Mocking External APIs with patch

```python
# src/myapp/weather.py
import httpx
from dataclasses import dataclass


@dataclass
class WeatherReport:
    """Weather data for a location."""

    city: str
    temperature: float
    condition: str
    humidity: int


class WeatherService:
    """Client for an external weather API."""

    def __init__(self, api_key: str, base_url: str = "https://api.weather.com") -> None:
        self.api_key = api_key
        self.base_url = base_url

    def get_weather(self, city: str) -> WeatherReport:
        """Fetch current weather for a city."""
        response = httpx.get(
            f"{self.base_url}/current",
            params={"city": city, "key": self.api_key},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()

        return WeatherReport(
            city=city,
            temperature=data["temp_c"],
            condition=data["condition"],
            humidity=data["humidity"],
        )

    def is_raining(self, city: str) -> bool:
        """Check if it's currently raining in a city."""
        report = self.get_weather(city)
        return report.condition.lower() in ("rain", "drizzle", "thunderstorm")


# tests/unit/test_weather.py
import pytest
from unittest.mock import MagicMock, patch
import httpx
from myapp.weather import WeatherService, WeatherReport


@pytest.fixture
def weather_service() -> WeatherService:
    """WeatherService with a test API key."""
    return WeatherService(api_key="test-key-123")


@pytest.fixture
def mock_weather_response() -> dict:
    """Standard mock API response data."""
    return {
        "temp_c": 22.5,
        "condition": "Partly Cloudy",
        "humidity": 65,
    }


class TestGetWeather:
    """Tests for WeatherService.get_weather()."""

    @patch("myapp.weather.httpx.get")
    def test_returns_weather_report(
        self,
        mock_get: MagicMock,
        weather_service: WeatherService,
        mock_weather_response: dict,
    ) -> None:
        """Test that API response is correctly parsed into WeatherReport."""
        # Configure the mock response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_weather_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = weather_service.get_weather("London")

        assert isinstance(result, WeatherReport)
        assert result.city == "London"
        assert result.temperature == 22.5
        assert result.condition == "Partly Cloudy"
        assert result.humidity == 65

    @patch("myapp.weather.httpx.get")
    def test_passes_correct_parameters(
        self,
        mock_get: MagicMock,
        weather_service: WeatherService,
        mock_weather_response: dict,
    ) -> None:
        """Test that the API is called with correct params."""
        mock_response = MagicMock()
        mock_response.json.return_value = mock_weather_response
        mock_get.return_value = mock_response

        weather_service.get_weather("Tokyo")

        mock_get.assert_called_once_with(
            "https://api.weather.com/current",
            params={"city": "Tokyo", "key": "test-key-123"},
            timeout=10.0,
        )

    @patch("myapp.weather.httpx.get")
    def test_handles_api_error(
        self,
        mock_get: MagicMock,
        weather_service: WeatherService,
    ) -> None:
        """Test that HTTP errors propagate correctly."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )
        mock_get.return_value = mock_response

        with pytest.raises(httpx.HTTPStatusError):
            weather_service.get_weather("London")


class TestIsRaining:
    """Tests for WeatherService.is_raining()."""

    @pytest.mark.parametrize(
        ("condition", "expected"),
        [
            ("Rain", True),
            ("Drizzle", True),
            ("Thunderstorm", True),
            ("Partly Cloudy", False),
            ("Sunny", False),
            ("Overcast", False),
        ],
    )
    @patch("myapp.weather.httpx.get")
    def test_rain_detection(
        self,
        mock_get: MagicMock,
        weather_service: WeatherService,
        condition: str,
        expected: bool,
    ) -> None:
        """Test rain detection across various conditions."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "temp_c": 15.0,
            "condition": condition,
            "humidity": 80,
        }
        mock_get.return_value = mock_response

        assert weather_service.is_raining("London") is expected
```

### Example 5: Autouse Fixtures and pytest Built-in Fixtures

```python
# tests/conftest.py
import pytest
import time
from pathlib import Path
from collections.abc import Generator


@pytest.fixture(autouse=True)
def log_test_timing(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Automatically log execution time for every test.

    autouse=True means this fixture runs for every test without
    being explicitly requested in the function signature.
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    # Warn if a test takes too long (might indicate a missing mock)
    if elapsed > 1.0:
        print(f"\n  SLOW TEST: {request.node.name} took {elapsed:.2f}s")


@pytest.fixture(autouse=True)
def reset_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure environment variables are clean for every test.

    monkeypatch is a built-in pytest fixture that undoes changes
    after each test automatically.
    """
    # Set a known environment for all tests
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    # Prevent tests from accidentally using production credentials
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("API_SECRET", raising=False)


# tests/unit/test_config.py
import os
from pathlib import Path


def test_app_env_is_test() -> None:
    """Verify that autouse fixture sets APP_ENV."""
    assert os.environ["APP_ENV"] == "test"


def test_production_creds_not_available() -> None:
    """Verify that production credentials are stripped."""
    assert "DATABASE_URL" not in os.environ
    assert "API_SECRET" not in os.environ


def test_tmp_path_usage(tmp_path: Path) -> None:
    """Demonstrate pytest's built-in tmp_path fixture.

    tmp_path provides a unique temporary directory for each test.
    pytest cleans it up automatically after the test session.
    """
    config_file = tmp_path / "config.json"
    config_file.write_text('{"debug": true, "port": 8080}')

    assert config_file.exists()
    assert "debug" in config_file.read_text()


def test_capsys_output_capture(capsys: pytest.CaptureFixture[str]) -> None:
    """Demonstrate pytest's built-in capsys fixture.

    capsys captures stdout and stderr so you can assert on print output.
    """
    print("Processing started")
    print("Processing complete")

    captured = capsys.readouterr()
    assert "Processing started" in captured.out
    assert "Processing complete" in captured.out
    assert captured.err == ""  # Nothing on stderr


def test_monkeypatch_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Demonstrate monkeypatching a module attribute.

    monkeypatch is often a lighter-weight alternative to unittest.mock.patch
    for simple cases where you just need to swap a value.
    """
    import myapp.config as config_module

    monkeypatch.setattr(config_module, "MAX_RETRIES", 1)
    assert config_module.MAX_RETRIES == 1
    # After the test, monkeypatch restores the original value
```

### Example 6: Complete conftest.py Hierarchy

```python
# project/conftest.py — root level, session-wide
import pytest
from dataclasses import dataclass


@dataclass
class TestConfig:
    """Configuration for the test environment."""
    database_url: str
    api_base_url: str
    debug: bool


@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    """Session-wide test configuration, created once."""
    return TestConfig(
        database_url="sqlite:///:memory:",
        api_base_url="http://localhost:8000",
        debug=True,
    )


# tests/conftest.py — shared across all test types
import pytest
from myapp.models import User, Product


@pytest.fixture
def sample_user() -> User:
    """A standard test user."""
    return User(name="Alice", email="alice@example.com", role="admin")


@pytest.fixture
def sample_product() -> Product:
    """A standard test product."""
    return Product(name="Widget", price=9.99, stock=100)


# Register custom markers to avoid warnings
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: requires external services")
    config.addinivalue_line("markers", "smoke: critical path tests")


# tests/unit/conftest.py — unit test fixtures (mocks)
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_db() -> MagicMock:
    """Mock database for unit tests."""
    db = MagicMock()
    db.query.return_value = []
    db.insert.return_value = True
    return db


@pytest.fixture
def mock_email() -> MagicMock:
    """Mock email client for unit tests."""
    client = MagicMock()
    client.send.return_value = True
    return client


@pytest.fixture
def mock_cache() -> MagicMock:
    """Mock cache for unit tests."""
    cache = MagicMock()
    cache.get.return_value = None
    cache.set.return_value = True
    return cache


# tests/integration/conftest.py — integration fixtures (real deps)
import pytest
import sqlite3
from collections.abc import Generator


@pytest.fixture(scope="module")
def real_db(test_config: "TestConfig") -> Generator[sqlite3.Connection, None, None]:
    """Real SQLite database for integration tests."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        );
    """)
    yield conn
    conn.close()
```

## Common Pitfalls

### Pitfall 1: Patching Where the Object Is Defined Instead of Where It Is Used

This is the most common mocking mistake. When a module does `from X import Y`, patching `X.Y` does not affect the copy in the importing module.

```python
# src/myapp/services.py
from myapp.email import send_email  # Creates a local reference

class NotificationService:
    def notify(self, email: str) -> bool:
        return send_email(email, "Hello", "Body")

# BAD — patching the definition site (does NOT affect services.py's reference)
from unittest.mock import patch

@patch("myapp.email.send_email")
def test_notify(mock_send):
    mock_send.return_value = True
    service = NotificationService()
    result = service.notify("alice@example.com")
    # This may still call the REAL send_email because services.py
    # holds its own reference from the 'from' import
    assert result is True  # Might fail or send a real email!

# GOOD — patching where the name is looked up (in the using module)
@patch("myapp.services.send_email")
def test_notify(mock_send):
    mock_send.return_value = True
    service = NotificationService()
    result = service.notify("alice@example.com")
    assert result is True
    mock_send.assert_called_once()
```

### Pitfall 2: Using MagicMock Without spec (Hiding Real Bugs)

Without `spec`, MagicMock accepts any attribute or method call — including typos and methods that do not exist on the real object. Your tests pass, but the code is broken.

```python
# BAD — mock without spec silently accepts nonexistent methods
from unittest.mock import MagicMock

def test_user_service():
    mock_repo = MagicMock()  # No spec — accepts anything
    mock_repo.find_by_idd.return_value = None  # TYPO: 'idd' not 'id'
    # Test passes because MagicMock doesn't care about the typo
    # But the real code calls find_by_id, so nothing was actually tested

# GOOD — mock with spec catches typos and interface changes
from myapp.repository import UserRepository

def test_user_service():
    mock_repo = MagicMock(spec=UserRepository)  # Constrained to real interface
    mock_repo.find_by_idd.return_value = None  # AttributeError! Typo caught.
    # Fix the typo:
    mock_repo.find_by_id.return_value = None    # OK — method exists
```

### Pitfall 3: Putting Assertions Inside the Test Setup Instead of After the Action

When assertions are mixed into setup or called before the action under test, failures give misleading information about what actually broke.

```python
# BAD — assertion about the setup, not the behavior
def test_user_creation(db):
    # Setup
    initial_count = db.count("users")
    assert initial_count == 0  # This is testing the fixture, not the code

    user = create_user(db, "Alice")
    assert user is not None   # Weak assertion — doesn't test anything specific

    final_count = db.count("users")
    assert final_count == 1   # Finally tests the real behavior

# GOOD — Arrange, Act, Assert (AAA) pattern with focused assertions
def test_user_creation_inserts_into_database(db):
    # Arrange
    user_data = {"name": "Alice", "email": "alice@example.com"}

    # Act
    created_user = create_user(db, **user_data)

    # Assert — specific, meaningful assertions about the behavior
    assert created_user.name == "Alice"
    assert created_user.email == "alice@example.com"
    assert db.count("users") == 1

    stored = db.find_by_email("alice@example.com")
    assert stored is not None
    assert stored.name == "Alice"
```

### Pitfall 4: Sharing Mutable State Between Tests via Module-Scoped Fixtures

When a module-scoped fixture returns a mutable object and tests modify it, earlier tests affect later tests. This creates order-dependent test failures — the hardest bugs to debug.

```python
# BAD — module-scoped fixture with mutable state
@pytest.fixture(scope="module")
def user_list():
    """Shared across all tests in the module."""
    return [
        {"name": "Alice", "role": "admin"},
        {"name": "Bob", "role": "user"},
    ]

def test_add_user(user_list):
    user_list.append({"name": "Carol", "role": "user"})
    assert len(user_list) == 3  # PASSES

def test_initial_count(user_list):
    assert len(user_list) == 2  # FAILS! Carol was added by previous test.
                                 # user_list now has 3 items.

# GOOD — function-scoped fixture for mutable data (fresh copy each test)
@pytest.fixture
def user_list():
    """Fresh copy for each test — no shared mutable state."""
    return [
        {"name": "Alice", "role": "admin"},
        {"name": "Bob", "role": "user"},
    ]

def test_add_user(user_list):
    user_list.append({"name": "Carol", "role": "user"})
    assert len(user_list) == 3  # PASSES

def test_initial_count(user_list):
    assert len(user_list) == 2  # PASSES — fresh list, Carol is not here
```

### Pitfall 5: Over-Mocking — Testing Mock Behavior Instead of Real Behavior

When you mock too many things, you end up testing that your mocks return what you told them to return. The test passes regardless of whether the actual code works.

```python
# BAD — testing that mock returns what you configured it to return
from unittest.mock import MagicMock

def test_calculate_total():
    mock_cart = MagicMock()
    mock_cart.items.return_value = [
        MagicMock(price=10.0, quantity=2),
        MagicMock(price=5.0, quantity=3),
    ]
    mock_cart.calculate_total.return_value = 35.0  # YOU set this value

    result = mock_cart.calculate_total()  # Just returns what you told it to
    assert result == 35.0  # Of course it equals 35.0 — you said so!
    # This test proves nothing about the real calculate_total logic

# GOOD — mock only the dependencies, test the real logic
from myapp.cart import ShoppingCart, CartItem

def test_calculate_total():
    # Use real objects for the code under test
    cart = ShoppingCart()
    cart.add_item(CartItem(name="Widget", price=10.0, quantity=2))
    cart.add_item(CartItem(name="Gadget", price=5.0, quantity=3))

    # Test the REAL calculate_total method
    result = cart.calculate_total()
    assert result == 35.0  # This actually tests the calculation logic

    # Only mock external dependencies (database, API, email)
    # Never mock the thing you're testing
```

## Key Takeaways

- **pytest's power comes from three features**: plain `assert` with introspection (better error messages with less code), fixture dependency injection (composable, declarative test setup), and parametrize (one test function, many test cases).
- **Fixtures replace setUp/tearDown** with a composable dependency graph. Each test declares what it needs by parameter name, and pytest injects the dependencies automatically. Yield fixtures handle teardown with guaranteed cleanup.
- **`conftest.py` is the glue** — place fixtures at the appropriate level of the directory hierarchy so they are available where needed without imports. Root conftest for project-wide resources, subdirectory conftest for specialized fixtures.
- **Mock where the name is used, not where it is defined** — the most critical rule for `unittest.mock.patch`. Always use `spec=RealClass` to catch interface mismatches.
- **Follow the AAA pattern** (Arrange, Act, Assert) and only mock external dependencies — never mock the code under test, or you are testing your test setup rather than your application.

## Exercises

1. **Write a fixture chain.** Create three fixtures: `db_connection` (session-scoped, creates an in-memory SQLite database with a `products` table), `db` (function-scoped, wraps `db_connection` with savepoint/rollback for test isolation), and `sample_products` (function-scoped, inserts 3 products and returns them). Write two tests: one that modifies data and one that verifies the database starts clean (proving isolation works).

2. **Parametrize a parser.** Write a function `parse_duration(s: str) -> int` that converts strings like `"5m"`, `"2h"`, `"30s"`, `"1d"` into total seconds. Then write a parametrized test with at least 8 cases covering valid inputs, edge cases (`"0s"`, `"24h"`), and invalid inputs (empty string, unknown unit, negative number). Use `pytest.raises` for the error cases.

3. **Mock an external dependency.** You have a `NotificationService` that sends messages via an `SMSClient.send(phone, message)` method. Write tests for `NotificationService.send_verification_code(phone)` that: (a) verify the correct phone number is passed to the SMS client, (b) verify the message contains a 6-digit code, (c) test behavior when the SMS client raises a `ConnectionError`. Use `patch` and `MagicMock` with `spec`.

4. **Design a conftest.py hierarchy.** You have a project with unit tests, integration tests, and end-to-end tests. Design the `conftest.py` files for each level. The root conftest should provide `app_config` (session-scoped). The unit conftest should provide `mock_database` and `mock_cache`. The integration conftest should provide `real_database` (module-scoped, using yield for cleanup). Write the fixture code and explain why you chose each scope.

5. **Debug a test isolation failure.** Two tests pass when run individually (`pytest test_a.py` and `pytest test_b.py`) but fail when run together (`pytest test_a.py test_b.py`). One test modifies a module-level dictionary, and the other test assumes the dictionary is in its default state. Explain why this happens, write code that reproduces the problem, and fix it using the appropriate pytest mechanism (fixture scope, monkeypatch, or test restructuring).

---
up:: [Schedule](../../Schedule.md)
#type/learning #source/self-study #status/seed
