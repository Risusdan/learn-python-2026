# Ruff — Ultra-Fast Python Linter and Formatter

> Ruff is a single Rust-powered tool that replaces flake8, isort, black, pyupgrade, and dozens of other linters — running 10-100x faster while catching more issues and auto-fixing most of them.

## Table of Contents
- [Core Concepts](#core-concepts)
- [Code Examples](#code-examples)
- [Common Pitfalls](#common-pitfalls)
- [Key Takeaways](#key-takeaways)
- [Exercises](#exercises)

## Core Concepts

### The Problem Ruff Solves

#### What

Before ruff, a typical Python project needed a stack of separate tools for code quality: flake8 for linting, isort for import sorting, black for formatting, pyupgrade for modernizing syntax, bandit for security checks, and pydocstyle for docstring conventions. Each tool had its own configuration format, its own CLI flags, its own version pinning, and its own speed characteristics. A pre-commit hook running all of them could take 30+ seconds on a medium-sized codebase.

#### How

Ruff consolidates all of these into a single binary written in Rust. It reimplements the rule sets from flake8 and its most popular plugins (flake8-bugbear, flake8-comprehensions, flake8-simplify, and many more), isort's import sorting, pyupgrade's syntax modernization, and black's formatting — all behind two commands: `ruff check` and `ruff format`. One tool to install, one configuration block in `pyproject.toml`, one CI step.

#### Why It Matters

Speed changes behavior. When your linter takes 30 seconds, you run it before commits. When it takes 30 milliseconds, you run it on every file save. Ruff is fast enough to be invisible — it finishes before you notice it started. This means you get instant feedback on mistakes, and your code stays clean continuously rather than in batch corrections. The consolidation also eliminates "tool version hell" — no more debugging why flake8 and isort disagree about import ordering.

### Installation and Basic Usage

#### What

Ruff is distributed as a standalone binary. You install it once and get both the linter (`ruff check`) and the formatter (`ruff format`). It works on individual files, directories, or entire projects.

#### How

Install ruff using uv (the modern Python package manager) or pip:

```bash
# Install with uv (recommended)
uv tool install ruff

# Or install with pip
pip install ruff

# Verify installation
ruff --version
```

The two primary commands:

```bash
# Lint your code — find problems
ruff check .

# Format your code — consistent style
ruff format .

# Lint and auto-fix what's possible
ruff check --fix .

# Check what formatting would change (dry run)
ruff format --check .
```

By default, `ruff check .` scans all Python files recursively from the current directory. It exits with a non-zero status code if any violations are found, making it CI-friendly out of the box.

#### Why It Matters

The two-command model (`check` + `format`) draws a clean line between two different concerns. Linting finds semantic problems — unused imports, undefined names, bad practices, potential bugs. Formatting handles pure aesthetics — indentation, line length, quote style, trailing commas. This separation matters because linting violations often need human judgment ("should I remove this import or is it used elsewhere?"), while formatting should be entirely automatic and non-negotiable.

### Ruff Check — The Linter

#### What

`ruff check` is the linting half of ruff. It analyzes your Python source code against a configurable set of rules and reports violations. Each rule has a code like `F401` (unused import) or `E501` (line too long), where the letter prefix indicates which rule set it belongs to.

#### How

```bash
# Check entire project
ruff check .

# Check specific file
ruff check src/main.py

# Check with auto-fix enabled
ruff check --fix .

# Show which rules would be fixed (preview without changing files)
ruff check --diff .

# Check and show rule explanations
ruff check --show-fixes .
```

When ruff finds a violation, it reports the file, line, column, rule code, and a human-readable message:

```
src/utils.py:3:1: F401 [*] `os` imported but unused
src/utils.py:15:89: E501 Line too long (102 > 88)
src/main.py:42:5: UP035 [*] Import from `typing` instead of `typing_extensions`
```

The `[*]` marker means the violation is auto-fixable. Running `ruff check --fix` will correct these automatically.

#### Why It Matters

The auto-fix capability is what makes ruff transformative for workflows. Most linters just report problems — you fix them manually. Ruff can fix the majority of what it finds: removing unused imports, sorting imports, modernizing syntax, simplifying expressions. This means you can write code quickly without worrying about style, then let `ruff check --fix` clean it up in milliseconds.

### Ruff Format — The Formatter

#### What

`ruff format` is ruff's code formatter, designed as a drop-in replacement for black. It reformats Python source code to a consistent style, handling indentation, line breaks, quote normalization, trailing commas, and whitespace. Like black, it is intentionally opinionated — there are very few configuration options because the whole point is to eliminate style debates.

#### How

```bash
# Format all Python files in the project
ruff format .

# Format a specific file
ruff format src/main.py

# Check if files are already formatted (CI mode — no changes)
ruff format --check .

# Show what would change without applying
ruff format --diff .
```

The formatter enforces:
- Consistent indentation (4 spaces)
- Line length limits (88 characters by default, matching black)
- Double quotes for strings (by default)
- Trailing commas in multi-line structures
- Consistent blank lines between definitions
- Parenthesized long expressions

#### Why It Matters

Formatters eliminate an entire category of code review friction. When every file in a project looks the same, diffs show only meaningful changes — not whitespace noise. The "opinionated" design is a feature: by removing most configuration knobs, ruff format guarantees that two developers working on the same codebase produce identical formatting without coordination. Black proved this model; ruff format continues it at higher speed.

### Rule Categories

#### What

Ruff organizes its rules into categories identified by letter prefixes. Each category corresponds to a linter or rule set from the broader Python ecosystem. Understanding these categories lets you selectively enable the checks most relevant to your project.

#### How

The major rule categories:

| Prefix | Origin | What It Checks |
|--------|--------|----------------|
| `F` | Pyflakes | Undefined names, unused imports/variables, redefined names |
| `E` / `W` | pycodestyle | PEP 8 style (errors and warnings) |
| `I` | isort | Import sorting order |
| `UP` | pyupgrade | Modernize syntax for your target Python version |
| `N` | pep8-naming | Naming conventions (snake_case, PascalCase) |
| `S` | flake8-bandit | Security issues (hardcoded passwords, SQL injection) |
| `B` | flake8-bugbear | Common bugs and design problems |
| `C4` | flake8-comprehensions | Unnecessary list/dict/set construction |
| `SIM` | flake8-simplify | Simplifiable expressions and control flow |
| `PT` | flake8-pytest-style | pytest best practices |
| `RUF` | Ruff-specific | Ruff's own rules (not from any upstream linter) |
| `D` | pydocstyle | Docstring conventions |
| `ANN` | flake8-annotations | Missing type annotations |
| `TCH` | flake8-type-checking | Imports that should be under `TYPE_CHECKING` |
| `ERA` | eradicate | Commented-out code |
| `ARG` | flake8-unused-arguments | Unused function arguments |
| `PLR` | Pylint refactor | Refactoring suggestions (too many arguments, too many branches) |
| `PLW` | Pylint warning | Pylint warnings |
| `PLE` | Pylint error | Pylint errors |
| `PERF` | Perflint | Performance anti-patterns |
| `FURB` | refurb | Modern Python idiom suggestions |

By default, ruff enables `F` (Pyflakes) and a subset of `E` (pycodestyle errors). Everything else must be opted into.

#### Why It Matters

The prefix system lets you adopt rules incrementally. Starting a new project? Enable everything. Adopting ruff on a legacy codebase with 10,000 violations? Start with `F` and `I`, fix those, then enable `UP`, then `B`, and so on. You can also enable entire categories while ignoring specific rules — for example, enable all `B` rules but ignore `B008` if it conflicts with a framework pattern like FastAPI's `Depends()`.

### Configuration in pyproject.toml

#### What

Ruff is configured through `pyproject.toml`, the standard Python project configuration file. All ruff settings live under the `[tool.ruff]` table, with sub-tables for linting (`[tool.ruff.lint]`) and formatting (`[tool.ruff.format]`). This centralized configuration replaces the scattered config files that flake8, isort, and black each required.

#### How

A comprehensive `pyproject.toml` configuration:

```toml
[tool.ruff]
# Apply to these Python files
include = ["*.py", "*.pyi"]

# Target Python version — affects which UP rules apply
target-version = "py312"

# Line length for both linter and formatter
line-length = 88

# Directories to exclude from checking
exclude = [
    ".venv",
    "migrations",
    "__pycache__",
    ".git",
]

[tool.ruff.lint]
# Enable rule categories
select = [
    "F",      # Pyflakes — unused imports, undefined names
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "I",      # isort — import sorting
    "UP",     # pyupgrade — modernize syntax
    "B",      # flake8-bugbear — common bugs
    "SIM",    # flake8-simplify — simplifiable code
    "C4",     # flake8-comprehensions — better comprehensions
    "PT",     # flake8-pytest-style — pytest best practices
    "RUF",    # Ruff-specific rules
]

# Ignore specific rules
ignore = [
    "E501",   # Line too long — handled by formatter
    "B008",   # Do not perform function call in argument defaults
              # (needed for FastAPI Depends())
]

# Allow auto-fix for all enabled rules
fixable = ["ALL"]
unfixable = []

[tool.ruff.lint.per-file-ignores]
# Tests can use assert and don't need docstrings
"tests/**/*.py" = ["S101", "D103"]
# __init__.py can have unused imports (they're re-exports)
"__init__.py" = ["F401"]

[tool.ruff.lint.isort]
# Force single-line imports (one import per line)
force-single-line = false
# Known first-party packages (for import grouping)
known-first-party = ["myproject"]

[tool.ruff.format]
# Use double quotes (consistent with black)
quote-style = "double"
# Unix-style line endings
line-ending = "lf"
# Format docstrings
docstring-code-format = true
```

The configuration hierarchy matters. Ruff looks for configuration in this order:
1. `pyproject.toml` in the current directory
2. `ruff.toml` or `.ruff.toml` in the current directory
3. Parent directories (walks up until it finds config or reaches root)

#### Why It Matters

Good configuration is the difference between ruff being helpful and ruff being noisy. The `select` list should be curated for your project — enabling everything at once on an existing codebase will produce thousands of violations that overwhelm rather than help. Start with the defaults plus `I` and `UP`, then expand. The `per-file-ignores` setting is essential for practical use: test files have different conventions than production code, and `__init__.py` files legitimately re-export imports.

### Auto-Fix and the --fix Flag

#### What

Many ruff rules are auto-fixable, meaning ruff can not only detect the problem but also rewrite your source code to correct it. The `--fix` flag enables this behavior. Fixable rules are marked with `[*]` in ruff's output.

#### How

```bash
# See violations without fixing
ruff check .

# Auto-fix all fixable violations
ruff check --fix .

# See what fixes would be applied (without applying)
ruff check --diff .

# Fix only specific rules
ruff check --fix --fixable I001 .

# Fix everything except specific rules
ruff check --fix --unfixable F401 .
```

Common auto-fixable violations:

| Rule | What It Fixes |
|------|--------------|
| `F401` | Removes unused imports |
| `I001` | Sorts imports into correct order |
| `UP006` | Replaces `List[int]` with `list[int]` |
| `UP007` | Replaces `Optional[str]` with `str \| None` |
| `UP035` | Moves imports from `typing` to `collections.abc` |
| `C4` rules | Converts unnecessary `list()` / `dict()` calls to comprehensions |
| `SIM` rules | Simplifies boolean expressions, ternaries, `if` blocks |

Some rules are "unsafe" fixes — they change code semantics in ways that might break things. For example, removing an unused import is safe only if the import has no side effects. Ruff distinguishes between safe and unsafe fixes, and by default only applies safe fixes. Use `--unsafe-fixes` to apply both.

#### Why It Matters

Auto-fix changes the economics of code quality. Without it, every linting violation means a manual edit — so developers configure fewer rules to avoid the burden. With auto-fix, you can enable aggressive rule sets knowing that most violations will be corrected automatically. The `--diff` flag is your safety net: always preview fixes on a new codebase before applying them. The safe/unsafe distinction protects you from fixes that could introduce bugs.

### Editor Integration

#### What

Ruff provides a Language Server Protocol (LSP) implementation called `ruff-lsp` (now built into `ruff server`), which integrates with VS Code, Neovim, Emacs, and any LSP-compatible editor. This gives you real-time linting feedback, code actions for auto-fixes, and format-on-save.

#### How

For VS Code, install the official "Ruff" extension (by Astral). Configure it in `.vscode/settings.json`:

```json
{
    "[python]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    }
}
```

This configuration means:
- On every file save, ruff formats the file (replacing black)
- Code actions fix all auto-fixable violations (replacing flake8 + manual fixes)
- Imports are organized automatically (replacing isort)

For Neovim with `nvim-lspconfig`:

```lua
require('lspconfig').ruff.setup({
    init_options = {
        settings = {
            lineLength = 88,
            lint = {
                select = { "F", "E", "W", "I", "UP", "B" },
            },
        },
    },
})
```

#### Why It Matters

Editor integration is where ruff's speed truly shines. Because ruff can lint a file in single-digit milliseconds, the editor can run it on every keystroke without any perceptible delay. This creates a tight feedback loop: you write a line with an unused import, and the editor immediately shows a diagnostic. You save, and the import disappears. Compare this to running flake8 in a separate terminal after writing a hundred lines — the faster feedback loop produces cleaner code with less effort.

### CI Integration

#### What

Running ruff in continuous integration (CI) ensures that no code quality regressions reach the main branch. The CI pipeline should check both linting and formatting, failing the build if either has violations.

#### How

A typical GitHub Actions workflow:

```yaml
name: Code Quality
on: [push, pull_request]

jobs:
  ruff:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/ruff-action@v3
        with:
          args: "check"
      - uses: astral-sh/ruff-action@v3
        with:
          args: "format --check"
```

The `astral-sh/ruff-action` is the official GitHub Action. It installs ruff and runs it in a single step. The `format --check` flag is critical — it checks whether files are formatted without modifying them, and exits non-zero if any file would be changed.

For pre-commit hooks (using the `pre-commit` framework):

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.9.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

#### Why It Matters

CI is your safety net. Even with editor integration, someone will occasionally commit code without saving, or use an editor without ruff configured. CI catches these cases. The key insight is that CI should be strict (no `--fix` for linting — fail and make the developer fix it locally) while pre-commit hooks can be lenient (auto-fix on commit so developers don't have to think about it). This dual strategy keeps the codebase clean without slowing down development.

### Migrating from flake8/isort/black

#### What

If you have an existing project using flake8, isort, and/or black, ruff can replace all three. The migration is usually straightforward because ruff intentionally reimplements the same rules with the same codes.

#### How

Ruff provides a built-in migration helper:

```bash
# Convert existing flake8 configuration to ruff
ruff check --config "extend-select = ['ALL']" --statistics .
```

Step-by-step migration:

1. **Remove old tools** from your dependencies:
   ```bash
   uv remove flake8 isort black flake8-bugbear flake8-comprehensions
   uv add --dev ruff
   ```

2. **Translate configuration**. Map your flake8 settings:
   ```ini
   # Old: .flake8 or setup.cfg
   [flake8]
   max-line-length = 88
   extend-ignore = E203, W503
   per-file-ignores =
       tests/*.py: S101
   ```
   ```toml
   # New: pyproject.toml
   [tool.ruff]
   line-length = 88

   [tool.ruff.lint]
   select = ["F", "E", "W"]
   ignore = ["E203", "W503"]

   [tool.ruff.lint.per-file-ignores]
   "tests/*.py" = ["S101"]
   ```

3. **Map isort settings**:
   ```toml
   # Old isort config
   [tool.isort]
   profile = "black"
   known_first_party = ["myproject"]

   # New ruff config
   [tool.ruff.lint]
   select = ["I"]

   [tool.ruff.lint.isort]
   known-first-party = ["myproject"]
   ```

4. **Run ruff and fix**:
   ```bash
   ruff check --fix .
   ruff format .
   ```

5. **Update CI and pre-commit** to use ruff instead of the old tools.

#### Why It Matters

Migration is the realistic scenario — most professional projects are not greenfield. Understanding how to translate existing configurations preserves team conventions while gaining ruff's speed advantage. The rule code compatibility (flake8's `F401` is the same as ruff's `F401`) makes the transition nearly seamless. The main risk is that ruff may detect violations that the old tools missed, which is actually a benefit once you work through the initial batch.

### Understanding Rule Output and Codes

#### What

Each ruff violation includes a rule code, a file location, and a message. Reading these effectively is a core skill — you need to quickly determine whether a violation is a real problem, a configuration issue, or a false positive.

#### How

Ruff's output format:

```
path/to/file.py:LINE:COL: CODE [*] Message
```

The components:
- **Path, line, column**: exact location of the violation
- **Code**: the rule identifier (e.g., `F401`, `UP007`, `B006`)
- **`[*]`**: present only if the violation is auto-fixable
- **Message**: human-readable description

To understand any rule, use `ruff rule`:

```bash
# Get detailed explanation of a rule
ruff rule F401

# List all available rules
ruff rule --all

# List rules in a specific category
ruff rule --all | grep "^UP"
```

The `ruff rule` command shows the rule's name, category, what it detects, why it matters, and examples of violating and compliant code. This is your go-to reference when you encounter an unfamiliar rule code.

#### Why It Matters

Understanding rule codes makes you faster at triaging violations. When you see `F841` you immediately know it is a Pyflakes rule about an unused local variable. When you see `UP007`, you know pyupgrade wants you to modernize a type annotation. This pattern recognition eliminates the need to look up every violation — you learn the most common 20-30 codes and handle them reflexively.

## Code Examples

### Example 1: Setting Up Ruff for a New Project

```python
# This example shows a complete project setup with ruff configuration.
# File: pyproject.toml (not Python, but essential for ruff usage)
#
# [project]
# name = "inventory-api"
# version = "0.1.0"
# requires-python = ">=3.12"
#
# [tool.ruff]
# target-version = "py312"
# line-length = 88
#
# [tool.ruff.lint]
# select = [
#     "F",    # Pyflakes
#     "E",    # pycodestyle errors
#     "W",    # pycodestyle warnings
#     "I",    # isort
#     "UP",   # pyupgrade
#     "B",    # flake8-bugbear
#     "SIM",  # flake8-simplify
#     "C4",   # flake8-comprehensions
#     "RUF",  # Ruff-specific
# ]
# ignore = ["E501"]  # line length handled by formatter
#
# [tool.ruff.lint.per-file-ignores]
# "tests/**/*.py" = ["S101"]
#
# [tool.ruff.format]
# quote-style = "double"
# docstring-code-format = true


# File: src/inventory/models.py
# This code is written to be ruff-compliant from the start.

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum


class ItemStatus(StrEnum):
    """Product status in the inventory system."""

    ACTIVE = "active"
    DISCONTINUED = "discontinued"
    OUT_OF_STOCK = "out_of_stock"


@dataclass
class InventoryItem:
    """A single item in the warehouse inventory."""

    sku: str
    name: str
    quantity: int
    unit_price: float
    status: ItemStatus = ItemStatus.ACTIVE
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def total_value(self) -> float:
        """Calculate total value of this item's stock."""
        return self.quantity * self.unit_price

    def is_low_stock(self, threshold: int = 10) -> bool:
        """Check if stock level is below the given threshold."""
        return self.quantity < threshold and self.status == ItemStatus.ACTIVE


def find_low_stock_items(
    items: list[InventoryItem],
    threshold: int = 10,
) -> list[InventoryItem]:
    """Return all items below the stock threshold.

    Uses a list comprehension instead of filter() — more Pythonic
    and flagged as preferred by ruff's C4 rules.
    """
    # Ruff C4 rules prefer comprehensions over list(filter(...))
    return [item for item in items if item.is_low_stock(threshold)]


def summarize_inventory(items: list[InventoryItem]) -> dict[str, float]:
    """Calculate inventory summary statistics.

    Demonstrates modern Python syntax that pyupgrade (UP) rules enforce:
    - dict[] instead of Dict[] (UP006)
    - str | None instead of Optional[str] (UP007)
    - Native type hints instead of typing imports
    """
    total_value = sum(item.total_value for item in items)
    active_count = sum(1 for item in items if item.status == ItemStatus.ACTIVE)

    return {
        "total_value": total_value,
        "active_items": float(active_count),
        "average_value": total_value / active_count if active_count else 0.0,
    }
```

### Example 2: Code Before and After Ruff Auto-Fix

```python
# BEFORE ruff check --fix
# This file has multiple violations that ruff can auto-fix.

import os                    # F401: unused import
import sys                   # F401: unused import
from typing import Optional  # UP007: use str | None
from typing import List      # UP006: use list[] instead
import json                  # I001: import not sorted
from pathlib import Path

def process_data(data: List[Optional[str]]) -> List[str]:
    result = list()          # C4: use [] instead of list()
    for item in data:
        if item is not None:
            if item != "":   # SIM: can be simplified
                result.append(item)
    return result

def get_config(path: Optional[str] = None) -> dict:
    if path == None:         # E711: use 'is None'
        path = "config.json"
    with open(path, "r") as f:  # UP015: unnecessary "r" mode
        data = json.load(f)
    return data


# AFTER ruff check --fix
# All auto-fixable violations are corrected.

import json                  # Sorted correctly (I001 fixed)
from pathlib import Path     # Unused os/sys removed (F401 fixed)


def process_data(data: list[str | None]) -> list[str]:
    # UP006: List -> list, UP007: Optional -> | None
    result = []              # C4: list() -> []
    for item in data:
        if item is not None:
            if item != "":
                result.append(item)
    return result


def get_config(path: str | None = None) -> dict:
    # UP007: Optional[str] -> str | None
    if path is None:         # E711: == None -> is None
        path = "config.json"
    with open(path) as f:    # UP015: removed unnecessary "r"
        data = json.load(f)
    return data
```

### Example 3: Configuring Per-File Ignores for Different Code Contexts

```python
# pyproject.toml excerpt:
#
# [tool.ruff.lint.per-file-ignores]
# "tests/**/*.py" = ["S101", "D103", "ANN"]
# "__init__.py" = ["F401"]
# "scripts/**/*.py" = ["T201"]  # allow print() in scripts
# "migrations/**/*.py" = ["E501", "RUF"]  # auto-generated, don't lint hard


# File: src/myapp/__init__.py
# F401 is ignored here — these imports are intentional re-exports
# that make "from myapp import Router, Config" work.

from myapp.config import Config
from myapp.router import Router
from myapp.middleware import CORSMiddleware

__all__ = ["Config", "Router", "CORSMiddleware"]


# File: tests/test_router.py
# S101 (assert usage) is ignored — pytest relies on bare assert.
# D103 (missing docstring) is ignored — test names are self-documenting.

from myapp.router import Router


def test_router_registers_route() -> None:
    router = Router()
    router.add_route("/health", lambda: {"status": "ok"})

    assert "/health" in router.routes
    assert router.routes["/health"]() == {"status": "ok"}


def test_router_rejects_duplicate_route() -> None:
    router = Router()
    router.add_route("/health", lambda: {"status": "ok"})

    try:
        router.add_route("/health", lambda: {"status": "new"})
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "already registered" in str(exc)


# File: scripts/seed_database.py
# T201 (print usage) is ignored — scripts use print for output.

from myapp.config import Config
from myapp.database import Database


def seed_data() -> None:
    """Populate database with initial data for development."""
    config = Config.from_env()
    db = Database(config.database_url)

    print("Seeding database...")  # T201 ignored in scripts/
    users = [
        {"name": "Alice", "role": "admin"},
        {"name": "Bob", "role": "user"},
    ]
    for user in users:
        db.insert("users", user)
        print(f"  Created user: {user['name']}")

    print(f"Done. Seeded {len(users)} users.")


if __name__ == "__main__":
    seed_data()
```

### Example 4: Using Ruff with isort Rules for Import Organization

```python
# Ruff's I (isort) rules enforce a specific import order:
#
# 1. __future__ imports
# 2. Standard library imports
# 3. Third-party imports
# 4. First-party imports (your project)
# 5. Local/relative imports
#
# Each group is separated by a blank line.
# Within each group, imports are sorted alphabetically.

# pyproject.toml:
# [tool.ruff.lint]
# select = ["I"]
#
# [tool.ruff.lint.isort]
# known-first-party = ["inventory"]
# force-sort-within-sections = true


# BEFORE ruff check --fix (imports are disorganized)
from inventory.models import InventoryItem
import json
from pathlib import Path
from datetime import datetime
import httpx
from inventory.database import get_connection
from collections.abc import Sequence
import sys
from pydantic import BaseModel


# AFTER ruff check --fix (imports properly organized)
import json                              # stdlib
import sys                               # stdlib
from collections.abc import Sequence     # stdlib
from datetime import datetime            # stdlib
from pathlib import Path                 # stdlib

import httpx                             # third-party
from pydantic import BaseModel           # third-party

from inventory.database import get_connection  # first-party
from inventory.models import InventoryItem     # first-party
```

### Example 5: Ruff's pyupgrade (UP) Rules in Action

```python
# pyupgrade rules modernize your code to use newer Python syntax.
# With target-version = "py312", ruff enforces Python 3.12+ idioms.


# UP006: Use builtin type for type hints (Python 3.9+)
# BAD:  from typing import List, Dict, Set, Tuple
# GOOD: use list, dict, set, tuple directly

def merge_records(
    primary: dict[str, int],
    secondary: dict[str, int],
    keys: set[str],
) -> list[tuple[str, int]]:
    """Merge two record dicts, returning key-value pairs for given keys."""
    merged = primary | secondary  # dict merge operator (3.9+)
    return [(k, merged[k]) for k in keys if k in merged]


# UP007: Use X | Y for union types (Python 3.10+)
# BAD:  from typing import Optional, Union
# GOOD: use | syntax

def find_user(user_id: int) -> dict[str, str] | None:
    """Look up a user by ID, returning None if not found."""
    users: dict[int, dict[str, str]] = {
        1: {"name": "Alice", "role": "admin"},
        2: {"name": "Bob", "role": "user"},
    }
    return users.get(user_id)


# UP035: Import from collections.abc instead of typing
# BAD:  from typing import Sequence, Mapping, Iterable
# GOOD: from collections.abc import Sequence, Mapping, Iterable

from collections.abc import Iterable, Mapping


def count_by_key(
    items: Iterable[Mapping[str, str]],
    key: str,
) -> dict[str, int]:
    """Count occurrences of each value for the given key."""
    counts: dict[str, int] = {}
    for item in items:
        value = item.get(key, "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


# UP032: Use f-string instead of format()
# BAD:  "Hello, {}!".format(name)
# GOOD: f"Hello, {name}!"

def format_summary(name: str, count: int, total: float) -> str:
    """Format an inventory summary line using f-strings."""
    return f"{name}: {count} items, ${total:,.2f} total value"


# UP040: Use PEP 695 type alias syntax (Python 3.12+)
# BAD:  UserId = int
#       from typing import TypeAlias; UserId: TypeAlias = int
# GOOD: type UserId = int

type UserId = int
type JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]


def get_user_name(user_id: UserId) -> str:
    """Demonstrate PEP 695 type alias usage."""
    return f"user_{user_id}"
```

### Example 6: Running Ruff in a CI Pipeline Script

```python
#!/usr/bin/env python3
"""Local script to run the same quality checks as CI.

Usage:
    python scripts/check_quality.py
    python scripts/check_quality.py --fix
"""

import subprocess
import sys
from dataclasses import dataclass


@dataclass
class CheckResult:
    """Result of a single quality check."""

    name: str
    command: list[str]
    passed: bool
    output: str


def run_check(name: str, command: list[str]) -> CheckResult:
    """Run a single quality check and capture its result."""
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    return CheckResult(
        name=name,
        command=command,
        passed=result.returncode == 0,
        output=result.stdout + result.stderr,
    )


def main() -> None:
    """Run all quality checks, report results, exit non-zero on failure."""
    fix_mode = "--fix" in sys.argv

    if fix_mode:
        # In fix mode, apply auto-fixes and format
        checks = [
            ("ruff check (fix)", ["ruff", "check", "--fix", "."]),
            ("ruff format", ["ruff", "format", "."]),
        ]
    else:
        # In check mode (CI), only report violations
        checks = [
            ("ruff check", ["ruff", "check", "."]),
            ("ruff format (check)", ["ruff", "format", "--check", "."]),
        ]

    results: list[CheckResult] = []
    for name, command in checks:
        result = run_check(name, command)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {name}")
        if not result.passed:
            print(result.output)

    # Summary
    print(f"\n{'='*50}")
    failed = [r for r in results if not r.passed]
    if failed:
        print(f"{len(failed)} check(s) failed:")
        for r in failed:
            print(f"  - {r.name}: {' '.join(r.command)}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
```

## Common Pitfalls

### Pitfall 1: Enabling All Rules at Once on an Existing Codebase

Enabling `select = ["ALL"]` on a codebase that was never linted produces an overwhelming wall of violations. Developers get frustrated, add hundreds of `ignore` rules, and end up with a worse configuration than if they had been selective.

```python
# BAD — enabling everything at once
# pyproject.toml:
# [tool.ruff.lint]
# select = ["ALL"]
# ignore = [
#     "D100", "D101", "D102", "D103", "D104",  # too many docstring rules
#     "ANN001", "ANN002", "ANN003", "ANN201",   # too many annotation rules
#     "S101",                                     # assert in tests
#     "T201",                                     # print statements
#     ... # 50 more ignores
# ]

# GOOD — incremental adoption, starting with high-value rules
# pyproject.toml:
# [tool.ruff.lint]
# select = [
#     "F",    # Pyflakes — catches real bugs (undefined names, unused imports)
#     "E",    # pycodestyle — basic style
#     "I",    # isort — import organization (fully auto-fixable)
#     "UP",   # pyupgrade — modernize syntax (fully auto-fixable)
# ]
#
# Phase 2 (after cleaning up phase 1):
# select = ["F", "E", "I", "UP", "B", "SIM", "C4"]
#
# Phase 3 (after stabilizing):
# select = ["F", "E", "I", "UP", "B", "SIM", "C4", "PT", "RUF", "S"]
```

### Pitfall 2: Ignoring E501 Without Understanding Why

Many developers carry over the habit of ignoring E501 (line too long) from their flake8 config. This is correct when using ruff format (which handles line length), but wrong if you are only using ruff check without ruff format.

```python
# BAD — ignoring E501 without using a formatter
# pyproject.toml:
# [tool.ruff.lint]
# ignore = ["E501"]
# (no [tool.ruff.format] usage, no format-on-save, no ruff format in CI)
#
# Result: lines grow unbounded, code becomes hard to read
very_long_variable = some_function(argument_one, argument_two, argument_three, argument_four, argument_five)

# GOOD — ignore E501 only when ruff format handles line length
# pyproject.toml:
# [tool.ruff.lint]
# ignore = ["E501"]  # formatter handles line wrapping
#
# [tool.ruff.format]
# quote-style = "double"
#
# CI runs both:
#   ruff check .
#   ruff format --check .
#
# Result: formatter breaks long lines automatically
very_long_variable = some_function(
    argument_one,
    argument_two,
    argument_three,
    argument_four,
    argument_five,
)
```

### Pitfall 3: Using --fix in CI Instead of Failing the Build

Running `ruff check --fix` in CI silently fixes violations and pushes corrected code. This means developers never learn about the issues, and the CI pipeline modifies source code — a dangerous practice.

```yaml
# BAD — CI silently fixes issues (developer never learns)
# .github/workflows/lint.yml
# steps:
#   - run: ruff check --fix .
#   - run: git add -A && git commit -m "auto-fix" && git push

# GOOD — CI reports issues, developer fixes locally
# .github/workflows/lint.yml
# steps:
#   - run: ruff check .        # Fail if any violations
#   - run: ruff format --check . # Fail if not formatted
#
# Meanwhile, pre-commit hook auto-fixes locally:
# .pre-commit-config.yaml:
# - id: ruff
#   args: [--fix]              # Auto-fix on commit (local only)
# - id: ruff-format
```

### Pitfall 4: Not Using per-file-ignores for Test Files

Test files have legitimately different conventions: they use bare `assert` (flagged by `S101`), they often lack docstrings (flagged by `D103`), and they may have longer function names (flagged by `E501`). Applying production rules to test files creates noise.

```python
# BAD — same rules for production and test code
# Results in hundreds of S101 violations in test files:
#   tests/test_user.py:15:5: S101 Use of assert detected
#   tests/test_user.py:22:5: S101 Use of assert detected
#   ... (hundreds more)

# GOOD — per-file-ignores for test conventions
# pyproject.toml:
# [tool.ruff.lint.per-file-ignores]
# "tests/**/*.py" = [
#     "S101",   # assert is how pytest works
#     "D103",   # test functions don't need docstrings
#     "ANN",    # type annotations less critical in tests
#     "ARG",    # fixture arguments may appear unused
# ]

# Now this test file is clean:
def test_user_creation() -> None:
    user = create_user("Alice", "admin")
    assert user.name == "Alice"       # S101 ignored in tests/
    assert user.role == "admin"       # S101 ignored in tests/
    assert user.is_active is True     # S101 ignored in tests/
```

### Pitfall 5: Conflicting Formatter and Linter Settings

If ruff's linter and formatter disagree on style (for example, different line lengths), they fight each other: the formatter reformats to one style, then the linter flags it.

```toml
# BAD — linter and formatter have different line lengths
# [tool.ruff]
# line-length = 120
#
# [tool.ruff.lint]
# select = ["E"]
# # E501 is enabled and uses line-length = 120
#
# [tool.ruff.format]
# line-length = 88  # WRONG — this key doesn't exist here;
#                    # format uses the top-level line-length
#
# Result: formatter wraps at 88 chars, linter allows up to 120 —
# no direct conflict here, but confusing if you intended 120 everywhere.

# GOOD — single source of truth for line length
# [tool.ruff]
# line-length = 88  # Used by both linter and formatter
#
# [tool.ruff.lint]
# select = ["F", "E", "W", "I", "UP", "B"]
# ignore = ["E501"]  # Let the formatter handle line length entirely
#
# [tool.ruff.format]
# quote-style = "double"
```

## Key Takeaways

- **Ruff replaces flake8 + isort + black + pyupgrade** with a single tool that is 10-100x faster, making it practical to lint on every keystroke rather than every commit.
- **Start with a curated `select` list** (`F`, `E`, `I`, `UP`, `B` is a strong default) and expand incrementally — enabling all rules at once on a legacy codebase creates overwhelming noise.
- **Auto-fix (`--fix`) is ruff's superpower** — it corrects the majority of violations automatically, changing linting from a burden into a background process.
- **Use `per-file-ignores`** to apply different rules to different contexts: test files, `__init__.py` re-exports, scripts, and auto-generated code all have legitimately different conventions.
- **Pair `ruff check` with `ruff format`** and ignore `E501` in the linter — let the formatter own line length, and let the linter own semantic correctness.

## Exercises

1. **Configure ruff for a new project.** Create a `pyproject.toml` with a `[tool.ruff]` section that targets Python 3.12, sets line length to 88, and enables the rule categories `F`, `E`, `W`, `I`, `UP`, `B`, `SIM`, and `RUF`. Add `per-file-ignores` that allow `assert` in test files and unused imports in `__init__.py`. Explain why you chose each rule category.

2. **Analyze ruff output.** Given this ruff output, explain what each violation means, whether it is auto-fixable, and what the corrected code would look like:
   ```
   app/main.py:1:1: F401 [*] `os` imported but unused
   app/main.py:5:1: I001 [*] Import block is un-sorted or un-formatted
   app/main.py:12:20: UP007 [*] Use `str | None` instead of `Optional[str]`
   app/main.py:25:9: B006 Do not use mutable data structures for argument defaults
   app/main.py:31:5: SIM108 [*] Use ternary operator instead of `if`-`else`-block
   ```

3. **Migrate from flake8.** You have a project with this `.flake8` configuration. Write the equivalent `[tool.ruff]` configuration in `pyproject.toml`:
   ```ini
   [flake8]
   max-line-length = 100
   extend-ignore = E203, W503
   per-file-ignores =
       tests/*.py: S101, D103
       __init__.py: F401
   max-complexity = 10
   ```

4. **Design a CI pipeline.** Write a GitHub Actions workflow that runs `ruff check` and `ruff format --check` on every pull request. The workflow should also run `ruff check --fix --diff` to show what auto-fixes are available (as a comment, not applied). Explain why CI should not auto-fix and push.

5. **Incremental adoption plan.** You are adding ruff to a 50,000-line Python project that has never used a linter. Running `ruff check --select ALL .` shows 4,200 violations. Design a phased adoption plan: which rule categories would you enable first, second, and third? How would you use `--fix` versus manual fixes? What `per-file-ignores` would you start with?

---
up:: [Schedule](../../Schedule.md)
#type/learning #source/self-study #status/seed
