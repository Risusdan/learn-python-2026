# Python Expert Learning Schedule

**Scope**: All required topics from the [Learning Path](LearningPath.md) (OPTIONAL topics excluded)

## Progress Tracker

| Week | Topic                                          | Status         |
| ---- | ---------------------------------------------- | -------------- |
| 1    | Basics Review                                  | ⬜ Not started |
| 2    | Data Structures + Exceptions                   | ⬜ Not started |
| 3    | Functions, Lambdas, Iterators, Decorators      | ⬜ Not started |
| 4    | OOP                                            | ⬜ Not started |
| 5    | Modules, Variable Scope, Regular Expressions   | ⬜ Not started |
| 6    | File Handling, Context Managers, Comprehensions, Generators | ⬜ Not started |
| 7    | Package Management + Static Typing             | ⬜ Not started |
| 8    | Code Quality: ruff + Testing: pytest           | ⬜ Not started |
| 9    | Concurrency                                    | ⬜ Not started |
| 10   | FastAPI Framework + Capstone Project            | ⬜ Not started |

---

## Week 1: Basics Review

*Review week — skim quickly if you already know this material.*

- [ ] Basic Syntax — indentation-based blocks, comments, `print()`
- [ ] Variables and Data Types — dynamic typing, `int`, `float`, `str`, `bool`, `None`
- [ ] Working with Strings — f-strings, slicing, common methods (`split`, `join`, `strip`)
- [ ] Conditionals — `if` / `elif` / `else`, truthiness, ternary expressions
- [ ] Loops — `for`, `while`, `break`, `continue`, `else` clause on loops
- [ ] Type Casting — `int()`, `float()`, `str()`, `bool()`, implicit vs explicit conversion

**Why It Matters**: Python's simplicity is deceptive — even basics have depth. Understanding truthiness (empty containers are falsy), string interning, and how `for` loops work with iterators (not indices) sets you apart. These foundations make everything that follows intuitive rather than magical.

---

## Week 2: Data Structures + Exceptions

- [ ] Lists — mutable sequences, indexing, slicing, common methods (`append`, `extend`, `pop`)
- [ ] Tuples — immutable sequences, packing/unpacking, named tuples
- [ ] Sets — unique elements, set operations (union, intersection, difference)
- [ ] Dictionaries — key-value pairs, `get()`, `setdefault()`, dictionary views
- [ ] Exceptions — `try` / `except` / `else` / `finally`, raising exceptions, custom exception classes

**Why It Matters**: Python's built-in data structures are highly optimized C implementations. Knowing *when* to use each one (list for ordered data, set for membership testing, dict for key lookups) directly affects performance. Exceptions in Python aren't just for errors — "easier to ask forgiveness than permission" (EAFP) is a core Python idiom, unlike the "look before you leap" style of other languages.

---

## Week 3: Functions, Lambdas, Iterators, Decorators

- [ ] Functions — `def`, parameters, return values, `*args`, `**kwargs`
- [ ] Builtin Functions — `map`, `filter`, `zip`, `enumerate`, `any`, `all`, `sorted`
- [ ] Lambdas — anonymous single-expression functions, use with `sorted()`, `map()`, `filter()`
- [ ] Iterators — the iterator protocol (`__iter__`, `__next__`), `StopIteration`
- [ ] Decorators — functions that wrap functions, `@decorator` syntax, preserving metadata with `functools.wraps`

**Why It Matters**: Functions in Python are first-class objects — they can be passed as arguments, returned from other functions, and stored in data structures. This is the foundation of decorators, which are everywhere in Python frameworks (`@app.route` in Flask/FastAPI, `@property` in classes, `@pytest.fixture` in testing). Understanding the iterator protocol explains how `for` loops, comprehensions, and generators all work under the hood.

---

## Week 4: OOP

- [ ] Classes — `class` keyword, `__init__`, instance vs class attributes
- [ ] Methods — instance methods, class methods (`@classmethod`), static methods (`@staticmethod`)
- [ ] Inheritance — single and multiple inheritance, `super()`, MRO (Method Resolution Order)
- [ ] Encapsulation — name mangling (`__private`), conventions (`_protected`), `@property`

**Why It Matters**: Python's OOP is flexible but opinionated. There are no true private attributes — the underscore convention (`_private`) is a social contract, not enforcement. Understanding MRO is critical when using multiple inheritance (which Python supports fully, unlike Java). The `@property` decorator lets you turn attribute access into method calls without changing the API — essential for maintaining backward compatibility.

---

## Week 5: Modules, Variable Scope, Regular Expressions

- [ ] Builtin Modules — `os`, `sys`, `json`, `datetime`, `pathlib`, `collections`
- [ ] Custom Modules — creating and importing your own modules, `__init__.py`, package structure
- [ ] Variable Scope — LEGB rule (Local, Enclosing, Global, Builtin), `global` and `nonlocal` keywords
- [ ] Regular Expressions — `re` module, patterns, `match`, `search`, `findall`, `sub`, groups

**Why It Matters**: The LEGB scope rule is Python's answer to "where does this variable come from?" — and getting it wrong causes subtle bugs (especially with mutable default arguments and closures). Modules are Python's primary code organization mechanism — understanding `__init__.py` and package structure is essential for any project beyond a single file. Regular expressions are a universal skill that appears in data validation, log parsing, and text processing across every domain.

---

## Week 6: File Handling, Context Managers, Comprehensions, Generators

- [ ] File Handling — `open()`, read/write modes, reading lines, working with paths via `pathlib`
- [ ] Context Managers — `with` statement, `__enter__` / `__exit__`, `contextlib.contextmanager`
- [ ] List Comprehensions — `[expr for item in iterable if condition]`, nested comprehensions
- [ ] Generator Expressions — `(expr for item in iterable)`, lazy evaluation, memory efficiency
- [ ] Paradigms — functional vs OOP style in Python, when to use each

**Why It Matters**: Context managers (`with` statement) are Python's equivalent of C++'s RAII — they guarantee cleanup regardless of exceptions. Files get closed, locks get released, database connections get returned to the pool. Generators are Python's killer feature for memory efficiency — they produce values one at a time instead of building entire lists in memory, making it possible to process datasets larger than RAM.

---

## Week 7: Package Management + Static Typing

- [ ] PyPI — the Python Package Index, finding and evaluating packages
- [ ] uv — fast Python package and project manager, replacing pip/venv/poetry in one tool
- [ ] pyproject.toml — modern Python project configuration, dependency specification
- [ ] `typing` module — type hints (`int`, `str`, `List[int]`, `Optional`, `Union`, `Dict`)
- [ ] mypy — static type checker, running checks, configuring strictness
- [ ] Pydantic — runtime data validation using type annotations, `BaseModel`, field validators

**Why It Matters**: `uv` is rapidly becoming the standard Python tool because it replaces pip, venv, and poetry with a single, much faster alternative. Type hints transform Python from "hope it works at runtime" to "catch bugs before running." mypy checks types statically (at analysis time), while Pydantic validates data at runtime (from APIs, files, user input) — together they cover both sides of type safety.

---

## Week 8: Code Quality: ruff + Testing: pytest

- [ ] ruff — ultra-fast Python linter and formatter, replacing flake8/isort/black in one tool
- [ ] Configuring ruff — `pyproject.toml` settings, rule selection, auto-fix
- [ ] pytest — writing tests with `assert`, test discovery, fixtures, parametrize
- [ ] Test organization — `tests/` directory structure, conftest.py, naming conventions
- [ ] Mocking — `unittest.mock.patch`, `MagicMock`, isolating units under test

**Why It Matters**: ruff and pytest are the modern Python quality stack — both are fast, well-designed, and widely adopted. ruff catches hundreds of common mistakes instantly (unused imports, style violations, potential bugs) and auto-fixes most of them. pytest's fixture system is revolutionary — instead of setUp/tearDown methods, you declare dependencies by name and pytest injects them automatically. This makes tests composable, readable, and maintainable.

---

## Week 9: Concurrency

- [ ] Threading — `threading` module, thread creation, when threading helps (I/O-bound tasks)
- [ ] GIL (Global Interpreter Lock) — what it is, why it exists, how it limits CPU-bound threading
- [ ] Multiprocessing — `multiprocessing` module, bypassing the GIL for CPU-bound work
- [ ] Asynchrony — `asyncio`, `async` / `await`, event loops, when async is the right choice
- [ ] Choosing the right model — threading for I/O, multiprocessing for CPU, asyncio for high-concurrency I/O

**Why It Matters**: The GIL is Python's most misunderstood feature. It prevents multiple threads from executing Python bytecode simultaneously — but threads still help for I/O-bound work (network calls, file reads) because the GIL is released during I/O. For CPU-bound work, `multiprocessing` spawns separate processes with their own GIL. `asyncio` handles thousands of concurrent connections in a single thread — the model behind modern Python web frameworks like FastAPI.

---

## Week 10: FastAPI Framework + Capstone Project

- [ ] FastAPI — project structure, creating routes, path/query parameters
- [ ] Request/Response — Pydantic models for validation, status codes, JSON responses
- [ ] Dependency Injection — FastAPI's `Depends()`, reusable dependencies
- [ ] Async endpoints — `async def` vs `def`, when async matters for performance
- [ ] Middleware and error handling — custom middleware, exception handlers
- [ ] **Capstone Project** — build a complete REST API that ties together everything learned

**Why It Matters**: FastAPI is the fastest-growing Python web framework because it combines the best ideas: automatic API documentation (OpenAPI/Swagger), Pydantic for validation, native async support, and dependency injection. Building a real API forces you to combine every skill — types, testing, project structure, async, error handling — into a cohesive application. This is where learning becomes practical ability.

---
up:: [MOC-Programming](../../../01-index/MOC-Programming.md)
#type/learning #source/self-study #status/seed
