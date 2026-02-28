# Functions and Lambdas

> Functions are Python's primary unit of code reuse and abstraction, and because they are first-class objects, they unlock patterns like callbacks, closures, and decorators that shape how real Python code is written.

## Table of Contents
- [Core Concepts](#core-concepts)
- [Code Examples](#code-examples)
- [Common Pitfalls](#common-pitfalls)
- [Key Takeaways](#key-takeaways)
- [Exercises](#exercises)

## Core Concepts

### Function Definition and the `def` Statement

#### What

A function in Python is created with the `def` keyword. When Python executes `def`, it does not run the function body — it creates a **function object** and binds it to the name you provide. That object is a regular Python value: it has a type (`function`), it lives in memory, and it can be assigned to variables, stored in lists, or passed as an argument.

#### How

```python
def greet(name: str) -> str:
    """Return a greeting for the given name."""
    return f"Hello, {name}"
```

When the interpreter hits `def greet(...)`, three things happen: (1) the bytecode for the function body is compiled, (2) a `function` object is created wrapping that bytecode plus the function's default argument values and closure cells, and (3) the name `greet` is bound to that object in the current namespace. You can verify this — `type(greet)` returns `<class 'function'>`, and `greet.__code__` exposes the compiled bytecode object.

#### Why It Matters

Understanding that `def` is an **executable statement** (not a declaration) explains many Python behaviors. Functions can be defined inside `if` blocks, inside loops, inside other functions. They can be redefined at any time. This is not a quirk — it is the foundation of closures and factory functions, which you will use constantly in production Python.

### Parameters: The Five Kinds

#### What

Python supports five parameter types, and they interact through a precise resolution order. From left to right in a function signature: positional-only (before `/`), regular positional-or-keyword, `*args` (catch-all positional), keyword-only (after `*` or `*args`), and `**kwargs` (catch-all keyword).

#### How

The full signature grammar looks like this:

```python
def example(
    pos_only: int,         # positional-only (before /)
    /,
    normal: str,           # positional-or-keyword
    *args: float,          # catch-all positional
    kw_only: bool = True,  # keyword-only (after * or *args)
    **kwargs: str,         # catch-all keyword
) -> None:
    ...
```

When you call a function, Python resolves arguments in this order:

1. Positional arguments fill slots left-to-right up to `*args`.
2. `*args` absorbs any remaining positional arguments as a tuple.
3. Keyword arguments fill named parameters or flow into `**kwargs`.
4. Default values fill any parameters that received no argument.

The `/` separator (PEP 570, Python 3.8+) marks everything before it as positional-only. This is how many builtin functions work — `len(obj)` does not accept `len(obj=mylist)`. You can use `/` in your own functions to keep parameter names as implementation details.

#### Why It Matters

Parameter design is API design. Positional-only parameters let you rename internal parameters without breaking callers. Keyword-only parameters (after `*`) prevent accidental positional usage of boolean flags — `connect(host, port, *, ssl=True)` is much harder to misuse than `connect(host, port, True)`. Mastering parameter types lets you write functions that are both flexible and safe.

### Default Parameter Values

#### What

Default values are evaluated **once**, at function definition time, not at each call. This is the single most common source of bugs for people coming from other languages.

#### How

```python
# The default value [] is created ONCE when def executes
def append_to(item: int, target: list[int] = []) -> list[int]:
    target.append(item)
    return target
```

Calling `append_to(1)` then `append_to(2)` returns `[1, 2]` the second time — the same list object is reused across calls. The default value is stored on the function object itself at `append_to.__defaults__`.

The standard fix uses `None` as a sentinel:

```python
def append_to(item: int, target: list[int] | None = None) -> list[int]:
    if target is None:
        target = []
    target.append(item)
    return target
```

#### Why It Matters

This is not a language bug — it is a direct consequence of `def` being an executable statement. The default value expression runs once when `def` runs. Understanding this teaches you to think about Python's execution model rather than memorizing rules. The `None` sentinel pattern appears everywhere in production code, standard library, and frameworks.

### `*args` and `**kwargs`

#### What

`*args` collects extra positional arguments into a tuple. `**kwargs` collects extra keyword arguments into a dictionary. Together, they let a function accept arbitrary arguments — essential for writing wrappers, decorators, and forwarding functions.

#### How

```python
def log_call(func_name: str, *args: object, **kwargs: object) -> None:
    """Log a function call with its arguments."""
    arg_str = ", ".join(repr(a) for a in args)
    kwarg_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    all_args = ", ".join(filter(None, [arg_str, kwarg_str]))
    print(f"{func_name}({all_args})")
```

The `*` and `**` operators also work on the **calling side** for unpacking:

```python
def point(x: float, y: float, z: float) -> str:
    return f"({x}, {y}, {z})"

coords = (1.0, 2.0, 3.0)
point(*coords)  # Unpacks tuple into positional args

config = {"x": 1.0, "y": 2.0, "z": 3.0}
point(**config)  # Unpacks dict into keyword args
```

#### Why It Matters

`*args` and `**kwargs` are the mechanism behind decorators, proxy objects, and framework magic. When you see `@app.route("/users")` in FastAPI, the decorator uses `*args`/`**kwargs` to forward your function's arguments transparently. Without these, writing generic wrappers would be impossible.

### Return Values

#### What

Every Python function returns a value. If no `return` statement is hit, the function returns `None`. A function can return multiple values by returning a tuple, which callers unpack with tuple unpacking.

#### How

```python
from math import sqrt

def quadratic(a: float, b: float, c: float) -> tuple[float, float]:
    """Solve ax^2 + bx + c = 0. Returns both roots."""
    discriminant = b**2 - 4 * a * c
    root = sqrt(discriminant)
    return (-b + root) / (2 * a), (-b - root) / (2 * a)

x1, x2 = quadratic(1, -5, 6)  # Tuple unpacking
```

When a function returns `x, y`, Python creates a single tuple `(x, y)`. The caller unpacking is syntactic sugar — `x1, x2 = result` is the same as `x1 = result[0]; x2 = result[1]`. For more than 2-3 return values, consider returning a `NamedTuple` or `dataclass` for clarity.

#### Why It Matters

The implicit `return None` means every function is an expression that produces a value. This is why you can do `result = some_function()` and always get something — it is never "void" like in C. Understanding this explains why methods like `list.sort()` return `None` (they mutate in-place; returning `None` is a deliberate signal that no new object was created).

### Functions as First-Class Objects

#### What

In Python, functions are objects with the same rights as any other object. They have a type, they have attributes, they can be assigned to variables, passed to other functions, returned from functions, and stored in data structures. This is what "first-class" means.

#### How

```python
def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    return a - b

# Functions stored in a dictionary
operations: dict[str, callable] = {
    "+": add,
    "-": subtract,
}

# Function passed as argument
def apply(func: callable, x: int, y: int) -> int:
    return func(x, y)

result = apply(operations["+"], 10, 3)  # 13
```

You can inspect function objects at runtime: `add.__name__` gives `"add"`, `add.__doc__` gives the docstring, `add.__code__.co_varnames` lists the local variable names. Functions even support arbitrary attributes — `add.call_count = 0` works.

#### Why It Matters

First-class functions are the foundation of functional programming patterns in Python. Callbacks, decorators, strategy patterns, command patterns — all rely on treating functions as values. When you pass `key=str.lower` to `sorted()`, you are passing a function object as an argument. This is not a special feature — it is how Python works by default.

### Closures

#### What

A closure is a function that captures variables from its enclosing scope. When an inner function references a variable from an outer function, Python creates a closure — the inner function "remembers" the enclosing variable even after the outer function has returned.

#### How

```python
def make_multiplier(factor: int) -> callable:
    """Return a function that multiplies its argument by factor."""
    def multiplier(x: int) -> int:
        return x * factor  # 'factor' is captured from the enclosing scope
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

double(5)   # 10
triple(5)   # 15
```

Under the hood, Python stores the captured variable in a **cell object** on the inner function. You can see this with `double.__closure__[0].cell_contents`, which returns `2`. The closure does not copy the value — it holds a reference to the cell, which means if the enclosed variable changes, the closure sees the change.

#### Why It Matters

Closures are the mechanism behind decorators, factory functions, and partial application. They let you create specialized functions from general ones without using classes. Understanding closures also explains one of Python's most confusing gotchas: the "late binding closure" problem in loops (covered in Common Pitfalls below).

### Builtin Functions: `map`, `filter`, `zip`, `enumerate`

#### What

Python provides a set of builtin higher-order functions that operate on iterables. These functions return **lazy iterators**, not lists — they produce values one at a time and consume minimal memory regardless of input size.

#### How

**`map(func, iterable)`** — applies `func` to every element:

```python
numbers = [1, 2, 3, 4]
squared = list(map(lambda x: x ** 2, numbers))  # [1, 4, 9, 16]
```

**`filter(func, iterable)`** — keeps elements where `func` returns truthy:

```python
numbers = [1, 2, 3, 4, 5, 6]
evens = list(filter(lambda x: x % 2 == 0, numbers))  # [2, 4, 6]
```

**`zip(*iterables)`** — pairs elements from multiple iterables, stopping at the shortest:

```python
names = ["Alice", "Bob", "Charlie"]
scores = [95, 87, 92]
paired = list(zip(names, scores))  # [("Alice", 95), ("Bob", 87), ("Charlie", 92)]
```

Use `itertools.zip_longest` if you need to fill in missing values for unequal-length iterables.

**`enumerate(iterable, start=0)`** — yields `(index, value)` pairs:

```python
for i, name in enumerate(names, start=1):
    print(f"{i}. {name}")
```

#### Why It Matters

These builtins embody Python's iterator protocol. They are lazy, composable, and memory-efficient. However, in modern Python, **comprehensions are generally preferred** over `map()` and `filter()` for readability:

```python
# Preferred — list comprehension
squared = [x ** 2 for x in numbers]
evens = [x for x in numbers if x % 2 == 0]
```

Use `map()` when you already have a named function (`map(str.upper, words)`) and a comprehension would just add noise. Use `zip()` and `enumerate()` everywhere — they have no comprehension equivalent and are universally considered Pythonic.

### Builtin Functions: `any`, `all`, `sorted`

#### What

`any()` returns `True` if any element is truthy. `all()` returns `True` if all elements are truthy. `sorted()` returns a new sorted list from any iterable. All three accept iterables and short-circuit or defer evaluation where possible.

#### How

```python
scores = [85, 92, 78, 95, 88]

# any: is there at least one perfect score?
has_perfect = any(s == 100 for s in scores)  # False

# all: did everyone pass (>= 60)?
all_passed = all(s >= 60 for s in scores)    # True

# sorted: rank from highest to lowest
ranked = sorted(scores, reverse=True)         # [95, 92, 88, 85, 78]
```

`any()` and `all()` are **short-circuiting**: `any()` stops at the first truthy value, `all()` stops at the first falsy value. They accept generator expressions directly — no need to build a list first.

`sorted()` accepts a `key` function that extracts a comparison key from each element:

```python
words = ["banana", "apple", "cherry"]
sorted(words, key=len)               # ["apple", "banana", "cherry"]
sorted(words, key=str.lower)         # ["apple", "banana", "cherry"]
```

The `key` parameter is where lambdas shine — providing a one-off sort criterion without defining a named function.

#### Why It Matters

`any()` and `all()` replace manual flag-checking loops. Instead of initializing a boolean and looping through a collection, you express the intent directly: "are any of these true?" This is more readable, more Pythonic, and less error-prone. `sorted()` with `key=` is the idiomatic way to sort in Python — never write a custom comparison function when you can extract a sort key instead.

### Lambda Functions

#### What

A lambda is an anonymous function defined with a single expression. It is syntactically restricted to one expression — no statements, no assignments, no multi-line logic. Lambdas are most useful as short, throwaway functions passed as arguments.

#### How

```python
# Named function equivalent
def square(x: int) -> int:
    return x ** 2

# Lambda equivalent
square = lambda x: x ** 2  # noqa: E731 — for demonstration only
```

The lambda syntax is `lambda parameters: expression`. The expression is implicitly returned. Lambdas support all parameter types — defaults, `*args`, `**kwargs`:

```python
# Lambda with default parameter
greet = lambda name, greeting="Hello": f"{greeting}, {name}"

# Lambda with *args
add_all = lambda *args: sum(args)
```

The real value of lambdas is **inline usage** where defining a named function would be overkill:

```python
# Sort a list of dicts by a specific key
users = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35},
]
sorted(users, key=lambda u: u["age"])

# Sort by multiple criteria: age descending, then name ascending
sorted(users, key=lambda u: (-u["age"], u["name"]))
```

#### Why It Matters

Lambdas exist because sometimes you need a tiny function exactly once. Writing `def compare_by_age(u): return u["age"]` is noise when `lambda u: u["age"]` is immediately clear. However, lambdas should stay short — the moment you are tempted to add logic, write a named function instead. PEP 8 explicitly discourages assigning lambdas to names (`square = lambda x: x ** 2`); use `def` for that.

### Type Hints on Functions

#### What

Type hints annotate a function's parameters and return type. They are **not enforced at runtime** by Python itself — they are metadata for humans, IDEs, and type checkers like mypy.

#### How

```python
from collections.abc import Sequence, Callable

def apply_to_all(
    func: Callable[[int], int],
    items: Sequence[int],
) -> list[int]:
    """Apply func to every item and return the results."""
    return [func(item) for item in items]
```

For complex types, use `collections.abc` types (`Sequence`, `Mapping`, `Callable`) in function signatures rather than concrete types (`list`, `dict`). This makes your functions more flexible — they accept any sequence, not just lists.

For functions that may return nothing meaningful, annotate with `-> None`. For functions that may return a value or `None`, use `str | None` (PEP 604 union syntax, Python 3.10+).

#### Why It Matters

Type hints transform Python from a dynamic-only language to one with optional static analysis. In a large codebase, they are indispensable — they document intent, catch bugs before runtime, and enable IDE autocompletion. Professional Python code always has type hints.

## Code Examples

### Example 1: Configuration Parser with Flexible Parameters

```python
from typing import Any


def parse_config(
    path: str,
    /,
    *,
    encoding: str = "utf-8",
    strict: bool = False,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse a key=value config file into a dictionary.

    Args:
        path: File path (positional-only — callers cannot use path= keyword).
        encoding: File encoding.
        strict: If True, raise on duplicate keys.
        defaults: Base dict to merge config into.

    Returns:
        Parsed configuration dictionary.
    """
    config: dict[str, Any] = dict(defaults) if defaults else {}

    with open(path, encoding=encoding) as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                raise ValueError(f"Line {line_num}: missing '=' separator")

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            if strict and key in config:
                raise KeyError(f"Line {line_num}: duplicate key '{key}'")

            config[key] = value

    return config


# Usage
config = parse_config(
    "app.conf",
    strict=True,
    defaults={"debug": "false", "port": "8080"},
)
```

This example demonstrates positional-only parameters (`path` before `/`), keyword-only parameters (everything after `*`), default values including the `None` sentinel pattern, and type hints using PEP 604 union syntax.

### Example 2: Function Dispatch Table

```python
from collections.abc import Callable

type MathOp = Callable[[float, float], float]


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a - b


def multiply(a: float, b: float) -> float:
    return a * b


def divide(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


# Functions as values in a dictionary
OPERATIONS: dict[str, MathOp] = {
    "+": add,
    "-": subtract,
    "*": multiply,
    "/": divide,
}


def calculate(expression: str) -> float:
    """Evaluate a simple 'a op b' expression.

    Demonstrates first-class functions: the operator string
    is used to look up a function object in a dictionary.
    """
    parts = expression.split()
    if len(parts) != 3:
        raise ValueError(f"Expected 'a op b', got: {expression!r}")

    left, op, right = parts

    if op not in OPERATIONS:
        raise ValueError(f"Unknown operator: {op!r}")

    func = OPERATIONS[op]  # Retrieve the function object
    return func(float(left), float(right))


# Usage
print(calculate("10 + 3"))   # 13.0
print(calculate("10 / 3"))   # 3.333...
```

### Example 3: Closure-Based Counter Factory

```python
from collections.abc import Callable


def make_counter(start: int = 0, step: int = 1) -> Callable[[], int]:
    """Return a function that returns the next count each time it's called.

    Demonstrates closures: the inner function captures 'current'
    from the enclosing scope and modifies it via 'nonlocal'.
    """
    current = start - step  # Offset so first call returns 'start'

    def counter() -> int:
        nonlocal current
        current += step
        return current

    return counter


# Each call to make_counter creates an independent closure
counter_a = make_counter()          # 0, 1, 2, 3, ...
counter_b = make_counter(100, 10)   # 100, 110, 120, ...

print(counter_a())  # 0
print(counter_a())  # 1
print(counter_b())  # 100
print(counter_b())  # 110
print(counter_a())  # 2 — counter_a is independent of counter_b
```

### Example 4: Composing Builtins and Lambdas for Data Processing

```python
from collections.abc import Sequence


def summarize_scores(
    students: Sequence[dict[str, str | int]],
    *,
    min_score: int = 0,
    sort_by: str = "score",
) -> list[dict[str, str | int]]:
    """Filter, sort, and annotate student score records.

    Demonstrates: filter, sorted, enumerate, any, all — all combined
    with lambdas for a realistic data processing pipeline.
    """
    # Filter: keep only students above the minimum score
    qualified = filter(lambda s: s["score"] >= min_score, students)

    # Sort by the requested field
    ranked = sorted(qualified, key=lambda s: s[sort_by], reverse=True)

    # Annotate with rank using enumerate
    for rank, student in enumerate(ranked, start=1):
        student["rank"] = rank

    return ranked


students = [
    {"name": "Alice", "score": 92},
    {"name": "Bob", "score": 67},
    {"name": "Charlie", "score": 85},
    {"name": "Diana", "score": 78},
    {"name": "Eve", "score": 95},
]

# Get top students with score >= 80
top = summarize_scores(students, min_score=80, sort_by="score")
for s in top:
    print(f"#{s['rank']} {s['name']}: {s['score']}")
# #1 Eve: 95
# #2 Alice: 92
# #3 Charlie: 85

# Use any/all for quick checks
has_failing = any(s["score"] < 60 for s in students)       # False
all_above_50 = all(s["score"] > 50 for s in students)      # True
```

### Example 5: Multi-Key Sorting with Lambdas and `operator.attrgetter`

```python
from dataclasses import dataclass
from operator import attrgetter


@dataclass
class Employee:
    name: str
    department: str
    salary: int
    years: int


employees = [
    Employee("Alice", "Engineering", 120_000, 5),
    Employee("Bob", "Engineering", 110_000, 3),
    Employee("Charlie", "Marketing", 95_000, 7),
    Employee("Diana", "Engineering", 120_000, 8),
    Employee("Eve", "Marketing", 105_000, 2),
]

# Sort by department (ascending), then salary (descending), then name
# Lambda approach — clear for simple cases
by_dept_salary = sorted(
    employees,
    key=lambda e: (e.department, -e.salary, e.name),
)

# operator.attrgetter approach — faster for attribute access, but no negation
by_name = sorted(employees, key=attrgetter("name"))

# Multi-level: stable sort trick for complex ordering
# Python's sort is stable — equal elements keep their original order.
# Sort by secondary key first, then primary key.
by_years_then_dept = sorted(employees, key=attrgetter("years"), reverse=True)
by_years_then_dept = sorted(by_years_then_dept, key=attrgetter("department"))

for emp in by_dept_salary:
    print(f"{emp.department:15} {emp.salary:>8,}  {emp.name}")
# Engineering     120,000  Alice
# Engineering     120,000  Diana
# Engineering     110,000  Bob
# Marketing       105,000  Eve
# Marketing        95,000  Charlie
```

### Example 6: Generic Retry Function Using `*args` / `**kwargs`

```python
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry(
    func: Callable[..., T],
    *args: object,
    max_attempts: int = 3,
    delay: float = 1.0,
    **kwargs: object,
) -> T:
    """Call func with retries on exception.

    Demonstrates *args/**kwargs for transparent argument forwarding.
    The caller passes the target function's arguments through retry(),
    which forwards them without knowing or caring what they are.
    """
    last_exception: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exception = exc
            print(f"Attempt {attempt}/{max_attempts} failed: {exc}")
            if attempt < max_attempts:
                time.sleep(delay)

    raise RuntimeError(
        f"All {max_attempts} attempts failed"
    ) from last_exception


# Usage — retry transparently forwards arguments to the target function
def fetch_data(url: str, *, timeout: int = 30) -> str:
    """Simulate a network call that might fail."""
    import random
    if random.random() < 0.5:
        raise ConnectionError(f"Failed to connect to {url}")
    return f"Data from {url}"

result = retry(fetch_data, "https://api.example.com", max_attempts=5, timeout=10)
```

## Common Pitfalls

### Pitfall 1: Mutable Default Arguments

The most common Python gotcha. Default values are evaluated once at function definition time, so mutable defaults are shared across all calls.

```python
# BAD — mutable default is shared across calls
def add_item(item: str, items: list[str] = []) -> list[str]:
    items.append(item)
    return items

print(add_item("a"))  # ["a"]
print(add_item("b"))  # ["a", "b"] — not ["b"]!

# GOOD — use None sentinel and create a new list each call
def add_item(item: str, items: list[str] | None = None) -> list[str]:
    if items is None:
        items = []
    items.append(item)
    return items

print(add_item("a"))  # ["a"]
print(add_item("b"))  # ["b"] — correct!
```

This applies to all mutable types: `list`, `dict`, `set`, and any custom mutable object. The `None` sentinel pattern is the universal fix.

### Pitfall 2: Late Binding Closures in Loops

Closures capture the **variable**, not the **value**. In a loop, all closures share the same loop variable, so they all see its final value.

```python
# BAD — all functions return 4 (the last value of i)
functions = []
for i in range(5):
    functions.append(lambda: i)

print([f() for f in functions])  # [4, 4, 4, 4, 4]

# GOOD — capture the current value as a default argument
functions = []
for i in range(5):
    functions.append(lambda i=i: i)  # Default arg evaluated at definition time

print([f() for f in functions])  # [0, 1, 2, 3, 4]
```

The `i=i` trick works because default arguments are evaluated when `lambda` (or `def`) is executed, capturing the current value of `i` at each iteration.

### Pitfall 3: Assigning Lambdas to Names

PEP 8 (E731) explicitly discourages assigning a lambda to a name. If you need a named function, use `def` — it gives you a proper name in tracebacks and supports docstrings.

```python
# BAD — lambda assigned to a name
square = lambda x: x ** 2  # E731 lint warning

# The traceback shows: <lambda>
# No docstring possible
# Harder to debug

# GOOD — use def for named functions
def square(x: int) -> int:
    """Return the square of x."""
    return x ** 2

# The traceback shows: square
# Has a docstring
# Full type hints on parameters and return
```

Lambdas are for anonymous inline use — `sorted(items, key=lambda x: x.name)`. If you are giving it a name, use `def`.

### Pitfall 4: Forgetting That `map` and `filter` Return Iterators

`map()` and `filter()` return lazy iterators, not lists. They can only be consumed once, and printing them shows the iterator object, not the contents.

```python
# BAD — treating map result as a list
result = map(str.upper, ["hello", "world"])
print(result)       # <map object at 0x...>  — not the values!
print(list(result)) # ["HELLO", "WORLD"]
print(list(result)) # [] — already exhausted!

# GOOD — consume into a list if you need to reuse, or iterate directly
words = ["hello", "world"]

# Option 1: convert to list if needed multiple times
upper_words = list(map(str.upper, words))

# Option 2: prefer comprehension (more Pythonic)
upper_words = [w.upper() for w in words]

# Option 3: iterate directly if single-use
for word in map(str.upper, words):
    print(word)
```

### Pitfall 5: Using `*args` / `**kwargs` When Explicit Parameters Are Better

`*args` and `**kwargs` are powerful but sacrifice clarity. If you know what arguments your function accepts, spell them out.

```python
# BAD — opaque signature hides the API
def create_user(**kwargs: object) -> dict[str, object]:
    """What keys does this accept? Who knows!"""
    return {
        "name": kwargs.get("name", "Anonymous"),
        "email": kwargs.get("email"),
        "role": kwargs.get("role", "user"),
    }

# GOOD — explicit parameters with type hints and defaults
def create_user(
    name: str = "Anonymous",
    *,
    email: str | None = None,
    role: str = "user",
) -> dict[str, str | None]:
    """Clear API: every parameter is documented in the signature."""
    return {"name": name, "email": email, "role": role}
```

Reserve `*args`/`**kwargs` for genuine forwarding scenarios (decorators, proxy functions) where you truly do not know the arguments in advance.

## Key Takeaways

- **Functions are objects**: `def` creates a function object and binds it to a name. Functions can be passed as arguments, returned from other functions, and stored in data structures — this is the foundation of closures, decorators, and higher-order programming.
- **Parameter design is API design**: Use positional-only (`/`) to protect internal names, keyword-only (`*`) to prevent positional misuse of flags, and `*args`/`**kwargs` only for genuine forwarding — explicit parameters are almost always better.
- **Default values are evaluated once**: Mutable defaults (`list`, `dict`, `set`) are shared across calls. Always use the `None` sentinel pattern for mutable defaults.
- **Prefer comprehensions over `map`/`filter`**: List comprehensions are more readable and Pythonic. Use `map()` only when you already have a named function. Always use `zip()` and `enumerate()` — they have no better alternative.
- **Lambdas are for inline use only**: If you are naming a lambda, use `def` instead. Lambdas shine as one-off `key=` arguments to `sorted()`, `max()`, `min()`, and similar functions.

## Exercises

1. **Parameter types**: Write a function `format_table(headers, *rows, separator="| ", padding=1)` that takes a list of header strings and any number of row tuples, then returns a formatted ASCII table as a string. Use keyword-only parameters for `separator` and `padding`. Include full type hints.

2. **Closure factory**: Write a function `make_validator(min_val, max_val)` that returns a function. The returned function should accept a single number and return `True` if it is within the range `[min_val, max_val]`, inclusive. Then use it to create validators for different ranges and test them.

3. **Builtin composition**: Given a list of dictionaries representing products (`{"name": str, "price": float, "in_stock": bool}`), write a single expression using `filter()`, `sorted()`, and a lambda to get all in-stock products sorted by price (ascending). Then rewrite the same logic using a list comprehension with `sorted()`.

4. **`any` / `all` refactoring**: Rewrite the following loop using `any()` or `all()` with a generator expression:
   ```python
   found = False
   for user in users:
       if user["active"] and user["role"] == "admin":
           found = True
           break
   ```

5. **Argument forwarding**: Write a `timed(func, *args, **kwargs)` function that calls `func` with the given arguments, measures how long the call takes (using `time.perf_counter`), prints the elapsed time, and returns the function's result. Then use it to time a call to `sorted()` on a large list.

---
up:: [Schedule](../../Schedule.md)
#type/learning #source/self-study #status/seed
