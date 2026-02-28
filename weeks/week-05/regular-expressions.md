# Regular Expressions

> Regular expressions are a domain-specific language for pattern matching embedded inside Python's `re` module — they let you describe *what text looks like* rather than writing procedural code to find it.

## Table of Contents
- [Core Concepts](#core-concepts)
- [Code Examples](#code-examples)
- [Common Pitfalls](#common-pitfalls)
- [Key Takeaways](#key-takeaways)
- [Exercises](#exercises)

## Core Concepts

### What Are Regular Expressions?

#### What

A regular expression (regex) is a string that describes a pattern of characters. Instead of searching for an exact string like `"error"`, you can search for patterns like "a word starting with 'error' followed by a colon and any number" — `r"error\w*:\s*\d+"`. The `re` module is Python's standard library implementation of regex.

#### How

At a high level, the regex engine reads your pattern and builds a state machine. It then runs that state machine against the input string, character by character, tracking whether the input matches the pattern. Python's `re` module uses a backtracking NFA (Nondeterministic Finite Automaton) engine — this means it tries possible matches and backtracks when a path fails. This is powerful but can be slow on pathological patterns (more on that in Common Pitfalls).

```python
import re

# Basic usage: does this string contain a date-like pattern?
text = "Meeting scheduled for 2026-02-28 at 10:00"
match = re.search(r"\d{4}-\d{2}-\d{2}", text)
if match:
    print(match.group())  # "2026-02-28"
```

#### Why It Matters

Regex is a universal skill. It appears in every programming language, in command-line tools (`grep`, `sed`, `awk`), in text editors, in database queries (`LIKE` vs `REGEXP`), and in web frameworks (URL routing). Learning regex once gives you a tool that works everywhere. In Python specifically, regex is the standard approach for data validation, log parsing, text extraction, and search-and-replace operations.

---

### Raw Strings and Why They Matter

#### What

In Python, backslash `\` is an escape character in both regular strings and regex patterns. This creates a conflict: `\n` in a Python string means "newline," but `\n` in a regex means "match a newline character." Raw strings (prefixed with `r`) tell Python to treat backslashes literally, passing them unmodified to the regex engine.

#### How

```python
import re

# Without raw string: \b is a Python escape (backspace character, \x08)
pattern_bad = "\bword\b"     # Python interprets \b as backspace!
print(repr(pattern_bad))      # '\x08word\x08' — NOT what you wanted

# With raw string: \b passes through to regex engine as word boundary
pattern_good = r"\bword\b"    # r prefix prevents Python from interpreting \b
print(repr(pattern_good))     # '\\bword\\b' — correct regex pattern
```

The rule is simple: **always use raw strings for regex patterns**. There is no downside and it prevents an entire category of bugs.

#### Why It Matters

The raw string issue is the number one regex gotcha in Python. Without `r`, patterns like `\d`, `\w`, `\s`, and `\b` may not behave as expected because Python processes the escapes before the regex engine sees them. Some work by accident (`\d` isn't a recognized Python escape, so it passes through), but `\b` silently becomes a backspace character. Always use `r"..."` and never think about this again.

---

### Pattern Syntax Reference

#### What

Regex patterns are built from literal characters and metacharacters. Metacharacters have special meaning — they represent character classes, quantifiers, anchors, and grouping. Mastering these building blocks lets you construct any pattern.

#### How

**Character Classes** — match a single character from a set:

| Pattern | Matches | Example |
|---------|---------|---------|
| `.` | Any character except newline | `a.c` matches `abc`, `a1c`, `a-c` |
| `\d` | Any digit `[0-9]` | `\d{3}` matches `123`, `456` |
| `\D` | Any non-digit `[^0-9]` | `\D+` matches `abc`, `---` |
| `\w` | Word character `[a-zA-Z0-9_]` | `\w+` matches `hello_42` |
| `\W` | Non-word character | `\W` matches `!`, ` `, `-` |
| `\s` | Whitespace `[ \t\n\r\f\v]` | `\s+` matches spaces, tabs, newlines |
| `\S` | Non-whitespace | `\S+` matches `hello`, `123` |
| `[abc]` | Any of a, b, or c | `[aeiou]` matches any vowel |
| `[^abc]` | Not a, b, or c | `[^0-9]` matches any non-digit |
| `[a-z]` | Range: a through z | `[A-Za-z]` matches any letter |

**Quantifiers** — how many times to match:

| Pattern | Meaning | Example |
|---------|---------|---------|
| `*` | 0 or more (greedy) | `\d*` matches `""`, `"1"`, `"123"` |
| `+` | 1 or more (greedy) | `\d+` matches `"1"`, `"123"`, not `""` |
| `?` | 0 or 1 (optional) | `colou?r` matches `color`, `colour` |
| `{n}` | Exactly n | `\d{4}` matches `2026`, not `26` |
| `{n,m}` | Between n and m | `\d{2,4}` matches `26`, `202`, `2026` |
| `{n,}` | n or more | `\d{3,}` matches `123`, `12345` |
| `*?` | 0 or more (non-greedy) | `".*?"` matches shortest quoted string |
| `+?` | 1 or more (non-greedy) | `\d+?` matches as few digits as possible |

**Anchors** — match positions, not characters:

| Pattern | Meaning | Example |
|---------|---------|---------|
| `^` | Start of string (or line with `re.MULTILINE`) | `^Hello` matches `"Hello world"` |
| `$` | End of string (or line with `re.MULTILINE`) | `world$` matches `"Hello world"` |
| `\b` | Word boundary | `\bword\b` matches `"word"` but not `"sword"` |
| `\B` | Non-word boundary | `\Bword` matches `"sword"` but not `"word"` |

**Grouping and Alternation**:

| Pattern | Meaning | Example |
|---------|---------|---------|
| `(...)` | Capturing group | `(\d{4})-(\d{2})` captures year and month |
| `(?:...)` | Non-capturing group | `(?:ab)+` matches `ababab` without capturing |
| `(?P<name>...)` | Named group | `(?P<year>\d{4})` — access by name |
| `\|` | Alternation (OR) | `cat\|dog` matches `"cat"` or `"dog"` |

#### Why It Matters

This table is your regex vocabulary. Every complex pattern is just a combination of these primitives. Memorizing the character classes (`\d`, `\w`, `\s`) and quantifiers (`*`, `+`, `?`, `{n,m}`) covers 90% of real-world patterns. The rest — lookaheads, backreferences — handle the remaining 10%.

---

### Greedy vs Non-Greedy (Lazy) Matching

#### What

By default, quantifiers (`*`, `+`, `{n,m}`) are greedy — they match as much text as possible while still allowing the overall pattern to succeed. Adding `?` after a quantifier makes it non-greedy (lazy) — it matches as little text as possible.

#### How

```python
import re

html = '<b>bold</b> and <i>italic</i>'

# Greedy: .* matches as much as possible
greedy = re.search(r"<.*>", html)
print(greedy.group())   # "<b>bold</b> and <i>italic</i>" — grabbed everything!

# Non-greedy: .*? matches as little as possible
lazy = re.search(r"<.*?>", html)
print(lazy.group())     # "<b>" — stopped at the first >

# findall with non-greedy to get all tags
tags = re.findall(r"<.*?>", html)
print(tags)             # ['<b>', '</b>', '<i>', '</i>']
```

The regex engine still tries to make the overall pattern succeed. Greedy means "try the longest match first, shorten if needed." Non-greedy means "try the shortest match first, extend if needed."

#### Why It Matters

Greedy vs non-greedy is the most common source of "my regex matches too much" bugs. The classic example is parsing HTML/XML tags — `<.*>` greedily matches from the first `<` to the last `>`, swallowing everything in between. Using `<.*?>` fixes it. As a rule of thumb: use non-greedy quantifiers when the text you want is *between* delimiters.

---

### `re` Module Functions

#### What

The `re` module provides several functions for different matching use cases. The four most important are `match`, `search`, `findall`, and `sub`. Each serves a distinct purpose, and confusing them is a common beginner mistake.

#### How

**`re.match(pattern, string)`** — matches only at the *beginning* of the string. Returns a `Match` object or `None`.

```python
import re

# match only checks the start of the string
print(re.match(r"\d+", "123abc"))    # <re.Match object; span=(0, 3), match='123'>
print(re.match(r"\d+", "abc123"))    # None — doesn't start with digits
```

**`re.search(pattern, string)`** — finds the first match *anywhere* in the string. Returns a `Match` object or `None`.

```python
# search scans the entire string
print(re.search(r"\d+", "abc123def456"))  # <re.Match object; match='123'>
```

**`re.findall(pattern, string)`** — finds *all* non-overlapping matches. Returns a list of strings (or list of tuples if there are groups).

```python
# findall returns all matches as a list
print(re.findall(r"\d+", "abc123def456"))      # ['123', '456']

# With groups: returns list of tuples
print(re.findall(r"(\w+)=(\d+)", "x=1 y=2"))  # [('x', '1'), ('y', '2')]
```

**`re.sub(pattern, replacement, string)`** — replaces matches with a replacement string. Returns the modified string.

```python
# Replace all digits with #
print(re.sub(r"\d", "#", "Call 555-1234"))      # "Call ###-####"

# Use backreferences in replacement
print(re.sub(r"(\w+) (\w+)", r"\2 \1", "John Smith"))  # "Smith John"
```

**`re.split(pattern, string)`** — splits the string at each match.

```python
# Split on any whitespace or comma
print(re.split(r"[,\s]+", "a, b,  c  d"))  # ['a', 'b', 'c', 'd']
```

**`re.fullmatch(pattern, string)`** — matches the *entire* string against the pattern. Equivalent to `^pattern$` but more explicit.

```python
# Validate that the entire string is a valid email-like format
print(re.fullmatch(r"\w+@\w+\.\w+", "user@example.com"))   # Match
print(re.fullmatch(r"\w+@\w+\.\w+", "not an email at all"))  # None
```

#### Why It Matters

The `match` vs `search` distinction trips up almost every Python regex beginner. `match` only checks the start of the string — it does NOT check if the pattern matches the entire string (use `fullmatch` for that). In practice, `search` is what you want 90% of the time. `findall` is the workhorse for extraction tasks. `sub` is the workhorse for transformation tasks.

---

### The `Match` Object

#### What

When `re.match()`, `re.search()`, or `re.fullmatch()` succeeds, it returns a `Match` object. This object contains the matched text, its position in the original string, and any captured groups.

#### How

```python
import re

text = "Error 404: Page not found at 2026-02-28 10:30"
pattern = r"Error (?P<code>\d+): (?P<message>.+?) at (?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2})"

m = re.search(pattern, text)
if m:
    # Full match
    print(m.group())       # "Error 404: Page not found at 2026-02-28 10:30"
    print(m.group(0))      # Same as above — group 0 is the entire match

    # Positional groups
    print(m.group(1))      # "404"        — first (...)
    print(m.group(2))      # "Page not found" — second (...)
    print(m.group(3))      # "2026-02-28 10:30"

    # Named groups
    print(m.group("code"))      # "404"
    print(m.group("message"))   # "Page not found"

    # All groups as a tuple
    print(m.groups())           # ('404', 'Page not found', '2026-02-28 10:30')

    # Named groups as a dict
    print(m.groupdict())        # {'code': '404', 'message': 'Page not found', 'timestamp': '2026-02-28 10:30'}

    # Position in original string
    print(m.start())   # 0
    print(m.end())     # 47
    print(m.span())    # (0, 47)
```

#### Why It Matters

Named groups (`(?P<name>...)`) combined with `.groupdict()` turn regex from "write-only string manipulation" into structured data extraction. Instead of remembering that group 3 is the timestamp, you access it as `m.group("timestamp")`. This is the difference between regex code that's maintainable and regex code that nobody can read six months later.

---

### Compiled Patterns

#### What

`re.compile(pattern)` pre-compiles a regex pattern into a `Pattern` object. You then call methods on this object instead of passing the pattern string each time. This is both a performance optimization and a readability improvement.

#### How

```python
import re

# Without compile: pattern is compiled every call (cached internally, but less readable)
for line in log_lines:
    if re.search(r"ERROR\s+\w+:\s+.+", line):
        process(line)

# With compile: pattern compiled once, reused many times
ERROR_PATTERN = re.compile(r"ERROR\s+\w+:\s+.+")

for line in log_lines:
    if ERROR_PATTERN.search(line):
        process(line)
```

Under the hood, `re.search(pattern, string)` calls `re.compile(pattern).search(string)` anyway. Python caches the last 512 compiled patterns in `re._cache`, so repeated calls with the same pattern string don't recompile. However, `re.compile()` makes patterns named, reusable, and avoids the cache-eviction issue with many different patterns.

#### Why It Matters

In production code, compiled patterns are preferred for two reasons. First, readability: `EMAIL_PATTERN.search(text)` is more meaningful than `re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)`. Second, performance: in tight loops processing millions of strings, the compile-once-use-many pattern avoids even the cache lookup overhead.

---

### Groups, Named Groups, and Backreferences

#### What

Parentheses `(...)` in a regex create capturing groups. Each group captures the text it matches, accessible by index or name. Backreferences (`\1`, `\2`, or `(?P=name)`) let you match the *same text* that a group previously captured.

#### How

```python
import re

# Capturing groups — accessed by index
date_pattern = r"(\d{4})-(\d{2})-(\d{2})"
m = re.search(date_pattern, "Today is 2026-02-28.")
if m:
    year, month, day = m.groups()  # ('2026', '02', '28')

# Named groups — accessed by name
date_pattern = r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"
m = re.search(date_pattern, "Today is 2026-02-28.")
if m:
    print(m.group("year"))   # '2026'
    print(m.groupdict())     # {'year': '2026', 'month': '02', 'day': '28'}

# Non-capturing groups — group for structure, don't capture
# Useful with alternation
pattern = r"(?:https?|ftp)://\S+"  # Match URLs without capturing the protocol
urls = re.findall(pattern, "Visit https://example.com or ftp://files.example.com")
print(urls)  # ['https://example.com', 'ftp://files.example.com']

# Backreferences — match the same text again
# Find duplicate words
dup_pattern = r"\b(\w+)\s+\1\b"
print(re.search(dup_pattern, "the the cat sat sat"))
# Matches "the the" — \1 matches whatever group 1 captured

# Named backreference
dup_pattern_named = r"\b(?P<word>\w+)\s+(?P=word)\b"
print(re.search(dup_pattern_named, "the the cat"))  # Matches "the the"
```

#### Why It Matters

Groups are what transform regex from "does this match?" to "extract structured data from text." Named groups make complex patterns self-documenting. Backreferences enable patterns that are impossible with simple string methods — like finding duplicate words, matching balanced quotes, or validating repeated substrings. The `findall` function's behavior changes with groups (returning tuples instead of strings), so understanding groups is essential for correct data extraction.

---

### Lookahead and Lookbehind Assertions

#### What

Lookahead and lookbehind are zero-width assertions — they check if a pattern exists ahead of or behind the current position without consuming characters (not included in the match). They answer "is this pattern next?" or "was this pattern before?" without moving the cursor.

#### How

| Syntax | Name | Meaning |
|--------|------|---------|
| `(?=...)` | Positive lookahead | Followed by `...` |
| `(?!...)` | Negative lookahead | NOT followed by `...` |
| `(?<=...)` | Positive lookbehind | Preceded by `...` |
| `(?<!...)` | Negative lookbehind | NOT preceded by `...` |

```python
import re

# Positive lookahead: match digits followed by "px" (but don't include "px")
text = "width: 100px; height: 200em; font-size: 16px"
pixels = re.findall(r"\d+(?=px)", text)
print(pixels)  # ['100', '16'] — only pixel values, "200" excluded

# Negative lookahead: match digits NOT followed by "px"
non_pixels = re.findall(r"\d+(?!px)\b", text)
print(non_pixels)  # ['200'] — matches 200em value

# Positive lookbehind: match text preceded by "$"
prices = "Items: $19.99, $5.50, €12.00"
dollar_amounts = re.findall(r"(?<=\$)\d+\.\d{2}", prices)
print(dollar_amounts)  # ['19.99', '5.50']

# Negative lookbehind: match ".py" NOT preceded by "test_"
files = "app.py test_app.py utils.py test_utils.py"
non_test = re.findall(r"(?<!test_)\b\w+\.py", files)
print(non_test)  # ['app.py', 'utils.py']
```

Important limitation: in Python's `re` module, lookbehinds must be fixed-width. You can use `(?<=abc)` or `(?<=\d{3})` but not `(?<=\d+)`. The third-party `regex` module lifts this restriction.

#### Why It Matters

Lookaheads and lookbehinds solve the "I need context but don't want it in my match" problem. Password validation is a classic use case: you need to check for digits, uppercase letters, and special characters, but you don't want to enforce their order. Each requirement becomes a lookahead: `(?=.*\d)(?=.*[A-Z])(?=.*[!@#$%])`. Without lookaheads, you'd need multiple separate checks.

---

### Flags (Modifiers)

#### What

Regex flags modify how the pattern engine behaves. They're passed as the `flags` parameter to `re` functions or embedded inline in the pattern with `(?flags)` syntax.

#### How

| Flag | Inline | Meaning |
|------|--------|---------|
| `re.IGNORECASE` | `(?i)` | Case-insensitive matching |
| `re.MULTILINE` | `(?m)` | `^` and `$` match at line boundaries, not just string boundaries |
| `re.DOTALL` | `(?s)` | `.` matches newlines too (normally it doesn't) |
| `re.VERBOSE` | `(?x)` | Allow comments and whitespace in pattern for readability |

```python
import re

# IGNORECASE — match regardless of case
print(re.findall(r"error", "Error ERROR error", re.IGNORECASE))
# ['Error', 'ERROR', 'error']

# MULTILINE — ^ and $ work per-line
multiline_text = """Line 1: error
Line 2: ok
Line 3: error"""
print(re.findall(r"^Line \d+: error$", multiline_text, re.MULTILINE))
# ['Line 1: error', 'Line 3: error']

# DOTALL — dot matches newline
html = "<div>\n  content\n</div>"
print(re.search(r"<div>.*</div>", html))                    # None — . doesn't match \n
print(re.search(r"<div>.*</div>", html, re.DOTALL).group()) # "<div>\n  content\n</div>"

# VERBOSE — write readable patterns with comments
PHONE_PATTERN = re.compile(r"""
    ^                   # Start of string
    (?:\+1[-.\s]?)?     # Optional country code (+1)
    \(?                 # Optional opening paren
    (?P<area>\d{3})     # Area code (3 digits)
    \)?                 # Optional closing paren
    [-.\s]?             # Optional separator
    (?P<exchange>\d{3}) # Exchange (3 digits)
    [-.\s]?             # Optional separator
    (?P<number>\d{4})   # Subscriber number (4 digits)
    $                   # End of string
""", re.VERBOSE)

m = PHONE_PATTERN.search("(555) 123-4567")
if m:
    print(m.groupdict())  # {'area': '555', 'exchange': '123', 'number': '4567'}
```

Combine flags with the pipe operator: `re.IGNORECASE | re.MULTILINE`.

#### Why It Matters

`re.VERBOSE` is the most underused flag. Complex regex patterns are notoriously hard to read. With `re.VERBOSE`, you can break the pattern across multiple lines and add comments — turning a cryptic one-liner into documented, maintainable code. Any pattern longer than about 30 characters should use `re.VERBOSE`.

---

### `re.sub` with Replacement Functions

#### What

`re.sub` doesn't just accept replacement strings — it also accepts functions. When the replacement is a callable, it receives the `Match` object for each match and returns the replacement string. This enables dynamic, context-dependent replacements.

#### How

```python
import re


def censor_email(match: re.Match) -> str:
    """Replace the local part of an email with asterisks, keep domain."""
    local = match.group("local")
    domain = match.group("domain")
    censored = local[0] + "*" * (len(local) - 1)
    return f"{censored}@{domain}"


EMAIL_PATTERN = re.compile(r"(?P<local>[\w.+-]+)@(?P<domain>[\w.-]+\.\w+)")

text = "Contact alice.smith@example.com or bob@company.org for details."
result = EMAIL_PATTERN.sub(censor_email, text)
print(result)
# "Contact a**********@example.com or b**@company.org for details."
```

This pattern is powerful because the replacement function has full access to the match, including all groups, positions, and the original string. You can do lookups, calculations, or any Python logic.

#### Why It Matters

Replacement functions are what make `re.sub` a general-purpose text transformer rather than just a find-and-replace tool. Real-world use cases include: template rendering (replacing `{{variable}}` with values from a dict), data anonymization (censoring PII while preserving format), and format conversion (converting date formats, unit conversions in text).

## Code Examples

### Example 1: Log Parser with Structured Extraction

```python
"""
Parse semi-structured log lines into typed data using named groups.
Demonstrates compiled patterns, named groups, and groupdict().
"""
import re
from dataclasses import dataclass
from datetime import datetime


@dataclass
class LogEntry:
    """A structured log entry parsed from raw text."""
    timestamp: datetime
    level: str
    service: str
    message: str
    trace_id: str | None = None


# Use VERBOSE for a readable, documented pattern
LOG_PATTERN = re.compile(r"""
    ^
    (?P<timestamp>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\.\d{3})  # ISO timestamp with ms
    \s+
    (?P<level>DEBUG|INFO|WARN|ERROR|FATAL)                         # Log level
    \s+
    \[(?P<service>[\w.-]+)\]                                       # Service name in brackets
    \s+
    (?:\[trace=(?P<trace_id>[\w-]+)\]\s+)?                         # Optional trace ID
    (?P<message>.+)                                                 # Rest is the message
    $
""", re.VERBOSE)


def parse_log_entry(line: str) -> LogEntry | None:
    """Parse a single log line into a LogEntry, or None if unparseable."""
    m = LOG_PATTERN.match(line)
    if not m:
        return None

    data = m.groupdict()
    return LogEntry(
        timestamp=datetime.fromisoformat(data["timestamp"]),
        level=data["level"],
        service=data["service"],
        message=data["message"],
        trace_id=data.get("trace_id"),
    )


def parse_logs(raw_log: str) -> list[LogEntry]:
    """Parse a multi-line log string into structured entries."""
    entries: list[LogEntry] = []
    for line in raw_log.strip().splitlines():
        if entry := parse_log_entry(line.strip()):
            entries.append(entry)
    return entries


# Usage
RAW_LOG = """\
2026-02-28 10:15:03.442 ERROR [auth-service] [trace=abc-123] Login failed: invalid credentials for user=alice
2026-02-28 10:15:05.110 INFO  [auth-service] Login succeeded for user=bob
2026-02-28 10:16:10.891 ERROR [payment-svc] [trace=def-456] Timeout after 30s connecting to payment gateway
2026-02-28 10:16:12.003 WARN  [auth-service] Rate limit approaching: 95/100 requests in window
2026-02-28 10:17:00.557 INFO  [payment-svc] [trace=ghi-789] Payment processed: order=12345 amount=$99.99
"""

entries = parse_logs(RAW_LOG)
for entry in entries:
    print(f"[{entry.level}] {entry.service}: {entry.message}")
    if entry.trace_id:
        print(f"       trace: {entry.trace_id}")

# Filter for errors only
errors = [e for e in entries if e.level == "ERROR"]
print(f"\nFound {len(errors)} error(s)")
```

### Example 2: Data Validation with Regex

```python
"""
Input validation functions using regex — the kind you'd use
in a web form backend or data pipeline.
"""
import re

# Compiled patterns as module-level constants (best practice)
EMAIL_PATTERN = re.compile(
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
)

# Password: 8+ chars, at least one uppercase, one lowercase, one digit, one special
PASSWORD_PATTERN = re.compile(r"""
    ^
    (?=.*[a-z])         # At least one lowercase
    (?=.*[A-Z])         # At least one uppercase
    (?=.*\d)            # At least one digit
    (?=.*[!@\#$%^&*])   # At least one special character
    .{8,}               # Minimum 8 characters total
    $
""", re.VERBOSE)

# Semantic versioning: major.minor.patch with optional pre-release
SEMVER_PATTERN = re.compile(r"""
    ^
    (?P<major>0|[1-9]\d*)       # Major version (no leading zeros)
    \.
    (?P<minor>0|[1-9]\d*)       # Minor version
    \.
    (?P<patch>0|[1-9]\d*)       # Patch version
    (?:-(?P<pre>[\da-zA-Z-]+    # Optional pre-release
      (?:\.[\da-zA-Z-]+)*))?    #   with dot-separated identifiers
    (?:\+(?P<build>[\da-zA-Z-]+ # Optional build metadata
      (?:\.[\da-zA-Z-]+)*))?    #   with dot-separated identifiers
    $
""", re.VERBOSE)

# IPv4 address
IPV4_PATTERN = re.compile(r"""
    ^
    (?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}   # First three octets + dots
    (?:25[0-5]|2[0-4]\d|[01]?\d\d?)             # Last octet
    $
""", re.VERBOSE)


def validate_email(email: str) -> bool:
    """Check if string looks like a valid email address."""
    return EMAIL_PATTERN.fullmatch(email) is not None


def validate_password(password: str) -> tuple[bool, list[str]]:
    """Validate password strength, returning (is_valid, list_of_issues)."""
    issues: list[str] = []
    if len(password) < 8:
        issues.append("Must be at least 8 characters")
    if not re.search(r"[a-z]", password):
        issues.append("Must contain a lowercase letter")
    if not re.search(r"[A-Z]", password):
        issues.append("Must contain an uppercase letter")
    if not re.search(r"\d", password):
        issues.append("Must contain a digit")
    if not re.search(r"[!@#$%^&*]", password):
        issues.append("Must contain a special character (!@#$%^&*)")
    return (len(issues) == 0, issues)


def parse_semver(version: str) -> dict[str, str | None] | None:
    """Parse a semantic version string into components."""
    m = SEMVER_PATTERN.fullmatch(version)
    if not m:
        return None
    return m.groupdict()


def validate_ipv4(address: str) -> bool:
    """Check if string is a valid IPv4 address."""
    return IPV4_PATTERN.fullmatch(address) is not None


# Demonstration
print(validate_email("user@example.com"))           # True
print(validate_email("not-an-email"))                # False

is_valid, issues = validate_password("weak")
print(f"Password valid: {is_valid}, issues: {issues}")
# Password valid: False, issues: ['Must be at least 8 characters', ...]

is_valid, issues = validate_password("Str0ng!Pass")
print(f"Password valid: {is_valid}, issues: {issues}")
# Password valid: True, issues: []

print(parse_semver("2.1.0-beta.1+build.42"))
# {'major': '2', 'minor': '1', 'patch': '0', 'pre': 'beta.1', 'build': 'build.42'}

print(validate_ipv4("192.168.1.1"))     # True
print(validate_ipv4("256.1.1.1"))       # False — 256 > 255
```

### Example 3: Text Transformation Pipeline

```python
"""
A pipeline of regex-based transformations for cleaning and normalizing text.
Demonstrates re.sub with both strings and replacement functions.
"""
import re
from collections.abc import Callable


# Type alias for a transformation function
TextTransform = Callable[[str], str]


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs into single space, strip leading/trailing."""
    return re.sub(r"[ \t]+", " ", text).strip()


def remove_html_tags(text: str) -> str:
    """Strip HTML tags, keeping the text content."""
    return re.sub(r"<[^>]+>", "", text)


def mask_credit_cards(text: str) -> str:
    """Replace credit card numbers with masked version, keeping last 4 digits."""
    def mask_match(m: re.Match) -> str:
        digits = re.sub(r"[\s-]", "", m.group())  # Remove spaces/hyphens
        last_four = digits[-4:]
        return f"****-****-****-{last_four}"

    # Match 16 digits with optional spaces or hyphens
    return re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", mask_match, text)


def convert_dates_to_iso(text: str) -> str:
    """Convert MM/DD/YYYY dates to ISO format YYYY-MM-DD."""
    def to_iso(m: re.Match) -> str:
        month, day, year = m.group("month"), m.group("day"), m.group("year")
        return f"{year}-{month}-{day}"

    return re.sub(
        r"(?P<month>\d{2})/(?P<day>\d{2})/(?P<year>\d{4})",
        to_iso,
        text,
    )


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)      # Remove non-word chars (except spaces/hyphens)
    text = re.sub(r"[\s_]+", "-", text)        # Replace spaces/underscores with hyphens
    text = re.sub(r"-+", "-", text)            # Collapse multiple hyphens
    return text.strip("-")


def apply_pipeline(text: str, transforms: list[TextTransform]) -> str:
    """Apply a sequence of text transformations."""
    for transform in transforms:
        text = transform(text)
    return text


# Usage
raw_text = """
  <p>Order placed on   02/28/2026 by   John Smith.</p>
  <p>Card:  4532-1234-5678-9012.  Amount: $99.99</p>
"""

clean = apply_pipeline(raw_text, [
    remove_html_tags,
    normalize_whitespace,
    mask_credit_cards,
    convert_dates_to_iso,
])
print(clean)
# "Order placed on 2026-02-28 by John Smith. Card: ****-****-****-9012. Amount: $99.99"

print(slugify("Hello, World! This is — a Test"))
# "hello-world-this-is-a-test"
```

### Example 4: Configuration File Parser

```python
"""
Parse a simple INI-like configuration file using regex.
Demonstrates re.match, groups, and building a parser from patterns.
"""
import re
from pathlib import Path


# Patterns for each line type
SECTION_PATTERN = re.compile(r"^\[(?P<name>[\w.-]+)\]\s*$")
KEY_VALUE_PATTERN = re.compile(r"""
    ^
    (?P<key>[\w.-]+)        # Key name
    \s*=\s*                  # Equals with optional whitespace
    (?P<value>.+?)           # Value (non-greedy to exclude trailing space)
    \s*                      # Trailing whitespace
    (?:\#.*)?                # Optional inline comment
    $
""", re.VERBOSE)
COMMENT_PATTERN = re.compile(r"^\s*(?:\#|;)")  # Lines starting with # or ;
BLANK_PATTERN = re.compile(r"^\s*$")           # Empty or whitespace-only lines


def parse_config(text: str) -> dict[str, dict[str, str]]:
    """Parse INI-like config text into nested dict: {section: {key: value}}."""
    config: dict[str, dict[str, str]] = {}
    current_section = "DEFAULT"
    config[current_section] = {}

    for line_num, line in enumerate(text.splitlines(), start=1):
        # Skip blanks and comments
        if BLANK_PATTERN.match(line) or COMMENT_PATTERN.match(line):
            continue

        # Check for section header
        if section_match := SECTION_PATTERN.match(line):
            current_section = section_match.group("name")
            config.setdefault(current_section, {})
            continue

        # Check for key=value pair
        if kv_match := KEY_VALUE_PATTERN.match(line):
            key = kv_match.group("key")
            value = kv_match.group("value")
            config[current_section][key] = value
            continue

        # If nothing matched, the line is malformed
        print(f"Warning: unparseable line {line_num}: {line!r}")

    return config


# Example config text
CONFIG_TEXT = """\
# Application configuration
app.name = MyApp
app.version = 2.1.0

[database]
host = localhost
port = 5432
name = mydb         # database name
user = admin
password = s3cret!

[logging]
level = INFO
file = /var/log/app.log
format = %(asctime)s %(levelname)s %(message)s

; Feature flags
[features]
dark_mode = true
beta_api = false
"""

config = parse_config(CONFIG_TEXT)
for section, values in config.items():
    print(f"\n[{section}]")
    for key, value in values.items():
        print(f"  {key} = {value}")
```

### Example 5: Markdown Link Extractor

```python
"""
Extract and categorize links from Markdown text.
Demonstrates findall, finditer, named groups, and lookaheads.
"""
import re
from dataclasses import dataclass


@dataclass
class MarkdownLink:
    """A link extracted from Markdown text."""
    text: str
    url: str
    is_image: bool
    line_number: int


# Pattern for Markdown links: [text](url) and ![alt](url)
# Uses lookbehind to detect image links (preceded by !)
LINK_PATTERN = re.compile(r"""
    (?P<is_image>!)?            # Optional ! prefix for images
    \[(?P<text>[^\]]+)\]        # Link text in square brackets
    \((?P<url>[^\)]+)\)         # URL in parentheses
""", re.VERBOSE)

# Pattern for reference-style links: [text][ref] and [ref]: url
REF_LINK_DEF_PATTERN = re.compile(r"""
    ^\[(?P<ref>[^\]]+)\]:\s+    # Reference label definition
    (?P<url>\S+)                # URL (no spaces)
    (?:\s+"(?P<title>[^"]*)")?  # Optional title in quotes
""", re.VERBOSE | re.MULTILINE)


def extract_inline_links(text: str) -> list[MarkdownLink]:
    """Extract all inline links and images from Markdown text."""
    links: list[MarkdownLink] = []

    for line_num, line in enumerate(text.splitlines(), start=1):
        for m in LINK_PATTERN.finditer(line):
            links.append(MarkdownLink(
                text=m.group("text"),
                url=m.group("url"),
                is_image=m.group("is_image") is not None,
                line_number=line_num,
            ))

    return links


def extract_reference_definitions(text: str) -> dict[str, str]:
    """Extract reference link definitions: {ref_label: url}."""
    return {
        m.group("ref"): m.group("url")
        for m in REF_LINK_DEF_PATTERN.finditer(text)
    }


def find_broken_links(text: str) -> list[str]:
    """Find reference-style links that have no matching definition."""
    # Find all reference usages: [text][ref]
    ref_usages = set(re.findall(r"\[[^\]]+\]\[([^\]]+)\]", text))

    # Find all definitions
    definitions = set(extract_reference_definitions(text).keys())

    # Broken = used but not defined
    return sorted(ref_usages - definitions)


# Usage
MARKDOWN = """\
# My Document

Check out [Python docs](https://docs.python.org) for reference.
Here's an image: ![Python logo](https://python.org/logo.png)

Read the [tutorial](../tutorial/basics.md) first, then the
[advanced guide](../tutorial/advanced.md).

For more info, see [the FAQ][faq] and [the wiki][wiki].

[faq]: https://example.com/faq
[wiki]: https://example.com/wiki "Project Wiki"
"""

# Extract inline links
links = extract_inline_links(MARKDOWN)
for link in links:
    prefix = "IMG" if link.is_image else "LINK"
    print(f"  [{prefix}] Line {link.line_number}: {link.text} -> {link.url}")

# Extract reference definitions
refs = extract_reference_definitions(MARKDOWN)
print("\nReference definitions:")
for ref, url in refs.items():
    print(f"  [{ref}] -> {url}")

# Check for broken references
broken = find_broken_links(MARKDOWN)
if broken:
    print(f"\nBroken references: {broken}")
else:
    print("\nNo broken references found")
```

### Example 6: Token-Based Lexer

```python
"""
A simple lexer (tokenizer) using regex — the first step of any parser.
Demonstrates re.Scanner-like pattern matching via finditer.
"""
import re
from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """Token types for a simple expression language."""
    NUMBER = auto()
    IDENTIFIER = auto()
    OPERATOR = auto()
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    STRING = auto()
    WHITESPACE = auto()
    UNKNOWN = auto()


@dataclass(frozen=True)
class Token:
    """A single token produced by the lexer."""
    type: TokenType
    value: str
    position: int


# Define token patterns — order matters (first match wins)
TOKEN_PATTERNS: list[tuple[TokenType, str]] = [
    (TokenType.NUMBER,      r"\d+(?:\.\d+)?"),          # Integer or float
    (TokenType.STRING,      r'"(?:[^"\\]|\\.)*"'),      # Double-quoted string with escapes
    (TokenType.IDENTIFIER,  r"[a-zA-Z_]\w*"),           # Variable/function names
    (TokenType.OPERATOR,    r"[+\-*/=<>!]=?|&&|\|\|"),  # Operators
    (TokenType.LPAREN,      r"\("),
    (TokenType.RPAREN,      r"\)"),
    (TokenType.COMMA,       r","),
    (TokenType.WHITESPACE,  r"\s+"),
]

# Combine all patterns into one master pattern using named groups
# Each group name encodes the token type
MASTER_PATTERN = re.compile(
    "|".join(
        f"(?P<{tok_type.name}>{pattern})"
        for tok_type, pattern in TOKEN_PATTERNS
    )
)


def tokenize(source: str) -> list[Token]:
    """Tokenize source code into a list of tokens."""
    tokens: list[Token] = []

    for m in MASTER_PATTERN.finditer(source):
        # Find which group matched
        kind = m.lastgroup
        value = m.group()
        position = m.start()

        if kind is None:
            tokens.append(Token(TokenType.UNKNOWN, value, position))
            continue

        tok_type = TokenType[kind]

        # Skip whitespace tokens
        if tok_type == TokenType.WHITESPACE:
            continue

        tokens.append(Token(tok_type, value, position))

    return tokens


# Usage
source_code = 'calculate(x + 3.14, "hello world", y * 2)'
tokens = tokenize(source_code)

for token in tokens:
    print(f"  {token.type.name:12s} | {token.value!r:20s} | pos={token.position}")

# Output:
#   IDENTIFIER   | 'calculate'          | pos=0
#   LPAREN       | '('                  | pos=9
#   IDENTIFIER   | 'x'                  | pos=10
#   OPERATOR     | '+'                  | pos=12
#   NUMBER       | '3.14'               | pos=14
#   COMMA        | ','                  | pos=18
#   STRING       | '"hello world"'      | pos=20
#   COMMA        | ','                  | pos=33
#   IDENTIFIER   | 'y'                  | pos=35
#   OPERATOR     | '*'                  | pos=37
#   NUMBER       | '2'                  | pos=39
#   RPAREN       | ')'                  | pos=40
```

## Common Pitfalls

### Pitfall 1: Forgetting Raw Strings

Backslash sequences are interpreted by Python before the regex engine sees them.

```python
import re

# BAD — \b is interpreted as backspace by Python, not word boundary
pattern = "\bword\b"
print(re.search(pattern, "a word here"))  # None — looking for \x08word\x08

# GOOD — raw string passes \b to regex engine as word boundary
pattern = r"\bword\b"
print(re.search(pattern, "a word here"))  # <re.Match object; match='word'>
```

This is insidious because some escapes accidentally work: `\d` is not a Python escape, so it passes through unchanged. But `\b` (backspace), `\t` (tab), `\n` (newline) all get transformed. Always use `r"..."`.

### Pitfall 2: Using `match` When You Mean `search`

`re.match` only checks the beginning of the string — it does not scan the entire string for a match.

```python
import re

text = "The error code is 404"

# BAD — match only looks at the start of the string
result = re.match(r"\d+", text)
print(result)  # None — text doesn't START with digits

# GOOD — search scans the entire string
result = re.search(r"\d+", text)
print(result.group())  # "404"

# ALSO BAD — assuming match checks the whole string
result = re.match(r"error", "error code 404")
print(result.group())  # "error" — but this also matches "error_handler: 200"
# match does NOT check if the WHOLE string matches

# GOOD — use fullmatch to check entire string
result = re.fullmatch(r"error", "error code 404")
print(result)  # None — "error code 404" is not entirely "error"
```

### Pitfall 3: Greedy Matching Eating Too Much

Default quantifiers are greedy — they match as much as possible.

```python
import re

# BAD — greedy .* matches from first < to LAST >
html = "<b>bold</b> and <i>italic</i>"
tags = re.findall(r"<(.+)>", html)
print(tags)  # ['b>bold</b> and <i>italic</i'] — one giant match

# GOOD — non-greedy .*? matches from < to the NEAREST >
tags = re.findall(r"<(.+?)>", html)
print(tags)  # ['b', '/b', 'i', '/i'] — individual tags

# ALSO GOOD — use negated character class instead of non-greedy
tags = re.findall(r"<([^>]+)>", html)
print(tags)  # ['b', '/b', 'i', '/i'] — often faster than non-greedy
```

The negated character class `[^>]+` approach is usually preferred over `.*?` because it's more explicit about what you're matching (any character that isn't `>`) and avoids backtracking.

### Pitfall 4: `findall` Behavior Changes with Groups

When your pattern contains capturing groups, `findall` returns the group contents, not the full match. With multiple groups, it returns tuples.

```python
import re

text = "age:25 height:180 weight:70"

# BAD — groups change findall's return value
pairs = re.findall(r"(\w+):(\d+)", text)
print(pairs)  # [('age', '25'), ('height', '180'), ('weight', '70')]
# Returns TUPLES of groups, not the full matches like "age:25"

# If you want full matches AND groups, use finditer
for m in re.finditer(r"(\w+):(\d+)", text):
    print(f"Full: {m.group()}, Key: {m.group(1)}, Value: {m.group(2)}")

# If you need groups but want the full match in findall,
# wrap the whole thing in a group
full_matches = re.findall(r"(\w+:\d+)", text)
print(full_matches)  # ['age:25', 'height:180', 'weight:70']

# Use non-capturing groups (?:...) to group without affecting findall
values = re.findall(r"\w+:(\d+)(?:kg|cm)?", text)
print(values)  # ['25', '180', '70'] — (?:...) doesn't count as a group
```

### Pitfall 5: Catastrophic Backtracking

Certain regex patterns can take exponential time on specific inputs. This happens when the engine has many overlapping ways to match, and none of them succeed.

```python
import re
import time

# BAD — catastrophic backtracking on non-matching input
# Pattern: (a+)+ with a string of a's followed by something that doesn't match
evil_pattern = re.compile(r"^(a+)+b$")

# This is fine for matching inputs
print(evil_pattern.match("aaaaab"))   # Matches

# But for non-matching inputs, time explodes exponentially
text = "a" * 25 + "c"  # No 'b' at end — forces backtracking
start = time.perf_counter()
result = evil_pattern.match(text)      # This takes several seconds!
elapsed = time.perf_counter() - start
print(f"Took {elapsed:.2f}s for just {len(text)} characters")

# GOOD — eliminate the nested quantifier
good_pattern = re.compile(r"^a+b$")   # Same logical meaning, no catastrophic backtracking
start = time.perf_counter()
result = good_pattern.match(text)
elapsed = time.perf_counter() - start
print(f"Fixed: {elapsed:.6f}s")        # Microseconds, not seconds
```

The rule: never nest quantifiers on overlapping patterns (`(a+)+`, `(a*)*`, `(a|b*)+`). If you must use complex patterns on untrusted input, set a timeout or use the `regex` library which supports atomic grouping and possessive quantifiers.

## Key Takeaways

- **Always use raw strings** (`r"..."`) for regex patterns. No exceptions, no excuses. It prevents an entire class of silent bugs from escaped backslash sequences.
- **`search` is your default, not `match`**: `match` only checks the start of the string. Use `search` to find patterns anywhere. Use `fullmatch` to validate that an entire string matches.
- **Named groups make regex maintainable**: `(?P<name>...)` combined with `.groupdict()` turns cryptic group indices into readable, self-documenting code. Use them for any pattern with more than one group.
- **`re.VERBOSE` is your friend**: Any pattern longer than 30 characters should be written with `re.VERBOSE` and comments. Future-you will thank present-you.
- **Prefer `[^X]+` over `.*?`**: When matching "everything except X," a negated character class is clearer and avoids backtracking issues. Use `[^>]+` instead of `.*?` when matching between delimiters.

## Exercises

### Exercise 1: Email Normalizer

Write a function `normalize_emails(text: str) -> str` that finds all email addresses in a text and lowercases them (email local parts and domains are case-insensitive in practice). The function should modify the text in-place, replacing each email with its lowercased version while leaving all other text unchanged.

### Exercise 2: Markdown Table Parser

Write a function `parse_markdown_table(text: str) -> list[dict[str, str]]` that takes a Markdown table as a string and returns a list of dictionaries, where each dictionary maps column headers to cell values. Handle the separator row (`| --- | --- |`) by skipping it. Strip whitespace from all headers and values.

```
| Name   | Age | City    |
| ------ | --- | ------- |
| Alice  | 30  | Taipei  |
| Bob    | 25  | Tokyo   |
```

Expected output: `[{"Name": "Alice", "Age": "30", "City": "Taipei"}, ...]`

### Exercise 3: Log Level Upgrader

Write a function `upgrade_log_levels(source_code: str) -> str` that takes Python source code as a string and replaces all `logging.warn(...)` calls with `logging.warning(...)` (since `warn` is deprecated). Be careful to only replace the method name, not any arguments or surrounding code. Handle both `logger.warn(` and `logging.warn(` patterns.

### Exercise 4: URL Parameter Extractor

Write a function `extract_query_params(url: str) -> dict[str, str]` that takes a URL string and returns a dictionary of query parameters. Do not use `urllib.parse` — implement this purely with regex. Handle edge cases: URLs without query strings, parameters without values (`?key`), and URL-encoded `+` signs (treat as spaces).

### Exercise 5: Code Comment Stripper

Write a function `strip_python_comments(source: str) -> str` that removes all comments from Python source code while preserving strings that contain `#` characters. For example, `x = "hello # world"  # this is a comment` should become `x = "hello # world"`. Handle both single-quoted and double-quoted strings, including escaped quotes within strings.

---
up:: [Schedule](../../Schedule.md)
#type/learning #source/self-study #status/seed
