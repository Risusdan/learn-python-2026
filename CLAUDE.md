# CLAUDE.md — learn-python-2026

## Project Structure

This is a self-study curriculum for learning Python from fundamentals to expert level.

- `README.md` — Repository overview
- `LearningPath.md` — Full topic tree with OPTIONAL markers
- `Schedule.md` — Week-by-week plan with checkboxes and "Why It Matters" context
- `weeks/week-NN/topic-name.md` — Individual lesson files (generated over time)

## How to Generate Lessons

1. **Always read `Schedule.md` first** to determine what topics to cover for the requested week.
2. Only cover topics listed in the schedule. Respect OPTIONAL markers in `LearningPath.md` — do not include OPTIONAL topics unless explicitly asked.
3. Create lesson files at `weeks/week-NN/topic-name.md` (e.g., `weeks/week-03/decorators.md`).
4. One file per major topic grouping within a week. If a week covers multiple distinct topics, split them into separate files.

## Lesson File Template

Every lesson file must follow this exact structure:

```markdown
# Topic Name

> One-sentence summary of what this topic is and why it matters.

## Core Concepts

[Explanation broken into logical subsections with ### headings as needed.
For every concept, cover three layers:
1. WHAT — what is this thing?
2. HOW — how does it work / how do you use it?
3. WHY — why does it exist? Why this approach over alternatives? Why does it matter?
The "why" is the most important layer — it builds lasting intuition.
Build intuition first, then add precision. Use analogies for abstract ideas.
Keep paragraphs short (3-5 sentences max).
Use Mermaid diagrams (```mermaid blocks) when visual representation helps —
e.g., data structure layouts, decorator wrapping order, MRO resolution, event loop flow.]

## Code Examples

[Annotated, idiomatic, production-quality code. Show how a professional
would actually write this — proper naming, error handling where appropriate,
clean structure. Each example should be self-contained and runnable.]

## Common Pitfalls

[Bad vs good code comparisons. Each pitfall gets:
- What the mistake is
- Why it's wrong
- The correct approach with code]

## Key Takeaways

- Bullet list of 3-5 most important points
- What to remember, what to internalize
```

## Writing Style

- **Audience**: Self-learner studying independently — no instructor, no classroom.
- **Tone**: Concise, expert, opinionated. Write like a senior engineer mentoring a colleague, not a textbook.
- **Structure**: Build intuition first, then add precision. Use analogies for abstract ideas.
- **Why-first**: For every concept, always explain *why* it exists, *why* this approach, and *why* it matters. The "why" is more important than the "what."
- **Paragraphs**: Keep short — 3-5 sentences max. Dense walls of text kill learning.
- **No exercises**: Do not include practice problems, homework, or challenges.
- **No external resources**: Do not include "Further Reading" sections or links to external material.

## Code Example Standards

- Write **idiomatic, production-quality code** — the kind a senior engineer would write at work.
- Show professional coding habits: meaningful names, type hints, proper error handling, clean structure.
- Every code block must be **self-contained and runnable** (include necessary imports).
- Use detailed inline comments to explain *why*, not just *what*.
- When showing a pattern or technique, show the **realistic use case**, not a toy example.

## Common Pitfalls Format

Each pitfall must include:

1. A brief description of the mistake
2. A `# BAD` code block showing the wrong way
3. An explanation of *why* it's wrong
4. A `# GOOD` code block showing the correct approach

```python
# BAD — [description of what's wrong]
[bad code]

# GOOD — [description of the fix]
[good code]
```

---

## Repo-Specific: Python

### Language & Version

- **Language**: Python (all code examples in Python)
- **Target version**: Python 3.12+ (use modern syntax and features)

### Coding Conventions (PEP 8 and beyond)

- Follow **PEP 8** strictly — `snake_case` for variables/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- **Type hints everywhere** — every function signature must have full type annotations (parameters and return type)
- Use `from __future__ import annotations` when using forward references or complex types
- Use `str | None` syntax (PEP 604) over `Optional[str]`
- Use **f-strings** for all string formatting — never `%` or `.format()`
- Use `pathlib.Path` over `os.path` — always
- Use `dataclasses` or Pydantic `BaseModel` over plain `__init__` for data containers
- Prefer `collections.abc` types for type hints (`Sequence`, `Mapping`) over concrete types (`list`, `dict`) in function signatures

### Modern Python Idioms

- Use **walrus operator** (`:=`) where it reduces redundancy (e.g., in `while` loops, `if` conditions)
- Use `match`/`case` (structural pattern matching, 3.10+) where it improves clarity over `if`/`elif` chains
- Use `contextlib.contextmanager` for simple context managers — full `__enter__`/`__exit__` only when needed
- Use `functools.cache` / `functools.lru_cache` for memoization
- Prefer list/dict/set comprehensions over `map()`/`filter()` — comprehensions are more Pythonic and readable
- Use `itertools` and `more-itertools` for complex iteration patterns instead of manual loops

### Python-Specific Teaching Notes

- Always explain what happens **under the hood** — Python's object model, reference counting, GIL implications
- When discussing data structures, mention the CPython implementation (dict is a hash table, list is a dynamic array)
- For OOP topics, explain Python's specific approach (duck typing, MRO via C3 linearization, descriptors behind `@property`)
- For concurrency topics, always clarify the GIL's impact and which approach fits which workload (threading for I/O, multiprocessing for CPU, asyncio for high-concurrency I/O)
- Show the REPL-friendly nature of Python — demonstrate that code can be tested interactively
