# Module 3 — Control Flow: Decisions and Repetition

> *"The art of programming is the art of organizing complexity."*
> — Edsger W. Dijkstra

---

## 3.1 Truthiness, Falsiness, and Short-Circuit Evaluation

### 🟢 Beginner: What Python Considers True and False

Every Python object can be evaluated in a boolean context. You don't need to write `if x == True:` — Python already knows what's "truthy" and "falsy."

```python
# The COMPLETE list of falsy values in Python:
falsy_values = [
    False,      # The boolean False
    None,       # The absence of a value
    0,          # Integer zero
    0.0,        # Float zero
    0j,         # Complex zero
    "",         # Empty string
    [],         # Empty list
    (),         # Empty tuple
    {},         # Empty dict
    set(),      # Empty set
    frozenset(),  # Empty frozenset
    range(0),   # Empty range
    b"",        # Empty bytes
]

# EVERYTHING ELSE is truthy
for val in falsy_values:
    assert not val, f"{val!r} should be falsy"

# Truthy examples
truthy_values = [True, 1, -1, 0.001, "hello", [0], {0: 0}, " "]
for val in truthy_values:
    assert val, f"{val!r} should be truthy"
```

**Using truthiness idiomatically:**

```python
# ❌ Beginner style — explicit comparisons
name: str = ""
if name != "":
    print(name)

items: list[int] = []
if len(items) > 0:
    print(items)

result = None
if result != None:
    print(result)

# ✅ Pythonic style — leverage truthiness
name = ""
if name:            # Empty string is falsy
    print(name)

items = []
if items:           # Empty list is falsy
    print(items)

result = None
if result is not None:  # For None, use 'is not' (identity check)
    print(result)
```

### 🟡 Intermediate: Short-Circuit Evaluation — `and`/`or` Return Values

This is the single most misunderstood behavior in Python for intermediate developers. `and` and `or` do **not** return `True`/`False`. They return one of their **operands**.

```python
# 'or' returns the FIRST TRUTHY value, or the LAST value if all falsy
print("hello" or "world")    # "hello"   — first is truthy, return it
print("" or "world")         # "world"   — first is falsy, try second
print("" or 0)               # 0         — both falsy, return last
print("" or [] or "backup")  # "backup"  — keeps going until truthy

# 'and' returns the FIRST FALSY value, or the LAST value if all truthy
print("hello" and "world")   # "world"   — first is truthy, check second
print("" and "world")        # ""        — first is falsy, return it
print("hello" and 42)        # 42        — both truthy, return last
print(1 and 2 and 3)         # 3         — all truthy, return last

# WHY THIS MATTERS — common patterns:
# Pattern 1: Default values with 'or'
username: str = "" or "Anonymous"   # "Anonymous"
port: int = 0 or 8080              # 8080  — but 0 might be valid!

# Pattern 2: Guard clauses with 'and'
data: dict | None = {"name": "Alice"}
name = data and data.get("name")   # "Alice" — safe, no KeyError
data = None
name = data and data.get("name")   # None — short-circuits, never calls .get()
```

**Gotcha: `or` replaces ALL falsy values, not just `None`:**

```python
# This is a subtle bug
config: dict[str, int] = {"timeout": 0}  # 0 is a valid timeout (no waiting)
timeout: int = config.get("timeout") or 30
print(timeout)  # 30! — 0 is falsy, so 'or' replaced it

# FIX: Be explicit about what you're checking
raw_timeout = config.get("timeout")
timeout = raw_timeout if raw_timeout is not None else 30
print(timeout)  # 0 — correct!

# Or even better, use dict.get() with a default
timeout = config.get("timeout", 30)  # Only uses 30 if key is MISSING
```

**Short-circuit evaluation means side effects may not execute:**

```python
def expensive_check() -> bool:
    print("  Running expensive check...")
    return True

# With 'or', second operand is skipped if first is truthy
result = True or expensive_check()
# "Running expensive check..." is NEVER printed

# With 'and', second operand is skipped if first is falsy
result = False and expensive_check()
# "Running expensive check..." is NEVER printed

# This is useful for guarding against errors:
items: list[int] = []
# Safe — if items is empty, 'and' short-circuits before items[0]
if items and items[0] > 10:
    print("First item is large")
```

### 🔴 Expert: How CPython Compiles Boolean Expressions

CPython compiles `and`/`or` into jump instructions, not function calls. This is why short-circuiting is essentially free.

```python
import dis

def boolean_example(a: object, b: object) -> object:
    return a or b

dis.dis(boolean_example)
# Bytecode:
#   LOAD_FAST    0 (a)
#   COPY         1              ← duplicate top of stack
#   POP_JUMP_IF_TRUE  target    ← if a is truthy, jump to return
#   POP_TOP                     ← discard the duplicate
#   LOAD_FAST    1 (b)          ← load b instead
# target:
#   RETURN_VALUE
```

```python
import dis

def complex_boolean(x: int, y: int, z: int) -> bool:
    return x > 0 and y > 0 and z > 0

dis.dis(complex_boolean)
# Compiles to a chain of conditional jumps:
#   LOAD_FAST  x
#   LOAD_CONST 0
#   COMPARE_OP >
#   POP_JUMP_IF_FALSE  fail_label   ← first False? skip the rest
#   LOAD_FAST  y
#   LOAD_CONST 0
#   COMPARE_OP >
#   POP_JUMP_IF_FALSE  fail_label   ← second False? skip the rest
#   LOAD_FAST  z
#   LOAD_CONST 0
#   COMPARE_OP >
#   RETURN_VALUE
# fail_label:
#   RETURN_VALUE (with False on stack)
```

**Custom truthiness via `__bool__` and `__len__`:**

```python
class Bag:
    """A container that is truthy when non-empty."""
    def __init__(self, items: list | None = None) -> None:
        self._items: list = items or []

    def __bool__(self) -> bool:
        """Called by bool(), if/else, and/or, not."""
        print(f"  __bool__ called, returning {len(self._items) > 0}")
        return len(self._items) > 0

    def __len__(self) -> int:
        """Fallback if __bool__ is not defined."""
        return len(self._items)

empty_bag = Bag()
full_bag = Bag([1, 2, 3])

if empty_bag:           # __bool__ called → False
    print("has items")
else:
    print("empty")      # ← this runs

if full_bag:            # __bool__ called → True
    print("has items")  # ← this runs
```

```
CPython truth-testing protocol:
1. Call obj.__bool__() if defined → must return True or False
2. Else call obj.__len__() if defined → truthy if len > 0
3. Else the object is ALWAYS truthy

If __bool__ returns a non-bool, CPython raises TypeError.
If __len__ returns negative, CPython raises ValueError.
```

---

## 3.2 `if/elif/else` — Conditionals and Pattern Matching

### 🟢 Beginner: The Basics

```python
temperature: int = 72

# Simple if
if temperature > 100:
    print("Boiling!")

# if/else
if temperature > 80:
    print("Hot")
else:
    print("Comfortable")  # ← this runs

# if/elif/else — checks top to bottom, runs the FIRST match
if temperature > 100:
    print("Boiling")
elif temperature > 80:
    print("Hot")
elif temperature > 60:
    print("Comfortable")  # ← this runs (first True condition)
elif temperature > 40:
    print("Cool")          # ← NOT reached even though it's also True
else:
    print("Cold")
```

**Ternary expression (conditional expression):**

```python
age: int = 20

# Instead of:
if age >= 18:
    status = "adult"
else:
    status = "minor"

# Write:
status: str = "adult" if age >= 18 else "minor"

# Can be nested (but please don't go deeper than one level)
label: str = "senior" if age >= 65 else "adult" if age >= 18 else "minor"

# ❌ This is unreadable — use if/elif/else instead
grade = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "F"
```

### 🟡 Intermediate: Pattern Matching with `match/case` (Python 3.10+)

Pattern matching is **not** a switch statement. It's a structural matching system that can destructure data.

```python
# Basic value matching (looks like switch/case, but it's more powerful)
def http_status(code: int) -> str:
    match code:
        case 200:
            return "OK"
        case 301:
            return "Moved Permanently"
        case 404:
            return "Not Found"
        case 500:
            return "Internal Server Error"
        case _:            # _ is the wildcard — matches anything
            return f"Unknown status: {code}"

print(http_status(404))  # "Not Found"
print(http_status(418))  # "Unknown status: 418"
```

**Structural matching — destructuring data:**

```python
# Matching tuples and sequences
def describe_point(point: tuple) -> str:
    match point:
        case (0, 0):
            return "Origin"
        case (x, 0):
            return f"On X-axis at x={x}"
        case (0, y):
            return f"On Y-axis at y={y}"
        case (x, y):
            return f"Point at ({x}, {y})"

print(describe_point((0, 0)))    # "Origin"
print(describe_point((5, 0)))    # "On X-axis at x=5"
print(describe_point((3, 4)))    # "Point at (3, 4)"
```

```python
# Matching dictionaries (partial match — extra keys are OK)
def process_event(event: dict) -> str:
    match event:
        case {"type": "click", "x": x, "y": y}:
            return f"Click at ({x}, {y})"
        case {"type": "keypress", "key": key}:
            return f"Key pressed: {key}"
        case {"type": "scroll", "direction": "up" | "down" as d}:
            return f"Scroll {d}"
        case _:
            return "Unknown event"

print(process_event({"type": "click", "x": 100, "y": 200, "button": "left"}))
# "Click at (100, 200)" — extra "button" key is fine
```

```python
# Guards — adding conditions to patterns
def categorize(value: int) -> str:
    match value:
        case x if x < 0:
            return "negative"
        case 0:
            return "zero"
        case x if x % 2 == 0:
            return f"positive even: {x}"
        case x:
            return f"positive odd: {x}"

print(categorize(-5))   # "negative"
print(categorize(0))    # "zero"
print(categorize(4))    # "positive even: 4"
print(categorize(7))    # "positive odd: 7"
```

**Gotcha: Pattern matching with classes:**

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

@dataclass
class Circle:
    center: Point
    radius: float

def describe_shape(shape: Point | Circle) -> str:
    match shape:
        case Circle(center=Point(x=0, y=0), radius=r):
            return f"Circle at origin with radius {r}"
        case Circle(center=Point(x=x, y=y), radius=r) if r > 10:
            return f"Large circle at ({x}, {y})"
        case Circle(center=center, radius=r):
            return f"Circle at {center} with radius {r}"
        case Point(x=x, y=y):
            return f"Just a point at ({x}, {y})"

print(describe_shape(Circle(Point(0, 0), 5)))
# "Circle at origin with radius 5"
print(describe_shape(Circle(Point(3, 4), 15)))
# "Large circle at (3, 4)"
```

### 🔴 Expert: CPython's Match Statement Implementation

Pattern matching compiles to a series of type checks, attribute lookups, and conditional jumps — NOT a hash table (unlike C's `switch`).

```python
import dis

def match_example(x: object) -> str:
    match x:
        case 1:
            return "one"
        case 2:
            return "two"
        case _:
            return "other"

dis.dis(match_example)
# Simplified bytecode:
#   LOAD_FAST  x
#   COPY
#   LOAD_CONST  1
#   COMPARE_OP  ==
#   POP_JUMP_IF_FALSE  next_case
#   ...return "one"...
# next_case:
#   COPY
#   LOAD_CONST  2
#   COMPARE_OP  ==
#   POP_JUMP_IF_FALSE  wildcard
#   ...return "two"...
# wildcard:
#   ...return "other"...
```

**Performance implication: `match/case` with value patterns is O(n) — it checks each case sequentially.** For many cases on a single value, a dictionary dispatch is faster:

```python
# O(n) — linear scan through cases
def status_match(code: int) -> str:
    match code:
        case 200: return "OK"
        case 301: return "Moved"
        case 302: return "Found"
        case 400: return "Bad Request"
        case 401: return "Unauthorized"
        case 403: return "Forbidden"
        case 404: return "Not Found"
        case 500: return "Server Error"
        case _:   return "Unknown"

# O(1) — dictionary lookup
STATUS_MAP: dict[int, str] = {
    200: "OK", 301: "Moved", 302: "Found",
    400: "Bad Request", 401: "Unauthorized",
    403: "Forbidden", 404: "Not Found", 500: "Server Error",
}

def status_dict(code: int) -> str:
    return STATUS_MAP.get(code, "Unknown")
```

**Where `match/case` truly shines (and dictionaries can't help):**

```python
# Pattern matching excels at STRUCTURAL decomposition
# This is awkward to express any other way:
def eval_expr(expr: tuple) -> float:
    """A simple recursive expression evaluator."""
    match expr:
        case ("+", left, right):
            return eval_expr(left) + eval_expr(right)
        case ("-", left, right):
            return eval_expr(left) - eval_expr(right)
        case ("*", left, right):
            return eval_expr(left) * eval_expr(right)
        case ("neg", operand):
            return -eval_expr(operand)
        case float(n) | int(n):
            return float(n)
        case _:
            raise ValueError(f"Unknown expression: {expr}")

# (2 + 3) * -4 = -20
tree = ("*", ("+", 2, 3), ("neg", 4))
print(eval_expr(tree))  # -20.0
```

---

## 3.3 `for` and `while` — Iteration Protocol Under the Hood

### 🟢 Beginner: Looping Over Things

```python
# for loop — iterate over any iterable
fruits: list[str] = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)

# range() — generate sequences of numbers
for i in range(5):         # 0, 1, 2, 3, 4
    print(i, end=" ")
print()

for i in range(2, 8):      # 2, 3, 4, 5, 6, 7
    print(i, end=" ")
print()

for i in range(0, 20, 3):  # 0, 3, 6, 9, 12, 15, 18
    print(i, end=" ")
print()

for i in range(10, 0, -2): # 10, 8, 6, 4, 2
    print(i, end=" ")
print()

# while loop — repeat while a condition is True
count: int = 0
while count < 5:
    print(count, end=" ")
    count += 1
# Output: 0 1 2 3 4
```

**Iterating with `enumerate()` and `zip()`:**

```python
# enumerate — when you need both index and value
languages: list[str] = ["Python", "Rust", "Go"]
for idx, lang in enumerate(languages):
    print(f"  {idx}: {lang}")
# 0: Python
# 1: Rust
# 2: Go

# zip — iterate over multiple sequences in parallel
names: list[str] = ["Alice", "Bob", "Charlie"]
scores: list[int] = [95, 87, 92]
grades: list[str] = ["A", "B+", "A-"]

for name, score, grade in zip(names, scores, grades):
    print(f"  {name}: {score} ({grade})")

# zip stops at the shortest — use zip_longest for padding
from itertools import zip_longest
short: list[int] = [1, 2]
long: list[int] = [10, 20, 30]
for a, b in zip_longest(short, long, fillvalue=0):
    print(f"  {a} + {b} = {a + b}")
# 1 + 10 = 11
# 2 + 20 = 22
# 0 + 30 = 30
```

### 🟡 Intermediate: The Iteration Protocol and Common Patterns

**The iteration protocol — what `for` actually does:**

```python
# When you write:
for item in [1, 2, 3]:
    print(item)

# Python actually does this:
_iter = iter([1, 2, 3])       # Calls list.__iter__() → returns iterator
while True:
    try:
        item = next(_iter)     # Calls iterator.__next__()
        print(item)
    except StopIteration:      # Iterator is exhausted
        break
```

**Gotcha: Modifying a collection while iterating over it:**

```python
# ❌ NEVER modify a list while iterating over it
numbers: list[int] = [1, 2, 3, 4, 5, 6]
for n in numbers:
    if n % 2 == 0:
        numbers.remove(n)   # Skips elements! Iterator gets confused
print(numbers)  # [1, 3, 5]? NO! → [1, 3, 5, 6] — 6 was skipped

# ❌ Also broken — index shifts on removal
numbers = [1, 2, 3, 4, 5, 6]
for i in range(len(numbers)):
    if numbers[i] % 2 == 0:
        numbers.pop(i)       # IndexError eventually!

# ✅ FIX 1: Build a new list (preferred)
numbers = [1, 2, 3, 4, 5, 6]
odds: list[int] = [n for n in numbers if n % 2 != 0]
print(odds)  # [1, 3, 5]

# ✅ FIX 2: Iterate over a copy
numbers = [1, 2, 3, 4, 5, 6]
for n in numbers[:]:        # [:] creates a shallow copy
    if n % 2 == 0:
        numbers.remove(n)
print(numbers)  # [1, 3, 5]

# ✅ FIX 3: Iterate backwards (for index-based removal)
numbers = [1, 2, 3, 4, 5, 6]
for i in range(len(numbers) - 1, -1, -1):
    if numbers[i] % 2 == 0:
        numbers.pop(i)
print(numbers)  # [1, 3, 5]
```

**Gotcha: Iterators are single-use:**

```python
numbers: list[int] = [1, 2, 3]

# A list is ITERABLE — you can iterate multiple times
for n in numbers:
    print(n, end=" ")  # 1 2 3
print()
for n in numbers:
    print(n, end=" ")  # 1 2 3 — works again!
print()

# An ITERATOR is single-use
it = iter(numbers)
for n in it:
    print(n, end=" ")  # 1 2 3
print()
for n in it:
    print(n, end=" ")  # nothing — iterator is exhausted!
print()

# This catches people with generators (which are iterators):
evens = (n for n in range(10) if n % 2 == 0)  # Generator expression
print(list(evens))  # [0, 2, 4, 6, 8]
print(list(evens))  # [] — exhausted!
```

**`while` loop patterns:**

```python
# Sentinel loop — read until a special value
entries: list[str] = []
while True:
    line: str = input("Enter text (or 'quit'): ")
    if line == "quit":
        break
    entries.append(line)

# Convergence loop — repeat until a condition is met
def sqrt_newton(n: float, tolerance: float = 1e-10) -> float:
    """Compute square root using Newton's method."""
    guess: float = n / 2.0
    while True:
        next_guess: float = (guess + n / guess) / 2.0
        if abs(next_guess - guess) < tolerance:
            return next_guess
        guess = next_guess

print(sqrt_newton(2))  # 1.4142135623730951
```

### 🔴 Expert: CPython's `for` Loop Bytecode and Optimizations

```python
import dis

def loop_example() -> int:
    total: int = 0
    for i in range(100):
        total += i
    return total

dis.dis(loop_example)
# Key bytecode:
#   GET_ITER                    ← calls iter(range(100))
# loop_start:
#   FOR_ITER  end_label         ← calls next(), jumps to end on StopIteration
#   STORE_FAST  i
#   LOAD_FAST   total
#   LOAD_FAST   i
#   BINARY_ADD
#   STORE_FAST  total
#   JUMP_BACKWARD  loop_start  ← unconditional jump back
# end_label:
#   ...
```

**`range()` is not a list — it's a lazy sequence:**

```python
import sys

# range() creates a tiny object regardless of size
r1 = range(10)
r2 = range(10_000_000_000)   # 10 billion

print(sys.getsizeof(r1))    # 48 bytes
print(sys.getsizeof(r2))    # 48 bytes — same!

# range supports O(1) membership testing
print(999_999_999 in r2)     # True — calculated, not iterated!

# range also supports O(1) indexing and slicing
print(r2[42])                # 42
print(r2[-1])                # 9999999999
print(r2[10:20])             # range(10, 20) — another range!
```

**How `range.__contains__` achieves O(1):**

```python
# For range(start, stop, step), membership check is:
# 1. Is start <= value < stop? (or reversed for negative step)
# 2. Is (value - start) % step == 0?
# Both are O(1) arithmetic operations — no iteration needed.

r = range(2, 100, 3)   # 2, 5, 8, 11, ...
print(11 in r)          # True:  2 <= 11 < 100 and (11-2) % 3 == 0
print(12 in r)          # False: (12-2) % 3 == 1, not 0
```

**Loop unrolling and optimization notes:**

```python
# CPython does NOT unroll loops or perform JIT compilation (as of 3.12)
# Performance-critical loops should:

# 1. Move attribute lookups outside the loop
data: list[int] = list(range(1000))

# ❌ Slow — attribute lookup on every iteration
result: list[int] = []
for x in data:
    result.append(x * 2)     # 'append' is looked up each time

# ✅ Faster — cache the method reference
result = []
append = result.append        # Bind once
for x in data:
    append(x * 2)             # Direct call, no attribute lookup

# ✅ Fastest — use a comprehension (compiled to tighter bytecode)
result = [x * 2 for x in data]

# 2. Use local variables instead of globals (LOAD_FAST vs LOAD_GLOBAL)
# Local variable access: ~50ns (LOAD_FAST)
# Global variable access: ~80ns (LOAD_GLOBAL → dict lookup)
# Attribute access: ~100ns (LOAD_ATTR → __dict__ or descriptor protocol)
```

---

## 3.4 The `else` Clause on Loops — Python's Most Misunderstood Feature

### 🟢 Beginner: What It Does

Both `for` and `while` loops can have an `else` clause. It runs when the loop finishes **normally** (without hitting a `break`).

```python
# The else clause runs when the loop completes without breaking
for n in [2, 4, 6, 8]:
    if n % 2 != 0:
        print(f"Found odd number: {n}")
        break
else:
    # This runs because no 'break' was executed
    print("All numbers are even!")
# Output: "All numbers are even!"

# Now with an odd number in the list
for n in [2, 4, 5, 8]:
    if n % 2 != 0:
        print(f"Found odd number: {n}")
        break
else:
    print("All numbers are even!")
# Output: "Found odd number: 5"
# The else clause does NOT run because 'break' was executed
```

### 🟡 Intermediate: The Mental Model — "nobreak"

The biggest source of confusion is the name `else`. It's better to think of it as `nobreak`:

```
for item in iterable:       │    while condition:
    if something:            │        if something:
        break                │            break
else:  # ← "nobreak"        │    else:  # ← "nobreak"
    # Ran to completion      │        # Condition became False naturally
```

**The classic use case — searching with a "not found" fallback:**

```python
def find_prime_factor(n: int) -> int | None:
    """Find the smallest prime factor of n, or None if n is prime."""
    for candidate in range(2, int(n ** 0.5) + 1):
        if n % candidate == 0:
            return candidate      # Found a factor — implicit break
    else:
        return None               # Loop completed — n is prime

print(find_prime_factor(15))   # 3
print(find_prime_factor(17))   # None — 17 is prime
```

**Without `for/else`, you'd need a flag variable:**

```python
# Without for/else — needs a sentinel flag
def find_prime_factor_no_else(n: int) -> int | None:
    found: bool = False
    result: int = 0
    for candidate in range(2, int(n ** 0.5) + 1):
        if n % candidate == 0:
            found = True
            result = candidate
            break
    if found:
        return result
    return None

# The for/else version is cleaner, but many teams avoid it
# because it confuses developers who don't know the feature.
```

**`while/else` — the condition-failed case:**

```python
def consume_queue(queue: list[str], max_attempts: int = 10) -> bool:
    """Process items until queue is empty or max attempts reached."""
    attempts: int = 0
    while queue and attempts < max_attempts:
        item: str = queue.pop(0)
        print(f"  Processing: {item}")
        attempts += 1
        if item == "POISON":
            print("  Poison pill received!")
            break
    else:
        # Loop ended naturally (queue empty OR max_attempts reached)
        print("  Queue processing completed normally.")
        return True
    # Only reached if break was hit
    print("  Queue processing aborted.")
    return False

consume_queue(["a", "b", "c"])
# Processing: a
# Processing: b
# Processing: c
# Queue processing completed normally.

consume_queue(["a", "POISON", "c"])
# Processing: a
# Processing: POISON
# Poison pill received!
# Queue processing aborted.
```

### 🔴 Expert: Bytecode and the PEP 315 Debate

```python
import dis

def loop_with_else() -> str:
    for i in range(5):
        if i == 10:
            break
    else:
        return "completed"
    return "broken"

dis.dis(loop_with_else)
# The 'else' block is positioned AFTER the loop's normal exit.
# The 'break' jumps PAST the else block.
#
#   GET_ITER
# loop:
#   FOR_ITER  after_else      ← normal exit jumps to else block
#   STORE_FAST  i
#   ...comparison...
#   POP_JUMP_IF_FALSE  loop
#   ...break...               ← JUMP_FORWARD past the else block
# else_block:
#   LOAD_CONST  "completed"
#   RETURN_VALUE
# after_else:
#   LOAD_CONST  "broken"
#   RETURN_VALUE
```

**The naming controversy:**

Python's `for/else` has been debated since its inception. Guido van Rossum has said he might not include it if redesigning the language. The confusion stems from `else` suggesting "if the for didn't execute" (like `if/else`), when it actually means "if the for wasn't broken out of."

Several alternative names have been proposed over the years: `nobreak`, `then`, `finally` (already taken), `completion`. None were adopted because changing keywords is nearly impossible in a mature language.

**Style guideline for production code:**

```python
# Many style guides (Google, etc.) discourage for/else
# because it's a readability trap for team members unfamiliar with it.

# If you DO use it, always add a comment:
for item in haystack:
    if item == needle:
        print(f"Found: {item}")
        break
else:  # nobreak — needle was not found in haystack
    print("Not found")

# Alternatively, extract to a function with early return:
def find_needle(haystack: list[str], needle: str) -> str | None:
    for item in haystack:
        if item == needle:
            return item
    return None  # Clearer intent than for/else
```

---

## 3.5 `break`, `continue`, and the Walrus Operator (`:=`) in Loops

### 🟢 Beginner: `break` and `continue`

```python
# break — exit the loop immediately
for i in range(10):
    if i == 5:
        break
    print(i, end=" ")
print()  # Output: 0 1 2 3 4

# continue — skip the rest of this iteration, go to next
for i in range(10):
    if i % 3 == 0:
        continue        # Skip multiples of 3
    print(i, end=" ")
print()  # Output: 1 2 4 5 7 8

# break and continue only affect the INNERMOST loop
for i in range(3):
    for j in range(3):
        if j == 1:
            break       # Only breaks the inner loop
        print(f"({i},{j})", end=" ")
    print()
# Output:
# (0,0)
# (1,0)
# (2,0)
```

### 🟡 Intermediate: The Walrus Operator `:=` (Python 3.8+)

The walrus operator assigns AND returns a value in a single expression.

```python
# Without walrus — read, then check
import re

line: str = "Error: connection timeout"
match = re.search(r"Error: (.+)", line)
if match:
    print(f"Found error: {match.group(1)}")

# With walrus — assign and check in one expression
if match := re.search(r"Error: (.+)", line):
    print(f"Found error: {match.group(1)}")
```

**Where the walrus operator truly shines — `while` loops:**

```python
# ❌ Without walrus — awkward read-ahead pattern
import sys

data: str = sys.stdin.readline()
while data:
    process(data)
    data = sys.stdin.readline()   # Duplicated read

# ✅ With walrus — read and check in the condition
while data := sys.stdin.readline():
    process(data)

# Another classic: reading chunks from a file
with open("large_file.txt", "rb") as f:
    while chunk := f.read(8192):  # Read 8KB at a time
        process_chunk(chunk)
```

**Walrus in comprehensions — filter and transform:**

```python
import math

# Without walrus — compute expensive function twice
results: list[float] = []
for x in range(100):
    val: float = math.sqrt(x)
    if val > 5:
        results.append(val)

# With walrus — compute once, filter on the result
results = [y for x in range(100) if (y := math.sqrt(x)) > 5]

# Real-world example: parse and filter in one pass
raw_data: list[str] = ["42", "invalid", "17", "bad", "99"]

valid_numbers: list[int] = []
for item in raw_data:
    try:
        if (num := int(item)) > 20:
            valid_numbers.append(num)
    except ValueError:
        pass
print(valid_numbers)  # [42, 99]
```

**Gotcha: Walrus operator scope in comprehensions:**

```python
# The walrus operator LEAKS into the enclosing scope!
# (unlike regular comprehension variables in Python 3)

result = [y := x * 2 for x in range(5)]
print(result)   # [0, 2, 4, 6, 8]
print(y)        # 8 — y leaked into the outer scope!

# This is intentional behavior per PEP 572, but it can be surprising
```

### 🔴 Expert: Bytecode for `break`, `continue`, and Walrus

```python
import dis

def walrus_loop() -> list[int]:
    results: list[int] = []
    data = iter(range(10))
    while (n := next(data, None)) is not None:
        if n % 2 == 0:
            results.append(n)
    return results

dis.dis(walrus_loop)
# The walrus operator compiles to:
#   LOAD_GLOBAL  next
#   LOAD_FAST    data
#   LOAD_CONST   None
#   CALL_FUNCTION  2
#   COPY                       ← duplicate the result
#   STORE_FAST   n             ← assign to n (the := part)
#   LOAD_CONST   None
#   IS_OP        1             ← 'is not None'
#   POP_JUMP_IF_FALSE  end     ← exit loop if None
```

**The `continue` statement and exception handling interaction:**

```python
# 'continue' inside try/finally is a subtle corner case
for i in range(5):
    try:
        if i == 2:
            continue        # Skip i=2
        print(f"  Processing {i}")
    finally:
        print(f"  Cleanup for {i}")  # ALWAYS runs, even on continue!

# Output:
#   Processing 0
#   Cleanup for 0
#   Processing 1
#   Cleanup for 1
#   Cleanup for 2         ← finally runs even though we continued!
#   Processing 3
#   Cleanup for 3
#   Processing 4
#   Cleanup for 4
```

**Nested loop breaking — the `for/else` alternative to flags:**

```python
# Problem: break out of nested loops

# ❌ Using a flag variable
found: bool = False
for i in range(10):
    for j in range(10):
        if i * j == 42:
            found = True
            break
    if found:
        break

# ✅ Option 1: Extract to a function (clearest)
def find_product_pair(target: int) -> tuple[int, int] | None:
    for i in range(10):
        for j in range(10):
            if i * j == target:
                return (i, j)
    return None

# ✅ Option 2: for/else (no extra function)
for i in range(10):
    for j in range(10):
        if i * j == 42:
            print(f"Found: {i} * {j} = 42")
            break
    else:
        continue  # Inner loop wasn't broken → continue outer loop
    break         # Inner loop WAS broken → break outer loop too

# ✅ Option 3: itertools.product for flat iteration
from itertools import product
for i, j in product(range(10), range(10)):
    if i * j == 42:
        print(f"Found: {i} * {j} = 42")
        break
```

---

## 🔧 Debug This: The Infinite Retry Loop

Your team wrote a retry mechanism for an API client. It has three bugs — one causes an infinite loop, one silently swallows errors, and one has a subtle off-by-one. Find them all:

```python
import time
import random

def flaky_api_call():
    """Simulates an API that fails 70% of the time."""
    if random.random() < 0.7:
        raise ConnectionError("Server unavailable")
    return {"status": "ok", "data": [1, 2, 3]}

def fetch_with_retry(max_retries=3, delay=1.0):
    """Fetch data from the API with exponential backoff."""
    retries = 0
    while retries <= max_retries:
        try:
            result = flaky_api_call()
            print(f"  Success on attempt {retries + 1}!")
            return result
        except Exception:
            retries += 1
            wait = delay * (2 ** retries)
            print(f"  Attempt {retries} failed. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    else:
        print(f"  All {max_retries} retries exhausted.")
        return None

# Test it
random.seed(42)
result = fetch_with_retry(max_retries=3, delay=0.1)
print(f"Result: {result}")
```

### Bugs to find:

```
1. ____________________________________________________
   Hint: If max_retries=3, how many TOTAL attempts happen?
   Is `<=` correct here? How does this interact with "retries exhausted"?

2. ____________________________________________________
   Hint: `except Exception` catches EVERYTHING. What if the API
   returns a JSON decode error? Should that be retried?

3. ____________________________________________________
   Hint: Look at the attempt number printed. After the first failure,
   retries becomes 1, but we print "Attempt 1 failed" — that looks
   like the first attempt. Is the counting off?

4. ____________________________________________________
   Hint: What about type hints, and should delay ever be negative?
```

### Solution (try first!)

```python
import time
import random
from typing import Any


def flaky_api_call() -> dict[str, Any]:
    """Simulates an API that fails 70% of the time."""
    if random.random() < 0.7:
        raise ConnectionError("Server unavailable")
    return {"status": "ok", "data": [1, 2, 3]}


def fetch_with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict[str, Any] | None:
    """Fetch data from the API with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (not counting the first).
        base_delay: Base delay in seconds. Must be positive.

    Returns:
        API response dict, or None if all retries exhausted.

    Raises:
        ValueError: If base_delay is not positive.
    """
    if base_delay <= 0:
        raise ValueError(f"base_delay must be positive, got {base_delay}")

    last_exception: Exception | None = None

    # Bug 1 FIX: Use < instead of <= for correct retry count.
    # With <=, we get max_retries + 1 total attempts, meaning
    # the "All retries exhausted" message lies about the count.
    # Now: attempt 0 (first try) + max_retries retries = max_retries + 1 total
    for attempt in range(1, max_retries + 2):  # 1-indexed, inclusive
        try:
            result: dict[str, Any] = flaky_api_call()
            print(f"  Success on attempt {attempt}!")
            return result
        except ConnectionError as exc:
            # Bug 2 FIX: Only catch the specific retryable exception.
            # JSONDecodeError, TypeError, etc. should propagate immediately.
            last_exception = exc
            if attempt <= max_retries:
                # Bug 3 FIX: Correct attempt numbering using the for variable
                wait: float = base_delay * (2 ** (attempt - 1))
                print(
                    f"  Attempt {attempt}/{max_retries + 1} failed. "
                    f"Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
            else:
                print(
                    f"  Attempt {attempt}/{max_retries + 1} failed. "
                    f"No retries remaining."
                )

    print(f"  All {max_retries + 1} attempts exhausted.")
    return None


# Test it
random.seed(42)
result: dict[str, Any] | None = fetch_with_retry(max_retries=3, base_delay=0.1)
print(f"Result: {result}")
```

---

## Summary: Module 3 Key Takeaways

```
┌──────────────────────────────────────────────────────────────────┐
│                    CONTROL FLOW CHEAT SHEET                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Truthiness:  0, None, "", [], {}, set() are falsy.              │
│               Everything else is truthy.                          │
│                                                                   │
│  and/or:      Return operands, not booleans. Short-circuit.      │
│               "x or default" replaces ALL falsy values, not      │
│               just None — use explicit checks for 0, "", etc.    │
│                                                                   │
│  match/case:  Structural pattern matching, NOT switch/case.      │
│               O(n) sequential checks. Use dict dispatch for      │
│               simple value mapping.                               │
│                                                                   │
│  for/else:    "else" = "nobreak". Runs when loop finishes        │
│               without hitting break. Add a comment if you use it.│
│                                                                   │
│  Iteration:   Never modify a collection while iterating it.      │
│               Iterators are single-use. range() is O(1) memory.  │
│                                                                   │
│  Walrus :=    Assign+test in one expression. Great for while     │
│               loops. Beware scope leaking in comprehensions.     │
│                                                                   │
│  Performance: Comprehensions > cached-method loops > naive loops.│
│               Local vars (LOAD_FAST) > globals (LOAD_GLOBAL).    │
│                                                                   │
│  Production rule: Explicit is better than clever.                │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Next up → Module 4: Console I/O — Talking to the World**

Say "Start Module 4" when you're ready.
