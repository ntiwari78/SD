# Module 6 — Comprehensions: Elegance Meets Performance

> *"There should be one — and preferably only one — obvious way to do it."*
> — The Zen of Python
>
> *"Comprehensions are that one way for building collections."*

---

## 6.1 List Comprehensions — Syntax, Readability Rules, and When to Stop

### 🟢 Beginner: From Loop to Comprehension

A list comprehension is a concise way to build a list by transforming and/or filtering an iterable.

```python
# The traditional loop way
squares_loop: list[int] = []
for x in range(10):
    squares_loop.append(x ** 2)

# The comprehension way — same result, one line
squares_comp: list[int] = [x ** 2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# The anatomy:
# [ expression  for variable in iterable ]
#   ↑            ↑              ↑
#   what to      loop           what to
#   produce      variable       iterate over
```

**Adding a filter with `if`:**

```python
# Only even squares
even_squares: list[int] = [x ** 2 for x in range(10) if x % 2 == 0]
# [0, 4, 16, 36, 64]

# Equivalent loop:
even_squares_loop: list[int] = []
for x in range(10):
    if x % 2 == 0:
        even_squares_loop.append(x ** 2)
```

**Using `if/else` in the expression (NOT the filter):**

```python
# if/else in the EXPRESSION — transforms every element
labels: list[str] = ["even" if x % 2 == 0 else "odd" for x in range(5)]
# ['even', 'odd', 'even', 'odd', 'even']

# IMPORTANT: position matters!
# [expr if cond else expr  for x in iter]  ← ternary expression (transforms)
# [expr                    for x in iter if cond]  ← filter (excludes)

# These are DIFFERENT:
a: list[int | str] = [x if x > 3 else "small" for x in range(6)]
# ['small', 'small', 'small', 'small', 4, 5]  ← every element transformed

b: list[int] = [x for x in range(6) if x > 3]
# [4, 5]  ← elements excluded entirely
```

**Practical examples:**

```python
# String processing
names: list[str] = ["  Alice ", "BOB", " charlie"]
cleaned: list[str] = [name.strip().title() for name in names]
# ['Alice', 'Bob', 'Charlie']

# Flattening a file into words
text: str = "hello world\nfoo bar\nbaz"
words: list[str] = [word for line in text.split("\n") for word in line.split()]
# ['hello', 'world', 'foo', 'bar', 'baz']

# Type conversion with error handling? Use a loop instead.
raw: list[str] = ["42", "bad", "17", "oops", "99"]
# ❌ Can't do try/except in a comprehension
# ✅ Use a helper function
def safe_int(s: str) -> int | None:
    try:
        return int(s)
    except ValueError:
        return None

numbers: list[int] = [n for s in raw if (n := safe_int(s)) is not None]
# [42, 17, 99]
```

### 🟡 Intermediate: When to Use (and Not Use) Comprehensions

**The readability cliff — the one-screen-width rule:**

```python
# ✅ GOOD — clear, fits in one line, immediately understandable
active_users: list[str] = [u.name for u in users if u.is_active]

# ✅ GOOD — multi-line for readability (still a single comprehension)
report: list[dict[str, str | float]] = [
    {
        "name": student.name,
        "grade": student.grade,
        "gpa": round(student.gpa, 2),
    }
    for student in students
    if student.enrolled
]

# ⚠️ BORDERLINE — getting complex, consider a loop
result: list[str] = [
    f"{item.name}: ${item.price * (1 - item.discount):.2f}"
    for item in catalog
    if item.in_stock and item.price > 0
]

# ❌ BAD — too complex, use a regular loop
# This is unreadable as a comprehension:
# result = [transform(x) for group in data for x in group.items
#           if x.valid and x.type in allowed_types
#           and not any(r.matches(x) for r in exclusion_rules)]

# ✅ Rewrite as a loop with comments:
result: list = []
for group in data:
    for x in group.items:
        if not x.valid:
            continue
        if x.type not in allowed_types:
            continue
        if any(r.matches(x) for r in exclusion_rules):
            continue
        result.append(transform(x))
```

**Comprehensions vs. `map()`/`filter()`:**

```python
# Comprehension — almost always preferred in Python
squares: list[int] = [x ** 2 for x in range(10)]
evens: list[int] = [x for x in range(20) if x % 2 == 0]

# map/filter — sometimes used in functional-style code
squares_map: list[int] = list(map(lambda x: x ** 2, range(10)))
evens_filter: list[int] = list(filter(lambda x: x % 2 == 0, range(20)))

# map() CAN be faster when using a built-in function (no lambda):
# This is slightly faster than [str(x) for x in range(1000)]
string_nums: list[str] = list(map(str, range(1000)))

# But for anything needing a lambda, comprehensions are both
# faster AND more readable.
```

**Side effects in comprehensions — DON'T:**

```python
# ❌ NEVER use comprehensions for side effects
# [print(x) for x in range(5)]  ← builds a list of Nones, wasteful

# ✅ Use a loop
for x in range(5):
    print(x)

# The ONLY purpose of a comprehension is to BUILD A COLLECTION.
# If you're not using the result, use a loop.
```

### 🔴 Expert: Comprehension Bytecode and Performance

```python
import dis

# Comprehensions compile to their own code object (like a nested function)
def comp_example() -> list[int]:
    return [x ** 2 for x in range(10)]

dis.dis(comp_example)
# LOAD_CONST     <code object <listcomp>>   ← separate code object!
# MAKE_FUNCTION  0
# LOAD_GLOBAL    range
# LOAD_CONST     10
# CALL_FUNCTION  1
# GET_ITER
# CALL_FUNCTION  1                           ← calls the listcomp function
# RETURN_VALUE
```

**Why comprehensions are faster than equivalent loops:**

```python
import timeit

# Loop version
def loop_squares(n: int) -> list[int]:
    result: list[int] = []
    for x in range(n):
        result.append(x ** 2)
    return result

# Comprehension version
def comp_squares(n: int) -> list[int]:
    return [x ** 2 for x in range(n)]

n: int = 10_000
loop_time: float = timeit.timeit(lambda: loop_squares(n), number=1000)
comp_time: float = timeit.timeit(lambda: comp_squares(n), number=1000)
```

```
Typical results:

    Loop:          ~3.2s
    Comprehension: ~2.1s  (~35% faster)

Why the speed difference?
1. No .append() attribute lookup per iteration
   - Loop: LOAD_ATTR 'append' + CALL_FUNCTION every iteration
   - Comp: Uses LIST_APPEND bytecode (direct C call, no Python overhead)

2. The comprehension's code object has tighter bytecode
   - Fewer LOAD/STORE operations per iteration
   - The iteration variable is LOAD_FAST (known at compile time)

3. CPython can optimize the LIST_APPEND path
   - Pre-sizes the internal array when the iterable length is known
   - Avoids repeated over-allocation resizes
```

**Comprehension pre-sizing optimization:**

```python
import sys

# When CPython can determine the iterable's length, it pre-allocates
# the exact right amount of memory for the result list

# From a range (length known):
result_sized: list[int] = [x for x in range(1000)]
print(sys.getsizeof(result_sized))   # ~8056 bytes (exactly 1000 pointers)

# From a generator (length unknown):
result_unsized: list[int] = [x for x in (x for x in range(1000))]
print(sys.getsizeof(result_unsized))  # ~8856 bytes (over-allocated)

# The pre-sized version wastes no memory on over-allocation
```

---

## 6.2 Set and Dict Comprehensions

### 🟢 Beginner: Building Sets and Dicts Concisely

```python
# Set comprehension — uses { } with a single expression
numbers: list[int] = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
unique_squares: set[int] = {x ** 2 for x in numbers}
print(unique_squares)  # {1, 4, 9, 16}  — duplicates automatically removed

# Dict comprehension — uses { } with key: value
words: list[str] = ["hello", "world", "python"]
word_lengths: dict[str, int] = {word: len(word) for word in words}
print(word_lengths)  # {'hello': 5, 'world': 5, 'python': 6}
```

**Filtering in set and dict comprehensions:**

```python
# Set: unique first letters of long words
sentence: str = "the quick brown fox jumps over the lazy dog"
long_initials: set[str] = {
    word[0].upper()
    for word in sentence.split()
    if len(word) > 3
}
print(long_initials)  # {'Q', 'B', 'J', 'O', 'L'}  (unordered)

# Dict: filter and transform
scores: dict[str, int] = {
    "Alice": 95, "Bob": 67, "Charlie": 82, "Diana": 45, "Eve": 91,
}

# Only passing students, with letter grades
grade_map: dict[str, str] = {
    name: ("A" if score >= 90 else "B" if score >= 80 else "C" if score >= 70 else "F")
    for name, score in scores.items()
    if score >= 70
}
print(grade_map)  # {'Alice': 'A', 'Charlie': 'B', 'Eve': 'A'}
```

### 🟡 Intermediate: Practical Patterns

**Inverting a dictionary:**

```python
# Simple inversion (assumes values are unique)
original: dict[str, int] = {"a": 1, "b": 2, "c": 3}
inverted: dict[int, str] = {v: k for k, v in original.items()}
print(inverted)  # {1: 'a', 2: 'b', 3: 'c'}

# Safe inversion (values may not be unique — collect keys into lists)
grades: dict[str, str] = {"Alice": "A", "Bob": "B", "Charlie": "A", "Diana": "B"}
by_grade: dict[str, list[str]] = {}
for name, grade in grades.items():
    by_grade.setdefault(grade, []).append(name)
print(by_grade)  # {'A': ['Alice', 'Charlie'], 'B': ['Bob', 'Diana']}

# One-liner alternative (but readability suffers):
from collections import defaultdict
by_grade_dd: defaultdict[str, list[str]] = defaultdict(list)
for name, grade in grades.items():
    by_grade_dd[grade].append(name)
```

**Transforming nested structures:**

```python
# API response → clean lookup table
raw_users: list[dict[str, str | int]] = [
    {"id": 1, "name": "Alice", "role": "admin"},
    {"id": 2, "name": "Bob", "role": "user"},
    {"id": 3, "name": "Charlie", "role": "admin"},
]

# Dict of id → name for admins only
admin_names: dict[int, str] = {
    user["id"]: user["name"]
    for user in raw_users
    if user["role"] == "admin"
}
print(admin_names)  # {1: 'Alice', 3: 'Charlie'}
```

**Set comprehension for data validation:**

```python
# Find all unique error codes in log entries
log_entries: list[dict[str, str | int]] = [
    {"level": "ERROR", "code": 404, "msg": "Not found"},
    {"level": "WARN", "code": 301, "msg": "Redirect"},
    {"level": "ERROR", "code": 500, "msg": "Server error"},
    {"level": "ERROR", "code": 404, "msg": "Not found"},  # Duplicate
]

error_codes: set[int] = {
    entry["code"]
    for entry in log_entries
    if entry["level"] == "ERROR"
}
print(error_codes)  # {404, 500}
```

### 🔴 Expert: Dict Comprehension Ordering and Collision Behavior

```python
# When a dict comprehension encounters duplicate keys,
# the LAST value wins (like dict construction)

data: list[tuple[str, int]] = [("a", 1), ("b", 2), ("a", 3), ("b", 4)]
result: dict[str, int] = {k: v for k, v in data}
print(result)  # {'a': 3, 'b': 4}  — last wins

# This is the same behavior as:
result2: dict[str, int] = dict(data)
print(result2)  # {'a': 3, 'b': 4}

# Gotcha: ALL expressions are evaluated, even for overwritten keys
call_count: int = 0
def expensive(x: int) -> int:
    global call_count
    call_count += 1
    return x * 100

data_with_dups: list[tuple[str, int]] = [("a", 1), ("a", 2), ("a", 3)]
result = {k: expensive(v) for k, v in data_with_dups}
print(result)        # {'a': 300}
print(call_count)    # 3 — expensive() was called 3 times, not 1!
# The first two results were computed and then discarded.
```

**Performance comparison — comprehension vs `dict()`:**

```python
import timeit

pairs: list[tuple[int, int]] = [(i, i * 2) for i in range(1000)]

# Dict comprehension
t1: float = timeit.timeit(lambda: {k: v for k, v in pairs}, number=10_000)

# dict() constructor
t2: float = timeit.timeit(lambda: dict(pairs), number=10_000)

# dict() with zip
keys: list[int] = list(range(1000))
vals: list[int] = [i * 2 for i in range(1000)]
t3: float = timeit.timeit(lambda: dict(zip(keys, vals)), number=10_000)
```

```
Typical results:

    Dict comprehension:  ~1.0x (baseline)
    dict(pairs):         ~0.9x (slightly faster — optimized C path)
    dict(zip(k, v)):     ~0.7x (fastest — zip is lazy and dict() is C-optimized)

Rule of thumb:
- dict(zip(keys, values)) is fastest for two parallel sequences
- Dict comprehension is fastest when you need filtering or transformation
- dict(iterable_of_pairs) is fastest for existing pair sequences
```

---

## 6.3 Nested Comprehensions — Flattening, Filtering, and the Readability Cliff

### 🟢 Beginner: Multiple `for` Clauses

```python
# A comprehension can have multiple 'for' clauses
# They nest LEFT TO RIGHT (same order as nested loops)

# Nested loop:
pairs: list[tuple[int, str]] = []
for x in range(3):
    for y in ["a", "b"]:
        pairs.append((x, y))
# [(0, 'a'), (0, 'b'), (1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]

# Same as comprehension:
pairs = [(x, y) for x in range(3) for y in ["a", "b"]]
#        ^^^^^^   ^^^ outer loop    ^^^ inner loop

# Read it as: "for each x, for each y, produce (x, y)"
```

**Flattening a 2D list:**

```python
matrix: list[list[int]] = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

# Flatten to 1D
flat: list[int] = [num for row in matrix for num in row]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

# With filtering — only even numbers
flat_evens: list[int] = [num for row in matrix for num in row if num % 2 == 0]
# [2, 4, 6, 8]
```

### 🟡 Intermediate: Building Matrices and Complex Nesting

**Creating a matrix with comprehensions:**

```python
# 3×4 matrix of zeros
zeros: list[list[int]] = [[0 for _ in range(4)] for _ in range(3)]
# [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]

# Identity matrix
n: int = 4
identity: list[list[int]] = [
    [1 if i == j else 0 for j in range(n)]
    for i in range(n)
]
# [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

# Multiplication table
table: list[list[int]] = [
    [i * j for j in range(1, 6)]
    for i in range(1, 6)
]
# [[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], ..., [5, 10, 15, 20, 25]]
```

**Transpose a matrix:**

```python
matrix: list[list[int]] = [
    [1, 2, 3],
    [4, 5, 6],
]

# Transpose using nested comprehension
transposed: list[list[int]] = [
    [row[j] for row in matrix]
    for j in range(len(matrix[0]))
]
# [[1, 4], [2, 5], [3, 6]]

# Elegant alternative using zip
transposed_zip: list[list[int]] = [list(col) for col in zip(*matrix)]
# [[1, 4], [2, 5], [3, 6]]

# The * unpacks matrix into separate arguments:
# zip([1,2,3], [4,5,6]) → (1,4), (2,5), (3,6)
```

**Gotcha: The order of `for` and `if` clauses:**

```python
# Multiple for + if clauses are evaluated left to right
# Each 'if' applies to the most recent 'for'

# This:
result: list[tuple[int, int]] = [
    (x, y)
    for x in range(5) if x % 2 == 0    # filter on x
    for y in range(5) if y > x          # filter on y (can reference x!)
]

# Is equivalent to:
result_loop: list[tuple[int, int]] = []
for x in range(5):
    if x % 2 == 0:             # filter on x
        for y in range(5):
            if y > x:          # filter on y
                result_loop.append((x, y))

print(result)
# [(0, 1), (0, 2), (0, 3), (0, 4), (2, 3), (2, 4), (4,)]
# Wait — where's (4, ...)? range(5) stops at 4, and y > 4 gives nothing.
```

**The readability cliff — when to stop nesting:**

```python
# ✅ One level: always fine
flat: list[int] = [x ** 2 for x in range(10)]

# ✅ Two levels: usually fine for flattening
flat_2d: list[int] = [x for row in matrix for x in row]

# ⚠️ Two levels with filters: getting dense
filtered: list[int] = [
    x for row in matrix
    for x in row
    if x > 3 and x % 2 == 0
]

# ❌ Three levels: almost never worth it as a comprehension
# cubes = [cell for layer in cube for row in layer for cell in row]
# Use a loop or itertools.chain.from_iterable instead:

from itertools import chain
cube: list[list[list[int]]] = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]

# ✅ Much clearer
flat_cube: list[int] = list(chain.from_iterable(
    chain.from_iterable(layer) for layer in cube
))
# Or just:
flat_cube = [cell for layer in cube for row in layer for cell in row]
# At 3 levels, readability is debatable — team preference applies.
```

### 🔴 Expert: Nested Comprehension Bytecode

```python
import dis

def nested_comp() -> list[tuple[int, int]]:
    return [(x, y) for x in range(3) for y in range(3)]

dis.dis(nested_comp)
# The nested comprehension compiles to a single <listcomp> code object
# with nested FOR_ITER loops — same as hand-written nested loops
# but with the LIST_APPEND optimization.
```

**Performance: nested comprehension vs. `itertools.product`:**

```python
import timeit
from itertools import product

# Nested comprehension
t1: float = timeit.timeit(
    lambda: [(x, y) for x in range(100) for y in range(100)],
    number=1000,
)

# itertools.product
t2: float = timeit.timeit(
    lambda: list(product(range(100), range(100))),
    number=1000,
)

# Direct nested loop
def loop_version() -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for x in range(100):
        for y in range(100):
            result.append((x, y))
    return result

t3: float = timeit.timeit(loop_version, number=1000)
```

```
Typical results for 100×100 = 10,000 pairs:

    Nested comprehension:    ~1.0x (baseline)
    itertools.product:       ~0.9x (slightly faster — C implementation)
    Nested loop:             ~1.4x (slower — append overhead)

For pure Cartesian products without filtering,
itertools.product is slightly faster and equally readable.
For filtered products, comprehensions win on both readability and speed.
```

---

## 6.4 Generator Expressions — Lazy Evaluation and Memory Footprint

### 🟢 Beginner: Parentheses Instead of Brackets

A generator expression looks just like a list comprehension but uses `()` instead of `[]`. It produces values **lazily** — one at a time, on demand.

```python
# List comprehension — builds entire list in memory
squares_list: list[int] = [x ** 2 for x in range(1_000_000)]
# Uses ~8 MB of memory (1M pointers + 1M int objects)

# Generator expression — produces values on demand
squares_gen = (x ** 2 for x in range(1_000_000))
# Uses ~120 bytes regardless of how many values it will produce!

# You can iterate over it
for square in squares_gen:
    if square > 100:
        print(f"First square > 100: {square}")
        break
# Only computed values up to 121 — didn't build the whole list!
```

**Where generators shine — passing to aggregate functions:**

```python
# sum(), min(), max(), any(), all() consume iterators
# No need to build an intermediate list!

numbers: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# ❌ Wasteful — builds a temporary list just to sum it
total: int = sum([x ** 2 for x in numbers])

# ✅ Memory efficient — generator, no intermediate list
total = sum(x ** 2 for x in numbers)
# Note: no double parentheses needed when it's the only argument

# More examples
has_negative: bool = any(x < 0 for x in numbers)
all_positive: bool = all(x > 0 for x in numbers)
max_square: int = max(x ** 2 for x in numbers)

# Joining strings
words: list[str] = ["hello", "world"]
result: str = " ".join(word.upper() for word in words)
```

### 🟡 Intermediate: Generator Exhaustion and Chaining

**Generators are single-use — the silent bug:**

```python
# A generator can only be iterated ONCE
gen = (x ** 2 for x in range(5))

print(list(gen))   # [0, 1, 4, 9, 16]
print(list(gen))   # [] — EMPTY! Generator is exhausted!

# This bites when you try to use a generator in multiple places
def process(data):
    filtered = (x for x in data if x > 0)

    # First pass — works
    count: int = sum(1 for _ in filtered)

    # Second pass — BROKEN! filtered is exhausted
    total: int = sum(filtered)  # Always 0!

    return total / count if count else 0

# ✅ Fix: use a list if you need multiple passes
def process_fixed(data: list[int]) -> float:
    filtered: list[int] = [x for x in data if x > 0]  # Materialize once
    return sum(filtered) / len(filtered) if filtered else 0.0
```

**Chaining generators — building data pipelines:**

```python
from typing import Iterator

# Each step is lazy — nothing happens until consumption
def read_lines(filename: str) -> Iterator[str]:
    """Lazily read lines from a file."""
    with open(filename) as f:
        for line in f:
            yield line.rstrip("\n")

# Build a pipeline of generators
lines = read_lines("server.log")
non_empty = (line for line in lines if line)
errors = (line for line in non_empty if "ERROR" in line)
timestamps = (line.split()[0] for line in errors)

# NOTHING has happened yet — no file I/O, no filtering
# Work only starts when we consume:
for ts in timestamps:
    print(ts)
# Each line flows through the entire pipeline one at a time
# Memory usage: O(1) regardless of file size!
```

**Memory comparison:**

```python
import sys

n: int = 1_000_000

# List comprehension — all in memory
list_comp: list[int] = [x ** 2 for x in range(n)]
print(f"  List: {sys.getsizeof(list_comp):>10,} bytes")  # ~8,000,056 bytes

# Generator expression — lazy, tiny footprint
gen_expr = (x ** 2 for x in range(n))
print(f"  Generator: {sys.getsizeof(gen_expr):>7,} bytes")  # ~200 bytes

# The generator doesn't store any values — it stores the recipe
# to compute them. Values are produced and discarded one at a time.
```

### 🔴 Expert: Generator Expression Internals

```python
import dis

def gen_example() -> ...:
    return (x ** 2 for x in range(10))

dis.dis(gen_example)
# LOAD_CONST     <code object <genexpr>>  ← generator code object
# MAKE_FUNCTION  0
# LOAD_GLOBAL    range
# LOAD_CONST     10
# CALL_FUNCTION  1
# GET_ITER
# CALL_FUNCTION  1                        ← creates generator object
# RETURN_VALUE

# The generator code object uses YIELD_VALUE instead of LIST_APPEND:
# Each iteration:
#   1. Compute x ** 2
#   2. YIELD_VALUE — suspend execution, return value to caller
#   3. When next() is called, resume from where we left off
```

**Generator expression vs. `yield` generator function:**

```python
# These are semantically equivalent:

# Generator expression
gen1 = (x ** 2 for x in range(10) if x % 2 == 0)

# Generator function
def gen2_func():
    for x in range(10):
        if x % 2 == 0:
            yield x ** 2

gen2 = gen2_func()

# Both produce the same values lazily
print(list(gen1))  # [0, 4, 16, 36, 64]
print(list(gen2))  # [0, 4, 16, 36, 64]

# Differences:
# - Generator expressions: concise, single expression
# - Generator functions: can have multiple yields, try/except,
#   complex logic, and send()/throw() support

# Use gen expressions for simple transforms/filters
# Use gen functions for complex stateful iteration
```

**When `list()` beats generators — the `len()` optimization:**

```python
import timeit

# When the consumer needs len() or random access,
# generators force materialization anyway

data: range = range(10_000)

# If you need len():
# ❌ Must consume entire generator to count
gen = (x for x in data if x % 2 == 0)
count: int = sum(1 for _ in gen)  # O(n) — iterates everything

# ✅ Just use a list
lst: list[int] = [x for x in data if x % 2 == 0]
count = len(lst)  # O(1) — stored attribute

# If you need to iterate multiple times: list
# If you need len(): list
# If you need indexing: list
# If you need single-pass and memory matters: generator
```

---

## 6.5 Comprehension Scoping (The Variable Leak That Was Fixed in Python 3)

### 🟢 Beginner: Comprehensions Have Their Own Scope

```python
# In Python 3, comprehension variables do NOT leak into the enclosing scope
x: int = 10
squares: list[int] = [x ** 2 for x in range(5)]
print(x)  # 10 — unchanged! The comprehension's 'x' was local to it

# In Python 2, the loop variable DID leak:
# (Python 2)
# x = 10
# squares = [x ** 2 for x in range(5)]
# print(x)  # 4 — LEAKED from the comprehension!
```

### 🟡 Intermediate: The Walrus Exception and Closure Gotchas

**The walrus operator `:=` DOES leak (by design):**

```python
# PEP 572: walrus operator targets the ENCLOSING scope
result: list[int] = [y := x ** 2 for x in range(5)]
print(result)  # [0, 1, 4, 9, 16]
print(y)       # 16 — y leaked! This is intentional.

# This is useful for "find and capture" patterns:
data: list[str] = ["hello", "world", "python", "programming"]

# Find the first word longer than 5 characters AND capture it
long_words: list[str] = [
    word for word in data
    if (matched := len(word)) > 5
]
print(long_words)  # ['python', 'programming']
print(matched)     # 11 — last value of matched (len("programming"))
```

**Closure gotcha with comprehensions:**

```python
# Classic late-binding closure problem
funcs: list = [lambda: x for x in range(5)]
print([f() for f in funcs])  # [4, 4, 4, 4, 4]  — all return 4!

# WHY: Each lambda captures the VARIABLE x, not its VALUE.
# By the time we call the lambdas, x = 4 (the last value).

# FIX 1: Default argument (captures value at creation time)
funcs = [lambda x=x: x for x in range(5)]
print([f() for f in funcs])  # [0, 1, 2, 3, 4]

# FIX 2: functools.partial
from functools import partial
def identity(n: int) -> int:
    return n
funcs = [partial(identity, x) for x in range(5)]
print([f() for f in funcs])  # [0, 1, 2, 3, 4]
```

**Comprehension scope with nested functions:**

```python
# Comprehensions create an implicit function scope
# This means they can access enclosing scope variables

multiplier: int = 10
result: list[int] = [x * multiplier for x in range(5)]
print(result)  # [0, 10, 20, 30, 40]

# And they participate in the closure chain:
def make_multipliers(n: int) -> list:
    return [lambda x: x * i for i in range(n)]
    # Same late-binding problem as above!

# FIX:
def make_multipliers_fixed(n: int) -> list:
    return [lambda x, i=i: x * i for i in range(n)]
```

### 🔴 Expert: How CPython Implements Comprehension Scoping

```python
import dis
import types

def outer() -> list[int]:
    x: int = 10
    result: list[int] = [i ** 2 for i in range(5)]
    return result

# The comprehension is compiled as a separate code object
code: types.CodeType = outer.__code__
for const in code.co_consts:
    if isinstance(const, types.CodeType) and "listcomp" in const.co_name:
        print(f"  Comprehension code object: {const.co_name}")
        print(f"  Local variables: {const.co_varnames}")
        # co_varnames includes '.0' (the implicit iterator argument)
        # and any loop variables (i)
        # But NOT x (which is in the enclosing scope)
        dis.dis(const)
        break
```

```
The comprehension compiles to a function that:
1. Takes one argument: the iterator (.0)
2. Has its own local namespace (loop variables)
3. Can access enclosing scope via LOAD_DEREF (closure)
4. Returns the built collection

This is why:
- Loop variables don't leak (they're locals of the inner function)
- Walrus := targets the enclosing scope (uses STORE_DEREF)
- Enclosing variables are accessible (via closure cells)
```

**The implicit `.0` argument:**

```python
# When CPython compiles [expr for x in iterable],
# it creates a function approximately like:
#
# def <listcomp>(.0):   ← .0 is iter(iterable), passed as argument
#     result = []
#     for x in .0:
#         result.append(expr)
#     return result
#
# And then calls it: <listcomp>(iter(iterable))
#
# This is why the iterable is evaluated in the ENCLOSING scope
# (it's passed as an argument), but the loop variable is local.

# Proof:
gen = (x for x in range(5))
print(gen.gi_code.co_varnames)  # ('.0', 'x')
# .0 is the iterator, x is the loop variable
```

---

## 🔧 Debug This: The Slow Data Pipeline

Your data science colleague wrote a pipeline to process sensor readings. It works but runs out of memory on large datasets and has several correctness bugs. Find all the issues:

```python
import statistics

def analyze_sensors(raw_readings):
    """Analyze sensor readings and return summary statistics."""

    # Step 1: Parse raw strings into floats
    parsed = [float(r) for r in raw_readings if r.strip()]

    # Step 2: Remove outliers (values beyond 3 standard deviations)
    mean = statistics.mean(parsed)
    stdev = statistics.stdev(parsed)
    cleaned = [x for x in parsed if abs(x - mean) <= 3 * stdev]

    # Step 3: Compute rolling averages (window of 5)
    rolling = [
        statistics.mean(cleaned[i:i+5])
        for i in range(len(cleaned))
    ]

    # Step 4: Find anomalies (where rolling avg differs from overall mean by >20%)
    overall_mean = statistics.mean(cleaned)
    anomalies = [
        (i, rolling[i])
        for i in range(len(rolling))
        if abs(rolling[i] - overall_mean) / overall_mean > 0.2
    ]

    # Step 5: Return unique anomaly timestamps
    unique_timestamps = list(set(a[0] for a in anomalies))

    return {
        "count": len(cleaned),
        "mean": overall_mean,
        "anomalies": unique_timestamps,
        "rolling_averages": rolling,
    }

# Test with sample data
readings = [str(x) for x in range(1, 101)] + ["", "  ", "bad_value"]
result = analyze_sensors(readings)
print(f"Count: {result['count']}, Anomalies: {len(result['anomalies'])}")
```

### Bugs to find:

```
1. ____________________________________________________
   Hint: What happens when raw_readings contains "bad_value"?
   The comprehension in Step 1 will crash. No error handling.

2. ____________________________________________________
   Hint: In Step 3, cleaned[i:i+5] when i is near the end gives
   windows smaller than 5. cleaned[98:103] is only 2 elements.
   Is that a valid "window of 5" average?

3. ____________________________________________________
   Hint: Step 4 divides by overall_mean. What if overall_mean is 0?
   ZeroDivisionError!

4. ____________________________________________________
   Hint: Step 5 uses set() on timestamps (integers). This removes
   duplicates but LOSES ORDER. Anomalies should be time-ordered.

5. ____________________________________________________
   Hint: The ENTIRE dataset is stored as a list at every step.
   parsed, cleaned, rolling, anomalies — four full copies in memory.
   With 10M readings, this uses ~320MB. Can generators help?

6. ____________________________________________________
   Hint: Type hints are missing throughout.
```

### Solution (try first!)

```python
import statistics
from collections.abc import Iterator
from typing import Any


def safe_float(value: str) -> float | None:
    """Convert string to float, returning None on failure."""
    try:
        stripped: str = value.strip()
        if not stripped:
            return None
        return float(stripped)
    except ValueError:
        return None


def rolling_mean(data: list[float], window: int) -> list[float]:
    """Compute rolling average with FULL windows only.

    Only produces averages where the full window fits.
    Returns (len(data) - window + 1) values.
    """
    if len(data) < window:
        return []
    # Efficient: use running sum instead of recomputing each window
    current_sum: float = sum(data[:window])
    result: list[float] = [current_sum / window]
    for i in range(window, len(data)):
        current_sum += data[i] - data[i - window]
        result.append(current_sum / window)
    return result


def analyze_sensors(
    raw_readings: list[str],
    outlier_sigma: float = 3.0,
    rolling_window: int = 5,
    anomaly_threshold: float = 0.2,
) -> dict[str, Any]:
    """Analyze sensor readings and return summary statistics.

    Args:
        raw_readings: List of string values to parse as floats.
        outlier_sigma: Standard deviations for outlier removal.
        rolling_window: Size of the rolling average window.
        anomaly_threshold: Fractional deviation from mean for anomaly detection.

    Returns:
        Dict with count, mean, anomalies, and rolling_averages.
    """
    # Bug 1 FIX: Handle parse errors gracefully
    parsed: list[float] = [
        val
        for r in raw_readings
        if (val := safe_float(r)) is not None
    ]

    if len(parsed) < 2:
        return {"count": 0, "mean": 0.0, "anomalies": [], "rolling_averages": []}

    # Step 2: Remove outliers
    mean: float = statistics.mean(parsed)
    stdev: float = statistics.stdev(parsed)
    if stdev == 0:
        cleaned = parsed  # All values identical — nothing to remove
    else:
        cleaned = [x for x in parsed if abs(x - mean) <= outlier_sigma * stdev]

    if not cleaned:
        return {"count": 0, "mean": 0.0, "anomalies": [], "rolling_averages": []}

    # Bug 2 FIX: Use full-window rolling averages (efficient running sum)
    rolling: list[float] = rolling_mean(cleaned, rolling_window)

    # Bug 3 FIX: Handle zero mean
    overall_mean: float = statistics.mean(cleaned)
    if overall_mean == 0:
        # Use absolute threshold when mean is zero
        anomalies: list[tuple[int, float]] = [
            (i, rolling[i])
            for i in range(len(rolling))
            if abs(rolling[i]) > anomaly_threshold
        ]
    else:
        anomalies = [
            (i, rolling[i])
            for i in range(len(rolling))
            if abs(rolling[i] - overall_mean) / abs(overall_mean) > anomaly_threshold
        ]

    # Bug 4 FIX: Preserve order (timestamps are already unique indices)
    unique_timestamps: list[int] = list(dict.fromkeys(a[0] for a in anomalies))

    return {
        "count": len(cleaned),
        "mean": overall_mean,
        "anomalies": unique_timestamps,
        "rolling_averages": rolling,
    }


# Test
readings: list[str] = [str(x) for x in range(1, 101)] + ["", "  ", "bad_value"]
result: dict[str, Any] = analyze_sensors(readings)
print(f"Count: {result['count']}, Anomalies: {len(result['anomalies'])}")
```

```
Bug Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. No error handling: float("bad_value") raises ValueError.
   Fix: safe_float() with try/except, or walrus + is not None.

2. Partial windows:   cleaned[98:103] when cleaned has 100 elements
   returns only 2 values. The "rolling average" of 2 values
   isn't comparable to one of 5. Fix: only produce full windows.
   Bonus: the naive approach recomputes sum each window (O(nk));
   running sum is O(n).

3. Zero division:     If all sensor readings are 0 (or sum to 0),
   dividing by overall_mean crashes. Fix: check for zero.

4. Lost ordering:     set() destroys timestamp order. Anomalies
   at timestamps [50, 10, 30] become {10, 30, 50} or any order.
   Fix: dict.fromkeys() preserves insertion order.

5. Memory bloat:      Four full-size lists in memory simultaneously.
   For streaming data, parsed → cleaned could be lazy. But rolling
   averages need random access, so full laziness isn't possible.
   Partial fix: process parsed → cleaned as generator, then
   materialize only for rolling computation.

6. Missing types:     No type hints, no docstring, no parameter docs.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Summary: Module 6 Key Takeaways

```
┌──────────────────────────────────────────────────────────────────┐
│                    COMPREHENSIONS CHEAT SHEET                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  SYNTAX:                                                         │
│    List:   [expr for x in iter if cond]                          │
│    Set:    {expr for x in iter if cond}                          │
│    Dict:   {key: val for x in iter if cond}                      │
│    Gen:    (expr for x in iter if cond)                          │
│                                                                   │
│  POSITION MATTERS:                                               │
│    [expr if/else  for x in iter]  ← ternary (transforms all)    │
│    [expr          for x in iter if cond]  ← filter (excludes)   │
│                                                                   │
│  NESTING: for clauses go LEFT TO RIGHT (outer to inner)          │
│    [expr for x in outer for y in inner]                          │
│    = for x in outer: for y in inner: append(expr)                │
│                                                                   │
│  PERFORMANCE:                                                    │
│    Comprehensions: ~35% faster than equivalent loops             │
│    (LIST_APPEND bytecode, no .append() lookup, pre-sizing)       │
│    Generators: O(1) memory regardless of input size              │
│                                                                   │
│  SCOPING (Python 3):                                             │
│    Loop variables don't leak into enclosing scope                │
│    Walrus := DOES leak (by design, PEP 572)                     │
│    Late-binding closures: use default args to capture values     │
│                                                                   │
│  READABILITY RULES:                                              │
│    1 for clause: always fine                                     │
│    2 for clauses: fine for flattening                            │
│    3+ for clauses: use a loop or itertools                       │
│    Side effects in comprehensions: NEVER                         │
│                                                                   │
│  GENERATORS vs LISTS:                                            │
│    Need len/index/multiple passes? → list comprehension          │
│    Single pass + memory matters? → generator expression          │
│    Passing to sum/any/all/join? → generator (no [brackets])     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Next up → Module 7: Functions & Namespaces — The LEGB Universe**

Say "Start Module 7" when you're ready.
