# Module 5 — The "Big Four" Containers: Lists, Tuples, Sets, Dicts

> *"Bad programmers worry about the code. Good programmers worry about data structures and their relationships."*
> — Linus Torvalds

---

## 5.1 Lists — Dynamic Arrays, Amortized Appends, and Over-Allocation

### 🟢 Beginner: Your Workhorse Collection

A list is an **ordered, mutable** sequence that can hold any mix of types (though in practice, keep them homogeneous).

```python
# Creating lists
empty: list = []
numbers: list[int] = [1, 2, 3, 4, 5]
mixed: list = [1, "hello", 3.14, True, None]  # Legal but discouraged
from_range: list[int] = list(range(10))        # [0, 1, 2, ..., 9]
from_string: list[str] = list("hello")         # ['h', 'e', 'l', 'l', 'o']

# Accessing elements
print(numbers[0])     # 1     (first)
print(numbers[-1])    # 5     (last)
print(numbers[1:4])   # [2, 3, 4]  (slice — new list)

# Modifying elements (lists are mutable!)
numbers[0] = 99
print(numbers)        # [99, 2, 3, 4, 5]

numbers[1:3] = [20, 30, 40]  # Replace a slice (can change length!)
print(numbers)        # [99, 20, 30, 40, 4, 5]
```

**Essential list methods:**

```python
items: list[int] = [3, 1, 4, 1, 5, 9, 2, 6]

# Adding elements
items.append(7)            # Add to end: [3, 1, 4, 1, 5, 9, 2, 6, 7]
items.insert(0, 0)         # Insert at index: [0, 3, 1, 4, 1, 5, 9, 2, 6, 7]
items.extend([8, 10])      # Add multiple: [..., 8, 10]
# items += [8, 10]         # Same as extend

# Removing elements
items.pop()                # Remove and return last: 10
items.pop(0)               # Remove and return at index: 0
items.remove(1)            # Remove first occurrence of value 1
# del items[2]             # Remove by index (no return value)

# Searching
print(items.index(5))      # Index of first occurrence (ValueError if missing)
print(items.count(1))      # How many times 1 appears
print(5 in items)          # True — membership test (O(n))

# Sorting (in-place — modifies the list, returns None!)
items.sort()               # Ascending
items.sort(reverse=True)   # Descending
items.sort(key=abs)        # Sort by absolute value

# sorted() creates a NEW list (original unchanged)
original: list[int] = [3, 1, 4]
new_sorted: list[int] = sorted(original)
print(original)            # [3, 1, 4] — unchanged
print(new_sorted)          # [1, 3, 4]

# Reversing
items.reverse()                    # In-place
reversed_copy: list[int] = items[::-1]  # New reversed list
```

### 🟡 Intermediate: Performance Characteristics and Gotchas

**Time complexity — what's fast and what's slow:**

```
┌──────────────────────────────────────────────────────────┐
│              LIST TIME COMPLEXITY                          │
├──────────────────────────────────┬────────────────────────┤
│  Operation                       │  Time Complexity       │
├──────────────────────────────────┼────────────────────────┤
│  list[i]                         │  O(1)  — direct index  │
│  list[i] = x                    │  O(1)                  │
│  list.append(x)                 │  O(1) amortized        │
│  list.pop()      (last)         │  O(1)                  │
│  list.pop(0)     (first)        │  O(n)  — shifts all!   │
│  list.insert(0, x)              │  O(n)  — shifts all!   │
│  list.insert(len, x)            │  O(1)  — same as append│
│  x in list                      │  O(n)  — linear scan   │
│  list.remove(x)                 │  O(n)  — find + shift  │
│  list.sort()                    │  O(n log n) — Timsort  │
│  list + list                    │  O(n+m) — new list     │
│  list.extend(iterable)          │  O(k) — k = len(iter)  │
│  list[a:b]                      │  O(b-a) — copy slice   │
│  len(list)                      │  O(1)  — stored field  │
│  del list[i]                    │  O(n)  — shifts after i│
└──────────────────────────────────┴────────────────────────┘
```

**Gotcha #1: `list.pop(0)` is O(n) — use `collections.deque` instead:**

```python
from collections import deque

# ❌ Using a list as a queue — O(n) dequeue
queue_bad: list[int] = [1, 2, 3, 4, 5]
queue_bad.pop(0)    # Shifts ALL remaining elements left

# ✅ Using deque — O(1) dequeue from both ends
queue_good: deque[int] = deque([1, 2, 3, 4, 5])
queue_good.popleft()   # O(1) — no shifting
queue_good.appendleft(0)  # O(1) — prepend
```

**Gotcha #2: `list * n` creates shallow copies of references:**

```python
# This looks like it creates a 3×3 grid...
grid: list[list[int]] = [[0] * 3] * 3
print(grid)  # [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  — looks right

grid[0][0] = 99
print(grid)  # [[99, 0, 0], [99, 0, 0], [99, 0, 0]]  — ALL rows changed!
# Because * 3 copies the REFERENCE to the same inner list

# ✅ Fix: use a comprehension to create independent lists
grid = [[0] * 3 for _ in range(3)]
grid[0][0] = 99
print(grid)  # [[99, 0, 0], [0, 0, 0], [0, 0, 0]]  — only first row changed
```

**Gotcha #3: `sort()` returns `None`, not the sorted list:**

```python
numbers: list[int] = [3, 1, 4]

# ❌ Common mistake — assigning the return value of .sort()
result = numbers.sort()
print(result)    # None!  The list is sorted in-place.
print(numbers)   # [1, 3, 4]  — sorted, but 'result' is None

# ✅ Use sorted() if you need the return value
result = sorted([3, 1, 4])  # Returns a new sorted list
```

**Custom sorting with `key=`:**

```python
# Sort strings by length
words: list[str] = ["banana", "pie", "apple", "kiwi"]
words.sort(key=len)
print(words)  # ['pie', 'kiwi', 'apple', 'banana']

# Sort objects by attribute
from dataclasses import dataclass

@dataclass
class Student:
    name: str
    gpa: float

students: list[Student] = [
    Student("Alice", 3.9),
    Student("Bob", 3.5),
    Student("Charlie", 3.9),
    Student("Diana", 3.7),
]

# Sort by GPA descending, then name ascending (stable sort!)
students.sort(key=lambda s: (-s.gpa, s.name))
for s in students:
    print(f"  {s.name}: {s.gpa}")
# Alice: 3.9       ← same GPA, alphabetical
# Charlie: 3.9
# Diana: 3.7
# Bob: 3.5

# For complex keys, use operator.attrgetter (faster than lambda)
from operator import attrgetter
students.sort(key=attrgetter("gpa"), reverse=True)
```

### 🔴 Expert: CPython's Dynamic Array Implementation

A Python list is a **dynamic array** — a contiguous block of pointers to PyObjects on the heap.

```
Memory layout of list [10, 20, 30]:

Stack                         Heap
┌─────────┐          ┌──────────────────────────────────┐
│  my_list ├─────────▶│       PyListObject                │
└─────────┘          │  ob_refcnt:  1                    │
                     │  ob_type:    → list                │
                     │  ob_size:    3   (current length)  │
                     │  allocated:  4   (capacity!)       │
                     │  ob_item: ──────────┐              │
                     └─────────────────────┼──────────────┘
                                           ▼
                     ┌─────────────────────────────────────┐
                     │ Pointer Array (ob_item)              │
                     │  [0] ──▶ PyLongObject(10)           │
                     │  [1] ──▶ PyLongObject(20)           │
                     │  [2] ──▶ PyLongObject(30)           │
                     │  [3]     (unused — overallocated)   │
                     └─────────────────────────────────────┘

Key insight: The list stores POINTERS to objects, not the objects
themselves. This is why lists can hold mixed types — every slot
is the same size (one pointer = 8 bytes on 64-bit).
```

**The over-allocation strategy — why `append()` is O(1) amortized:**

```python
import sys

# Watch the memory grow as we append
sizes: list[tuple[int, int]] = []
items: list[int] = []
for i in range(100):
    items.append(i)
    sizes.append((len(items), sys.getsizeof(items)))

# Print the growth pattern
prev_size: int = 0
for length, byte_size in sizes:
    if byte_size != prev_size:
        print(f"  Length: {length:3d}  →  Size: {byte_size:5d} bytes  "
              f"(capacity ≈ {(byte_size - 56) // 8})")
        prev_size = byte_size
```

```
Typical output on 64-bit CPython 3.12:

    Length:   1  →  Size:    88 bytes  (capacity ≈ 4)
    Length:   5  →  Size:   120 bytes  (capacity ≈ 8)
    Length:   9  →  Size:   184 bytes  (capacity ≈ 16)
    Length:  17  →  Size:   248 bytes  (capacity ≈ 24)
    Length:  25  →  Size:   312 bytes  (capacity ≈ 32)
    Length:  33  →  Size:   408 bytes  (capacity ≈ 44)
    ...

The growth factor is roughly 1.125x (12.5% over-allocation).
Formula in CPython (Objects/listobject.c):
    new_allocated = (newsize >> 3) + (newsize < 9 ? 3 : 6) + newsize
```

**Why this matters — `append` vs `insert(0)` at scale:**

```python
import time

def benchmark_append(n: int) -> float:
    """Append n items — O(n) total (O(1) amortized per append)."""
    start: float = time.perf_counter()
    items: list[int] = []
    for i in range(n):
        items.append(i)
    return time.perf_counter() - start

def benchmark_insert_front(n: int) -> float:
    """Insert n items at front — O(n²) total (O(n) per insert)."""
    start: float = time.perf_counter()
    items: list[int] = []
    for i in range(n):
        items.insert(0, i)
    return time.perf_counter() - start

# n = 100,000:  append ~0.01s,  insert(0) ~3.0s
# The 300x difference is the cost of O(n) shifting on every insert.
```

**Timsort — CPython's sorting algorithm:**

```python
# Python's list.sort() and sorted() use Timsort (Tim Peters, 2002)
# Key properties:
# - Stable sort (equal elements preserve their original order)
# - O(n log n) worst case
# - O(n) on already-sorted or nearly-sorted data (detects "runs")
# - Hybrid of merge sort and insertion sort
# - Uses at most n/2 extra memory for merging

# Timsort exploits real-world data patterns:
import random
import time

# Random data
random_data: list[int] = random.sample(range(1_000_000), 1_000_000)
start: float = time.perf_counter()
sorted(random_data)
random_time: float = time.perf_counter() - start

# Nearly sorted data (10% of elements displaced)
nearly_sorted: list[int] = list(range(1_000_000))
indices: list[int] = random.sample(range(1_000_000), 100_000)
for i in range(0, len(indices) - 1, 2):
    nearly_sorted[indices[i]], nearly_sorted[indices[i+1]] = \
        nearly_sorted[indices[i+1]], nearly_sorted[indices[i]]

start = time.perf_counter()
sorted(nearly_sorted)
nearly_time: float = time.perf_counter() - start

# nearly_sorted is significantly faster because Timsort
# detects and merges pre-existing sorted "runs"
```

---

## 5.2 Tuples — Immutability, Hashability, and Named Tuples

### 🟢 Beginner: The Immutable Sequence

A tuple is like a list, but **immutable** — once created, it cannot be changed.

```python
# Creating tuples
point: tuple[int, int] = (3, 4)
singleton: tuple[str] = ("hello",)   # Note the trailing comma!
not_a_tuple: str = ("hello")          # This is just a string in parentheses
empty: tuple = ()

# Parentheses are optional (the comma makes the tuple)
coordinates: tuple[float, float, float] = 1.0, 2.0, 3.0
print(type(coordinates))  # <class 'tuple'>

# Accessing elements (same as lists)
print(point[0])      # 3
print(point[-1])     # 4
print(point[0:1])    # (3,) — slice returns a tuple

# Immutability
# point[0] = 99      # TypeError: 'tuple' object does not support item assignment

# Tuple unpacking — one of Python's best features
x, y = point
print(f"x={x}, y={y}")  # x=3, y=4

# Extended unpacking with *
first, *rest = (1, 2, 3, 4, 5)
print(first)  # 1
print(rest)   # [2, 3, 4, 5] — note: rest is a LIST

# Swap without a temp variable (uses tuple packing/unpacking)
a, b = 1, 2
a, b = b, a      # Tuple unpacking magic
print(a, b)       # 2, 1
```

**When to use tuples vs. lists:**

```python
# Tuples: fixed structure, heterogeneous data, dict keys, return values
rgb: tuple[int, int, int] = (255, 128, 0)
person: tuple[str, int] = ("Alice", 30)

def get_min_max(data: list[int]) -> tuple[int, int]:
    return min(data), max(data)

lo, hi = get_min_max([3, 1, 4, 1, 5])

# Lists: variable length, homogeneous data, will be modified
temperatures: list[float] = [72.1, 68.4, 75.3]
temperatures.append(71.2)
```

### 🟡 Intermediate: Hashability, Named Tuples, and the Mutability Trap

**Tuples are hashable (if all their elements are hashable):**

```python
# Tuples can be dictionary keys and set members
location: tuple[float, float] = (40.7128, -74.0060)
city_map: dict[tuple[float, float], str] = {
    (40.7128, -74.0060): "New York",
    (51.5074, -0.1278): "London",
}

# Lists CANNOT be dictionary keys
# {[1, 2]: "value"}  → TypeError: unhashable type: 'list'

# But a tuple containing a list is NOT hashable either!
# mixed = ([1, 2], 3)
# hash(mixed)  → TypeError: unhashable type: 'list'
```

**The mutability trap — tuples containing mutable objects:**

```python
# A tuple is immutable, but its contents might not be!
sneaky: tuple[list[int], list[int]] = ([1, 2], [3, 4])

# Can't replace the lists themselves
# sneaky[0] = [99]  → TypeError

# But CAN mutate the lists in-place!
sneaky[0].append(99)
print(sneaky)  # ([1, 2, 99], [3, 4])

# The += operator is especially confusing here
t: tuple = ([1, 2],)
# t[0] += [3]  →  This BOTH succeeds AND raises TypeError!
# 1. [1, 2].__iadd__([3]) mutates the list to [1, 2, 3]  ← succeeds
# 2. t[0] = <result> tries to assign back to the tuple     ← TypeError!
# The list IS modified despite the error!
```

**Named tuples — tuples with field names:**

```python
from typing import NamedTuple

# Modern approach: class-based NamedTuple
class Point(NamedTuple):
    x: float
    y: float
    z: float = 0.0   # Default value

p = Point(1.0, 2.0)
print(p.x)          # 1.0 — access by name
print(p[0])          # 1.0 — still works by index
print(p)             # Point(x=1.0, y=2.0, z=0.0)

# Unpacking still works
x, y, z = p

# Immutable like regular tuples
# p.x = 99  → AttributeError

# _replace creates a new instance with modified fields
p2: Point = p._replace(x=99.0)
print(p2)  # Point(x=99.0, y=2.0, z=0.0)

# Convert to dict
print(p._asdict())   # {'x': 1.0, 'y': 2.0, 'z': 0.0}
```

```python
# collections.namedtuple — older functional approach
from collections import namedtuple

Color = namedtuple("Color", ["red", "green", "blue"])
white = Color(255, 255, 255)
print(white.red)  # 255

# Use NamedTuple (typing) for new code — better IDE support and type hints
```

### 🔴 Expert: CPython Tuple Internals and the Free List

```
Memory layout of tuple (10, 20, 30):

┌───────────────────────────────────────┐
│          PyTupleObject                 │
├───────────────────────────────────────┤
│  ob_refcnt    (8 bytes)               │
│  ob_type      (8 bytes)  → tuple type │
│  ob_size      (8 bytes)  = 3          │
│  ob_item[0]   (8 bytes)  → int(10)    │
│  ob_item[1]   (8 bytes)  → int(20)    │
│  ob_item[2]   (8 bytes)  → int(30)    │
└───────────────────────────────────────┘

Unlike lists, tuples store the pointer array INLINE
(no separate allocation). This means:
- One fewer memory allocation per tuple
- Better cache locality
- Slightly smaller memory footprint
```

```python
import sys

# Tuple vs. List memory comparison
lst: list[int] = [1, 2, 3]
tup: tuple[int, ...] = (1, 2, 3)

print(f"  list: {sys.getsizeof(lst)} bytes")  # ~88 bytes (header + ptr array + slack)
print(f"  tuple: {sys.getsizeof(tup)} bytes") # ~64 bytes (header + inline ptrs)
# Tuple is ~25% smaller for the same data
```

**The tuple free list — CPython's recycling optimization:**

```python
# CPython maintains a "free list" of recently freed tuples
# (up to length 20, with up to 2000 per length).
# When you create a small tuple, CPython first checks the free list
# before allocating new memory.

import timeit

# Creating tuples is faster than creating lists partly because of this
tuple_time: float = timeit.timeit("(1, 2, 3)", number=10_000_000)
list_time: float = timeit.timeit("[1, 2, 3]", number=10_000_000)

# Tuples are also constant-folded by the compiler:
# a = (1, 2, 3)  → single LOAD_CONST bytecode (built at compile time)
# b = [1, 2, 3]  → BUILD_LIST bytecode (built at runtime)
```

**Empty tuple singleton:**

```python
# There is exactly ONE empty tuple object in all of CPython
a: tuple = ()
b: tuple = ()
c: tuple = tuple()

print(a is b)   # True — same object
print(a is c)   # True — all empty tuples are the same object

# This is safe because tuples are immutable — sharing is harmless
```

---

## 5.3 Sets — Hash Tables Without Values

### 🟢 Beginner: Unique, Unordered Collections

A set is an **unordered** collection of **unique, hashable** elements.

```python
# Creating sets
fruits: set[str] = {"apple", "banana", "cherry"}
from_list: set[int] = set([1, 2, 2, 3, 3, 3])  # {1, 2, 3} — duplicates removed
empty_set: set = set()   # NOT {} — that's an empty dict!

# Membership testing — O(1) average (the killer feature)
print("apple" in fruits)     # True  — extremely fast
print("mango" in fruits)     # False

# Length
print(len(fruits))           # 3

# Iterating (order is NOT guaranteed)
for fruit in fruits:
    print(fruit)

# Adding and removing
fruits.add("mango")           # Add one element
fruits.update(["kiwi", "pear"])  # Add multiple
fruits.discard("banana")      # Remove (no error if missing)
fruits.remove("cherry")       # Remove (KeyError if missing!)
popped: str = fruits.pop()    # Remove and return arbitrary element
```

**Set operations — the math you forgot you needed:**

```python
a: set[int] = {1, 2, 3, 4, 5}
b: set[int] = {4, 5, 6, 7, 8}

# Union — all elements from both sets
print(a | b)         # {1, 2, 3, 4, 5, 6, 7, 8}
print(a.union(b))    # Same

# Intersection — elements in BOTH sets
print(a & b)         # {4, 5}
print(a.intersection(b))

# Difference — elements in a but NOT in b
print(a - b)         # {1, 2, 3}
print(a.difference(b))

# Symmetric difference — elements in EITHER but not both
print(a ^ b)         # {1, 2, 3, 6, 7, 8}
print(a.symmetric_difference(b))

# Subset and superset
print({1, 2} <= {1, 2, 3})   # True — subset
print({1, 2, 3} >= {1, 2})   # True — superset
print({1, 2} < {1, 2, 3})    # True — proper subset
```

### 🟡 Intermediate: Practical Patterns and frozenset

**De-duplication:**

```python
# Remove duplicates while preserving order (Python 3.7+)
items: list[int] = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]

# ❌ set() removes duplicates but loses order
unique_unordered: set[int] = set(items)

# ✅ dict.fromkeys() preserves insertion order (Python 3.7+)
unique_ordered: list[int] = list(dict.fromkeys(items))
print(unique_ordered)  # [3, 1, 4, 5, 9, 2, 6]
```

**Finding commonalities and differences:**

```python
# Real-world: find users with overlapping permissions
admins: set[str] = {"alice", "bob", "charlie"}
editors: set[str] = {"bob", "diana", "eve"}
viewers: set[str] = {"charlie", "eve", "frank"}

# Users who are both admin and editor
admin_editors: set[str] = admins & editors
print(admin_editors)  # {'bob'}

# Users with any role
all_users: set[str] = admins | editors | viewers
print(all_users)

# Users who are ONLY viewers (not admin or editor)
only_viewers: set[str] = viewers - admins - editors
print(only_viewers)  # {'frank'}
```

**`frozenset` — the immutable, hashable set:**

```python
# Regular sets are mutable and NOT hashable
# You can't use a set as a dict key or put a set inside another set
# s = {frozenset({1, 2}), frozenset({3, 4})}  ← this works
# s = {{1, 2}, {3, 4}}  ← TypeError: unhashable type: 'set'

# frozenset is immutable and hashable
fs: frozenset[int] = frozenset([1, 2, 3])
# fs.add(4)  → AttributeError — no mutating methods

# Use case: set of sets
graph: dict[frozenset[str], float] = {
    frozenset({"A", "B"}): 5.0,    # Edge A-B with weight 5
    frozenset({"B", "C"}): 3.0,    # Edge B-C with weight 3
}
# frozenset({"A", "B"}) == frozenset({"B", "A"}) — order doesn't matter!
print(graph[frozenset({"B", "A"})])  # 5.0
```

### 🔴 Expert: CPython's Hash Table Implementation for Sets

CPython sets use a **hash table with open addressing** (probing), almost identical to dicts but without storing values.

```
Hash table for set {10, 20, 30} (simplified):

Table size: 8 (always a power of 2)
Load factor: 3/8 = 0.375 (resize at 2/3 ≈ 0.667)

Index │ Hash          │ Pointer       │ Status
──────┼───────────────┼───────────────┼─────────
  0   │ —             │ NULL          │ empty
  1   │ —             │ NULL          │ empty
  2   │ hash(10)=10   │ → int(10)     │ occupied
  3   │ —             │ NULL          │ empty
  4   │ hash(20)=20   │ → int(20)     │ occupied
  5   │ —             │ NULL          │ empty
  6   │ hash(30)=30   │ → int(30)     │ occupied
  7   │ —             │ NULL          │ empty

Slot index = hash(key) % table_size
For integers: hash(n) = n (for small n)
So: hash(10) % 8 = 2, hash(20) % 8 = 4, hash(30) % 8 = 6
```

**Collision resolution — linear probing:**

```python
# When two keys hash to the same slot, CPython uses a perturbation-based
# probing sequence (not simple linear probing):
#
# j = (5 * j + 1 + perturb) % table_size
# perturb >>= 5  (perturb starts as the full hash value)
#
# This distributes collisions better than linear probing.

# Demonstration of collision
a: int = 1
b: int = 9  # hash(1) % 8 = 1, hash(9) % 8 = 1  → collision!

s: set[int] = {a, b}  # 'b' will probe to the next available slot
```

**Set operation algorithms:**

```python
# Union: O(len(a) + len(b))
# - Copy the larger set
# - Insert each element from the smaller set

# Intersection: O(min(len(a), len(b)))
# - Iterate the SMALLER set
# - Check membership in the LARGER set (O(1) per check)

# This is why a & b can be much faster than you'd expect
# when one set is much smaller than the other

big: set[int] = set(range(1_000_000))
small: set[int] = {42, 999_999}

# This is O(2), not O(1_000_000)!
result: set[int] = small & big  # Iterates small, checks in big
```

---

## 5.4 Dicts — Hash Maps, Open Addressing, and Collision Resolution

### 🟢 Beginner: Key-Value Storage

A dictionary maps **unique keys** to **values**. Keys must be hashable.

```python
# Creating dictionaries
empty: dict = {}
person: dict[str, str | int] = {
    "name": "Alice",
    "age": 30,
    "city": "NYC",
}

# From sequences of pairs
pairs: dict[str, int] = dict([("a", 1), ("b", 2), ("c", 3)])
from_kwargs: dict[str, int] = dict(a=1, b=2, c=3)  # Keys must be valid identifiers
from_zip: dict[str, int] = dict(zip(["x", "y", "z"], [1, 2, 3]))

# Accessing values
print(person["name"])              # "Alice"
# print(person["email"])           # KeyError!
print(person.get("email"))         # None (no error)
print(person.get("email", "N/A"))  # "N/A" (custom default)

# Setting values
person["email"] = "alice@example.com"   # Add new key
person["age"] = 31                       # Update existing key

# Removing
del person["city"]                 # Remove key (KeyError if missing)
email: str = person.pop("email")   # Remove and return (KeyError if missing)
safe: str = person.pop("phone", "none")  # Remove with default
```

**Essential dict methods:**

```python
config: dict[str, int | str] = {"host": "localhost", "port": 8080, "debug": True}

# Iteration
for key in config:                 # Iterates over KEYS (default)
    print(key)

for key, value in config.items():  # Key-value pairs
    print(f"  {key}: {value}")

for value in config.values():      # Values only
    print(value)

# Membership (checks KEYS, not values)
print("host" in config)            # True
print("localhost" in config)       # False — "localhost" is a value, not a key

# Merging (Python 3.9+)
defaults: dict[str, int | str] = {"port": 80, "timeout": 30}
overrides: dict[str, int | str] = {"port": 8080, "debug": True}

merged: dict = defaults | overrides    # Right side wins on conflicts
print(merged)  # {'port': 8080, 'timeout': 30, 'debug': True}

# In-place merge
defaults |= overrides                  # Mutates defaults

# setdefault — get value or set it if missing
cache: dict[str, list[int]] = {}
cache.setdefault("users", []).append(42)
# Equivalent to:
# if "users" not in cache:
#     cache["users"] = []
# cache["users"].append(42)
```

### 🟡 Intermediate: defaultdict, Counter, OrderedDict

```python
from collections import defaultdict, Counter, OrderedDict

# defaultdict — auto-creates missing keys
word_groups: defaultdict[int, list[str]] = defaultdict(list)
for word in ["hello", "cat", "world", "dog", "hi"]:
    word_groups[len(word)].append(word)
print(dict(word_groups))
# {5: ['hello', 'world'], 3: ['cat', 'dog', 'hi'], 2: ['hi']}

# Counter — count occurrences
text: str = "abracadabra"
counts: Counter[str] = Counter(text)
print(counts)                  # Counter({'a': 5, 'b': 2, 'r': 2, 'c': 1, 'd': 1})
print(counts.most_common(3))   # [('a', 5), ('b', 2), ('r', 2)]

# Counter arithmetic
c1: Counter[str] = Counter("aabbb")
c2: Counter[str] = Counter("aab")
print(c1 - c2)       # Counter({'b': 2}) — subtract counts (drops non-positive)
print(c1 + c2)       # Counter({'a': 4, 'b': 4})
print(c1 & c2)       # Counter({'a': 2, 'b': 1}) — min of each
print(c1 | c2)       # Counter({'b': 3, 'a': 2}) — max of each
```

**Dict comprehensions and the merge pattern:**

```python
# Invert a dictionary
original: dict[str, int] = {"a": 1, "b": 2, "c": 3}
inverted: dict[int, str] = {v: k for k, v in original.items()}
print(inverted)  # {1: 'a', 2: 'b', 3: 'c'}

# Filtered dict
scores: dict[str, int] = {"Alice": 95, "Bob": 67, "Charlie": 82, "Diana": 45}
passing: dict[str, int] = {k: v for k, v in scores.items() if v >= 70}
print(passing)  # {'Alice': 95, 'Charlie': 82}

# Group by pattern
from collections import defaultdict

words: list[str] = ["apple", "banana", "avocado", "blueberry", "cherry", "apricot"]
by_letter: dict[str, list[str]] = defaultdict(list)
for word in words:
    by_letter[word[0]].append(word)
print(dict(by_letter))
# {'a': ['apple', 'avocado', 'apricot'], 'b': ['banana', 'blueberry'], 'c': ['cherry']}
```

**Gotcha: Dictionary ordering and mutation during iteration:**

```python
# Since Python 3.7, dicts preserve INSERTION ORDER (guaranteed)
d: dict[str, int] = {"b": 2, "a": 1, "c": 3}
print(list(d.keys()))   # ['b', 'a', 'c'] — insertion order

# ❌ NEVER modify a dict while iterating it
d = {"a": 1, "b": 2, "c": 3, "d": 4}
# for k in d:
#     if d[k] % 2 == 0:
#         del d[k]   # RuntimeError: dictionary changed size during iteration

# ✅ Collect keys first, then delete
keys_to_delete: list[str] = [k for k, v in d.items() if v % 2 == 0]
for k in keys_to_delete:
    del d[k]

# ✅ Or create a new dict
d = {k: v for k, v in d.items() if v % 2 != 0}
```

### 🔴 Expert: CPython's Compact Dict Implementation (3.6+)

Since CPython 3.6, dictionaries use a **compact, ordered** hash table design. There are two separate arrays:

```
BEFORE Python 3.6 — Sparse Table (wasteful):
┌─────────────────────────────────────────────────┐
│  Each slot stored: [hash | key_ptr | value_ptr]  │
│  8 slots × 24 bytes = 192 bytes                  │
│  Only 3 slots used = 72 bytes of actual data      │
│  Wasted: 120 bytes (62.5% waste!)                 │
└─────────────────────────────────────────────────┘

AFTER Python 3.6 — Compact + Index Table:
┌──────────────────────────────┐
│  Index Table (sparse)        │    1 byte per slot (if table < 256)
│  [_, _, 0, _, 1, _, 2, _]    │    Stores indices into entries array
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────────────────────┐
│  Entries Array (dense, compact)               │
│  [0] hash=hash("b")  key="b"  value=2        │
│  [1] hash=hash("a")  key="a"  value=1        │
│  [2] hash=hash("c")  key="c"  value=3        │
└──────────────────────────────────────────────┘
  Entries are in INSERTION ORDER (this is why dicts are ordered!)

Benefits:
- 20-25% less memory than the old layout
- Iteration is O(n) over the dense entries array (no skipping empty slots)
- Insertion order is preserved by the entries array
```

**Hash collision resolution in dicts:**

```python
# CPython uses open addressing with a perturbation-based probe sequence
# Starting slot: i = hash(key) % table_size
# On collision: i = (5 * i + 1 + perturb) % table_size; perturb >>= 5

# This probe sequence visits every slot in the table exactly once
# (because table_size is always a power of 2 and 5 is coprime to any power of 2)

# Demo: watch for collisions
class VerboseHash:
    """An object that prints when it's hashed or compared."""
    def __init__(self, value: int, hash_value: int) -> None:
        self.value = value
        self._hash = hash_value

    def __hash__(self) -> int:
        print(f"  __hash__({self.value}) = {self._hash}")
        return self._hash

    def __eq__(self, other: object) -> bool:
        if isinstance(other, VerboseHash):
            print(f"  __eq__({self.value}, {other.value})")
            return self.value == other.value
        return NotImplemented

    def __repr__(self) -> str:
        return f"V({self.value})"

# Two objects with the same hash → collision
a = VerboseHash(1, hash_value=42)
b = VerboseHash(2, hash_value=42)  # Same hash!
c = VerboseHash(3, hash_value=42)  # Same hash!

d: dict = {}
d[a] = "first"    # hash → slot, store
d[b] = "second"   # hash → same slot, probe to next, compare, store
d[c] = "third"    # hash → same slot, probe, compare, probe, compare, store

# Looking up 'b' requires:
# 1. hash(b) → find slot
# 2. slot occupied by 'a' → compare b == a → False → probe
# 3. next slot has 'b' → compare b == b → True → return "second"
print(d[b])
```

**Dict resize thresholds:**

```python
# Dicts resize when they're more than 2/3 full
# New size is 2x for small dicts, less aggressive for large dicts
#
# Table sizes: 8, 16, 32, 64, 128, 256, ...
# A fresh dict starts with table size 8
# With 6 entries (6/8 > 2/3), it resizes to 16

import sys

d: dict = {}
prev_size: int = 0
for i in range(100):
    d[i] = i
    size: int = sys.getsizeof(d)
    if size != prev_size:
        print(f"  {i+1} entries: {size} bytes")
        prev_size = size
```

**The `__missing__` dunder — custom dict behavior:**

```python
class DefaultDict(dict):
    """Dict that calls a factory for missing keys (like defaultdict)."""

    def __init__(self, factory, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.factory = factory

    def __missing__(self, key):
        """Called by __getitem__ when key is not found."""
        self[key] = self.factory()
        return self[key]

d = DefaultDict(list)
d["fruits"].append("apple")
d["fruits"].append("banana")
print(d)  # {'fruits': ['apple', 'banana']}
```

---

## 5.5 Choosing the Right Container

### 🟢 Beginner: The Decision Tree

```
"What container should I use?"

Need key-value pairs?
├── Yes → dict
│   ├── Need default values for missing keys? → defaultdict
│   ├── Need to count things? → Counter
│   └── Otherwise → dict
│
└── No → sequence or set?
    │
    ├── Need unique elements only?
    │   ├── Need to modify it? → set
    │   └── Need it hashable/immutable? → frozenset
    │
    └── Need ordered sequence?
        ├── Will it change (add/remove/modify)?
        │   ├── Add/remove from BOTH ends? → deque
        │   └── Otherwise → list
        └── Will it stay fixed?
            ├── Need it hashable? → tuple
            ├── Need named fields? → NamedTuple
            └── Otherwise → tuple
```

### 🟡 Intermediate: Time Complexity Comparison

```
┌──────────────────────────────────────────────────────────────────┐
│                OPERATION COMPARISON TABLE                          │
├──────────────────┬────────┬────────┬────────┬────────┬──────────┤
│  Operation        │  list  │ tuple  │  set   │  dict  │  deque   │
├──────────────────┼────────┼────────┼────────┼────────┼──────────┤
│  Index [i]        │  O(1)  │  O(1)  │  N/A   │  N/A   │  O(n)   │
│  Append/Add end   │  O(1)* │  N/A   │  O(1)* │  O(1)* │  O(1)   │
│  Prepend/Add start│  O(n)  │  N/A   │  N/A   │  N/A   │  O(1)   │
│  Insert middle    │  O(n)  │  N/A   │  N/A   │  N/A   │  O(n)   │
│  Delete by index  │  O(n)  │  N/A   │  N/A   │  N/A   │  O(n)   │
│  Delete by value  │  O(n)  │  N/A   │  O(1)* │  O(1)* │  O(n)   │
│  x in container   │  O(n)  │  O(n)  │  O(1)* │  O(1)* │  O(n)   │
│  len()            │  O(1)  │  O(1)  │  O(1)  │  O(1)  │  O(1)   │
│  Sort             │O(nlogn)│  N/A   │  N/A   │  N/A   │  N/A    │
│  min()/max()      │  O(n)  │  O(n)  │  O(n)  │  O(n)  │  O(n)   │
│  Memory (relative)│  High  │  Low   │  High  │  High  │  High   │
├──────────────────┴────────┴────────┴────────┴────────┴──────────┤
│  * = amortized;  O(1)* can be O(n) worst case (hash collision)   │
│  N/A = operation not supported                                    │
└──────────────────────────────────────────────────────────────────┘
```

```python
# Real-world choice examples:

from collections import deque

# 1. BFS queue → deque (O(1) popleft)
queue: deque[int] = deque([1])
queue.append(2)
node: int = queue.popleft()   # O(1), not O(n) like list.pop(0)

# 2. Seen-set for graph traversal → set (O(1) lookup)
visited: set[int] = set()
if node not in visited:       # O(1)
    visited.add(node)

# 3. Frequency counting → Counter
from collections import Counter
words: list[str] = ["the", "cat", "sat", "on", "the", "mat"]
freq: Counter[str] = Counter(words)

# 4. Config with defaults → dict | ChainMap
from collections import ChainMap
defaults: dict[str, int] = {"timeout": 30, "retries": 3}
user_config: dict[str, int] = {"timeout": 60}
config = ChainMap(user_config, defaults)
print(config["timeout"])   # 60 (user override)
print(config["retries"])   # 3  (default)

# 5. Return multiple values → tuple or NamedTuple
from typing import NamedTuple

class ParseResult(NamedTuple):
    success: bool
    value: str
    errors: list[str]

result = ParseResult(True, "hello", [])
if result.success:
    print(result.value)
```

### 🔴 Expert: Memory Layout Comparison

```python
import sys

# Memory comparison for storing 1000 integers

# List: header + pointer array + over-allocation slack
lst: list[int] = list(range(1000))
print(f"  list:  {sys.getsizeof(lst):>6} bytes (container only)")

# Tuple: header + inline pointer array (no slack)
tup: tuple = tuple(range(1000))
print(f"  tuple: {sys.getsizeof(tup):>6} bytes (container only)")

# Set: hash table (sparse) + element storage
st: set[int] = set(range(1000))
print(f"  set:   {sys.getsizeof(st):>6} bytes (container only)")

# Dict: index table + entries array
dt: dict[int, None] = dict.fromkeys(range(1000))
print(f"  dict:  {sys.getsizeof(dt):>6} bytes (container only)")
```

```
Typical output:

    list:    8056 bytes   (8 bytes per pointer)
    tuple:   8040 bytes   (same, but no slack)
    set:    32984 bytes   (hash table overhead)
    dict:   36960 bytes   (hash table + entries)

NOTE: These are CONTAINER sizes only. The actual int objects
(stored on the heap) add ~28 bytes each regardless of container.
For range(1000), many small ints are cached, so real overhead
varies.

Key insight: sets and dicts use ~4x more memory than lists/tuples
for the same number of elements. The tradeoff is O(1) lookup.
```

---

## 🔧 Debug This: The Broken Inventory System

Your e-commerce team's inventory system has bugs across all four container types. Find them all:

```python
def process_inventory(raw_orders, catalog):
    """Process orders against inventory catalog."""

    # Bug zone 1: De-duplicate orders
    unique_orders = list(set(raw_orders))  # "preserves order"

    # Bug zone 2: Build price lookup
    prices = {}
    for item in catalog:
        name, price, stock = item
        prices[name.lower()] = price

    # Bug zone 3: Calculate totals
    totals = []
    for order in unique_orders:
        item_name, quantity = order
        price = prices.get(item_name, 0)
        line_total = price * quantity
        totals.append((item_name, quantity, line_total))

    # Bug zone 4: Track out-of-stock items
    in_stock = {item[0] for item in catalog if item[2] > 0}
    ordered_items = {order[0] for order in unique_orders}
    out_of_stock = ordered_items - in_stock

    # Bug zone 5: Summary
    grand_total = sum(t[2] for t in totals)
    return {
        "line_items": totals,
        "grand_total": grand_total,
        "out_of_stock": out_of_stock,
    }

# Test data
catalog = [
    ("Widget", 9.99, 100),
    ("Gadget", 24.99, 0),    # Out of stock!
    ("widget", 9.99, 50),    # Duplicate with different case
]

orders = [
    ("Widget", 2),
    ("Gadget", 1),
    ("Widget", 2),  # Duplicate order
    ("Doohickey", 3),  # Not in catalog
]

result = process_inventory(orders, catalog)
print(f"Total: ${result['grand_total']:.2f}")
print(f"Out of stock: {result['out_of_stock']}")
```

### Bugs to find:

```
1. ____________________________________________________
   Hint: set(raw_orders) doesn't preserve insertion order.
   And are tuples of (name, quantity) the right way to de-dup?
   What if someone orders Widget×2 twice?

2. ____________________________________________________
   Hint: prices keys are lowercased, but order item names are not.
   prices.get("Widget") != prices.get("widget").

3. ____________________________________________________
   Hint: "Doohickey" isn't in the catalog. price defaults to 0,
   but should we silently accept orders for nonexistent items?

4. ____________________________________________________
   Hint: in_stock uses catalog item names as-is (mixed case),
   but ordered_items also uses original case. Will the set
   difference work correctly with "Widget" vs "widget"?

5. ____________________________________________________
   Hint: The catalog has "Widget" and "widget" as separate entries.
   After lowercasing in the prices dict, one overwrites the other.
   Is the stock count being aggregated?
```

### Solution (try first!)

```python
from typing import NamedTuple


class CatalogItem(NamedTuple):
    name: str
    price: float
    stock: int


class LineItem(NamedTuple):
    item_name: str
    quantity: int
    line_total: float


class OrderSummary(NamedTuple):
    line_items: list[LineItem]
    grand_total: float
    out_of_stock: list[str]
    not_found: list[str]


def process_inventory(
    raw_orders: list[tuple[str, int]],
    catalog: list[tuple[str, float, int]],
) -> OrderSummary:
    """Process orders against inventory catalog.

    Normalizes item names to lowercase for matching.
    Aggregates duplicate orders (same item) by summing quantities.
    """
    # Bug 1 FIX: Aggregate duplicate orders by item, preserving order
    # Instead of set-based dedup (loses order AND treats Widget×2 + Widget×2
    # as one order), we MERGE quantities for the same item.
    from collections import OrderedDict
    merged_orders: dict[str, int] = {}
    for item_name, quantity in raw_orders:
        key: str = item_name.lower()  # Bug 2 FIX: normalize here too
        merged_orders[key] = merged_orders.get(key, 0) + quantity

    # Bug 5 FIX: Aggregate catalog entries with the same normalized name
    prices: dict[str, float] = {}
    stock: dict[str, int] = {}
    for name, price, item_stock in catalog:
        normalized: str = name.lower()
        prices[normalized] = price  # Last price wins (or could validate equality)
        stock[normalized] = stock.get(normalized, 0) + item_stock  # SUM stock

    # Bug 3 FIX: Separate "not found" from "out of stock"
    line_items: list[LineItem] = []
    not_found: list[str] = []
    out_of_stock: list[str] = []

    for item_key, quantity in merged_orders.items():
        if item_key not in prices:
            not_found.append(item_key)
            continue

        if stock.get(item_key, 0) <= 0:
            out_of_stock.append(item_key)
            # Still calculate the line total (backorder scenario)

        price: float = prices[item_key]
        line_total: float = round(price * quantity, 2)
        line_items.append(LineItem(item_key, quantity, line_total))

    grand_total: float = round(sum(li.line_total for li in line_items), 2)

    return OrderSummary(
        line_items=line_items,
        grand_total=grand_total,
        out_of_stock=out_of_stock,
        not_found=not_found,
    )


# Test data
catalog: list[tuple[str, float, int]] = [
    ("Widget", 9.99, 100),
    ("Gadget", 24.99, 0),
    ("widget", 9.99, 50),  # Aggregated with "Widget" → stock=150
]

orders: list[tuple[str, int]] = [
    ("Widget", 2),
    ("Gadget", 1),
    ("Widget", 2),     # Merged: Widget total = 4
    ("Doohickey", 3),  # Not in catalog
]

result: OrderSummary = process_inventory(orders, catalog)
for li in result.line_items:
    print(f"  {li.item_name}: {li.quantity} × ${li.line_total / li.quantity:.2f} = ${li.line_total:.2f}")
print(f"Grand Total: ${result.grand_total:.2f}")
print(f"Out of stock: {result.out_of_stock}")
print(f"Not found: {result.not_found}")
```

```
Bug Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. set() dedup:      Doesn't preserve order AND treats two
   Widget×2 orders as one (tuples are equal). Should aggregate
   quantities instead.

2. Case mismatch:    prices dict uses .lower() keys, but
   orders use original case. prices.get("Widget") returns None
   because the key is "widget".

3. Silent zero:      Nonexistent items silently get price=0.
   Should be flagged as errors or separated into "not_found".

4. Case in sets:     in_stock has "Widget" and "widget" as
   separate entries. ordered_items has "Widget". The set
   difference doesn't match correctly across cases.

5. Overwrite:        "Widget" and "widget" overwrite each other
   in prices dict after lowercasing. Stock should be SUMMED
   (100 + 50 = 150), not overwritten.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Summary: Module 5 Key Takeaways

```
┌──────────────────────────────────────────────────────────────────┐
│                    BIG FOUR CONTAINERS CHEAT SHEET                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  LIST:                                                           │
│    Dynamic array. O(1) index/append, O(n) insert/remove/search. │
│    Over-allocates ~12.5% for amortized O(1) append.              │
│    Never use list.pop(0) — use deque.popleft().                  │
│    [[0]*3]*3 shares rows! Use comprehension instead.             │
│                                                                   │
│  TUPLE:                                                          │
│    Immutable sequence. ~25% less memory than list.               │
│    Hashable (if contents are). Use for dict keys, return values. │
│    NamedTuple for self-documenting structured data.              │
│    Beware: tuple containing mutable objects CAN be mutated.      │
│                                                                   │
│  SET:                                                            │
│    Hash table. O(1) add/remove/contains.                         │
│    Perfect for: dedup, membership tests, set algebra.            │
│    ~4x memory vs list. frozenset for immutable/hashable sets.    │
│    Intersection iterates the SMALLER set (optimization).         │
│                                                                   │
│  DICT:                                                           │
│    Compact hash map. O(1) get/set/delete. Preserves order (3.7+)│
│    defaultdict, Counter, ChainMap for specialized patterns.      │
│    Never modify during iteration. Use comprehension to filter.   │
│    Case sensitivity: normalize keys on insert AND lookup.        │
│                                                                   │
│  Production rule: Choose by ACCESS PATTERN, not by content.     │
│    Need O(1) lookup? → set or dict                               │
│    Need ordering? → list or tuple                                │
│    Need both? → dict (ordered + O(1) lookup since 3.7)          │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Next up → Module 6: Comprehensions — Elegance Meets Performance**

Say "Start Module 6" when you're ready.
