# Module 1 — Foundations: The Bedrock

> *"Every production outage I've debugged started with someone not understanding the basics deeply enough."*
> — A Principal Engineer, probably

---

## 1.1 Python's Type System

### 🟢 Beginner: What Are Types?

Every value in Python is an **object**, and every object has a **type**. The type determines what you can *do* with a value.

```python
# The fundamental built-in types
name: str = "Alice"          # Text
age: int = 30                # Whole numbers (arbitrary precision!)
height: float = 5.9          # Decimal numbers (IEEE 754 double)
is_student: bool = True      # True or False
nothing: None = None         # The absence of a value
z: complex = 3 + 4j          # Complex numbers (yes, Python has these)
```

**Key intuition:** Python is *dynamically typed* (types are checked at runtime) but *strongly typed* (you can't silently mix incompatible types).

```python
# This works — Python converts int to float automatically
result: float = 10 + 3.14    # 13.14

# This fails — Python won't guess what you mean
broken = "age: " + 30        # TypeError: can only concatenate str to str
fixed: str = "age: " + str(30)   # "age: 30"
```

**The `type()` and `isinstance()` duo:**

```python
print(type(42))              # <class 'int'>
print(type(3.14))            # <class 'float'>
print(type(True))            # <class 'bool'>  — surprise! bool is a subclass of int

# isinstance() is almost always preferred over type()
print(isinstance(True, int))  # True — because bool inherits from int
print(type(True) == int)      # False — exact type check fails
```

### 🟡 Intermediate: The Numeric Tower and Gotchas

**Gotcha #1: `bool` is an `int`**

```python
# This is valid Python and it's terrifying
total: int = True + True + False  # 2
# Because True == 1 and False == 0

# This leads to subtle bugs in data processing
counts: list[int] = [True, 1, False, 0, True]
print(sum(counts))  # 3 — mixes boolean flags with integers silently
```

**Gotcha #2: Integer interning (small integer cache)**

```python
a: int = 256
b: int = 256
print(a is b)   # True — CPython caches integers from -5 to 256

a = 257
b = 257
print(a is b)   # False (usually) — outside the cache range
# NEVER use 'is' to compare values. Use '=='
```

**Gotcha #3: `None` is a singleton**

```python
# This is the ONE correct use of 'is'
x = None
if x is None:       # ✅ Correct — None is a singleton
    print("empty")

if x == None:       # ❌ Works but bad practice — some objects override __eq__
    print("empty")
```

**The numeric tower in practice:**

```python
# Python auto-promotes: int → float → complex
result = 1 + 2.0          # float: 3.0
result = 1 + 2.0 + 3j     # complex: (3+3j)

# But narrowing requires explicit conversion
value: float = 3.7
truncated: int = int(value)     # 3 — truncates toward zero, does NOT round
rounded: int = round(value)     # 4
import math
floored: int = math.floor(value)  # 3
ceiled: int = math.ceil(value)    # 4
```

### 🔴 Expert: CPython's Object Model in C

Every Python object in CPython is a C struct. At minimum, it contains:

```
┌─────────────────────────────────────┐
│         PyObject (C struct)         │
├─────────────────────────────────────┤
│  ob_refcnt   │  Reference count     │  ← How many names point to this
│  ob_type     │  Pointer to type     │  ← Points to the type object
├─────────────────────────────────────┤
│  ... type-specific data ...         │  ← For int: the actual number
└─────────────────────────────────────┘
```

**How CPython stores integers:**

Small integers (-5 to 256) live in a pre-allocated array (`small_ints[]` in `Objects/longobject.c`). When you write `x = 42`, Python doesn't allocate new memory — it returns a pointer to the existing `42` object.

For large integers, CPython uses a variable-length representation. A Python `int` is essentially an array of 30-bit "digits" (stored in `uint32_t`), so `2**1000` is stored as ~34 of these digits. This is why Python integers have *arbitrary precision* — they grow as needed.

```
# Memory layout of a large integer
┌──────────────┬──────────────┬─────────┬─────────┬─────────┐
│  ob_refcnt   │   ob_type    │ ob_size │ digit[0]│ digit[1]│...
│   (8 bytes)  │  (8 bytes)   │(8 bytes)│(4 bytes)│(4 bytes)│
└──────────────┴──────────────┴─────────┴─────────┴─────────┘
                                  ↑
                         Number of 30-bit digits
                         (negative = negative number)
```

**Reference counting in action:**

```python
import sys

a: list[int] = [1, 2, 3]
print(sys.getrefcount(a))  # 2 (one for 'a', one for the getrefcount arg)

b = a                       # refcount → 3
c = a                       # refcount → 4
del b                       # refcount → 3
del c                       # refcount → 2
# When refcount hits 0, memory is freed immediately
# (unless there's a reference cycle — that's what the GC handles)
```

---

## 1.2 Arithmetic Nuances: IEEE 754 and Floating Point Traps

### 🟢 Beginner: Basic Arithmetic

```python
# The seven arithmetic operators
a, b = 17, 5

print(a + b)    # 22    Addition
print(a - b)    # 12    Subtraction
print(a * b)    # 85    Multiplication
print(a / b)    # 3.4   True division (ALWAYS returns float)
print(a // b)   # 3     Floor division (rounds toward negative infinity)
print(a % b)    # 2     Modulo (remainder)
print(a ** b)   # 1419857  Exponentiation
```

**The division trap that broke Python 2:**

```python
# In Python 3, / always gives float
print(10 / 3)    # 3.3333...
print(10 // 3)   # 3

# Floor division rounds toward NEGATIVE INFINITY, not zero
print(-7 // 2)   # -4  (not -3!)
print(7 // -2)   # -4  (not -3!)
```

### 🟡 Intermediate: Why `0.1 + 0.2 != 0.3`

This is not a Python bug. It's how *all* languages using IEEE 754 floating point work.

```python
print(0.1 + 0.2)           # 0.30000000000000004
print(0.1 + 0.2 == 0.3)    # False!
```

**Why?** `0.1` in binary is a repeating fraction (like 1/3 in decimal). The 64-bit float stores an approximation.

```
Decimal 0.1 in binary:
0.0001100110011001100110011001100110011001100110011... (repeating)

Stored as (52-bit mantissa):
0.1000000000000000055511151231257827021181583404541015625
```

**The fix: Never compare floats with `==`**

```python
import math

# Option 1: math.isclose (Python 3.5+)
print(math.isclose(0.1 + 0.2, 0.3))  # True
# Default: rel_tol=1e-09, abs_tol=0.0

# Option 2: decimal module for exact decimal arithmetic
from decimal import Decimal, getcontext

getcontext().prec = 28  # 28 significant digits

price: Decimal = Decimal("0.1") + Decimal("0.2")
print(price == Decimal("0.3"))  # True!

# WARNING: Decimal("0.1") != Decimal(0.1)
print(Decimal(0.1))    # 0.1000000000000000055511151231257827...
print(Decimal("0.1"))  # 0.1  ← Use string constructor!
```

**When to use what:**

| Use Case | Type | Why |
|---|---|---|
| Counting things | `int` | Exact, arbitrary precision |
| Scientific computation | `float` | Fast, hardware-accelerated |
| Money / billing | `Decimal` | Exact decimal representation |
| Symbolic math | `fractions.Fraction` | Exact rational arithmetic |

```python
from fractions import Fraction

# Fraction gives exact results
f: Fraction = Fraction(1, 10) + Fraction(2, 10)
print(f)               # 3/10
print(f == Fraction(3, 10))  # True — exact comparison works
```

### 🔴 Expert: IEEE 754 Deep Dive

A 64-bit float (`double`) has three components:

```
┌───┬───────────────┬──────────────────────────────────────────────────┐
│ S │   Exponent    │                   Mantissa                       │
│1b │   11 bits     │                   52 bits                        │
└───┴───────────────┴──────────────────────────────────────────────────┘

Value = (-1)^S × 2^(Exponent - 1023) × 1.Mantissa
```

**Special values:**

```python
# Infinity
print(float('inf'))           # inf
print(float('inf') + 1)      # inf
print(float('inf') - float('inf'))  # nan

# NaN (Not a Number) — the only value not equal to itself
import math
nan = float('nan')
print(nan == nan)             # False!
print(math.isnan(nan))        # True — always use this

# Negative zero exists
print(-0.0 == 0.0)            # True (they compare equal)
print(math.copysign(1, -0.0)) # -1.0 (but the sign is preserved)

# Largest and smallest representable values
import sys
print(sys.float_info.max)     # 1.7976931348623157e+308
print(sys.float_info.min)     # 2.2250738585072014e-308  (smallest normal)
print(sys.float_info.epsilon) # 2.220446049250313e-16    (machine epsilon)
```

**Catastrophic cancellation — when subtraction destroys precision:**

```python
# Computing (10**20 + 1) - 10**20
a: float = 1e20
b: float = a + 1
print(b - a)  # 0.0 — the +1 was lost when stored in b!

# The Kahan summation algorithm compensates for this
def kahan_sum(values: list[float]) -> float:
    """Compensated summation for better floating-point accuracy."""
    total: float = 0.0
    compensation: float = 0.0
    for value in values:
        y: float = value - compensation
        t: float = total + y
        compensation = (t - total) - y  # Recovers lost low-order bits
        total = t
    return total

# Demonstration
values: list[float] = [1e16, 1.0, -1e16]
print(sum(values))        # 0.0  — naive sum loses the 1.0
print(kahan_sum(values))  # 1.0  — compensated sum preserves it
```

---

## 1.3 Operator Precedence & Associativity

### 🟢 Beginner: The Rules of the Road

Python evaluates expressions in a specific order. You don't need to memorize the full table — just know the common pitfalls.

```python
# Multiplication before addition (like math class)
result: int = 2 + 3 * 4     # 14, not 20

# Exponentiation before negation (this one trips people up)
result = -2 ** 2             # -4, not 4!
# Because it's parsed as -(2 ** 2), not (-2) ** 2
result = (-2) ** 2           # 4 — use parentheses to be safe

# Chained comparisons are special in Python
x: int = 5
print(1 < x < 10)           # True — equivalent to (1 < x) and (x < 10)
print(1 < x > 3)            # True — both conditions checked
```

**The precedence hierarchy (simplified, high to low):**

```
1.  ()              Parentheses (always wins)
2.  **              Exponentiation (right-to-left!)
3.  +x, -x, ~x     Unary operators
4.  *, /, //, %     Multiplication family
5.  +, -            Addition family
6.  <<, >>          Bitwise shifts
7.  &               Bitwise AND
8.  ^               Bitwise XOR
9.  |               Bitwise OR
10. ==, !=, <, >, <=, >=, is, in   Comparisons
11. not             Logical NOT
12. and             Logical AND
13. or              Logical OR
14. :=              Walrus operator (lowest)
```

### 🟡 Intermediate: Associativity and the Traps

**Right-to-left associativity of `**`:**

```python
# ** is RIGHT-associative (the only binary operator that is)
print(2 ** 3 ** 2)    # 512, not 64
# Parsed as 2 ** (3 ** 2) = 2 ** 9 = 512
# NOT (2 ** 3) ** 2 = 8 ** 2 = 64
```

**Short-circuit evaluation — `and`/`or` return values, not booleans:**

```python
# 'or' returns the first truthy value (or the last value)
name: str = "" or "Anonymous"  # "Anonymous"
port: int = 0 or 8080          # 8080

# 'and' returns the first falsy value (or the last value)
result = "hello" and "world"   # "world"
result = "" and "world"        # ""

# Common pattern: default values (pre-walrus operator era)
config: dict = {}
timeout: int = config.get("timeout") or 30
# But beware: this replaces 0 with 30 too!
# Better: config.get("timeout", 30)
```

**Bitwise operator precedence — a notorious trap:**

```python
# You'd expect this to work like math
flags: int = 0b1010
mask: int = 0b1100
if flags & mask == 0b1000:   # WRONG! == binds tighter than &
    print("match")
# Parsed as: flags & (mask == 0b1000) → flags & False → 0

if (flags & mask) == 0b1000:  # CORRECT — always parenthesize bitwise ops
    print("match")
```

### 🔴 Expert: CPython's Expression Evaluation

CPython compiles expressions to bytecode that runs on a stack machine. You can inspect this:

```python
import dis

def example() -> int:
    return 2 + 3 * 4

dis.dis(example)
# Output:
#   LOAD_CONST   2
#   LOAD_CONST   3
#   LOAD_CONST   4
#   BINARY_MULTIPLY         ← 3 * 4 first (higher precedence)
#   BINARY_ADD              ← then 2 + 12
#   RETURN_VALUE
```

**Constant folding — the compiler is smarter than you think:**

```python
import dis

def folded() -> int:
    return 2 + 3 * 4  # Compiler computes this at compile time!

dis.dis(folded)
# In CPython 3.12+:
#   LOAD_CONST  14          ← The compiler already calculated 14
#   RETURN_VALUE

# The peephole optimizer also handles:
# - String concatenation of literals: "hello" + " " + "world" → "hello world"
# - Tuple of constants: (1, 2, 3) → single LOAD_CONST
# - Power of small ints: 2 ** 8 → 256
```

---

## 1.4 Built-in Functions vs. Built-in Modules

### 🟢 Beginner: What's Available Without Importing?

Python gives you ~70 built-in functions that are *always* available. No `import` needed.

```python
# The ones you'll use every day
length: int = len([1, 2, 3])        # 3
total: int = sum([1, 2, 3])         # 6
biggest: int = max(1, 5, 3)         # 5
smallest: int = min(1, 5, 3)        # 1
absolute: int = abs(-42)            # 42
text: str = str(42)                 # "42"
number: int = int("42")             # 42
decimal: float = float("3.14")      # 3.14
items: list = list(range(5))        # [0, 1, 2, 3, 4]
pairs: list = list(zip([1, 2], ["a", "b"]))  # [(1, 'a'), (2, 'b')]

# Type checking
print(type(42))                     # <class 'int'>
print(isinstance(42, (int, float))) # True — checks against multiple types

# The help system
help(len)  # Prints documentation for len()
dir(str)   # Lists all attributes/methods of str
```

**Built-in modules require `import`:**

```python
import math
import os
import sys
import json
import datetime

# Why aren't these built-in functions?
# Because Python follows "batteries included, but not forced on you"
# Only the most universally needed tools are built-in
```

### 🟡 Intermediate: The Useful Built-ins Nobody Teaches You

```python
# enumerate — stop writing index counters
fruits: list[str] = ["apple", "banana", "cherry"]
for i, fruit in enumerate(fruits, start=1):
    print(f"{i}. {fruit}")
# 1. apple
# 2. banana
# 3. cherry

# zip — iterate in parallel
names: list[str] = ["Alice", "Bob"]
scores: list[int] = [95, 87]
for name, score in zip(names, scores, strict=True):  # strict=True in 3.10+
    print(f"{name}: {score}")
# strict=True raises ValueError if lengths differ

# any() and all() — short-circuit on iterables
numbers: list[int] = [2, 4, 6, 8]
print(all(n % 2 == 0 for n in numbers))  # True — all even
print(any(n > 5 for n in numbers))       # True — at least one > 5

# sorted() vs .sort() — one returns new list, other mutates in place
data: list[int] = [3, 1, 4, 1, 5]
new_list: list[int] = sorted(data)     # data unchanged, new_list = [1, 1, 3, 4, 5]
data.sort()                             # data is now [1, 1, 3, 4, 5], returns None

# map() and filter() — functional style (often replaced by comprehensions)
doubled: list[int] = list(map(lambda x: x * 2, [1, 2, 3]))     # [2, 4, 6]
evens: list[int] = list(filter(lambda x: x % 2 == 0, [1, 2, 3, 4]))  # [2, 4]
# Prefer comprehensions:
doubled = [x * 2 for x in [1, 2, 3]]
evens = [x for x in [1, 2, 3, 4] if x % 2 == 0]

# vars() — inspect an object's namespace
class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

p = Point(1.0, 2.0)
print(vars(p))  # {'x': 1.0, 'y': 2.0}
```

### 🔴 Expert: How Built-ins Live in CPython

Built-in functions are not Python functions — they're implemented in C.

```python
# A built-in function is a 'builtin_function_or_method' object
print(type(len))  # <class 'builtin_function_or_method'>
print(type(print))  # <class 'builtin_function_or_method'>

# A regular Python function is different
def my_func() -> None:
    pass
print(type(my_func))  # <class 'function'>
```

**Where built-ins come from — the `builtins` module:**

```python
import builtins

# Every "built-in" is actually an attribute of the builtins module
print(builtins.len([1, 2, 3]))  # 3

# You can even override built-ins (please don't in production)
original_print = builtins.print

def custom_print(*args, **kwargs) -> None:
    original_print("[LOG]", *args, **kwargs)

builtins.print = custom_print
print("hello")  # [LOG] hello

# Restore it
builtins.print = original_print
```

**The LEGB lookup chain for built-ins:**

```
When Python encounters a name like `len`:
1. Local scope   → not found
2. Enclosing     → not found
3. Global        → not found
4. Built-in      → found in builtins module ✓

This is why you can shadow built-ins:
    list = [1, 2, 3]  # Now 'list' in Global scope shadows builtins.list
    list("hello")      # TypeError! You just broke list()
```

---

## 1.5 Variables, Names, and the Object Model

### 🟢 Beginner: Names, Not Boxes

In Python, variables are **names** that **refer to** objects. They are NOT boxes that contain values.

```python
a: int = 42    # Create an int object 42, bind the name 'a' to it
b = a          # Bind the name 'b' to the SAME object

print(id(a))   # e.g., 140234866357008
print(id(b))   # Same! Both names point to the same object
```

Think of it like name tags, not boxes:

```
WRONG mental model (boxes):          CORRECT mental model (tags):
┌─────┐  ┌─────┐                    ┌────┐
│a: 42│  │b: 42│                    │ 42 │ ← object in memory
└─────┘  └─────┘                    └────┘
  (two separate copies)              ↑    ↑
                                     a    b   ← two names, one object
```

**Why this matters — mutable vs. immutable:**

```python
# With immutable objects (int, str, tuple), it doesn't matter
a: int = 10
b = a
a = 20         # Rebinds 'a' to a NEW object (20)
print(b)       # Still 10 — b still points to original object

# With mutable objects (list, dict, set), it's critical
x: list[int] = [1, 2, 3]
y = x          # y and x point to the SAME list
x.append(4)    # Mutate the list through x
print(y)       # [1, 2, 3, 4] — y sees the change!

# To make an independent copy:
y = x.copy()           # Shallow copy
y = list(x)            # Also shallow copy
import copy
y = copy.deepcopy(x)   # Deep copy (copies nested objects too)
```

### 🟡 Intermediate: Mutable Default Arguments — The Classic Trap

```python
# THE BUG
def add_item(item: str, items: list[str] = []) -> list[str]:
    items.append(item)
    return items

print(add_item("a"))   # ['a']        — looks fine
print(add_item("b"))   # ['a', 'b']   — wait, where did 'a' come from?!
print(add_item("c"))   # ['a', 'b', 'c']  — it keeps accumulating!

# WHY: The default [] is created ONCE when the function is defined,
# not each time the function is called. All calls share the same list.

# THE FIX: Use None as sentinel
def add_item_fixed(item: str, items: list[str] | None = None) -> list[str]:
    if items is None:
        items = []    # Fresh list created on each call
    items.append(item)
    return items
```

**Identity vs. Equality:**

```python
a: list[int] = [1, 2, 3]
b: list[int] = [1, 2, 3]
c = a

# == checks VALUE equality (calls __eq__)
print(a == b)    # True — same contents

# 'is' checks IDENTITY (same object in memory)
print(a is b)    # False — different objects
print(a is c)    # True — same object

# Rule of thumb:
# Use 'is' ONLY for: None, True, False, and sentinel objects
# Use '==' for everything else
```

### 🔴 Expert: The Heap, Stack, and Reference Counting

```
╔══════════════════════════════════════════════════════════════════╗
║                     MEMORY LAYOUT                                ║
╠════════════════════════╦═════════════════════════════════════════╣
║   STACK (per frame)    ║            HEAP (shared)                ║
║                        ║                                         ║
║  ┌──────────────┐      ║   ┌─────────────────────────────────┐   ║
║  │ Frame: main  │      ║   │ PyListObject                    │   ║
║  │              │      ║   │   refcnt: 2                     │   ║
║  │  x ──────────╫──────╫──▶│   size: 3                       │   ║
║  │  y ──────────╫──────╫──▶│   items: [ptr, ptr, ptr]        │   ║
║  │              │      ║   │            │    │    │           │   ║
║  └──────────────┘      ║   └────────────┼────┼────┼───────────┘   ║
║                        ║                ▼    ▼    ▼               ║
║                        ║   ┌────┐  ┌────┐  ┌────┐               ║
║                        ║   │ 1  │  │ 2  │  │ 3  │  (int objs)  ║
║                        ║   └────┘  └────┘  └────┘               ║
╚════════════════════════╩═════════════════════════════════════════╝

After: y = x
Both 'x' and 'y' on the stack point to the same PyListObject.
refcnt is 2 (two references).

After: del x
refcnt drops to 1. Object survives (y still references it).

After: del y
refcnt drops to 0. CPython immediately frees the PyListObject.
The int objects 1, 2, 3 may survive (small int cache) or be freed.
```

**The cyclic garbage collector — when refcounting isn't enough:**

```python
import gc

# Reference cycle: a refers to b, b refers to a
a: list = []
b: list = []
a.append(b)   # a → b
b.append(a)   # b → a

del a
del b
# refcount of both lists is 1 (they reference each other)
# refcounting alone can't free them!

# The cyclic GC runs periodically and detects these cycles
gc.collect()  # Force a collection cycle

# You can inspect GC behavior
print(gc.get_threshold())  # (700, 10, 10) — generation thresholds
print(gc.get_count())      # Allocation counts per generation
```

**Object interning beyond small integers:**

```python
# CPython also interns some strings
a: str = "hello"
b: str = "hello"
print(a is b)  # True — string literals that look like identifiers are interned

a = "hello world"
b = "hello world"
print(a is b)  # May be True or False — depends on compilation context

# You can force interning
import sys
a = sys.intern("hello world")
b = sys.intern("hello world")
print(a is b)  # True — guaranteed to be the same object

# Why intern? Dictionary lookups can short-circuit the string comparison
# if both keys are the same object (identity check before equality check)
```

---

## 🔧 Debug This: The Billing System Bug

You're building a billing system. A customer reports they were charged an incorrect amount. Here's the code — find all the bugs:

```python
def calculate_total(prices, tax_rate=0.08, discount=0.1):
    """Calculate the total cost with tax and discount."""
    subtotal = 0
    for price in prices:
        subtotal += price

    # Apply discount
    if discount:
        subtotal = subtotal - subtotal * discount

    # Apply tax
    total = subtotal + subtotal * tax_rate

    # Round to cents
    total = round(total, 2)

    # Check if total matches expected
    expected = 0.1 + 0.2
    if total == expected:
        print("Exact match!")

    return total

# Test case
items = [19.99, 29.99, 0.01]
print(f"Total: ${calculate_total(items)}")  # Expected: $48.59 after 10% discount + 8% tax

# Bug report: Customer with $0.00 discount was still getting 10% off!
print(f"No discount: ${calculate_total(items, discount=0.0)}")
```

### Bugs to find:

```
1. ____________________________________________________
   Hint: What does `if discount:` evaluate to when discount is 0.0?

2. ____________________________________________________
   Hint: What happens with floating point addition of prices?

3. ____________________________________________________
   Hint: The `expected` comparison — will it ever be True?

4. ____________________________________________________
   Hint: What type hinting is missing? What about the default mutable argument pattern?
```

### Solution (try to find them yourself first!)

```python
from decimal import Decimal
from typing import Sequence


def calculate_total(
    prices: Sequence[Decimal],
    tax_rate: Decimal = Decimal("0.08"),
    discount: Decimal = Decimal("0.1"),
) -> Decimal:
    """Calculate the total cost with tax and discount.

    Args:
        prices: Sequence of item prices as Decimal.
        tax_rate: Tax rate as Decimal (default 8%).
        discount: Discount rate as Decimal (default 10%).
                  Pass Decimal("0") for no discount.

    Returns:
        Total cost rounded to 2 decimal places.
    """
    subtotal: Decimal = sum(prices, Decimal("0"))

    # Bug 1 FIX: Compare against specific value, not truthiness
    # 0.0 is falsy, so `if discount:` skips a $0 discount correctly...
    # BUT `if discount:` would also skip if someone passes Decimal("0.0")
    # Explicit comparison is clearer and safer:
    if discount > Decimal("0"):
        subtotal = subtotal - subtotal * discount

    total: Decimal = subtotal + subtotal * tax_rate

    # Bug 2 FIX: Using Decimal eliminates float accumulation errors
    total = total.quantize(Decimal("0.01"))

    # Bug 3 FIX: Never compare floats with ==
    # (This comparison was meaningless anyway — removed)

    # Bug 4 FIX: Added type hints and used Decimal throughout
    return total


items: list[Decimal] = [
    Decimal("19.99"),
    Decimal("29.99"),
    Decimal("0.01"),
]
print(f"Total: ${calculate_total(items)}")
print(f"No discount: ${calculate_total(items, discount=Decimal('0'))}")
```

---

## Summary: Module 1 Key Takeaways

```
┌─────────────────────────────────────────────────────────────────┐
│                     FOUNDATIONS CHEAT SHEET                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Types:     Everything is an object. bool subclasses int.       │
│  Floats:    Never compare with ==. Use Decimal for money.       │
│  Names:     Variables are references, not boxes.                │
│  Mutables:  Shared references can bite. Copy when needed.       │
│  is vs ==:  'is' for identity (None), '==' for equality.       │
│  Defaults:  Never use mutable default arguments.                │
│  Built-ins: 70+ functions always available. Don't shadow them.  │
│  Memory:    Reference counting + cyclic GC.                     │
│                                                                  │
│  Production rule: When in doubt, be explicit.                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Next up → Module 2: Strings & RegEx — Text as a First-Class Citizen**

Say "Start Module 2" when you're ready.
