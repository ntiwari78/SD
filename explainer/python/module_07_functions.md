# Module 7 — Functions & Namespaces: The LEGB Universe

> *"Functions are first-class citizens. Treat them that way."*

---

## 7.1 Defining Functions, Parameters, and Return Values

### 🟢 Beginner: Your First Functions

```python
# Basic function definition
def greet(name: str) -> str:
    """Return a greeting for the given name."""
    return f"Hello, {name}!"

print(greet("Alice"))  # "Hello, Alice!"

# Functions without a return statement return None
def say_hello(name: str) -> None:
    print(f"Hello, {name}!")

result = say_hello("Bob")
print(result)  # None

# Multiple return values (really returning a tuple)
def divide(a: float, b: float) -> tuple[float, float]:
    """Return quotient and remainder."""
    quotient: float = a // b
    remainder: float = a % b
    return quotient, remainder  # Packs into a tuple

q, r = divide(17, 5)  # Tuple unpacking
print(f"17 ÷ 5 = {q} remainder {r}")  # 17 ÷ 5 = 3.0 remainder 2.0
```

**The four parameter types:**

```python
# 1. Positional parameters (the basics)
def add(a: int, b: int) -> int:
    return a + b

add(1, 2)       # a=1, b=2

# 2. Default parameters (optional with fallback)
def greet(name: str, greeting: str = "Hello") -> str:
    return f"{greeting}, {name}!"

greet("Alice")              # "Hello, Alice!"
greet("Alice", "Hi")        # "Hi, Alice!"
greet("Alice", greeting="Hey")  # "Hey, Alice!"

# 3. *args — variable positional arguments (collected as a tuple)
def total(*numbers: float) -> float:
    """Sum any number of arguments."""
    return sum(numbers)

total(1, 2, 3)        # 6
total(1, 2, 3, 4, 5)  # 15
total()                # 0

# 4. **kwargs — variable keyword arguments (collected as a dict)
def build_profile(**kwargs: str) -> dict[str, str]:
    """Build a user profile from keyword arguments."""
    return kwargs

build_profile(name="Alice", role="admin", team="backend")
# {'name': 'Alice', 'role': 'admin', 'team': 'backend'}
```

**Combining all parameter types:**

```python
def api_call(
    endpoint: str,                    # Positional (required)
    method: str = "GET",              # Default
    *path_parts: str,                 # *args
    timeout: int = 30,               # Keyword-only (after *)
    **headers: str,                   # **kwargs
) -> dict:
    """Demonstrate all parameter types in one function."""
    return {
        "endpoint": endpoint,
        "method": method,
        "path": "/".join(path_parts),
        "timeout": timeout,
        "headers": headers,
    }

result = api_call(
    "https://api.example.com",
    "POST",
    "users", "42", "profile",
    timeout=60,
    Authorization="Bearer token123",
    Accept="application/json",
)
# endpoint: https://api.example.com
# method: POST
# path: users/42/profile
# timeout: 60
# headers: {Authorization: Bearer token123, Accept: application/json}
```

### 🟡 Intermediate: Keyword-Only, Positional-Only, and Unpacking

**Keyword-only arguments (after `*`):**

```python
# Parameters after * can ONLY be passed by keyword
def safe_divide(a: float, b: float, *, round_digits: int = 2) -> float:
    """Divide a by b with optional rounding.

    round_digits MUST be passed as a keyword argument.
    """
    result: float = a / b
    return round(result, round_digits)

safe_divide(10, 3)                     # 3.33
safe_divide(10, 3, round_digits=4)     # 3.3333
# safe_divide(10, 3, 4)               # TypeError! round_digits is keyword-only
```

**Positional-only arguments (before `/`, Python 3.8+):**

```python
# Parameters before / can ONLY be passed positionally
def pow(base: float, exp: float, /, *, mod: int | None = None) -> float:
    """Compute base**exp, optionally modulo mod.

    base and exp are positional-only (like C's pow()).
    mod is keyword-only.
    """
    result: float = base ** exp
    if mod is not None:
        result = result % mod
    return result

pow(2, 10)              # 1024.0
pow(2, 10, mod=100)     # 24.0
# pow(base=2, exp=10)   # TypeError! base and exp are positional-only
```

**Unpacking arguments with `*` and `**`:**

```python
# * unpacks iterables into positional arguments
def add(a: int, b: int, c: int) -> int:
    return a + b + c

numbers: list[int] = [1, 2, 3]
print(add(*numbers))     # 6  — same as add(1, 2, 3)

# ** unpacks dicts into keyword arguments
config: dict[str, int | str] = {"host": "localhost", "port": 8080}
def connect(host: str, port: int) -> str:
    return f"Connecting to {host}:{port}"

print(connect(**config))  # "Connecting to localhost:8080"

# Combining both
def func(a: int, b: int, c: int = 0, *, d: int = 0) -> None:
    print(f"a={a}, b={b}, c={c}, d={d}")

args: list[int] = [1, 2]
kwargs: dict[str, int] = {"c": 3, "d": 4}
func(*args, **kwargs)  # a=1, b=2, c=3, d=4
```

**Gotcha: Mutable default arguments (revisited with deeper understanding):**

```python
# WHY it happens: defaults are evaluated ONCE at function DEFINITION time
# and stored in func.__defaults__

def append_to(item: int, lst: list[int] = []) -> list[int]:
    lst.append(item)
    return lst

# Inspect the default
print(append_to.__defaults__)  # ([],)  — the actual default object

append_to(1)  # [1]
append_to(2)  # [1, 2]  — same list object!

# The default is STILL the same object, now mutated:
print(append_to.__defaults__)  # ([1, 2],)

# FIX: Use None sentinel
def append_to_fixed(item: int, lst: list[int] | None = None) -> list[int]:
    if lst is None:
        lst = []
    lst.append(item)
    return lst
```

### 🔴 Expert: Function Objects in CPython

Functions in Python are full objects with attributes, stored on the heap like everything else.

```python
def example(x: int, y: int = 10) -> int:
    """An example function."""
    z: int = x + y
    return z

# Function object attributes
print(example.__name__)        # 'example'
print(example.__doc__)         # 'An example function.'
print(example.__defaults__)    # (10,)  — tuple of default values
print(example.__annotations__) # {'x': int, 'y': int, 'return': int}
print(example.__code__)        # <code object example at 0x...>

# The code object — the compiled bytecode
code = example.__code__
print(code.co_varnames)   # ('x', 'y', 'z')  — local variable names
print(code.co_argcount)   # 2 — number of positional arguments
print(code.co_consts)     # (None, ...) — constants used in the function
print(code.co_stacksize)  # stack depth needed to execute
print(code.co_nlocals)    # 3 — total local variables (including args)
```

**How CPython stores function objects:**

```
┌──────────────────────────────────────────────────────────┐
│                  PyFunctionObject                         │
├──────────────────────────────────────────────────────────┤
│  ob_refcnt         │  Reference count                    │
│  ob_type           │  → function type                    │
│  func_code         │  → PyCodeObject (bytecode)          │
│  func_globals      │  → module's global dict             │
│  func_defaults     │  → tuple of default values          │
│  func_kwdefaults   │  → dict of keyword-only defaults    │
│  func_closure      │  → tuple of cell objects (closures) │
│  func_doc          │  → docstring                        │
│  func_name         │  → function name string             │
│  func_dict         │  → function's __dict__ (attributes) │
│  func_annotations  │  → type annotations dict            │
│  func_qualname     │  → qualified name                   │
└──────────────────────────────────────────────────────────┘

The key insight: func_code (PyCodeObject) is SEPARATE from the
function itself. Multiple functions can share the same code object
(e.g., in a factory pattern). The code object is immutable and
contains only bytecode + constants.
```

**Parameter passing: neither "by value" nor "by reference":**

```python
# Python uses "pass by object reference" (or "pass by assignment")
# The function receives a reference to the same object — NOT a copy

def modify_list(lst: list[int]) -> None:
    lst.append(99)          # Mutates the original (same object)
    lst = [1, 2, 3]         # Rebinds LOCAL name only — original unchanged

original: list[int] = [10, 20]
modify_list(original)
print(original)  # [10, 20, 99]  — .append() mutated it; = didn't

# Equivalent pseudocode of what happens:
# 1. lst = original  (both names point to [10, 20])
# 2. lst.append(99)  (mutates the shared object → [10, 20, 99])
# 3. lst = [1, 2, 3] (lst now points to a NEW list; original unchanged)
```

---

## 7.2 The LEGB Rule — Local, Enclosing, Global, Built-in

### 🟢 Beginner: Where Python Looks for Names

When you use a variable name, Python searches four scopes in order:

```
L — Local:     Inside the current function
E — Enclosing: Inside any enclosing functions (closures)
G — Global:    At the module level (the file's top-level)
B — Built-in:  Python's built-in names (len, print, etc.)
```

```python
# Built-in scope
# print, len, int, str, etc. are here

# Global scope (module level)
x: str = "global"

def outer():
    # Enclosing scope (for inner())
    x: str = "enclosing"

    def inner():
        # Local scope
        x: str = "local"
        print(x)  # "local" — found in Local scope first

    inner()
    print(x)  # "enclosing" — inner()'s x was local to inner

outer()
print(x)  # "global" — outer()'s x was local to outer
```

**Each scope is checked in L → E → G → B order:**

```python
x: str = "global"

def show_x():
    # No local x, no enclosing scope → finds x in Global
    print(x)  # "global"

show_x()

def shadow_x():
    x: str = "local"  # Creates a LOCAL x (shadows global)
    print(x)  # "local"

shadow_x()
print(x)  # "global" — unchanged
```

### 🟡 Intermediate: `global`, `nonlocal`, and the UnboundLocalError

**The UnboundLocalError trap:**

```python
count: int = 0

def increment():
    count += 1    # UnboundLocalError: local variable 'count'
                  # referenced before assignment!

# WHY: Python sees 'count += 1' (which is count = count + 1)
# and decides at COMPILE TIME that 'count' is a LOCAL variable
# (because it's assigned to). But then at RUNTIME, the right side
# tries to read 'count' before it's been assigned locally.
```

**Fix 1: `global` — access the module-level variable:**

```python
count: int = 0

def increment() -> None:
    global count       # Tell Python: 'count' is the GLOBAL one
    count += 1

increment()
increment()
print(count)  # 2

# ⚠️ global is generally considered bad practice
# It makes functions dependent on external state (hard to test/debug)
# Better: pass the value in and return the new value
def increment_pure(count: int) -> int:
    return count + 1
```

**Fix 2: `nonlocal` — access the enclosing function's variable:**

```python
def make_counter(start: int = 0):
    """Create a counter function with its own state."""
    count: int = start

    def increment() -> int:
        nonlocal count   # Tell Python: 'count' is in the ENCLOSING scope
        count += 1
        return count

    return increment

counter = make_counter()
print(counter())  # 1
print(counter())  # 2
print(counter())  # 3

# Each make_counter() call creates a NEW closure with its own 'count'
counter_a = make_counter(0)
counter_b = make_counter(100)
print(counter_a())  # 1
print(counter_b())  # 101
print(counter_a())  # 2   — independent!
```

**LEGB with class scopes — the surprising exception:**

```python
# Class bodies do NOT create an enclosing scope for LEGB!
x: int = 10

class MyClass:
    x: int = 20  # Class attribute

    # ❌ This FAILS — class scope is NOT 'E' in LEGB
    # items = [x for _ in range(3)]  # NameError: name 'x' is not defined
    # The comprehension's implicit function can't see the class scope!

    # ✅ Fix: reference explicitly
    items: list[int] = [20 for _ in range(3)]  # Hardcode, or:

    @staticmethod
    def get_items(val: int = 20) -> list[int]:
        return [val for _ in range(3)]

print(MyClass.x)     # 20
print(MyClass.items)  # [20, 20, 20]
```

### 🔴 Expert: CPython's Scope Analysis at Compile Time

CPython determines scope at **compile time**, not runtime. This is why `UnboundLocalError` happens even if the assignment is never executed:

```python
x: int = 10

def tricky():
    print(x)      # UnboundLocalError!
    if False:
        x = 20    # This line is NEVER executed, but Python still
                   # sees it at compile time and marks x as local

# The compiler scans the entire function body for assignments
# to determine which names are local. This is called "scope analysis."
```

```python
import dis

x: int = 10

def reads_global():
    return x

def writes_local():
    x = 20
    return x

dis.dis(reads_global)
# LOAD_GLOBAL  x        ← Python knows x is global (no assignment)

dis.dis(writes_local)
# LOAD_CONST   20
# STORE_FAST   x        ← Python knows x is local (assigned in function)
# LOAD_FAST    x         ← fast local lookup
# RETURN_VALUE
```

**Cell objects — how closures work in CPython:**

```python
def outer(x: int):
    def inner():
        return x  # x is a "free variable" — accessed via a cell object
    return inner

fn = outer(42)

# The closure is stored as a tuple of cell objects
print(fn.__closure__)         # (<cell at 0x...: int object at 0x...>,)
print(fn.__closure__[0].cell_contents)  # 42

# Cell objects allow the inner function to access the enclosing scope's
# variables even after the enclosing function has returned.
# They're essentially shared mutable pointers.
```

```
How cells connect outer and inner scopes:

  outer() frame (while running):                inner() closure:
  ┌───────────────┐                             ┌───────────────┐
  │ x ─────────┐  │                             │ __closure__   │
  └────────────┼──┘                             │  [0] ─────┐  │
               ▼                                └──────────┼──┘
          ┌──────────┐                                     │
          │ Cell obj  │ ◄──────────────────────────────────┘
          │  content ─┼──▶ int(42)
          └──────────┘

  Both outer's local 'x' and inner's free 'x' point to the SAME
  cell object. When outer assigns x, the cell's content changes.
  When inner reads x, it reads the cell's content.
```

---

## 7.3 Closures — Capturing State Without Classes

### 🟢 Beginner: What Is a Closure?

A closure is a function that **remembers** variables from the scope where it was created, even after that scope has finished.

```python
def make_greeter(greeting: str):
    """Return a function that greets with the given greeting."""

    def greeter(name: str) -> str:
        return f"{greeting}, {name}!"  # 'greeting' is "remembered"

    return greeter

hello = make_greeter("Hello")
howdy = make_greeter("Howdy")

print(hello("Alice"))  # "Hello, Alice!"
print(howdy("Bob"))    # "Howdy, Bob!"

# 'greeting' is different for each closure:
# hello remembers "Hello"
# howdy remembers "Howdy"
```

### 🟡 Intermediate: Practical Closure Patterns

**Pattern 1: Configuration factories:**

```python
def make_validator(
    min_val: float,
    max_val: float,
    name: str = "value",
) -> callable:
    """Create a validation function with configured bounds."""

    def validate(value: float) -> bool:
        if not min_val <= value <= max_val:
            raise ValueError(
                f"{name} must be between {min_val} and {max_val}, got {value}"
            )
        return True

    return validate

validate_age = make_validator(0, 150, "Age")
validate_temp = make_validator(-459.67, 1_000_000, "Temperature")

validate_age(25)     # True
# validate_age(200)  # ValueError: Age must be between 0 and 150, got 200
```

**Pattern 2: Accumulator / running state:**

```python
def make_running_average():
    """Create a function that maintains a running average."""
    total: float = 0.0
    count: int = 0

    def add(value: float) -> float:
        nonlocal total, count
        total += value
        count += 1
        return total / count

    return add

avg = make_running_average()
print(avg(10))   # 10.0
print(avg(20))   # 15.0
print(avg(30))   # 20.0
print(avg(40))   # 25.0
```

**Pattern 3: Callback registration:**

```python
from typing import Callable

def make_event_handler(
    event_name: str,
    log_func: Callable[[str], None] = print,
) -> Callable[..., None]:
    """Create a handler that logs events with context."""
    call_count: int = 0

    def handler(*args: object, **kwargs: object) -> None:
        nonlocal call_count
        call_count += 1
        log_func(f"[{event_name}] Call #{call_count}: args={args}, kwargs={kwargs}")

    # Attach metadata to the handler
    handler.event_name = event_name  # type: ignore[attr-defined]
    handler.get_count = lambda: call_count  # type: ignore[attr-defined]

    return handler

on_click = make_event_handler("click")
on_click(x=100, y=200)  # [click] Call #1: args=(), kwargs={'x': 100, 'y': 200}
on_click(x=150, y=250)  # [click] Call #2: ...
print(on_click.get_count())  # 2
```

**Gotcha: Late binding in closures:**

```python
# The classic trap (revisited from Module 6)
def make_multipliers() -> list[Callable[[int], int]]:
    multipliers: list = []
    for i in range(5):
        def mult(x: int) -> int:
            return x * i  # 'i' is captured by REFERENCE, not by VALUE
        multipliers.append(mult)
    return multipliers

funcs = make_multipliers()
print([f(10) for f in funcs])  # [40, 40, 40, 40, 40] — all use i=4!

# FIX 1: Default argument (eagerly captures i's current value)
def make_multipliers_fixed() -> list[Callable[[int], int]]:
    multipliers: list = []
    for i in range(5):
        def mult(x: int, i: int = i) -> int:  # i=i captures current value
            return x * i
        multipliers.append(mult)
    return multipliers

# FIX 2: Factory function (creates a new scope each iteration)
def make_multipliers_factory() -> list[Callable[[int], int]]:
    def make_mult(factor: int) -> Callable[[int], int]:
        def mult(x: int) -> int:
            return x * factor
        return mult
    return [make_mult(i) for i in range(5)]
```

### 🔴 Expert: Closure Implementation Details

```python
import types

def outer(x: int):
    y: int = x * 2

    def inner(z: int) -> int:
        return x + y + z  # x and y are free variables

    return inner

fn = outer(10)

# Inspect the closure
code: types.CodeType = fn.__code__
print(f"  Free variables: {code.co_freevars}")  # ('x', 'y')
print(f"  Local variables: {code.co_varnames}")  # ('z',)

# The closure tuple — one cell per free variable
for i, cell in enumerate(fn.__closure__):
    print(f"  Cell {i} ({code.co_freevars[i]}): {cell.cell_contents}")
# Cell 0 (x): 10
# Cell 1 (y): 20
```

**How `nonlocal` works at the bytecode level:**

```python
import dis

def outer():
    count: int = 0

    def inner():
        nonlocal count
        count += 1
        return count

    return inner

# outer's bytecode uses LOAD_DEREF/STORE_DEREF for 'count'
# because 'count' is shared with inner via a cell object
dis.dis(outer)
# LOAD_CONST    0
# STORE_DEREF   count      ← stores into cell object, not local!
# LOAD_CLOSURE  count
# BUILD_TUPLE   1
# LOAD_CONST    <code inner>
# MAKE_FUNCTION 8          ← flag 8 = has closure
# STORE_FAST    inner
# LOAD_FAST     inner
# RETURN_VALUE

fn = outer()
dis.dis(fn)
# LOAD_DEREF    count      ← reads from cell object
# LOAD_CONST    1
# BINARY_ADD
# STORE_DEREF   count      ← writes back to cell object
# LOAD_DEREF    count
# RETURN_VALUE
```

---

## 7.4 Decorators — From `@staticmethod` to Writing Your Own

### 🟢 Beginner: What Decorators Do

A decorator is a function that **wraps** another function to add behavior.

```python
# The @ syntax is just syntactic sugar
@decorator
def my_func():
    pass

# Is exactly the same as:
def my_func():
    pass
my_func = decorator(my_func)
```

```python
# A simple logging decorator
from typing import Callable, Any

def log_calls(func: Callable) -> Callable:
    """Print a message every time the decorated function is called."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        print(f"  Calling {func.__name__}({args}, {kwargs})")
        result = func(*args, **kwargs)
        print(f"  {func.__name__} returned {result!r}")
        return result

    return wrapper

@log_calls
def add(a: int, b: int) -> int:
    return a + b

add(3, 4)
# Calling add((3, 4), {})
# add returned 7
```

**Common built-in decorators:**

```python
class MyClass:
    class_var: int = 0

    def __init__(self, value: int) -> None:
        self.value = value

    # Regular method — gets 'self' automatically
    def get_value(self) -> int:
        return self.value

    # @classmethod — gets 'cls' instead of 'self'
    @classmethod
    def from_string(cls, s: str) -> "MyClass":
        return cls(int(s))

    # @staticmethod — gets neither 'self' nor 'cls'
    @staticmethod
    def is_valid(value: int) -> bool:
        return value >= 0

    # @property — makes a method look like an attribute
    @property
    def doubled(self) -> int:
        return self.value * 2

obj = MyClass(21)
print(obj.get_value())       # 21
print(obj.doubled)           # 42  (no parentheses — it's a property!)
print(MyClass.is_valid(-1))  # False
print(MyClass.from_string("99").value)  # 99
```

### 🟡 Intermediate: Writing Production-Grade Decorators

**Problem: Naive decorators break function metadata:**

```python
from typing import Callable, Any

def my_decorator(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name: str) -> str:
    """Return a greeting."""
    return f"Hello, {name}!"

print(greet.__name__)  # 'wrapper'  ← WRONG! Should be 'greet'
print(greet.__doc__)   # None       ← WRONG! Should be 'Return a greeting.'
help(greet)            # Shows wrapper's signature, not greet's
```

**Fix: Always use `functools.wraps`:**

```python
from functools import wraps
from typing import Callable, Any, TypeVar, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")

def my_decorator(func: Callable[P, T]) -> Callable[P, T]:
    @wraps(func)  # ← Copies __name__, __doc__, __annotations__, etc.
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def greet(name: str) -> str:
    """Return a greeting."""
    return f"Hello, {name}!"

print(greet.__name__)  # 'greet'  ✅
print(greet.__doc__)   # 'Return a greeting.'  ✅
```

**Decorators with arguments:**

```python
from functools import wraps
from typing import Callable, Any
import time

def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator factory: retry the function on exception."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_exception  # type: ignore[misc]

        return wrapper
    return decorator

# Usage — note the parentheses!
@retry(max_attempts=5, delay=0.5)
def fetch_data(url: str) -> str:
    """Fetch data from a URL with automatic retry."""
    ...

# What happens under the hood:
# 1. retry(max_attempts=5, delay=0.5) → returns 'decorator'
# 2. decorator(fetch_data) → returns 'wrapper'
# 3. fetch_data = wrapper
```

**Timing decorator — a practical example:**

```python
from functools import wraps
from typing import Callable, Any
import time

def timer(func: Callable) -> Callable:
    """Measure and print execution time of a function."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start: float = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed: float = time.perf_counter() - start
        print(f"  {func.__name__} took {elapsed:.4f}s")
        return result

    return wrapper

@timer
def slow_function(n: int) -> int:
    """Compute sum the slow way."""
    return sum(range(n))

slow_function(10_000_000)  # slow_function took 0.1234s
```

**Stacking decorators:**

```python
@decorator_a
@decorator_b
@decorator_c
def my_func():
    pass

# Is equivalent to:
# my_func = decorator_a(decorator_b(decorator_c(my_func)))
# Execution order: c wraps first, then b wraps c's result, then a wraps b's.
# When called: a's wrapper runs first, then b's, then c's, then my_func.
```

### 🔴 Expert: Descriptor Protocol and Class-Based Decorators

**Class-based decorators:**

```python
from functools import wraps, update_wrapper
from typing import Any, Callable

class CountCalls:
    """Decorator that counts how many times a function is called."""

    def __init__(self, func: Callable) -> None:
        update_wrapper(self, func)  # Copy metadata
        self.func = func
        self.count: int = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.count += 1
        return self.func(*args, **kwargs)

    def reset(self) -> None:
        self.count = 0

@CountCalls
def say_hello(name: str) -> str:
    return f"Hello, {name}!"

say_hello("Alice")
say_hello("Bob")
print(say_hello.count)  # 2  — class instance attribute
say_hello.reset()
print(say_hello.count)  # 0
```

**The decorator and descriptor interaction — why `@staticmethod` works:**

```python
# When you access a function through a class, Python's descriptor
# protocol kicks in. Functions are descriptors:

class MyClass:
    def method(self):
        pass

# MyClass.__dict__['method'] is a plain function object
# But MyClass.method is a BOUND method (function.__get__ was called)

func = MyClass.__dict__['method']
print(type(func))              # <class 'function'>
print(type(MyClass.method))    # <class 'function'>  (unbound in Py3)
print(type(MyClass().method))  # <class 'method'>    (bound)

# @staticmethod and @classmethod work by replacing the function
# with a descriptor that changes __get__ behavior:
#
# staticmethod.__get__() → returns the raw function (no self/cls)
# classmethod.__get__() → returns a bound method with cls as first arg
# function.__get__() → returns a bound method with instance as first arg
```

**Decorators that work with and without arguments:**

```python
from functools import wraps
from typing import Callable, overload

# The elegant pattern for optional-argument decorators
def optional_arg_decorator(decorator_func: Callable) -> Callable:
    """Meta-decorator that allows decorators to be used with or without args."""
    @wraps(decorator_func)
    def wrapper(*args: Any, **kwargs: Any) -> Callable:
        if len(args) == 1 and callable(args[0]) and not kwargs:
            # Called without arguments: @decorator
            return decorator_func(args[0])
        else:
            # Called with arguments: @decorator(args)
            def real_decorator(func: Callable) -> Callable:
                return decorator_func(func, *args, **kwargs)
            return real_decorator
    return wrapper

@optional_arg_decorator
def debug(func: Callable, prefix: str = "DEBUG") -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        print(f"[{prefix}] Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

# Both usages work:
@debug                    # Without arguments
def hello() -> str:
    return "hello"

@debug("TRACE")           # With arguments
def goodbye() -> str:
    return "goodbye"

hello()    # [DEBUG] Calling hello
goodbye()  # [TRACE] Calling goodbye
```

---

## 7.5 `functools` Deep Dive: `lru_cache`, `partial`, `wraps`, and `singledispatch`

### 🟢 Beginner: Caching with `lru_cache`

```python
from functools import lru_cache

# Without caching — exponential time O(2^n)
def fib_slow(n: int) -> int:
    if n < 2:
        return n
    return fib_slow(n - 1) + fib_slow(n - 2)

# fib_slow(35) takes ~3 seconds!

# With caching — linear time O(n)
@lru_cache(maxsize=128)
def fib_fast(n: int) -> int:
    if n < 2:
        return n
    return fib_fast(n - 1) + fib_fast(n - 2)

print(fib_fast(100))  # 354224848179261915075 — instant!

# Check cache stats
print(fib_fast.cache_info())
# CacheInfo(hits=98, misses=101, maxsize=128, currsize=101)

# Clear the cache
fib_fast.cache_clear()
```

### 🟡 Intermediate: `partial`, `singledispatch`, and `reduce`

**`functools.partial` — freeze some arguments:**

```python
from functools import partial

# Create specialized versions of general functions
def power(base: float, exponent: float) -> float:
    return base ** exponent

square: Callable[[float], float] = partial(power, exponent=2)
cube: Callable[[float], float] = partial(power, exponent=3)

print(square(5))   # 25.0
print(cube(3))     # 27.0

# Practical use: configuring callbacks
import json

# Create a JSON serializer with specific settings
pretty_json = partial(json.dumps, indent=2, sort_keys=True, ensure_ascii=False)
print(pretty_json({"name": "Alice", "age": 30}))
```

**`functools.singledispatch` — method overloading by type:**

```python
from functools import singledispatch

@singledispatch
def process(data) -> str:
    """Default handler for unknown types."""
    raise TypeError(f"Unsupported type: {type(data)}")

@process.register(str)
def _(data: str) -> str:
    return f"String of length {len(data)}: {data!r}"

@process.register(int)
def _(data: int) -> str:
    return f"Integer: {data} (hex: {data:#x})"

@process.register(list)
def _(data: list) -> str:
    return f"List with {len(data)} items: {data}"

print(process("hello"))     # String of length 5: 'hello'
print(process(42))          # Integer: 42 (hex: 0x2a)
print(process([1, 2, 3]))   # List with 3 items: [1, 2, 3]
# process(3.14)             # TypeError: Unsupported type: <class 'float'>
```

**`functools.reduce` — fold a sequence into a single value:**

```python
from functools import reduce
from operator import mul

# reduce(f, [a, b, c, d]) = f(f(f(a, b), c), d)

# Product of a list (no built-in for this)
numbers: list[int] = [1, 2, 3, 4, 5]
product: int = reduce(mul, numbers)  # 1*2*3*4*5 = 120

# With initial value
product_with_init: int = reduce(mul, numbers, 1)  # Start with 1

# Building a nested dict from a key path
def set_nested(d: dict, keys: list[str], value: object) -> dict:
    """Set a value in a nested dict using a list of keys."""
    reduce(lambda d, k: d.setdefault(k, {}), keys[:-1], d)[keys[-1]] = value
    return d

config: dict = {}
set_nested(config, ["database", "connection", "host"], "localhost")
print(config)  # {'database': {'connection': {'host': 'localhost'}}}
```

### 🔴 Expert: `lru_cache` Implementation and `cache` (3.9+)

```python
from functools import lru_cache, cache

# @cache is just @lru_cache(maxsize=None) — unbounded cache
@cache
def expensive(n: int) -> int:
    """Cache ALL results forever (use with caution — memory leak risk)."""
    return sum(range(n))

# lru_cache uses a doubly-linked list + dict for O(1) operations:
#
# ┌─────────────────────────────────────────────────┐
# │             lru_cache internals                  │
# │                                                  │
# │  dict: {args_key → linked_list_node}            │
# │                                                  │
# │  Doubly-linked list (most recent → least recent)│
# │  [newest] ↔ [node] ↔ [node] ↔ [oldest]         │
# │                                                  │
# │  On cache hit:                                   │
# │    1. Find node in dict — O(1)                   │
# │    2. Move node to front — O(1)                  │
# │    3. Return cached value                        │
# │                                                  │
# │  On cache miss:                                  │
# │    1. Call the function                          │
# │    2. Add node to front — O(1)                   │
# │    3. If full, remove oldest (tail) — O(1)       │
# │    4. Remove from dict — O(1)                    │
# └─────────────────────────────────────────────────┘
```

**`lru_cache` gotchas:**

```python
from functools import lru_cache

# Gotcha 1: Arguments must be HASHABLE (used as dict keys)
@lru_cache(maxsize=128)
def process(data: list) -> int:  # ← TypeError! lists are not hashable
    return sum(data)

# Fix: use tuples instead
@lru_cache(maxsize=128)
def process(data: tuple[int, ...]) -> int:
    return sum(data)

process((1, 2, 3))  # Works!

# Gotcha 2: lru_cache considers keyword vs positional as DIFFERENT
@lru_cache
def add(a: int, b: int) -> int:
    print("  Computing...")
    return a + b

add(1, 2)        # Computing... → 3 (cache miss)
add(1, 2)        # 3 (cache hit!)
add(a=1, b=2)    # Computing... → 3 (cache miss! different key)
add(b=2, a=1)    # Computing... → 3 (cache miss! different key order!)

# Gotcha 3: Class methods — cache is shared across ALL instances
class MyClass:
    @lru_cache(maxsize=32)
    def method(self, x: int) -> int:
        return x ** 2

# 'self' is part of the cache key, AND it prevents garbage collection
# of instances (the cache holds references to self!)
# Fix: use __hash__/__eq__ carefully, or use a separate cache dict
```

**Type-safe decorators with `ParamSpec` (Python 3.10+):**

```python
from functools import wraps
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")       # Captures the parameter signature
T = TypeVar("T")         # Captures the return type

def logged(func: Callable[P, T]) -> Callable[P, T]:
    """A decorator that preserves the exact type signature."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)

    return wrapper

@logged
def add(a: int, b: int) -> int:
    return a + b

# Type checkers (mypy, pyright) now know:
# add(a: int, b: int) -> int
result: int = add(1, 2)    # ✅ Type checks pass
# add("a", "b")            # ❌ Type error caught by mypy
```

---

## 🔧 Debug This: The Broken Memoization System

Your team implemented a caching system for API responses. It has several bugs. Find them all:

```python
import time

# Global cache
cache = {}

def memoize(func):
    """Cache function results based on arguments."""
    def wrapper(*args, **kwargs):
        key = (args, kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

@memoize
def fetch_user(user_id, include_details=False):
    """Simulate an API call that takes 2 seconds."""
    time.sleep(0.1)  # Simulate network delay
    return {
        "id": user_id,
        "name": f"User {user_id}",
        "details": {"role": "admin"} if include_details else None,
        "fetched_at": time.time(),
    }

# Test
print(fetch_user(1))
print(fetch_user(1))  # Should be cached — but is it?

print(fetch_user(1, include_details=True))
print(fetch_user(1, True))  # Same call, different cache key?

# Clear cache for user 1?
# How do we invalidate?
```

### Bugs to find:

```
1. ____________________________________________________
   Hint: kwargs is a dict. Can you use a dict as part of a
   dict key? (Hint: dicts are not hashable.)

2. ____________________________________________________
   Hint: fetch_user(1, True) and fetch_user(1, include_details=True)
   should be the same call. Are they cached as the same key?

3. ____________________________________________________
   Hint: The global cache is shared across ALL memoized functions.
   If two different functions are decorated, their caches collide.

4. ____________________________________________________
   Hint: @wraps is missing. What happens to __name__, __doc__?

5. ____________________________________________________
   Hint: The cached response contains a mutable dict. If the caller
   modifies the returned dict, the CACHE is modified too!

6. ____________________________________________________
   Hint: No cache invalidation, no TTL, no size limit. Memory leak.
```

### Solution (try first!)

```python
import time
import copy
from functools import wraps
from typing import Any, Callable, TypeVar, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


def memoize(
    ttl: float | None = None,
    maxsize: int | None = 128,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Memoize decorator with per-function cache, TTL, and size limit.

    Args:
        ttl: Time-to-live in seconds. None for no expiry.
        maxsize: Maximum cache entries. None for unbounded.

    Returns:
        Decorator function.
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        # Bug 3 FIX: Per-function cache (not global)
        func_cache: dict[tuple, tuple[T, float]] = {}
        cache_order: list[tuple] = []  # For LRU eviction

        @wraps(func)  # Bug 4 FIX: Preserve function metadata
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Bug 1 & 2 FIX: Normalize the key
            # Convert kwargs to a sorted tuple of pairs (hashable)
            # and merge positional + keyword args using the function's signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()  # Fill in default values
            key: tuple = tuple(sorted(bound.arguments.items()))

            # Check cache (with TTL)
            if key in func_cache:
                result, cached_time = func_cache[key]
                if ttl is None or (time.time() - cached_time) < ttl:
                    # Bug 5 FIX: Return a deep copy so callers can't mutate cache
                    return copy.deepcopy(result)
                else:
                    # Expired
                    del func_cache[key]
                    cache_order.remove(key)

            # Cache miss — call the function
            result = func(*args, **kwargs)
            func_cache[key] = (result, time.time())
            cache_order.append(key)

            # Bug 6 FIX: Enforce size limit (evict oldest)
            if maxsize is not None and len(func_cache) > maxsize:
                oldest: tuple = cache_order.pop(0)
                func_cache.pop(oldest, None)

            return copy.deepcopy(result)

        # Utility methods on the wrapper
        wrapper.cache_clear = lambda: (func_cache.clear(), cache_order.clear())  # type: ignore
        wrapper.cache_info = lambda: {  # type: ignore
            "size": len(func_cache),
            "maxsize": maxsize,
            "ttl": ttl,
        }

        return wrapper
    return decorator


@memoize(ttl=60.0, maxsize=100)
def fetch_user(user_id: int, include_details: bool = False) -> dict[str, Any]:
    """Simulate an API call."""
    time.sleep(0.1)
    return {
        "id": user_id,
        "name": f"User {user_id}",
        "details": {"role": "admin"} if include_details else None,
        "fetched_at": time.time(),
    }


# Now these are treated as the same cache key:
print(fetch_user(1, include_details=True))
print(fetch_user(1, True))  # Cache hit! Same normalized key

# Mutating the result doesn't corrupt the cache:
result = fetch_user(1)
result["name"] = "HACKED"
print(fetch_user(1)["name"])  # "User 1" — cache is safe
```

---

## Summary: Module 7 Key Takeaways

```
┌──────────────────────────────────────────────────────────────────┐
│                  FUNCTIONS & NAMESPACES CHEAT SHEET               │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  PARAMETERS:                                                     │
│    Positional → Default → *args → Keyword-only → **kwargs        │
│    Before / = positional-only (3.8+)                             │
│    After * = keyword-only                                        │
│    NEVER use mutable defaults (use None sentinel)                │
│                                                                   │
│  LEGB RULE:                                                      │
│    Local → Enclosing → Global → Built-in                         │
│    Scope determined at COMPILE TIME (not runtime)                │
│    Class bodies are NOT enclosing scopes for comprehensions      │
│    global: access module-level variable (avoid if possible)      │
│    nonlocal: access enclosing function's variable                │
│                                                                   │
│  CLOSURES:                                                       │
│    Functions that remember enclosing scope variables             │
│    Variables captured by REFERENCE (late binding!)               │
│    Fix late binding: default args or factory functions           │
│    Implemented via cell objects in __closure__                   │
│                                                                   │
│  DECORATORS:                                                     │
│    @decorator = func = decorator(func)                           │
│    ALWAYS use @functools.wraps to preserve metadata              │
│    Decorator with args: needs two levels of nesting              │
│    Stacking: @a @b @c → a(b(c(func)))                           │
│                                                                   │
│  FUNCTOOLS:                                                      │
│    lru_cache: O(1) memoization with LRU eviction                │
│    cache: unbounded lru_cache (watch memory!)                    │
│    partial: freeze arguments for specialized functions           │
│    singledispatch: method overloading by argument type           │
│    wraps: ALWAYS use in custom decorators                        │
│                                                                   │
│  Production rules:                                               │
│    Pass by object reference, not value or reference.             │
│    Prefer pure functions (no global state, no side effects).     │
│    Use ParamSpec + TypeVar for type-safe decorators.             │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Next up → Module 8: Modules & Packages — Building Distributable Code**

Say "Start Module 8" when you're ready.
