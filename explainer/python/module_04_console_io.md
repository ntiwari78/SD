# Module 4 — Console I/O: Talking to the World

> *"The most fundamental I/O is text in, text out. Master it, and everything else — files, sockets, APIs — is just a variation."*

---

## 4.1 `input()`, `print()`, and `sys.stdin`/`sys.stdout`

### 🟢 Beginner: Reading and Writing Text

```python
# print() — send text to the screen (stdout)
print("Hello, World!")                      # Hello, World!
print("Name:", "Alice", "Age:", 30)         # Name: Alice Age: 30
print(1, 2, 3, sep=", ")                   # 1, 2, 3
print("Loading", end="")                    # No newline at end
print("...done!")                           # Loading...done!

# input() — read text from the keyboard (stdin)
name: str = input("Enter your name: ")     # Blocks until user presses Enter
print(f"Hello, {name}!")

# input() ALWAYS returns a string — you must convert
age_str: str = input("Enter your age: ")   # e.g., user types "25"
age: int = int(age_str)                     # Convert to int
print(f"Next year you'll be {age + 1}")

# Combine in one line
height: float = float(input("Height in meters: "))
```

**Common `print()` parameters:**

```python
# sep — separator between arguments (default: " ")
print("2025", "01", "15", sep="-")          # 2025-01-15
print(*[1, 2, 3], sep=" → ")               # 1 → 2 → 3

# end — string appended after all arguments (default: "\n")
for i in range(5):
    print(i, end=" ")
print()                                      # 0 1 2 3 4

# file — redirect output to a file object
with open("output.txt", "w") as f:
    print("This goes to a file", file=f)

# flush — force write to stream immediately
import time
for i in range(5):
    print(f"\rProgress: {i+1}/5", end="", flush=True)
    time.sleep(0.5)
print()  # Final newline
```

**Handling bad input gracefully:**

```python
def get_integer(prompt: str) -> int:
    """Repeatedly prompt until the user enters a valid integer."""
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("  That's not a valid integer. Try again.")

def get_choice(prompt: str, valid: set[str]) -> str:
    """Prompt until the user enters one of the valid choices."""
    while True:
        choice: str = input(prompt).strip().lower()
        if choice in valid:
            return choice
        print(f"  Please enter one of: {', '.join(sorted(valid))}")

# Usage
age: int = get_integer("Enter your age: ")
color: str = get_choice("Pick a color (red/green/blue): ", {"red", "green", "blue"})
```

### 🟡 Intermediate: `sys.stdin`, `sys.stdout`, and Redirection

Under the hood, `print()` writes to `sys.stdout` and `input()` reads from `sys.stdin`. These are file-like objects you can manipulate.

```python
import sys

# print() is essentially:
sys.stdout.write("Hello\n")    # Equivalent to print("Hello")

# input(prompt) is essentially:
sys.stderr.write("Enter: ")    # Actually writes prompt to stdout, not stderr
line: str = sys.stdin.readline().rstrip("\n")

# The three standard streams
# sys.stdin   — standard input  (keyboard by default)
# sys.stdout  — standard output (terminal by default)
# sys.stderr  — standard error  (terminal by default, not redirected by >)
```

**Redirecting stdout — capture print output:**

```python
import sys
from io import StringIO

# Capture all print output into a string
captured = StringIO()
original_stdout = sys.stdout
sys.stdout = captured

print("This is captured")
print("So is this")

sys.stdout = original_stdout  # Restore
output: str = captured.getvalue()
print(f"Captured: {output!r}")
# Captured: 'This is captured\nSo is this\n'
```

```python
# Better approach — use contextlib.redirect_stdout
from contextlib import redirect_stdout
from io import StringIO

buffer = StringIO()
with redirect_stdout(buffer):
    print("Safely captured")
    print("Without manual restore")
# stdout is automatically restored when the 'with' block exits

print(f"Got: {buffer.getvalue()!r}")
```

**Reading all of stdin at once (useful for piped input):**

```python
import sys

# When your script is used in a pipeline:
# echo "hello\nworld" | python my_script.py

# Read all lines
lines: list[str] = sys.stdin.readlines()    # List of lines with \n

# Or iterate line by line (memory efficient for large input)
for line in sys.stdin:
    process(line.rstrip("\n"))

# Or read everything as one string
all_input: str = sys.stdin.read()
```

**Detecting if stdin/stdout is a terminal or a pipe:**

```python
import sys

if sys.stdin.isatty():
    # Interactive mode — prompt the user
    name: str = input("Enter your name: ")
else:
    # Piped mode — read silently
    name = sys.stdin.readline().strip()

if sys.stdout.isatty():
    # Terminal — use colors and formatting
    print("\033[1;32mSuccess!\033[0m")  # Green bold text
else:
    # Piped to file/process — plain text only
    print("Success!")
```

### 🔴 Expert: CPython's I/O Stack and the C Runtime

```
┌─────────────────────────────────────────────────────────┐
│                  CPython I/O Architecture                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   Python code:  print("hello")                          │
│        │                                                 │
│        ▼                                                 │
│   TextIOWrapper  (sys.stdout)                           │
│   ├── encoding: "utf-8"                                 │
│   ├── line_buffering: True (when isatty)                │
│   └── wraps:                                            │
│        ▼                                                 │
│   BufferedWriter  (binary buffered layer)               │
│   ├── buffer_size: 8192 bytes (default)                 │
│   └── wraps:                                            │
│        ▼                                                 │
│   FileIO  (raw unbuffered I/O)                          │
│   ├── fd: 1 (stdout file descriptor)                    │
│   └── calls:                                            │
│        ▼                                                 │
│   OS write() syscall → kernel → terminal/pipe/file      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

```python
import sys

# Inspect the I/O stack
stdout = sys.stdout
print(f"  Type: {type(stdout)}")              # TextIOWrapper
print(f"  Encoding: {stdout.encoding}")        # utf-8
print(f"  Line buffering: {stdout.line_buffering}")  # True if terminal

# Access the underlying binary buffer
binary_layer = stdout.buffer
print(f"  Binary type: {type(binary_layer)}")  # BufferedWriter

# Access the raw file descriptor layer
raw_layer = binary_layer.raw
print(f"  Raw type: {type(raw_layer)}")        # FileIO
print(f"  File descriptor: {raw_layer.fileno()}")  # 1 (stdout)
```

**`print()` implementation — what actually happens:**

```python
# Simplified version of CPython's builtin_print (Python/bltinmodule.c)
def my_print(
    *args: object,
    sep: str | None = None,
    end: str | None = None,
    file=None,
    flush: bool = False,
) -> None:
    """Reimplementation showing what print() does internally."""
    if sep is None:
        sep = " "
    if end is None:
        end = "\n"
    if file is None:
        file = sys.stdout
        if file is None:
            return  # stdout can be None if closed

    # Convert all args to strings and join with separator
    output: str = sep.join(str(arg) for arg in args) + end

    # Write to the file object
    file.write(output)

    # Flush if requested
    if flush:
        file.flush()
```

**`input()` implementation details:**

```python
# input(prompt) does the following in CPython:
# 1. If sys.stdin is a real TTY:
#    a. Write prompt to stdout (not stderr!)
#    b. Call GNU readline (if available) for line editing
#    c. Read a line from stdin
#    d. Strip the trailing \n
# 2. If sys.stdin is NOT a TTY (piped):
#    a. Write prompt to stderr (so it doesn't mix with piped data)
#    b. Read a line from stdin
#    c. Strip the trailing \n
# 3. If stdin is closed: raise EOFError

# This is why 'input()' plays nicely with pipes:
# echo "Alice" | python -c "name = input('Name: '); print(f'Hi {name}')"
# "Name: " appears on stderr, "Alice" comes from stdin, "Hi Alice" goes to stdout
```

---

## 4.2 String Formatting Showdown: f-strings vs. `.format()` vs. `%`

### 🟢 Beginner: Three Ways to Format Strings

Python has three string formatting systems. Each has its place, but **f-strings are the modern default**.

```python
name: str = "Alice"
age: int = 30
gpa: float = 3.856

# Method 1: f-strings (Python 3.6+) — THE DEFAULT CHOICE
msg: str = f"Name: {name}, Age: {age}, GPA: {gpa:.2f}"
# "Name: Alice, Age: 30, GPA: 3.86"

# Method 2: str.format() — older but still useful
msg = "Name: {}, Age: {}, GPA: {:.2f}".format(name, age, gpa)
# "Name: Alice, Age: 30, GPA: 3.86"

# Method 3: % formatting — C-style, legacy
msg = "Name: %s, Age: %d, GPA: %.2f" % (name, age, gpa)
# "Name: Alice, Age: 30, GPA: 3.86"
```

**When to use each:**

```python
# ✅ f-strings: everyday formatting, inline expressions
total: float = 99.95
tax: float = 0.08
print(f"Total: ${total * (1 + tax):.2f}")  # Total: $107.95

# ✅ .format(): when the template is separate from the data
template: str = "Dear {name},\nYour order #{order_id} is {status}."
# Template defined elsewhere (config file, database, etc.)
msg = template.format(name="Bob", order_id=42, status="shipped")

# ✅ % formatting: logging module (it uses % by convention)
import logging
logging.warning("User %s failed login attempt %d", "alice", 3)
# The % formatting is LAZY here — string is only built if the
# log level is active. f-strings would always build the string.

# ❌ Don't use % for new code outside of logging
```

### 🟡 Intermediate: The Format Specification Mini-Language

All three methods support the same format spec syntax (after the `:`):

```
{value:[[fill]align][sign][z][#][0][width][grouping][.precision][type]}
```

```python
# ── WIDTH AND ALIGNMENT ──────────────────────────────────────
n: int = 42

print(f"|{n:10d}|")       # |        42|  Right-aligned (default for numbers)
print(f"|{n:<10d}|")      # |42        |  Left-aligned
print(f"|{n:^10d}|")      # |    42    |  Center-aligned
print(f"|{n:0>10d}|")     # |0000000042|  Right-aligned, fill with zeros
print(f"|{n:*^10d}|")     # |****42****|  Center, fill with asterisks

s: str = "hello"
print(f"|{s:15s}|")       # |hello          |  Left-aligned (default for strings)
print(f"|{s:>15s}|")      # |          hello|  Right-aligned
print(f"|{s:.3s}|")       # |hel|                Truncate to 3 chars


# ── NUMBER FORMATTING ────────────────────────────────────────
pi: float = 3.14159265359

print(f"{pi:.2f}")         # 3.14         Fixed point, 2 decimal places
print(f"{pi:.6f}")         # 3.141593     Fixed point, 6 decimal places
print(f"{pi:e}")           # 3.141593e+00 Scientific notation
print(f"{pi:.2e}")         # 3.14e+00     Scientific, 2 decimal places
print(f"{pi:.4g}")         # 3.142        General — auto-picks f or e
print(f"{pi:%}")           # 314.159265%  Percentage
print(f"{pi:.1%}")         # 314.2%       Percentage, 1 decimal

big: int = 1234567890
print(f"{big:,}")          # 1,234,567,890   Comma separator
print(f"{big:_}")          # 1_234_567_890   Underscore separator


# ── SIGN CONTROL ─────────────────────────────────────────────
pos: int = 42
neg: int = -42

print(f"{pos:+d}")         # +42    Always show sign
print(f"{neg:+d}")         # -42
print(f"{pos: d}")         #  42    Space for positive, - for negative
print(f"{neg: d}")         # -42


# ── BASE CONVERSION ──────────────────────────────────────────
n = 255

print(f"{n:b}")            # 11111111        Binary
print(f"{n:#b}")           # 0b11111111      Binary with prefix
print(f"{n:o}")            # 377             Octal
print(f"{n:#o}")           # 0o377           Octal with prefix
print(f"{n:x}")            # ff              Hex lowercase
print(f"{n:#X}")           # 0XFF            Hex uppercase with prefix
print(f"{n:08b}")          # 11111111        Binary, zero-padded to 8


# ── DATE FORMATTING ──────────────────────────────────────────
from datetime import datetime

now: datetime = datetime(2025, 6, 15, 14, 30, 0)
print(f"{now:%Y-%m-%d}")              # 2025-06-15
print(f"{now:%B %d, %Y at %I:%M %p}")  # June 15, 2025 at 02:30 PM
print(f"{now:%A}")                     # Sunday
```

**Building formatted tables:**

```python
# Practical example: aligned table output
students: list[tuple[str, int, float]] = [
    ("Alice Johnson", 95, 3.92),
    ("Bob Smith", 87, 3.45),
    ("Charlie Brown", 91, 3.78),
    ("Diana Prince", 99, 4.00),
]

# Header
print(f"{'Name':<20s} {'Score':>6s} {'GPA':>6s}")
print("-" * 34)

# Rows
for name, score, gpa in students:
    print(f"{name:<20s} {score:>6d} {gpa:>6.2f}")

# Output:
# Name                  Score    GPA
# ----------------------------------
# Alice Johnson             95   3.92
# Bob Smith                 87   3.45
# Charlie Brown             91   3.78
# Diana Prince              99   4.00
```

### 🔴 Expert: Performance Comparison and Bytecode

```python
import dis

# f-strings compile to FORMAT_VALUE/BUILD_STRING opcodes
def fstring_example(name: str, age: int) -> str:
    return f"Name: {name}, Age: {age}"

dis.dis(fstring_example)
# LOAD_CONST     "Name: "
# LOAD_FAST      name
# FORMAT_VALUE   0          ← converts to str inline
# LOAD_CONST     ", Age: "
# LOAD_FAST      age
# FORMAT_VALUE   0
# BUILD_STRING   4          ← concatenates all parts at once
# RETURN_VALUE

# .format() compiles to a method call — slower
def format_example(name: str, age: int) -> str:
    return "Name: {}, Age: {}".format(name, age)

dis.dis(format_example)
# LOAD_CONST     "Name: {}, Age: {}"
# LOAD_ATTR      format     ← attribute lookup
# LOAD_FAST      name
# LOAD_FAST      age
# CALL_FUNCTION  2          ← function call overhead
# RETURN_VALUE
```

**Benchmarking the three methods:**

```python
import timeit

setup: str = "name = 'Alice'; age = 30; gpa = 3.856"

# f-string
t1: float = timeit.timeit(
    "f'Name: {name}, Age: {age}, GPA: {gpa:.2f}'",
    setup=setup, number=1_000_000
)

# .format()
t2: float = timeit.timeit(
    "'Name: {}, Age: {}, GPA: {:.2f}'.format(name, age, gpa)",
    setup=setup, number=1_000_000
)

# % formatting
t3: float = timeit.timeit(
    "'Name: %s, Age: %d, GPA: %.2f' % (name, age, gpa)",
    setup=setup, number=1_000_000
)
```

```
Typical results (CPython 3.12, relative to f-string):

    Method              Time (relative)     Why
    ─────────────────────────────────────────────────
    f-string            1.0x (fastest)      Compiled to opcodes, no function call
    % formatting        ~1.3x               C implementation, but still a call
    .format()           ~1.5x               Python method call + parsing

    RULE: f-strings are fastest AND most readable. Use them by default.
    Exception: logging uses % for lazy evaluation.
    Exception: templates defined separately need .format() or Template.
```

**`string.Template` — safe formatting for user-supplied templates:**

```python
from string import Template

# .format() and f-strings are DANGEROUS with untrusted templates:
evil_template: str = "{.__class__.__mro__}"  # Can leak object internals
# evil_template.format(42)  → reveals class hierarchy!

# Template uses $ substitution and is safe with untrusted input
tmpl = Template("Hello, $name! You have $$${amount} in your account.")
result: str = tmpl.substitute(name="Alice", amount="1000")
print(result)  # Hello, Alice! You have $1000 in your account.

# safe_substitute doesn't raise on missing keys
result = tmpl.safe_substitute(name="Alice")
print(result)  # Hello, Alice! You have $$${amount} in your account.

# Template only allows $identifier and ${identifier} — no attribute access,
# no method calls, no arbitrary expressions. Safe for user-provided formats.
```

---

## 4.3 Advanced f-string Tricks

### 🟢 Beginner: Expressions Inside f-strings

```python
# f-strings evaluate ANY valid Python expression
import math

radius: float = 5.0
print(f"Area: {math.pi * radius ** 2:.2f}")         # Area: 78.54
print(f"Circumference: {2 * math.pi * radius:.2f}")  # Circumference: 31.42

# Conditional expressions
score: int = 85
print(f"Grade: {'Pass' if score >= 60 else 'Fail'}")  # Grade: Pass

# Method calls
name: str = "hello world"
print(f"Title: {name.title()}")    # Title: Hello World
print(f"Upper: {name.upper()}")    # Upper: HELLO WORLD

# List indexing and dictionary access
colors: list[str] = ["red", "green", "blue"]
print(f"First color: {colors[0]}")    # First color: red

config: dict[str, int] = {"port": 8080}
print(f"Port: {config['port']}")      # Port: 8080
# Note: use different quotes inside the braces
```

### 🟡 Intermediate: The Debug Specifier and Nesting

**The `=` specifier (Python 3.8+) — instant debug printing:**

```python
# The = specifier shows the expression AND its value
x: int = 42
y: float = 3.14
name: str = "Alice"

print(f"{x = }")                    # x = 42
print(f"{y = :.1f}")               # y = 3.1
print(f"{name = !r}")              # name = 'Alice'  (repr)
print(f"{x + y = }")               # x + y = 45.14
print(f"{name.upper() = }")        # name.upper() = 'HELLO'
print(f"{len(name) = }")           # len(name) = 5

# Spaces around = are preserved
print(f"{x=}")                      # x=42
print(f"{x =}")                     # x =42
print(f"{x = }")                    # x = 42

# Incredibly useful for debugging
items: list[int] = [3, 1, 4, 1, 5]
print(f"{sorted(items) = }")        # sorted(items) = [1, 1, 3, 4, 5]
print(f"{sum(items)/len(items) = :.2f}")  # sum(items)/len(items) = 2.80
```

**Nested f-strings (Python 3.12+ made this cleaner):**

```python
# You can nest f-strings for dynamic formatting
width: int = 15
value: str = "centered"
print(f"{value:^{width}}")          #    centered

# Dynamic precision
data: list[tuple[str, float, int]] = [
    ("pi", 3.14159265, 2),
    ("e", 2.71828182, 4),
    ("phi", 1.61803398, 6),
]

for name, val, prec in data:
    print(f"  {name}: {val:.{prec}f}")
# pi: 3.14
# e: 2.7183
# phi: 1.618034
```

**Conversion flags: `!s`, `!r`, `!a`:**

```python
class User:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"User({self.name!r})"

u = User("Alice")

print(f"{u}")       # Alice           — calls __str__
print(f"{u!s}")     # Alice           — explicitly calls __str__
print(f"{u!r}")     # User('Alice')   — calls __repr__
print(f"{u!a}")     # User('Alice')   — calls ascii() (escapes non-ASCII)

# !r is invaluable for debugging — shows the actual representation
path: str = "hello\tworld\n"
print(f"Path: {path}")       # Path: hello	world    (tab/newline rendered)
print(f"Path: {path!r}")     # Path: 'hello\tworld\n' (escaped)
```

**Multi-line f-strings:**

```python
from datetime import datetime

user: str = "Alice"
role: str = "Admin"
last_login: datetime = datetime(2025, 6, 15, 14, 30)

# Using parentheses for implicit concatenation (preferred)
message: str = (
    f"User Report\n"
    f"{'='*40}\n"
    f"  Name:       {user}\n"
    f"  Role:       {role}\n"
    f"  Last Login: {last_login:%Y-%m-%d %H:%M}\n"
    f"{'='*40}"
)
print(message)

# Using triple-quoted f-string (watch indentation!)
message = f"""
User Report
{'='*40}
  Name:       {user}
  Role:       {role}
  Last Login: {last_login:%Y-%m-%d %H:%M}
{'='*40}
""".strip()
```

### 🔴 Expert: f-string Compilation Internals

**f-strings are compiled at parse time, not runtime:**

```python
import ast

# The AST shows f-strings as JoinedStr nodes
source: str = "f'Hello, {name}!'"
tree = ast.parse(source, mode="eval")
print(ast.dump(tree, indent=2))
# Expression(
#   body=JoinedStr(
#     values=[
#       Constant(value='Hello, '),
#       FormattedValue(
#         value=Name(id='name', ctx=Load()),
#         conversion=-1,          # -1=none, 115=!s, 114=!r, 97=!a
#         format_spec=None
#       ),
#       Constant(value='!')
#     ]
#   )
# )
```

**f-string security — expression injection:**

```python
# f-strings evaluate expressions at the CALL SITE — they're not injectable
# in the way .format() templates are. The expression is baked into bytecode.

# But .format_map() with user data CAN be dangerous:
class MaliciousDict(dict):
    def __getitem__(self, key: str):
        if key == "__class__":
            return "HACKED"
        return super().__getitem__(key)

# safe — f-string expression is compiled into bytecode
name: str = "Alice"
print(f"Hello {name}")  # Can't inject; expression is fixed at compile time

# dangerous — .format() with untrusted template strings
user_template: str = "{0.__class__.__bases__[0]}"  # Accesses object internals
# print(user_template.format(42))  → <class 'object'>
# An attacker could explore the entire class hierarchy!

# RULE: Never use .format() with user-supplied template strings.
# Use string.Template or sanitize the template.
```

**The `__format__` protocol — how objects control their formatting:**

```python
class Temperature:
    """A temperature value that formats in C, F, or K."""

    def __init__(self, celsius: float) -> None:
        self.celsius = celsius

    def __format__(self, spec: str) -> str:
        """
        Format specs:
            'c' or '' → Celsius
            'f'       → Fahrenheit
            'k'       → Kelvin
        Precision can be prepended: '.2c', '.1f'
        """
        # Parse precision from spec
        precision: int = 1
        unit: str = "c"

        if spec:
            if spec[-1] in "cfk":
                unit = spec[-1]
                spec = spec[:-1]
            if spec.startswith("."):
                precision = int(spec[1:])

        if unit == "c":
            return f"{self.celsius:.{precision}f}°C"
        elif unit == "f":
            fahrenheit: float = self.celsius * 9 / 5 + 32
            return f"{fahrenheit:.{precision}f}°F"
        elif unit == "k":
            kelvin: float = self.celsius + 273.15
            return f"{kelvin:.{precision}f}K"
        else:
            raise ValueError(f"Unknown format spec: {spec!r}")

temp = Temperature(100)
print(f"Boiling: {temp}")          # Boiling: 100.0°C
print(f"Boiling: {temp:c}")        # Boiling: 100.0°C
print(f"Boiling: {temp:.2f}")      # Boiling: 212.00°F
print(f"Boiling: {temp:.3k}")      # Boiling: 373.150K
```

---

## 4.4 Buffering, Flushing, and Real-Time Output

### 🟢 Beginner: Why Output Sometimes Appears Late

```python
import time

# This might NOT show progress in real-time:
for i in range(5):
    print(f"Step {i + 1}/5...", end="")
    time.sleep(1)
print(" Done!")
# Problem: All "Step" messages may appear at once after 5 seconds

# Fix: flush the output buffer
for i in range(5):
    print(f"\rStep {i + 1}/5...", end="", flush=True)
    time.sleep(1)
print(" Done!")
# Now each step appears in real-time
```

### 🟡 Intermediate: Understanding the Buffer Hierarchy

```python
import sys

# Python has THREE buffering modes:
# 1. Unbuffered: every write() goes to the OS immediately
# 2. Line-buffered: flushes on every newline (default for terminal stdout)
# 3. Fully-buffered: flushes when buffer is full (default for pipes/files)

# Check current buffering
if sys.stdout.isatty():
    print("Connected to terminal: LINE buffered")
    # Each print() with \n flushes automatically
    # But print(..., end="") does NOT flush!
else:
    print("Piped/redirected: FULLY buffered (8KB default)")
    # Output accumulates until 8192 bytes, then flushes all at once
```

**Three ways to force unbuffered output:**

```python
# Method 1: flush=True on each print call
print("Immediate!", flush=True)

# Method 2: Run Python with -u flag (unbuffered stdin/stdout/stderr)
# python -u my_script.py

# Method 3: Set PYTHONUNBUFFERED environment variable
# export PYTHONUNBUFFERED=1

# Method 4: Reconfigure stdout at runtime (Python 3.7+)
import sys
sys.stdout.reconfigure(line_buffering=True)
# Now stdout is line-buffered even when piped
```

**Building a progress bar with buffering awareness:**

```python
import time
import sys
import shutil


def progress_bar(
    current: int,
    total: int,
    width: int | None = None,
    prefix: str = "Progress",
) -> None:
    """Display a progress bar with percentage."""
    if width is None:
        terminal_width: int = shutil.get_terminal_size().columns
        width = min(50, terminal_width - len(prefix) - 15)

    fraction: float = current / total
    filled: int = int(width * fraction)
    bar: str = "█" * filled + "░" * (width - filled)
    percent: float = fraction * 100

    sys.stdout.write(f"\r{prefix} |{bar}| {percent:5.1f}%")
    sys.stdout.flush()

    if current == total:
        sys.stdout.write("\n")


# Usage
total: int = 50
for i in range(total + 1):
    progress_bar(i, total)
    time.sleep(0.05)
```

### 🔴 Expert: File Descriptor Buffering and the Kernel

```
┌──────────────────────────────────────────────────────────────────┐
│                    BUFFERING LAYERS                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Python Layer:                                                    │
│  ┌─────────────┐                                                 │
│  │ TextIO      │ ← Characters, encoding, newline translation     │
│  │ (8KB buf)   │                                                 │
│  └──────┬──────┘                                                 │
│         ▼                                                         │
│  ┌─────────────┐                                                 │
│  │ BufferedIO   │ ← Bytes, coalesces small writes                │
│  │ (8KB buf)   │                                                 │
│  └──────┬──────┘                                                 │
│         ▼                                                         │
│  ┌─────────────┐                                                 │
│  │ RawIO       │ ← Direct write() syscall                       │
│  │ (unbuffered)│                                                 │
│  └──────┬──────┘                                                 │
│         ▼                                                         │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─  Kernel boundary  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│         ▼                                                         │
│  ┌─────────────┐                                                 │
│  │ Kernel buf  │ ← OS-level pipe/socket buffer                   │
│  │ (64KB typ.) │                                                 │
│  └──────┬──────┘                                                 │
│         ▼                                                         │
│  Terminal / File / Pipe / Socket                                  │
│                                                                   │
│  A single print("x") might pass through FOUR buffers             │
│  before the user sees it!                                         │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Controlling every layer:**

```python
import io
import os
import sys

# Layer 1: Open with explicit buffering
# buffering=0  → unbuffered (binary mode only)
# buffering=1  → line buffered (text mode only)
# buffering=N  → N-byte buffer
f = open("output.txt", "w", buffering=1)    # Line buffered
f = open("output.bin", "wb", buffering=0)   # Unbuffered binary

# Layer 2: Bypass Python buffering entirely with os.write()
os.write(1, b"Direct to fd 1 (stdout)\n")   # No Python buffering at all

# Layer 3: Sync to disk (bypass kernel buffer too)
f = open("critical.log", "w")
f.write("Important data\n")
f.flush()            # Python buffer → kernel buffer
os.fsync(f.fileno())  # Kernel buffer → physical disk
```

**The `stderr` exception — why it's always unbuffered:**

```python
import sys

# stderr is ALWAYS unbuffered (line_buffering=True when attached to TTY,
# or write_through=True internally)
# This ensures error messages appear IMMEDIATELY, even during crashes

print("This may be buffered", file=sys.stdout)
print("This is immediate", file=sys.stderr)

# That's why you see error tracebacks instantly, even if stdout output
# is delayed. Python deliberately made this choice for debuggability.

# You can verify:
print(f"stdout line_buffering: {sys.stdout.line_buffering}")
print(f"stderr line_buffering: {sys.stderr.line_buffering}")
```

---

## 🔧 Debug This: The Progress Bar That Doesn't Progress

Your colleague wrote a download progress reporter. It has four bugs. The user reports "It just sits there for 30 seconds, then shows 100% and exits."

```python
import time
import sys

def download_file(url, size_mb=100):
    """Simulate downloading a file with progress reporting."""
    downloaded = 0
    chunk_size = 10  # MB per chunk

    while downloaded < size_mb:
        # Simulate network delay
        time.sleep(0.3)
        downloaded += chunk_size

        # Calculate progress
        percent = downloaded / size_mb * 100

        # Display progress
        bar_length = 30
        filled = int(bar_length * percent)
        bar = "=" * filled + "-" * (bar_length - filled)
        print(f"\rDownloading: [{bar}] {percent}%", end="")

    print(f"\nDownload complete: {url}")

# Run it
download_file("https://example.com/large-file.zip")
```

### Bugs to find:

```
1. ____________________________________________________
   Hint: What happens when percent is, say, 50? What's
   int(30 * 50)? The bar_length calculation is wrong.

2. ____________________________________________________
   Hint: The progress output uses end="". What's missing
   for real-time display?

3. ____________________________________________________
   Hint: What if size_mb is not evenly divisible by chunk_size?
   What if size_mb=95? When does the loop end?

4. ____________________________________________________
   Hint: percent can exceed 100 if downloaded > size_mb.
   Is the output clamped?

5. ____________________________________________________
   Hint: Type hints, docstring, and the percent formatting
   is showing raw floats like 33.33333333.
```

### Solution (try first!)

```python
import sys
import time


def download_file(
    url: str,
    size_mb: float = 100.0,
    chunk_size: float = 10.0,
) -> None:
    """Simulate downloading a file with a real-time progress bar.

    Args:
        url: The URL being downloaded (for display).
        size_mb: Total file size in megabytes.
        chunk_size: Size of each download chunk in megabytes.
    """
    downloaded: float = 0.0
    bar_length: int = 30

    while downloaded < size_mb:
        time.sleep(0.3)
        downloaded += chunk_size

        # Bug 3 & 4 FIX: Clamp downloaded to size_mb
        downloaded = min(downloaded, size_mb)

        # Bug 1 FIX: percent was 0-100, but we need 0.0-1.0 for bar fill
        fraction: float = downloaded / size_mb
        percent: float = fraction * 100

        filled: int = int(bar_length * fraction)  # ← fraction, not percent!
        bar: str = "█" * filled + "░" * (bar_length - filled)

        # Bug 2 FIX: Add flush=True for real-time output
        # Bug 5 FIX: Format percent to 1 decimal place
        print(
            f"\rDownloading: [{bar}] {percent:5.1f}%",
            end="",
            flush=True,  # ← THIS is why it appeared all at once!
        )

    print(f"\nDownload complete: {url}")


download_file("https://example.com/large-file.zip")
```

```
Bug Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Bar calculation:  int(30 * 50) = 1500, not 15.
   Used percent (0-100) instead of fraction (0.0-1.0).
   Result: filled overflows bar_length → garbled output.

2. Missing flush:    end="" suppresses newline, which means
   line-buffered stdout never flushes. All output appears
   at the end when print("\n") finally triggers a flush.

3. Overshoot:        If size_mb=95, downloaded goes
   90 → 100, and loop continues because 100 < 95 is False...
   wait, 100 > 95, so it exits. But percent shows 105.3%.

4. No clamping:      downloaded can exceed size_mb, giving
   percent > 100 and filled > bar_length.

5. Missing types:    No type hints, no docstring for params,
   and percent displays as 33.33333333333... instead of 33.3%.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Summary: Module 4 Key Takeaways

```
┌──────────────────────────────────────────────────────────────────┐
│                    CONSOLE I/O CHEAT SHEET                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  print():    Writes to sys.stdout. Use sep, end, flush params.   │
│  input():    Always returns str. Convert types explicitly.       │
│  Streams:    sys.stdin, sys.stdout, sys.stderr are file objects. │
│                                                                   │
│  FORMATTING (use f-strings by default):                          │
│    f"{value:>10.2f}"  → right-align, width 10, 2 decimal places │
│    f"{value:,}"       → comma thousands separator                │
│    f"{value:#x}"      → hex with 0x prefix                      │
│    f"{value = }"      → debug: shows "value = 42"               │
│    f"{value!r}"       → calls repr() instead of str()           │
│                                                                   │
│  WHEN TO USE EACH:                                               │
│    f-strings    → everyday formatting (fastest, most readable)   │
│    .format()    → templates defined separately from data         │
│    % formatting → logging module (lazy evaluation)               │
│    Template     → user-supplied format strings (safe)            │
│                                                                   │
│  BUFFERING:                                                      │
│    Terminal stdout  → line buffered (flushes on \n)              │
│    Piped stdout     → fully buffered (flushes at 8KB)            │
│    stderr           → always unbuffered (immediate)              │
│    Fix: flush=True, python -u, or PYTHONUNBUFFERED=1            │
│                                                                   │
│  Production rule: Always use flush=True with end="" in loops.   │
│  Never use .format() with untrusted template strings.            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Next up → Module 5: The "Big Four" Containers (Lists, Tuples, Sets, Dicts)**

Say "Start Module 5" when you're ready.
