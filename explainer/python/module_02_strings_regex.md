# Module 2 — Strings & RegEx: Text as a First-Class Citizen

> *"I had a problem, so I used a regex. Now I have two problems."*
> — Jamie Zawinski (paraphrased)

---

## 2.1 String Internals: Immutability, Interning, and Unicode

### 🟢 Beginner: Strings Are Sequences of Characters

A string in Python is an ordered, immutable sequence of Unicode characters.

```python
# Creating strings — four equivalent ways
single: str = 'hello'
double: str = "hello"
triple_single: str = '''hello'''
triple_double: str = """hello"""

# Triple-quoted strings preserve newlines
poem: str = """Roses are red,
Violets are blue,
Python is great,
And so are you."""

# Strings are sequences — you can index and iterate
name: str = "Python"
print(name[0])     # 'P'   — zero-indexed
print(name[-1])    # 'n'   — negative indexing from the end
print(len(name))   # 6

for char in name:
    print(char, end=" ")   # P y t h o n
```

**Immutability — strings cannot be changed in place:**

```python
greeting: str = "Hello"
# greeting[0] = "J"    # TypeError: 'str' object does not support item assignment

# Instead, create a NEW string
greeting = "J" + greeting[1:]   # "Jello"
# The original "Hello" object is now unreferenced and will be garbage collected
```

**Essential string methods (the ones you'll use daily):**

```python
text: str = "  Hello, World!  "

# Searching
print(text.find("World"))       # 9  (index of first occurrence, -1 if not found)
print(text.index("World"))      # 9  (same but raises ValueError if not found)
print("World" in text)          # True  (membership test — preferred)
print(text.count("l"))          # 3

# Transforming (always returns a NEW string)
print(text.strip())             # "Hello, World!"      (remove whitespace)
print(text.lstrip())            # "Hello, World!  "    (left strip only)
print(text.upper())             # "  HELLO, WORLD!  "
print(text.lower())             # "  hello, world!  "
print(text.title())             # "  Hello, World!  "
print(text.replace("World", "Python"))  # "  Hello, Python!  "

# Splitting and joining
csv_line: str = "apple,banana,cherry"
fruits: list[str] = csv_line.split(",")     # ['apple', 'banana', 'cherry']
rejoined: str = " | ".join(fruits)          # 'apple | banana | cherry'

# Testing
print("hello123".isalnum())     # True  (letters and digits only)
print("hello".isalpha())        # True  (letters only)
print("12345".isdigit())        # True  (digits only)
print("hello".startswith("hel"))  # True
print("hello".endswith("llo"))    # True
```

### 🟡 Intermediate: Unicode, Encoding, and the Pain of Text

**Python 3 strings are Unicode by default.** Every character is a Unicode code point.

```python
# Unicode is everywhere
emoji: str = "Hello 🌍"
print(len(emoji))         # 8 — each emoji is ONE character (one code point)

chinese: str = "你好世界"
print(len(chinese))       # 4 — four characters

# Accessing code points
print(ord("A"))           # 65
print(ord("🌍"))          # 127757
print(chr(65))            # 'A'
print(chr(127757))        # '🌍'

# Unicode escape sequences
print("\u0041")           # 'A'      (4-digit hex)
print("\U0001F30D")       # '🌍'     (8-digit hex for chars above U+FFFF)
print("\N{EARTH GLOBE EUROPE-AFRICA}")  # '🌍'  (by name)
```

**Encoding vs. Decoding — the critical distinction:**

```
str (Unicode)  ──encode()──►  bytes (raw data)
bytes (raw data)  ──decode()──►  str (Unicode)
```

```python
# Encoding: str → bytes
text: str = "Café"
utf8_bytes: bytes = text.encode("utf-8")      # b'Caf\xc3\xa9'  (5 bytes)
latin1_bytes: bytes = text.encode("latin-1")   # b'Caf\xe9'      (4 bytes)
ascii_bytes: bytes = text.encode("ascii", errors="replace")  # b'Caf?'

print(len(text))          # 4 characters
print(len(utf8_bytes))    # 5 bytes — 'é' takes 2 bytes in UTF-8
print(len(latin1_bytes))  # 4 bytes — 'é' takes 1 byte in Latin-1

# Decoding: bytes → str
raw: bytes = b'Caf\xc3\xa9'
decoded: str = raw.decode("utf-8")    # "Café"
# wrong_decode = raw.decode("latin-1")  # "CafÃ©" — mojibake! Wrong encoding.
```

**Gotcha: `len()` counts characters, not bytes or display width:**

```python
# Some "characters" are actually combining sequences
e_acute_1: str = "é"        # U+00E9 (precomposed)
e_acute_2: str = "e\u0301"  # U+0065 + U+0301 (decomposed: e + combining accent)

print(len(e_acute_1))  # 1
print(len(e_acute_2))  # 2  — looks the same, different length!
print(e_acute_1 == e_acute_2)  # False!

# Fix: Normalize before comparing
import unicodedata
norm_1: str = unicodedata.normalize("NFC", e_acute_1)
norm_2: str = unicodedata.normalize("NFC", e_acute_2)
print(norm_1 == norm_2)  # True
```

**String interning — when Python reuses string objects:**

```python
# CPython interns strings that look like identifiers
a: str = "hello"
b: str = "hello"
print(a is b)  # True — same object, interned

# Strings with spaces or special chars are NOT auto-interned
a = "hello world"
b = "hello world"
print(a is b)  # Depends on context (may be True in REPL, False elsewhere)

# RULE: Never rely on 'is' for string comparison. Always use '=='
```

### 🔴 Expert: CPython's Compact Unicode Representation (PEP 393)

Before Python 3.3, every character was stored as either 2 or 4 bytes (UCS-2 or UCS-4, a compile-time choice). Since PEP 393, CPython uses a **flexible string representation** that adapts to the content:

```
┌─────────────────────────────────────────────────────────┐
│              CPython String Storage (PEP 393)            │
├──────────────┬──────────────────────────────────────────┤
│  Kind        │  Bytes/Char  │  Max Code Point           │
├──────────────┼──────────────┼───────────────────────────┤
│  Latin-1     │     1        │  U+00FF  (255)            │
│  UCS-2       │     2        │  U+FFFF  (65535)          │
│  UCS-4       │     4        │  U+10FFFF (1,114,111)     │
└──────────────┴──────────────┴───────────────────────────┘
```

```python
import sys

# Latin-1 only: 1 byte per character
ascii_str: str = "hello"
print(sys.getsizeof(ascii_str))   # 54 bytes (49 header + 5 × 1 byte + null)

# Has a character > U+FF: upgrades to UCS-2 (2 bytes each)
chinese: str = "你好"
print(sys.getsizeof(chinese))     # 76 bytes (header + 2 × 2 bytes)

# Has emoji > U+FFFF: upgrades to UCS-4 (4 bytes each)
emoji: str = "hi🌍"
print(sys.getsizeof(emoji))       # 64 bytes (header + 3 × 4 bytes)

# The ENTIRE string upgrades — one emoji forces UCS-4 for all chars
mixed: str = "a" * 1000 + "🌍"
# All 1001 characters stored as 4 bytes each = 4004 bytes of char data
```

**Memory layout of a CPython string object:**

```
┌──────────────────────────────────────────────────┐
│               PyUnicodeObject                     │
├──────────────────────────────────────────────────┤
│  ob_refcnt      (8 bytes)  reference count        │
│  ob_type        (8 bytes)  → str type object      │
│  length         (8 bytes)  number of code points  │
│  hash           (8 bytes)  cached hash value      │
│  state          (4 bytes)  kind, interned, etc.   │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
│  data[]         (variable) the actual characters  │
│                  1, 2, or 4 bytes per character    │
│  \0             (1 char)   null terminator         │
└──────────────────────────────────────────────────┘
```

**String concatenation performance — why `+` in a loop is O(n²):**

```python
import time

def concat_with_plus(n: int) -> str:
    """O(n²) — each + creates a new string and copies all previous content."""
    result: str = ""
    for i in range(n):
        result += str(i)  # Copy entire result + new piece each time
    return result

def concat_with_join(n: int) -> str:
    """O(n) — join() allocates once and fills."""
    parts: list[str] = []
    for i in range(n):
        parts.append(str(i))
    return "".join(parts)  # Single allocation

def concat_with_comprehension(n: int) -> str:
    """O(n) — same as join but more Pythonic."""
    return "".join(str(i) for i in range(n))

# CPython has an optimization for += on strings with refcount 1
# (it reallocs in place), but DON'T rely on it — it's an implementation
# detail and doesn't work when another reference exists.
```

```
Concatenation performance for n = 100,000:

    Method              Time        Memory Pattern
    ─────────────────────────────────────────────────
    + in loop           ~0.5s       Repeated alloc+copy (O(n²))
    "".join(list)       ~0.02s      Single alloc (O(n))
    f-string            N/A         Best for small, fixed concatenations

    RULE: Use "".join() for building strings in loops.
          Use f-strings for inline formatting.
```

---

## 2.2 Slicing Mechanics and the Stride Trick

### 🟢 Beginner: The Slice Syntax

```python
#         +---+---+---+---+---+---+
#         | P | y | t | h | o | n |
#         +---+---+---+---+---+---+
# Index:    0   1   2   3   4   5
# Neg:     -6  -5  -4  -3  -2  -1

text: str = "Python"

# Basic slicing: [start:stop]  (stop is EXCLUSIVE)
print(text[0:2])    # "Py"      — characters at index 0 and 1
print(text[2:])     # "thon"    — from index 2 to end
print(text[:3])     # "Pyt"     — from start to index 2
print(text[:])      # "Python"  — full copy

# Negative indexing
print(text[-3:])    # "hon"     — last 3 characters
print(text[:-2])    # "Pyth"    — everything except last 2
print(text[-4:-1])  # "tho"     — from -4 up to (not including) -1
```

**The slice never raises IndexError:**

```python
text: str = "Hi"
print(text[0:100])  # "Hi"    — silently clamps to string length
print(text[50:100]) # ""      — empty string, no error
print(text[-100:1]) # "H"     — silently clamps negative too

# But direct indexing DOES raise:
# text[50]  → IndexError
```

### 🟡 Intermediate: The Stride (Step) Parameter

```python
# Full syntax: [start:stop:step]
numbers: str = "0123456789"

# Every other character
print(numbers[::2])     # "02468"

# Every third character
print(numbers[::3])     # "0369"

# Reverse a string
print(numbers[::-1])    # "9876543210"

# Reverse with stride
print(numbers[::-2])    # "97531"

# Start and stop with stride
print(numbers[1:8:2])   # "1357"
```

**Slice objects — slices are first-class objects:**

```python
# You can create reusable slice objects
first_three: slice = slice(0, 3)
last_two: slice = slice(-2, None)

data: str = "abcdefgh"
print(data[first_three])  # "abc"
print(data[last_two])     # "gh"

# This is how NumPy and Pandas use slicing under the hood
# The slice object has .start, .stop, .step attributes
s: slice = slice(1, 10, 2)
print(s.indices(6))  # (1, 6, 2) — adjusted for a sequence of length 6
```

**Gotcha: Slicing creates copies for strings and lists, but NOT for NumPy arrays:**

```python
# Strings: always a copy (strings are immutable anyway)
original: str = "hello"
sliced: str = original[:3]  # New string "hel"

# Lists: also a copy (shallow)
original_list: list[int] = [1, 2, 3, 4, 5]
sliced_list: list[int] = original_list[1:4]
sliced_list[0] = 99
print(original_list)  # [1, 2, 3, 4, 5] — unchanged

# NumPy (will matter later): slices are VIEWS, not copies!
# import numpy as np
# arr = np.array([1, 2, 3, 4, 5])
# view = arr[1:4]
# view[0] = 99
# print(arr)  # [1, 99, 3, 4, 5] — original changed!
```

### 🔴 Expert: How Slicing Works Under the Hood

When you write `text[1:5]`, Python calls `text.__getitem__(slice(1, 5, None))`.

```python
# You can implement slicing in your own classes
class MySequence:
    def __init__(self, data: str) -> None:
        self._data = data

    def __getitem__(self, key: int | slice) -> str:
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self._data))
            print(f"Slice requested: start={start}, stop={stop}, step={step}")
            return self._data[key]
        elif isinstance(key, int):
            print(f"Index requested: {key}")
            return self._data[key]
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

seq = MySequence("Hello, World!")
print(seq[7:12])    # Slice requested: start=7, stop=12, step=1 → "World"
print(seq[::2])     # Slice requested: start=0, stop=13, step=2 → "Hlo ol!"
```

**CPython string slicing memory behavior:**

```python
import sys

# CPython may or may not share memory for slices
# Unlike some languages (Java pre-7u6, Go), Python string slices
# are always independent copies. This prevents memory leaks where
# a small slice keeps a huge string alive.

big: str = "x" * 1_000_000
small: str = big[:10]

# 'small' does NOT hold a reference to 'big'
del big
# The 1MB string is now eligible for garbage collection
# 'small' is an independent 10-character string
```

---

## 2.3 The `re` Engine: From `match` vs. `search` to Compiled Patterns

### 🟢 Beginner: Your First Regular Expression

A regular expression (regex) is a pattern that describes a set of strings.

```python
import re

text: str = "My phone number is 555-1234 and my zip is 90210."

# re.search() — find the FIRST match anywhere in the string
match = re.search(r"\d+", text)  # \d+ = one or more digits
if match:
    print(match.group())   # "555"
    print(match.start())   # 19 (index where match begins)
    print(match.end())     # 22 (index where match ends)
    print(match.span())    # (19, 22)

# re.findall() — find ALL matches, return as list of strings
all_numbers: list[str] = re.findall(r"\d+", text)
print(all_numbers)  # ['555', '1234', '90210']

# re.sub() — search and replace
cleaned: str = re.sub(r"\d+", "XXX", text)
print(cleaned)  # "My phone number is XXX-XXX and my zip is XXX."
```

**The essential pattern vocabulary:**

```
PATTERN    MEANING                         EXAMPLE
──────────────────────────────────────────────────────────
.          Any character except newline     a.c → "abc", "a1c"
\d         Digit [0-9]                      \d\d → "42"
\D         Non-digit [^0-9]                 \D+ → "hello"
\w         Word char [a-zA-Z0-9_]           \w+ → "hello_42"
\W         Non-word char                    \W → "!", " "
\s         Whitespace [ \t\n\r\f\v]         \s+ → "   "
\S         Non-whitespace                   \S+ → "hello"
\b         Word boundary                    \bcat\b → "cat" not "scatter"

^          Start of string                  ^Hello
$          End of string                    world$
*          0 or more                        ab* → "a", "ab", "abbb"
+          1 or more                        ab+ → "ab", "abbb"
?          0 or 1                           colou?r → "color", "colour"
{n}        Exactly n                        \d{3} → "123"
{n,m}      Between n and m                  \d{2,4} → "12", "1234"
[abc]      Character class                  [aeiou] → any vowel
[^abc]     Negated class                    [^0-9] → non-digit
(...)      Capture group                    (\d+)-(\d+) → groups
|          Alternation (OR)                 cat|dog
```

**Always use raw strings (`r"..."`) for regex patterns:**

```python
# Without raw string, backslashes get interpreted by Python first
print("\n")    # Actual newline character
print(r"\n")   # Literal backslash + n

# This matters for regex
re.search("\d+", "123")    # Works, but Python sees "\d" as unknown escape
re.search(r"\d+", "123")   # Correct — raw string preserves backslash
```

### 🟡 Intermediate: `match` vs. `search` vs. `fullmatch` and Groups

**The three search functions — know when to use each:**

```python
import re

text: str = "Error 404: Page not found"

# re.match() — anchored at the START of the string
print(re.match(r"\d+", text))          # None — "Error" is not digits
print(re.match(r"Error", text))        # Match! Starts with "Error"

# re.search() — finds first match ANYWHERE
print(re.search(r"\d+", text))         # Match: "404"

# re.fullmatch() — must match the ENTIRE string
print(re.fullmatch(r"\d+", "404"))     # Match: "404"
print(re.fullmatch(r"\d+", "404!"))    # None — "!" doesn't match
```

**Named groups and backreferences:**

```python
import re

# Capture groups with ()
pattern: str = r"(\d{3})-(\d{4})"
match = re.search(pattern, "Call 555-1234 now")
if match:
    print(match.group(0))  # "555-1234"  — entire match
    print(match.group(1))  # "555"       — first group
    print(match.group(2))  # "1234"      — second group
    print(match.groups())  # ('555', '1234')

# Named groups with (?P<name>...)
pattern = r"(?P<area>\d{3})-(?P<number>\d{4})"
match = re.search(pattern, "Call 555-1234 now")
if match:
    print(match.group("area"))     # "555"
    print(match.group("number"))   # "1234"
    print(match.groupdict())       # {'area': '555', 'number': '1234'}

# Backreferences — match the SAME text again
# Find repeated words
pattern = r"\b(\w+)\s+\1\b"  # \1 refers back to group 1
text = "the the quick brown fox fox"
print(re.findall(pattern, text))   # ['the', 'fox']
```

**Compiling patterns for reuse:**

```python
import re

# Compile once, use many times — faster when pattern is used repeatedly
EMAIL_PATTERN: re.Pattern[str] = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE
)

emails: list[str] = [
    "user@example.com",
    "bad@.com",
    "Test.User@Sub.Domain.org",
]

for email in emails:
    if EMAIL_PATTERN.fullmatch(email):
        print(f"  Valid: {email}")
    else:
        print(f"  Invalid: {email}")

# Flags can be combined with |
pattern: re.Pattern[str] = re.compile(
    r"hello world",
    re.IGNORECASE | re.MULTILINE
)
```

**`re.finditer()` — memory-efficient iteration over matches:**

```python
import re

# findall() returns a list (all in memory)
# finditer() returns an iterator (lazy, one at a time)
text: str = "Error at line 42, warning at line 87, error at line 103"
pattern: re.Pattern[str] = re.compile(r"(?:error|warning) at line (\d+)", re.IGNORECASE)

for match in pattern.finditer(text):
    print(f"  {match.group()} → line number: {match.group(1)}")
# error at line 42 → line number: 42
# warning at line 87 → line number: 87
# error at line 103 → line number: 103
```

### 🔴 Expert: The NFA Engine and Compiled Pattern Objects

Python's `re` module uses a **backtracking NFA (Nondeterministic Finite Automaton)** engine, which means certain patterns can have exponential worst-case time complexity.

```python
import re
import time

# Measure regex compilation cost
pattern_str: str = r"(?:(?:\d+\.){3}\d+)"  # IPv4-ish pattern

start: float = time.perf_counter()
for _ in range(10_000):
    re.search(pattern_str, "192.168.1.1")  # Recompiles each time (cached, but still)
uncompiled_time: float = time.perf_counter() - start

compiled: re.Pattern[str] = re.compile(pattern_str)
start = time.perf_counter()
for _ in range(10_000):
    compiled.search("192.168.1.1")  # No compilation overhead
compiled_time: float = time.perf_counter() - start

# re module caches the last 512 patterns (re._MAXCACHE)
# But compiled patterns skip the cache lookup entirely
print(f"  Uncompiled: {uncompiled_time:.4f}s")
print(f"  Compiled:   {compiled_time:.4f}s")
```

**Inspecting compiled pattern internals:**

```python
import re
import sre_parse
import sre_compile

# The compilation pipeline:
# 1. re.compile(pattern_string)
# 2. sre_parse.parse() → parse tree
# 3. sre_compile.compile() → bytecode
# 4. _sre.SRE_Pattern (C object)

pattern: str = r"(\d{3})-(\d{4})"

# Step 1: Parse tree
parsed = sre_parse.parse(pattern)
print(list(parsed))
# [(SUBPATTERN, (1, 0, 0, [(MAX_REPEAT, (3, 3, [(IN, [(CATEGORY, CATEGORY_DIGIT)])]))])),
#  (LITERAL, 45),  ← ord('-')
#  (SUBPATTERN, (2, 0, 0, [(MAX_REPEAT, (4, 4, [(IN, [(CATEGORY, CATEGORY_DIGIT)])]))]))]

# The compiled pattern has:
compiled: re.Pattern[str] = re.compile(pattern)
print(compiled.pattern)   # r'(\d{3})-(\d{4})' — original string
print(compiled.flags)     # 32 (re.UNICODE is default in Py3)
print(compiled.groups)    # 2 (number of capture groups)
print(compiled.groupindex)  # {} (named groups would appear here)
```

---

## 2.4 Advanced Regex: Lookaheads, Lookbehinds, and Backtracking Catastrophe

### 🟢 Beginner: Non-Capturing Groups

Sometimes you need grouping but don't need to capture:

```python
import re

# Capturing group — stored for later retrieval
match = re.search(r"(cat|dog) food", "I bought cat food")
if match:
    print(match.group(1))  # "cat"

# Non-capturing group (?:...) — groups without storing
# Useful when you only need the full match
matches: list[str] = re.findall(r"(?:cat|dog) food", "cat food and dog food")
print(matches)  # ['cat food', 'dog food']
# Without (?:), findall returns captured groups: ['cat', 'dog']
```

### 🟡 Intermediate: Lookaheads and Lookbehinds

Lookaheads and lookbehinds are **zero-width assertions** — they check a condition without consuming characters.

```python
import re

# (?=...)  Positive lookahead — "followed by"
# Match digits ONLY IF followed by "px"
text: str = "width: 100px; height: 200em; margin: 50px"
matches: list[str] = re.findall(r"\d+(?=px)", text)
print(matches)  # ['100', '50'] — 200 excluded (followed by 'em')

# (?!...)  Negative lookahead — "NOT followed by"
# Match digits NOT followed by "px"
matches = re.findall(r"\d+(?!px)", text)
print(matches)  # ['10', '200', '5'] — partial matches! (gotcha below)

# Better: use word boundaries
matches = re.findall(r"\b\d+(?!px)\b", text)
# Still tricky — the \b after digits doesn't prevent partial matches well
# More precise approach:
matches = re.findall(r"\d+(?=em\b)", text)
print(matches)  # ['200']
```

```python
import re

# (?<=...) Positive lookbehind — "preceded by"
# Extract prices (numbers preceded by $)
text: str = "Costs: $100, €200, $50"
matches: list[str] = re.findall(r"(?<=\$)\d+", text)
print(matches)  # ['100', '50'] — only dollar amounts

# (?<!...) Negative lookbehind — "NOT preceded by"
# Numbers NOT preceded by $
matches = re.findall(r"(?<!\$)\b\d+", text)
print(matches)  # ['200']
```

**LIMITATION: Lookbehinds must be fixed-width in Python's `re`:**

```python
import re

# This WORKS — fixed width lookbehind
re.findall(r"(?<=\$\$)\d+", "$$100")    # ['100']

# This FAILS — variable width lookbehind
# re.findall(r"(?<=\$+)\d+", "$100 $$200")
# re.error: look-behind requires fixed-width pattern

# Workaround: use the `regex` third-party module (supports variable lookbehinds)
# Or restructure with a capturing group:
matches: list[str] = re.findall(r"\$+(\d+)", "$100 $$200")
print(matches)  # ['100', '200']
```

**Password validation — combining multiple lookaheads:**

```python
import re

def validate_password(password: str) -> bool:
    """
    Password must:
    - Be at least 8 characters
    - Contain at least one uppercase letter
    - Contain at least one lowercase letter
    - Contain at least one digit
    - Contain at least one special character
    """
    pattern: str = (
        r"^"
        r"(?=.*[A-Z])"       # Lookahead: at least one uppercase
        r"(?=.*[a-z])"       # Lookahead: at least one lowercase
        r"(?=.*\d)"          # Lookahead: at least one digit
        r"(?=.*[!@#$%^&*])"  # Lookahead: at least one special char
        r".{8,}"             # Then match 8+ of any characters
        r"$"
    )
    return bool(re.match(pattern, password))

print(validate_password("Abc1!xyz"))    # True
print(validate_password("abc1!xyz"))    # False — no uppercase
print(validate_password("Short1!"))     # False — only 7 chars
```

### 🔴 Expert: Catastrophic Backtracking and ReDoS

The NFA engine tries all possible paths through the pattern. Certain patterns create **exponential** backtracking.

```python
import re
import time

# DANGEROUS PATTERN: nested quantifiers
evil_pattern: str = r"(a+)+"
# For input "aaaaaaaaaaaaaaaaab", the engine tries 2^n combinations
# because each 'a' could belong to any repetition of the outer group

# Demonstration (be careful — this can freeze your process)
for n in range(15, 26):
    test_input: str = "a" * n + "b"
    start: float = time.perf_counter()
    re.match(evil_pattern, test_input)
    elapsed: float = time.perf_counter() - start
    print(f"  n={n:2d}: {elapsed:.4f}s")
    if elapsed > 2.0:
        print("  Stopping — exponential growth detected!")
        break

# Output shows exponential growth:
#   n=15: 0.0010s
#   n=18: 0.0080s
#   n=20: 0.0310s
#   n=22: 0.1240s    ← doubling every +2
#   n=24: 0.4960s
#   n=25: 0.9920s
```

**Common catastrophic patterns and their fixes:**

```
DANGEROUS                    SAFE ALTERNATIVE              WHY
─────────────────────────────────────────────────────────────────
(a+)+                        a+                            Remove nesting
(a|a)+                       a+                            Remove ambiguity
(.*a){10}                    (?:[^a]*a){10}                Make inner match specific
(\w+\s*)+                    [\w\s]+                       Combine into one class
(a+b?)+                      Use atomic group / possessive Prevent backtracking
```

**Atomic grouping workaround (Python's `re` doesn't support `(?>...)`):**

```python
import re

# Python 3.11+ supports atomic groups: (?>...)
# For older versions, use the `regex` module or restructure

# Python 3.11+:
# pattern = r"(?>a+)b"  # Atomic: once a+ matches, it can't give back chars

# Workaround for older Python: make patterns unambiguous
# Instead of (a+)+, just use a+
# Instead of (\w+\.)+, use [\w.]+

# The re module also has a timeout-like safeguard: re.MAXREPEAT
# But the real fix is ALWAYS to write non-pathological patterns
```

**Real-world ReDoS (Regular expression Denial of Service) example:**

```python
import re

# Email-like pattern that's vulnerable
bad_email_pattern: str = r"^([a-zA-Z0-9]+\.)+[a-zA-Z]{2,}$"

# Normal input: fast
re.match(bad_email_pattern, "user.name.example.com")  # Quick

# Malicious input: exponential backtracking
# "aaaaaaaaaaaaaaaaaaaaa!" — many 'a's that match [a-zA-Z0-9]+
# but no valid TLD, so engine backtracks through all split points
malicious: str = "a" * 25 + "!"
# re.match(bad_email_pattern, malicious)  # This would hang!

# SAFE version: use possessive quantifier or restructure
safe_email_pattern: str = r"^[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*\.[a-zA-Z]{2,}$"
# The key fix: [a-zA-Z0-9]+ can't overlap with \. so no ambiguity
```

---

## 2.5 When Not to Use Regex (and What to Use Instead)

### 🟢 Beginner: String Methods Are Often Enough

```python
# DON'T use regex for simple tasks

text: str = "Hello, World!"

# ❌ Regex overkill
import re
if re.search(r"World", text):
    print("found")

# ✅ Just use 'in'
if "World" in text:
    print("found")

# ❌ Regex for splitting on a fixed delimiter
parts: list[str] = re.split(r",", "a,b,c")

# ✅ str.split()
parts = "a,b,c".split(",")

# ❌ Regex for starts/endswith
if re.match(r"^Hello", text):
    pass

# ✅ str.startswith()
if text.startswith("Hello"):
    pass
```

### 🟡 Intermediate: The Right Tool for Each Job

```python
# Structured data? Use a proper parser, not regex.

# ❌ Parsing HTML with regex (the famous bad idea)
html: str = '<a href="https://example.com">Click</a>'
# This "works" for simple cases but WILL break:
links = re.findall(r'href="([^"]+)"', html)

# ✅ Use an HTML parser
from html.parser import HTMLParser

class LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.links.append(value)

parser = LinkExtractor()
parser.feed(html)
print(parser.links)  # ['https://example.com']

# ❌ Parsing JSON with regex
# ✅ Use json.loads()

# ❌ Parsing CSV with regex
# ✅ Use the csv module

# ❌ Parsing URLs with regex
# ✅ Use urllib.parse
from urllib.parse import urlparse
result = urlparse("https://user:pass@example.com:8080/path?q=1#frag")
print(result.hostname)  # 'example.com'
print(result.port)      # 8080
print(result.path)      # '/path'
```

**When regex IS the right choice:**

```python
import re

# 1. Pattern-based validation
phone_pattern: re.Pattern[str] = re.compile(r"^\(\d{3}\) \d{3}-\d{4}$")
print(bool(phone_pattern.match("(555) 123-4567")))  # True

# 2. Complex search-and-replace
# Convert "lastName, firstName" to "firstName lastName"
names: str = "Doe, John\nSmith, Jane\nBrown, Bob"
converted: str = re.sub(r"(\w+), (\w+)", r"\2 \1", names)
print(converted)
# John Doe
# Jane Smith
# Bob Brown

# 3. Tokenizing / lexing
code: str = "x = 42 + y * 3.14"
tokens: list[tuple[str, str]] = re.findall(
    r"(\d+\.?\d*|[a-zA-Z_]\w*|[+\-*/=])",
    code
)
# Pairs: [('x', ''), ('=', ''), ('42', ''), ('+', ''), ...]

# 4. Log parsing
log_line: str = '2024-01-15 14:30:22 [ERROR] Connection timeout (retries: 3)'
pattern: re.Pattern[str] = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"\[(?P<level>\w+)\]\s+"
    r"(?P<message>.+)"
)
match = pattern.match(log_line)
if match:
    print(match.groupdict())
    # {'date': '2024-01-15', 'time': '14:30:22',
    #  'level': 'ERROR', 'message': 'Connection timeout (retries: 3)'}
```

### 🔴 Expert: Alternatives to `re` — the `regex` Module and PEG Parsers

```python
# The third-party 'regex' module (pip install regex) adds features:
# - Variable-length lookbehinds
# - Atomic groups (?>...)
# - Possessive quantifiers (a++, a*+)
# - Unicode properties (\p{Greek}, \p{Emoji})
# - Fuzzy matching (allow N errors)

# Example of fuzzy matching (not available in re):
# import regex
# matches = regex.findall(r"(?:color){e<=1}", "colour and coler and color")
# Finds all strings within edit distance 1 of "color"

# For complex structured parsing, consider:
# - PLY (Python Lex-Yacc) for formal grammars
# - lark for Earley/LALR parsers
# - parsimonious for PEG parsers
# - pyparsing for readable parser combinators

# Example: parsing arithmetic with pyparsing (conceptual)
# from pyparsing import (Word, nums, oneOf, infixNotation, opAssoc)
# integer = Word(nums).setParseAction(lambda t: int(t[0]))
# expr = infixNotation(integer, [
#     (oneOf("* /"), 2, opAssoc.LEFT),
#     (oneOf("+ -"), 2, opAssoc.LEFT),
# ])
# result = expr.parseString("3 + 4 * 2")  # Correctly gives 11
```

---

## 🔧 Debug This: The Email Validator That Accepts `user@.com`

Your team wrote an email validation function. QA reports that it accepts invalid emails. Find all the bugs:

```python
import re

def validate_email(email):
    """Validate an email address. Returns True if valid."""
    pattern = r"[\w.]+@[\w.]+\.\w+"
    return bool(re.match(pattern, email))

# Test suite — all of these should work correctly
test_cases = [
    ("user@example.com", True),
    ("first.last@company.org", True),
    ("user@.com", False),           # BUG: returns True!
    ("@example.com", False),        # BUG: returns True!
    (".user@example.com", False),   # BUG: returns True!
    ("user@example..com", False),   # BUG: returns True!
    ("user@example.com..", False),  # BUG: returns True!
    ("user@example.com extra", True),  # Wait — should trailing text matter?
    ("USER@Example.COM", True),
    ("u@e.co", True),
]

print("Email Validation Results:")
print("-" * 50)
for email, expected in test_cases:
    result = validate_email(email)
    status = "✓" if result == expected else "✗ BUG"
    print(f"  {status:5s} | {email:30s} | expected={expected}, got={result}")
```

### Bugs to find:

```
1. ____________________________________________________
   Hint: Does [\w.]+ prevent a dot at the start?

2. ____________________________________________________
   Hint: Can [\w.]+ match an empty string before @?

3. ____________________________________________________
   Hint: Does the pattern prevent consecutive dots?

4. ____________________________________________________
   Hint: re.match() only checks the START. What about the end?

5. ____________________________________________________
   Hint: Are type hints and the return type annotated?
```

### Solution (try first!)

```python
import re


def validate_email(email: str) -> bool:
    """Validate an email address.

    Rules enforced:
    - Local part: starts/ends with alphanumeric, allows dots (not consecutive)
    - Domain: starts/ends with alphanumeric, allows dots and hyphens
    - TLD: 2-63 alphabetic characters
    - Case insensitive

    Note: This is a simplified validator. For production use,
    consider the 'email-validator' package or sending a confirmation email.

    Args:
        email: The email address string to validate.

    Returns:
        True if the email matches the validation pattern.
    """
    pattern: str = (
        r"^"                        # Anchor to start
        r"[a-zA-Z0-9]"             # Must start with alphanumeric
        r"(?:[a-zA-Z0-9.]*[a-zA-Z0-9])?"  # Middle can have dots, must end alnum
        r"@"
        r"[a-zA-Z0-9]"             # Domain starts with alphanumeric
        r"(?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?"  # Domain middle
        r"\."                       # At least one dot in domain
        r"[a-zA-Z]{2,63}"          # TLD: 2-63 letters
        r"$"                        # Anchor to end
    )
    # Reject consecutive dots anywhere
    if ".." in email:
        return False

    return bool(re.match(pattern, email, re.IGNORECASE))


# Bug fixes:
# 1. Added ^ and $ anchors (fullmatch would also work)
# 2. Local part must start and end with alphanumeric — rejects ".user@" and "@"
# 3. Explicit ".." check — rejects "user@example..com"
# 4. $ anchor prevents trailing garbage after a valid-looking prefix
# 5. Added type hints, docstring, and re.IGNORECASE flag
```

---

## Summary: Module 2 Key Takeaways

```
┌──────────────────────────────────────────────────────────────────┐
│                    STRINGS & REGEX CHEAT SHEET                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Strings:   Immutable sequences of Unicode code points.          │
│  Encoding:  str.encode() → bytes, bytes.decode() → str.         │
│  PEP 393:   CPython uses 1/2/4 bytes per char based on content. │
│  Building:  Use "".join() in loops, f-strings for formatting.    │
│  Slicing:   [start:stop:step], never raises IndexError.         │
│                                                                   │
│  Regex:     Use re.search() not re.match() for general search.  │
│  Groups:    (?P<name>...) for named, (?:...) for non-capturing.  │
│  Lookahead: (?=...) and (?!...) — zero-width, don't consume.    │
│  Compile:   re.compile() for patterns used more than once.       │
│  ReDoS:     Never nest quantifiers: (a+)+ is exponential.       │
│  Rule:      If str methods can do it, skip regex entirely.       │
│                                                                   │
│  Production rule: Use regex for pattern matching, proper         │
│  parsers for structured data (HTML, JSON, CSV, URLs).            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Next up → Module 3: Control Flow — Decisions and Repetition**

Say "Start Module 3" when you're ready.
