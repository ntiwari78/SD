# Module 8 — Modules & Packages: Building Distributable Code

> *"The purpose of software engineering is to control complexity, not to create it."*
> — Pamela Zave

---

## 8.1 `import` Mechanics — `sys.path`, Relative vs. Absolute Imports

### 🟢 Beginner: How `import` Works

A **module** is simply a `.py` file. A **package** is a directory containing modules (and usually an `__init__.py`).

```python
# Importing a module
import math
print(math.sqrt(16))   # 4.0

# Importing specific names from a module
from math import sqrt, pi
print(sqrt(16))         # 4.0
print(pi)               # 3.141592653589793

# Aliasing (renaming on import)
import numpy as np                    # Convention
from collections import defaultdict as dd
from datetime import datetime as dt

# Import everything (generally discouraged)
from math import *     # Imports all public names — pollutes namespace
```

**Where does Python find modules?**

```python
import sys

# sys.path is a list of directories Python searches (in order)
for p in sys.path:
    print(f"  {p}")

# Search order:
# 1. The directory containing the input script (or current directory)
# 2. PYTHONPATH environment variable (if set)
# 3. Standard library directories
# 4. Site-packages (pip-installed packages)
```

**The module search algorithm in full:**

```
When you write: import foo

Python checks (in order):
1. sys.modules    ← Already imported? Return the cached module.
2. sys.meta_path  ← Finder objects (import hooks)
3. sys.path       ← File system search:
   a. foo.py (module)
   b. foo/ directory with __init__.py (regular package)
   c. foo/ directory without __init__.py (namespace package, 3.3+)
   d. foo.so / foo.pyd (C extension)
```

### 🟡 Intermediate: `sys.modules`, Reloading, and Import Side Effects

**`sys.modules` — the import cache:**

```python
import sys

# Every imported module is cached in sys.modules
import json
print("json" in sys.modules)    # True
print(sys.modules["json"])       # <module 'json' from '...'>

# Importing the same module again returns the cached version
import json as json2
print(json is json2)             # True — exact same object!

# This is why module-level code runs only ONCE:
# my_module.py:
#   print("Module loaded!")  ← This prints only on first import
#   DATA = expensive_computation()  ← This runs only once
```

**Reloading modules (for development only):**

```python
import importlib
import my_module

# If you change my_module.py and want to reload:
importlib.reload(my_module)
# Warning: This doesn't update 'from my_module import X' references!
# Those still point to the old objects.

# reload() pitfalls:
# - Doesn't affect already-created objects
# - Doesn't update 'from' imports
# - Can cause subtle state bugs
# - NEVER use in production — restart the process instead
```

**Absolute vs. Relative imports:**

```
Project structure:
    myproject/
    ├── __init__.py
    ├── main.py
    ├── utils/
    │   ├── __init__.py
    │   ├── helpers.py
    │   └── validators.py
    └── models/
        ├── __init__.py
        └── user.py
```

```python
# Inside myproject/models/user.py:

# Absolute import — full path from the project root
from myproject.utils.helpers import format_name
from myproject.utils.validators import validate_email

# Relative import — relative to current package location
from ..utils.helpers import format_name       # Go up one level (myproject), then into utils
from ..utils.validators import validate_email
from . import __init__                         # Current package (models/)

# Relative import syntax:
# .   = current package
# ..  = parent package
# ... = grandparent package (and so on)

# ✅ Best practice: Use absolute imports for readability
# Use relative imports only within a package for internal references
```

**Gotcha: Running a module directly breaks relative imports:**

```python
# If you run: python myproject/models/user.py
# Relative imports FAIL because Python doesn't know the package context

# Fix 1: Run as a module
# python -m myproject.models.user

# Fix 2: Run from the project root with the package structure
# The -m flag tells Python to treat the argument as a module path
```

### 🔴 Expert: The Import System Internals

**The full import protocol (PEP 302, PEP 451):**

```python
import sys
import importlib

# The import system uses "finders" and "loaders"
# Finders: locate the module
# Loaders: create the module object and execute its code

# sys.meta_path contains the active finders:
for finder in sys.meta_path:
    print(f"  {type(finder).__name__}")
# BuiltinImporter     — handles built-in modules (sys, _io, etc.)
# FrozenImporter      — handles frozen modules (used in py2exe)
# PathFinder          — handles file-system-based modules

# You can write custom import hooks:
class DebugImporter:
    """Log every import attempt."""

    def find_module(self, fullname: str, path=None):
        print(f"  [DebugImporter] Looking for: {fullname}")
        return None  # Return None = "I can't handle this, try next finder"

# Install it:
# sys.meta_path.insert(0, DebugImporter())
# import json   # Prints: [DebugImporter] Looking for: json
```

**Module spec objects (PEP 451):**

```python
import importlib.util

# Every module has a __spec__ that describes how it was loaded
import json
spec = json.__spec__

print(f"  Name: {spec.name}")                   # json
print(f"  Origin: {spec.origin}")               # /usr/lib/python3.x/json/__init__.py
print(f"  Package: {spec.parent}")              # json
print(f"  Loader: {type(spec.loader).__name__}") # SourceFileLoader
print(f"  Submodule search: {spec.submodule_search_locations}")  # ['/usr/.../json']
```

**What happens during `import foo` — the complete sequence:**

```
1. Check sys.modules['foo']  →  if found, return it (done!)
2. For each finder in sys.meta_path:
   a. finder.find_spec('foo', None, None)  →  ModuleSpec or None
3. If no spec found: raise ModuleNotFoundError
4. spec.loader.create_module(spec)  →  module object (or None for default)
5. sys.modules['foo'] = module     ← cached BEFORE execution!
6. spec.loader.exec_module(module) ← execute the module's code
7. Return sys.modules['foo']

Step 5 is crucial: The module is cached BEFORE its code runs.
This is what allows circular imports to (partially) work.
```

**The `__import__` builtin — what `import` actually calls:**

```python
# The import statement is syntactic sugar for __import__

# import foo.bar
# is roughly equivalent to:
foo = __import__('foo.bar', globals(), locals(), [], 0)

# from foo.bar import baz
# is roughly equivalent to:
_temp = __import__('foo.bar', globals(), locals(), ['baz'], 0)
baz = _temp.baz

# You can override __import__ for total control:
# builtins.__import__ = my_custom_import
# But importlib is the preferred way to customize imports
```

---

## 8.2 `__init__.py`, `__main__.py`, and `__all__`

### 🟢 Beginner: What These Files Do

```
mypackage/
├── __init__.py     ← Makes this directory a package; runs on import
├── __main__.py     ← Runs when you do: python -m mypackage
├── module_a.py
└── module_b.py
```

**`__init__.py` — the package initializer:**

```python
# mypackage/__init__.py

# This file runs when anyone does: import mypackage
# Common uses:

# 1. Package-level imports (convenience re-exports)
from .module_a import ClassA
from .module_b import function_b

# Now users can do:
# from mypackage import ClassA, function_b
# Instead of:
# from mypackage.module_a import ClassA

# 2. Package-level variables
__version__: str = "1.0.0"
__author__: str = "Alice"

# 3. Package initialization code
print(f"mypackage v{__version__} loaded")  # Runs once on first import
```

**`__main__.py` — the package entry point:**

```python
# mypackage/__main__.py

# This runs when you execute: python -m mypackage
# (Similar to if __name__ == "__main__" but for packages)

from . import main_function

if __name__ == "__main__":
    main_function()
```

```bash
# Running a package as a script
python -m mypackage          # Executes mypackage/__main__.py
python -m mypackage.module_a  # Executes module_a.py directly
```

**The `if __name__ == "__main__"` guard:**

```python
# my_script.py

def main() -> None:
    print("Running as main script!")

def helper() -> str:
    return "I'm a helper function"

# This block runs ONLY when the file is executed directly
# NOT when it's imported as a module
if __name__ == "__main__":
    main()

# When executed directly: __name__ == "__main__"
# When imported: __name__ == "my_script"

# python my_script.py       → prints "Running as main script!"
# import my_script           → nothing printed (guard prevents it)
# my_script.helper()         → "I'm a helper function" (still accessible)
```

### 🟡 Intermediate: `__all__` and Controlling Public APIs

**`__all__` — explicitly define the public API:**

```python
# mypackage/utils.py

# __all__ controls what 'from utils import *' exports
__all__: list[str] = ["public_function", "PublicClass"]

def public_function() -> str:
    """This IS exported by 'from utils import *'."""
    return "public"

def _private_function() -> str:
    """This is NOT exported (underscore convention)."""
    return "private"

def another_function() -> str:
    """This is NOT exported (not in __all__)."""
    return "another"

class PublicClass:
    """This IS exported."""
    pass

class InternalClass:
    """This is NOT exported (not in __all__)."""
    pass
```

```python
# What different import styles expose:

# from utils import *
# → Only public_function and PublicClass (whatever's in __all__)

# import utils
# → Everything is accessible: utils.public_function, utils.another_function, etc.

# from utils import another_function
# → Works! __all__ only affects 'import *', not explicit imports
```

**`__all__` in `__init__.py` — package-level API:**

```python
# mypackage/__init__.py
from .module_a import ClassA, ClassB
from .module_b import function_b
from .module_c import utility

__all__: list[str] = [
    "ClassA",       # Re-exported from module_a
    "function_b",   # Re-exported from module_b
    # ClassB and utility are importable but not in the "public API"
]

# Users who do 'from mypackage import *' get a clean, curated API
# Users who know what they want can still import anything directly
```

**The `_` and `__` naming conventions:**

```python
# Single underscore prefix: "internal, use at your own risk"
def _helper() -> None:
    pass
# Not exported by 'from module import *' (unless in __all__)
# But fully accessible via import module; module._helper()

# Double underscore prefix: name mangling (for classes only)
class MyClass:
    def __secret(self) -> None:
        pass
# Accessible as obj._MyClass__secret (mangled to avoid subclass conflicts)

# Single underscore: "I don't care about this value"
for _ in range(10):
    pass
_, important = some_function()  # Ignore first return value
```

### 🔴 Expert: Package `__path__` and Import Hooks

**`__path__` — how Python finds submodules:**

```python
import json
print(json.__path__)  # ['/usr/lib/python3.x/json']

# __path__ is a list of directories where submodules are searched
# For a regular package, it's [the package directory]
# For a namespace package, it can span MULTIPLE directories

# You can modify __path__ to add search locations:
# json.__path__.append('/my/custom/json/extensions/')
# Now 'import json.my_extension' would search that directory too
```

**Writing a custom finder/loader (import hook):**

```python
import importlib.abc
import importlib.machinery
import sys
import types


class DictModuleFinder(importlib.abc.MetaPathFinder):
    """Import modules from a dictionary instead of the file system."""

    def __init__(self, module_sources: dict[str, str]) -> None:
        self.module_sources = module_sources

    def find_spec(
        self,
        fullname: str,
        path: list[str] | None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname in self.module_sources:
            return importlib.machinery.ModuleSpec(
                fullname,
                DictModuleLoader(self.module_sources[fullname]),
            )
        return None


class DictModuleLoader(importlib.abc.Loader):
    """Load module from a source code string."""

    def __init__(self, source: str) -> None:
        self.source = source

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        return None  # Use default module creation

    def exec_module(self, module: types.ModuleType) -> None:
        exec(compile(self.source, module.__name__, "exec"), module.__dict__)


# Usage: register virtual modules
virtual_modules: dict[str, str] = {
    "virtual_math": """
PI = 3.14159
def double(x):
    return x * 2
""",
    "virtual_greet": """
def hello(name):
    return f"Hello, {name}!"
""",
}

sys.meta_path.insert(0, DictModuleFinder(virtual_modules))

# Now these work even though no files exist:
import virtual_math
print(virtual_math.PI)           # 3.14159
print(virtual_math.double(21))   # 42

import virtual_greet
print(virtual_greet.hello("World"))  # Hello, World!
```

---

## 8.3 Creating a Package with `pyproject.toml` (The Modern Way)

### 🟢 Beginner: Project Layout

```
my-awesome-lib/
├── pyproject.toml          ← Project metadata + build config
├── README.md
├── LICENSE
├── src/                    ← Source layout (recommended)
│   └── awesome/
│       ├── __init__.py
│       ├── core.py
│       └── utils.py
└── tests/
    ├── __init__.py
    ├── test_core.py
    └── test_utils.py
```

**The `pyproject.toml` file (PEP 621):**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-awesome-lib"
version = "0.1.0"
description = "A short description of your library"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "Alice Developer", email = "alice@example.com"},
]
keywords = ["example", "library"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
]

# Dependencies
dependencies = [
    "requests>=2.28",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "mypy>=1.0",
    "ruff>=0.1",
]
docs = [
    "sphinx>=6.0",
]

# Console script entry points
[project.scripts]
awesome-cli = "awesome.core:main"
# Now 'awesome-cli' command runs awesome.core.main()

[project.urls]
Homepage = "https://github.com/alice/my-awesome-lib"
Issues = "https://github.com/alice/my-awesome-lib/issues"
```

### 🟡 Intermediate: Building and Installing

```bash
# Development install (editable mode — changes to source take effect immediately)
pip install -e ".[dev]"     # Install package + dev dependencies

# Build distributable package
pip install build
python -m build
# Creates:
#   dist/my_awesome_lib-0.1.0.tar.gz     (source distribution)
#   dist/my_awesome_lib-0.1.0-py3-none-any.whl  (wheel — faster to install)

# Upload to PyPI
pip install twine
twine upload dist/*

# Upload to TestPyPI first (for testing)
twine upload --repository testpypi dist/*
```

**`src` layout vs. flat layout:**

```
# Flat layout (simpler, older style):
my-project/
├── pyproject.toml
├── awesome/
│   ├── __init__.py
│   └── core.py
└── tests/

# src layout (recommended — prevents accidental local imports):
my-project/
├── pyproject.toml
├── src/
│   └── awesome/
│       ├── __init__.py
│       └── core.py
└── tests/

# Why src layout?
# With flat layout, 'import awesome' works from the project root
# because Python adds CWD to sys.path. This means tests might
# accidentally import the local source instead of the installed version.
# src layout prevents this — you MUST install the package to import it.
```

**Configuring tools in `pyproject.toml`:**

```toml
# Ruff linter/formatter
[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

# Mypy type checker
[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true

# Pytest
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

# All tools read from pyproject.toml — one config file for everything!
```

### 🔴 Expert: Dynamic Versioning, Entry Points, and Setuptools Internals

**Dynamic version from source code:**

```toml
# pyproject.toml
[project]
dynamic = ["version"]

[tool.setuptools.dynamic]
version = {attr = "awesome.__version__"}
```

```python
# src/awesome/__init__.py
__version__: str = "0.1.0"

# Now the version is defined in ONE place (the Python source)
# and pyproject.toml reads it at build time
```

**Entry points — the plugin system:**

```toml
# pyproject.toml
[project.entry-points."myapp.plugins"]
csv-handler = "awesome.plugins.csv:CsvPlugin"
json-handler = "awesome.plugins.json:JsonPlugin"
```

```python
# Discovering plugins at runtime
from importlib.metadata import entry_points

# Python 3.12+:
plugins = entry_points(group="myapp.plugins")
for ep in plugins:
    print(f"  Plugin: {ep.name} → {ep.value}")
    plugin_class = ep.load()   # Actually imports and returns the object
    instance = plugin_class()

# This is how pytest discovers plugins, Flask discovers extensions,
# and console_scripts work. It's the standard plugin mechanism.
```

**How `setuptools` processes `pyproject.toml`:**

```
Build pipeline:
1. python -m build
2. Build frontend reads [build-system] to find backend
3. Backend (setuptools) reads [project] metadata
4. Backend generates:
   - PKG-INFO / METADATA (package metadata)
   - RECORD (list of installed files)
   - entry_points.txt (console scripts, plugins)
   - top_level.txt (package names)
5. Output: .whl (wheel) and/or .tar.gz (sdist)

When pip install runs:
1. Download or build wheel
2. Extract to site-packages/
3. Create console script wrappers in bin/
4. Register in INSTALLER/METADATA files
```

---

## 8.4 Namespace Packages and Plugin Architectures

### 🟢 Beginner: What's a Namespace Package?

A namespace package is a package that can be **split across multiple directories** — no `__init__.py` needed (Python 3.3+, PEP 420).

```
# Two separate installations that contribute to the same namespace:

# Installation 1: /site-packages/mynamespace/module_a.py
# Installation 2: /other/path/mynamespace/module_b.py

# Both contribute to the 'mynamespace' package:
import mynamespace.module_a   # Found in /site-packages/
import mynamespace.module_b   # Found in /other/path/

# This works because there's NO __init__.py in either directory
# Python treats 'mynamespace' as a namespace package that spans
# multiple directories
```

### 🟡 Intermediate: Building a Plugin System

```python
# plugin_system/core.py
"""A simple plugin architecture using entry points."""

from importlib.metadata import entry_points
from typing import Protocol


class Plugin(Protocol):
    """Interface that all plugins must implement."""
    name: str

    def process(self, data: str) -> str:
        ...


def discover_plugins(group: str = "myapp.plugins") -> dict[str, Plugin]:
    """Discover and load all installed plugins."""
    plugins: dict[str, Plugin] = {}
    for ep in entry_points(group=group):
        try:
            plugin_class = ep.load()
            plugin = plugin_class()
            plugins[ep.name] = plugin
        except Exception as e:
            print(f"  Warning: Failed to load plugin {ep.name}: {e}")
    return plugins


def run_pipeline(data: str, plugins: dict[str, Plugin]) -> str:
    """Run data through all plugins in sequence."""
    result: str = data
    for name, plugin in plugins.items():
        print(f"  Running plugin: {name}")
        result = plugin.process(result)
    return result
```

**Alternative: directory-based plugin discovery:**

```python
import importlib
import pkgutil
from pathlib import Path
from typing import Any


def load_plugins_from_directory(plugin_dir: Path) -> list[Any]:
    """Load all Python modules from a directory as plugins."""
    plugins: list[Any] = []

    for finder, name, is_pkg in pkgutil.iter_modules([str(plugin_dir)]):
        module = importlib.import_module(name)
        if hasattr(module, "Plugin"):
            plugins.append(module.Plugin())

    return plugins
```

### 🔴 Expert: The Namespace Package Resolution Algorithm

```python
import sys

# Namespace packages have a special __path__ that uses _NamespacePath
# This path dynamically searches sys.path on every submodule import

# Regular package:
import json
print(type(json.__path__))  # list — static, computed once

# Namespace package (if one exists):
# print(type(ns_pkg.__path__))  # _NamespacePath — dynamic, recalculated

# The resolution algorithm:
# 1. Python searches sys.path for directories matching the package name
# 2. If ANY directory has __init__.py → regular package (stops searching)
# 3. If NO directory has __init__.py but some exist → namespace package
#    (collects ALL matching directories into __path__)
# 4. If nothing found → ImportError
```

**Performance implication of namespace packages:**

```python
# Namespace packages are slower to import because:
# 1. Python must scan ALL of sys.path (not just find the first match)
# 2. __path__ is recalculated on each submodule access
# 3. No __init__.py means no package initialization code

# For application code, ALWAYS use regular packages (__init__.py)
# Use namespace packages only for multi-package plugin ecosystems
# where packages are installed separately

# Check if a package is a namespace package:
import importlib.util

spec = importlib.util.find_spec("json")
if spec and spec.origin:
    print("Regular package")
else:
    print("Namespace package (or not found)")
```

---

## 8.5 Circular Imports — Why They Happen and Three Ways to Fix Them

### 🟢 Beginner: What's a Circular Import?

```python
# module_a.py
from module_b import function_b   # module_b tries to import from module_a!

def function_a() -> str:
    return "A"

# module_b.py
from module_a import function_a   # module_a tries to import from module_b!

def function_b() -> str:
    return "B"

# Result: ImportError: cannot import name 'function_a' from partially
# initialized module 'module_a' (most likely due to a circular import)
```

**Why it happens — the import sequence:**

```
1. import module_a
2. Python starts executing module_a.py
3. First line: from module_b import function_b
4. Python starts executing module_b.py
5. First line: from module_a import function_a
6. Python checks sys.modules — module_a IS there (partially initialized!)
7. Tries to get function_a from module_a — but it hasn't been defined yet!
8. ImportError!
```

### 🟡 Intermediate: Three Fixes for Circular Imports

**Fix 1: Import at the module level, use full paths (defer attribute access):**

```python
# module_a.py
import module_b   # Import the MODULE, not specific names

def function_a() -> str:
    return "A calls " + module_b.function_b()  # Access at CALL time, not import time

# module_b.py
import module_a

def function_b() -> str:
    return "B calls " + module_a.function_a()

# This works because by the time function_a() or function_b() is CALLED,
# both modules are fully loaded. The import only needs the module object
# (which exists even when partially initialized).
```

**Fix 2: Move the import inside the function (lazy import):**

```python
# module_a.py
def function_a() -> str:
    from module_b import function_b  # Import only when called
    return "A calls " + function_b()

# module_b.py
def function_b() -> str:
    from module_a import function_a  # Import only when called
    return "B calls " + function_a()

# This works because the import happens at RUNTIME, not at module load time.
# By that point, both modules are fully initialized.

# Downside: Slightly slower (import check on every call, though cached in sys.modules)
# Downside: Import errors are deferred to runtime instead of caught at startup
```

**Fix 3: Restructure to eliminate the cycle (the REAL fix):**

```python
# The circular dependency usually means the architecture is wrong.
# Extract the shared dependency into a third module.

# BEFORE (circular):
# module_a.py: uses function_b from module_b
# module_b.py: uses function_a from module_a

# AFTER (restructured):
# shared.py: contains shared utilities that both need
# module_a.py: imports from shared
# module_b.py: imports from shared

# shared.py
def shared_function() -> str:
    return "shared"

# module_a.py
from shared import shared_function

def function_a() -> str:
    return "A uses " + shared_function()

# module_b.py
from shared import shared_function

def function_b() -> str:
    return "B uses " + shared_function()
```

**Fix 3b: Use `TYPE_CHECKING` for type annotation cycles:**

```python
from __future__ import annotations  # PEP 563: postponed evaluation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # This block ONLY runs during type checking (mypy/pyright)
    # It is NEVER executed at runtime
    from module_b import ClassB

class ClassA:
    def process(self, other: "ClassB") -> None:  # String annotation
        ...

# With 'from __future__ import annotations', all annotations are
# treated as strings automatically, so you don't even need quotes.
# This completely avoids the runtime circular import for type hints.
```

### 🔴 Expert: Partial Initialization and the Import Lock

**How partially-initialized modules work:**

```python
# When Python starts importing module_a:
# 1. Creates an EMPTY module object
# 2. Adds it to sys.modules['module_a']
# 3. THEN starts executing module_a.py line by line

# If module_b imports module_a during step 3:
# Python finds module_a in sys.modules (from step 2)
# Returns the PARTIAL module (only names defined so far)

# Demo:
# order_demo_a.py
print("module_a: starting")
import order_demo_b
print("module_a: finished")
MY_VAR = "defined in a"

# order_demo_b.py
print("module_b: starting")
import order_demo_a
print(f"module_b: can see module_a? {dir(order_demo_a)}")
# At this point, module_a only has names defined BEFORE 'import order_demo_b'
print("module_b: finished")

# Output:
# module_a: starting
# module_b: starting
# module_b: can see module_a? ['__name__', '__doc__', ...]  ← MY_VAR NOT here yet!
# module_b: finished
# module_a: finished
```

**The Global Import Lock (GIL for imports):**

```python
import importlib

# CPython uses a per-module lock for thread safety during imports
# (Changed in Python 3.3 from a single global lock to per-module locks)

# This means:
# - Two threads importing DIFFERENT modules can proceed in parallel
# - Two threads importing the SAME module: one waits for the other
# - Deadlocks are possible if two threads have circular imports!

# The lock prevents the same module from being initialized twice
# in a multithreaded environment

# You can check if an import lock is held:
# importlib._bootstrap._thread  (internal, don't rely on this)
```

**Lazy module loading pattern (for large packages):**

```python
# Some packages (like numpy, pandas) take a long time to import
# because they load many submodules. You can defer this:

import importlib
import types


class LazyModule(types.ModuleType):
    """Module that loads itself on first attribute access."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._real_module: types.ModuleType | None = None

    def _load(self) -> types.ModuleType:
        if self._real_module is None:
            self._real_module = importlib.import_module(self.__name__)
        return self._real_module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)

    def __dir__(self):
        return dir(self._load())


# Usage:
# Instead of: import heavy_library  (takes 2 seconds)
# Do: heavy_library = LazyModule("heavy_library")  (instant)
# First actual use triggers the real import

# Python 3.7+ has a built-in mechanism via module __getattr__:
# mypackage/__init__.py
def __getattr__(name: str):
    """Lazy-load submodules on first access."""
    if name == "heavy_submodule":
        from . import heavy_submodule
        return heavy_submodule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

---

## 🔧 Debug This: The Broken Package

You're setting up a new Python package. It has multiple issues preventing it from working correctly. Find all the bugs:

```
Project structure:
    my_tool/
    ├── pyproject.toml
    ├── my_tool/
    │   ├── __init__.py
    │   ├── cli.py
    │   ├── core.py
    │   └── utils/
    │       ├── helpers.py      ← Note: no __init__.py!
    │       └── validators.py
    └── tests/
        └── test_core.py
```

```python
# my_tool/__init__.py
from core import Engine
from utils.helpers import format_output
__version__ = "1.0.0"

# my_tool/core.py
from utils.helpers import format_output
from utils.validators import validate_input

class Engine:
    def run(self, data):
        validate_input(data)
        return format_output(data)

# my_tool/cli.py
import sys
sys.path.insert(0, ".")  # "Fix" import issues
from my_tool.core import Engine

def main():
    engine = Engine()
    print(engine.run(sys.argv[1]))

# my_tool/utils/helpers.py
def format_output(data):
    return f"Result: {data}"

# my_tool/utils/validators.py
from my_tool.core import Engine  # "Need to check Engine type"

def validate_input(data):
    if not isinstance(data, str):
        raise TypeError("Data must be a string")
    if len(data) > 1000:
        raise ValueError("Data too long")
```

```toml
# pyproject.toml
[project]
name = "my_tool"
version = "1.0.0"

[project.scripts]
my-tool = "my_tool.cli:main"
```

### Bugs to find:

```
1. ____________________________________________________
   Hint: my_tool/utils/ has no __init__.py. Can you import
   from it as a regular package?

2. ____________________________________________________
   Hint: __init__.py uses 'from core import Engine'. Is 'core'
   a top-level module? What should the import path be?

3. ____________________________________________________
   Hint: core.py uses 'from utils.helpers import ...' — same
   problem as #2. What's the correct import for within a package?

4. ____________________________________________________
   Hint: validators.py imports from my_tool.core. core.py imports
   from validators. That's a circular import!

5. ____________________________________________________
   Hint: cli.py uses sys.path.insert(0, "."). This is a fragile
   hack that breaks when run from different directories.

6. ____________________________________________________
   Hint: pyproject.toml is missing [build-system]. Also, where
   are the type hints?

7. ____________________________________________________
   Hint: __init__.py defines __version__ = "1.0.0" and so does
   pyproject.toml. Which is the source of truth?
```

### Solution (try first!)

```
Fixed structure:
    my_tool/
    ├── pyproject.toml
    ├── src/
    │   └── my_tool/
    │       ├── __init__.py
    │       ├── cli.py
    │       ├── core.py
    │       └── utils/
    │           ├── __init__.py       ← Bug 1 FIX: Added!
    │           ├── helpers.py
    │           └── validators.py
    └── tests/
        └── test_core.py
```

```toml
# pyproject.toml — Bug 6 FIX: Added build-system; Bug 7 FIX: dynamic version
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-tool"
dynamic = ["version"]
requires-python = ">=3.10"

[tool.setuptools.dynamic]
version = {attr = "my_tool.__version__"}

[project.scripts]
my-tool = "my_tool.cli:main"
```

```python
# src/my_tool/__init__.py — Bug 2 FIX: Use relative imports
from .core import Engine
from .utils.helpers import format_output

__all__: list[str] = ["Engine", "format_output"]
__version__: str = "1.0.0"  # Single source of truth


# src/my_tool/core.py — Bug 3 FIX: Relative imports
from .utils.helpers import format_output
from .utils.validators import validate_input


class Engine:
    def run(self, data: str) -> str:
        validate_input(data)
        return format_output(data)


# src/my_tool/cli.py — Bug 5 FIX: No sys.path hack needed
import sys
from my_tool.core import Engine


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: my-tool <data>", file=sys.stderr)
        sys.exit(1)
    engine = Engine()
    print(engine.run(sys.argv[1]))


# src/my_tool/utils/__init__.py — Bug 1 FIX: Created this file
"""Utility functions for my_tool."""


# src/my_tool/utils/helpers.py
def format_output(data: str) -> str:
    return f"Result: {data}"


# src/my_tool/utils/validators.py — Bug 4 FIX: Removed circular import
# The Engine import was unnecessary — validate_input doesn't need it
def validate_input(data: object) -> None:
    """Validate input data.

    Raises:
        TypeError: If data is not a string.
        ValueError: If data exceeds maximum length.
    """
    if not isinstance(data, str):
        raise TypeError(f"Data must be a string, got {type(data).__name__}")
    if len(data) > 1000:
        raise ValueError(f"Data too long: {len(data)} chars (max 1000)")
```

```
Bug Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Missing __init__.py:    utils/ is either a namespace package
   (slower, no init code) or simply broken on some setups.
   Fix: add utils/__init__.py.

2. Wrong import in __init__.py: 'from core' is an absolute import
   for a top-level 'core' module. Should be 'from .core' (relative)
   to import from within the package.

3. Same issue in core.py: 'from utils.helpers' should be
   'from .utils.helpers' (relative import within package).

4. Circular import: validators.py imports core.py which imports
   validators.py. The Engine import was unnecessary — remove it.

5. sys.path hack: Inserting "." breaks when CWD changes.
   Fix: install the package properly (pip install -e .) and use
   normal imports.

6. Missing build-system: pyproject.toml needs [build-system]
   for pip to know how to build the package. Missing type hints.

7. Duplicate version: Version defined in both pyproject.toml and
   __init__.py. Use dynamic versioning (single source of truth).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Summary: Module 8 Key Takeaways

```
┌──────────────────────────────────────────────────────────────────┐
│                  MODULES & PACKAGES CHEAT SHEET                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  IMPORT ORDER:                                                   │
│    sys.modules (cache) → sys.meta_path (finders) → sys.path     │
│    Module cached BEFORE execution (enables partial init)         │
│                                                                   │
│  IMPORT STYLES:                                                  │
│    import pkg.mod           → access as pkg.mod.name             │
│    from pkg.mod import name → access as name (direct)            │
│    from .mod import name    → relative import (within package)   │
│    Prefer absolute imports. Use relative for internal refs.      │
│                                                                   │
│  PACKAGE FILES:                                                  │
│    __init__.py    → Package initializer; re-export public API    │
│    __main__.py    → Entry point for python -m package            │
│    __all__        → Controls 'from pkg import *'                 │
│    pyproject.toml → Single config file for metadata + tools      │
│                                                                   │
│  PACKAGING (modern):                                             │
│    Use src/ layout to prevent accidental local imports           │
│    pyproject.toml + setuptools (or hatch/flit/poetry)           │
│    pip install -e ".[dev]" for development                       │
│    Entry points for CLI commands and plugin systems              │
│                                                                   │
│  CIRCULAR IMPORTS — Three fixes:                                 │
│    1. Import module, not name (defer attribute access)           │
│    2. Import inside function (lazy import)                       │
│    3. Restructure: extract shared code (the REAL fix)            │
│    For type hints only: TYPE_CHECKING + string annotations       │
│                                                                   │
│  NAMESPACE PACKAGES:                                             │
│    No __init__.py. Span multiple directories.                    │
│    Slower than regular packages. Use only for plugin ecosystems. │
│                                                                   │
│  Production rules:                                               │
│    NEVER use sys.path.insert() hacks.                            │
│    NEVER use importlib.reload() in production.                   │
│    ONE source of truth for version (__init__.py + dynamic toml). │
│    Test imports by installing the package, not running files.    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Next up → Module 9: OOP Mastery — Modeling the Real World**

Say "Start Module 9" when you're ready.
