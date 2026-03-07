# Module 9 — OOP Mastery: Modeling the Real World

> *"Favor composition over inheritance."*
> — Design Patterns: Elements of Reusable Object-Oriented Software (Gang of Four)

---

## 9.1 Classes and Objects — `__init__`, `self`, and Instance vs. Class Attributes

### 🟢 Beginner: Your First Class

A **class** is a blueprint. An **object** (instance) is a thing built from that blueprint.

```python
class Dog:
    """A simple dog class."""

    # Class attribute — shared by ALL instances
    species: str = "Canis familiaris"

    def __init__(self, name: str, age: int) -> None:
        """Initialize a new Dog instance.

        Args:
            name: The dog's name.
            age: The dog's age in years.
        """
        # Instance attributes — unique to EACH instance
        self.name = name
        self.age = age

    def bark(self) -> str:
        """Return a bark string."""
        return f"{self.name} says Woof!"

    def describe(self) -> str:
        """Return a description of this dog."""
        return f"{self.name} is {self.age} years old"


# Creating instances
buddy = Dog("Buddy", 5)
charlie = Dog("Charlie", 3)

print(buddy.bark())        # "Buddy says Woof!"
print(charlie.describe())  # "Charlie is 3 years old"

# Class attribute is shared
print(buddy.species)       # "Canis familiaris"
print(charlie.species)     # "Canis familiaris"
print(Dog.species)         # "Canis familiaris"
```

**What is `self`?**

```python
# 'self' is just a convention — it's the instance the method is called on.
# When you write buddy.bark(), Python translates it to Dog.bark(buddy)

class Point:
    def __init__(self, x: float, y: float) -> None:
        self.x = x  # self.x is an attribute of THIS specific instance
        self.y = y

    def distance_to(self, other: "Point") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

p1 = Point(0, 0)
p2 = Point(3, 4)
print(p1.distance_to(p2))  # 5.0

# These are equivalent:
print(p1.distance_to(p2))      # Method call syntax
print(Point.distance_to(p1, p2))  # Explicit — 'self' is p1
```

### 🟡 Intermediate: Instance vs. Class Attributes — The Trap

```python
class Student:
    # Class attribute
    school: str = "MIT"
    grades: list[int] = []  # ← DANGER! Mutable class attribute

    def __init__(self, name: str) -> None:
        self.name = name    # Instance attribute

# The school attribute works fine (immutable)
alice = Student("Alice")
bob = Student("Bob")

alice.school = "Stanford"   # Creates an INSTANCE attribute on alice
print(alice.school)         # "Stanford" (instance)
print(bob.school)           # "MIT" (class — unchanged)
print(Student.school)       # "MIT" (class — unchanged)

# The grades attribute is BROKEN (mutable, shared!)
alice.grades.append(95)     # Modifies the CLASS attribute!
print(bob.grades)           # [95] ← Bob sees Alice's grade!
print(Student.grades)       # [95] ← The class attribute was mutated

# WHY? alice.grades doesn't create an instance attribute — it finds
# the class attribute and mutates it in place.
```

```python
# ✅ FIX: Initialize mutable attributes in __init__
class StudentFixed:
    school: str = "MIT"  # Immutable class attribute — fine

    def __init__(self, name: str) -> None:
        self.name = name
        self.grades: list[int] = []  # Instance attribute — each student gets their own

alice = StudentFixed("Alice")
bob = StudentFixed("Bob")
alice.grades.append(95)
print(bob.grades)  # [] — independent!
```

**Attribute lookup chain:**

```
When you access obj.attr, Python searches:
1. obj.__dict__          (instance attributes)
2. type(obj).__dict__    (class attributes)
3. Base classes           (via MRO — see Section 9.3)
4. __getattr__() if defined (fallback)

When you SET obj.attr = value:
- ALWAYS creates/updates in obj.__dict__ (instance level)
- NEVER modifies the class attribute
```

```python
class Demo:
    x: int = 10  # Class attribute

d = Demo()
print(d.__dict__)      # {}          — no instance attributes yet
print(d.x)             # 10          — found in class dict

d.x = 20               # Creates INSTANCE attribute
print(d.__dict__)      # {'x': 20}   — now has instance attribute
print(d.x)             # 20          — instance shadows class
print(Demo.x)          # 10          — class attribute unchanged

del d.x                # Removes instance attribute
print(d.x)             # 10          — falls back to class attribute
```

### 🔴 Expert: CPython's Object Layout

```
Instance of class Dog (simplified):

┌──────────────────────────────────────────┐
│           PyObject (Dog instance)         │
├──────────────────────────────────────────┤
│  ob_refcnt    │  Reference count          │
│  ob_type      │  → Dog (the class object) │
│  __dict__     │  → {"name": "Buddy",      │
│               │      "age": 5}            │
│  __weakref__  │  → weak reference list    │
└──────────────────────────────────────────┘

The Dog CLASS itself is also an object:

┌──────────────────────────────────────────────┐
│           Type Object (Dog class)             │
├──────────────────────────────────────────────┤
│  ob_type      │  → type (metaclass)           │
│  tp_name      │  "Dog"                        │
│  tp_bases     │  → (object,)                  │
│  tp_dict      │  → {"species": "Canis...",    │
│               │      "bark": <function>,      │
│               │      "__init__": <function>}  │
│  tp_mro       │  → (Dog, object)              │
└──────────────────────────────────────────────┘
```

**`__slots__` — eliminating `__dict__` for memory efficiency:**

```python
import sys

class PointDict:
    """Regular class — each instance has a __dict__."""
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

class PointSlots:
    """Slotted class — no __dict__, fixed attributes."""
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

pd = PointDict(1.0, 2.0)
ps = PointSlots(1.0, 2.0)

print(f"  With __dict__:  {sys.getsizeof(pd)} bytes")   # ~48 bytes + dict (~104)
print(f"  With __slots__: {sys.getsizeof(ps)} bytes")    # ~48 bytes (no dict)

# __slots__ benefits:
# 1. ~40-50% less memory per instance
# 2. Slightly faster attribute access (descriptor vs dict lookup)
# 3. Prevents accidental attribute creation (typos caught!)

# ps.z = 3   # AttributeError: 'PointSlots' object has no attribute 'z'

# __slots__ tradeoffs:
# - Can't add arbitrary attributes
# - Can't use __dict__-based features easily (vars(), etc.)
# - Inheritance with slots is tricky
# - No weak references unless you add '__weakref__' to slots
```

---

## 9.2 Inheritance vs. Composition — When "Is-A" Becomes a Trap

### 🟢 Beginner: Inheritance Basics

```python
class Animal:
    """Base class for all animals."""

    def __init__(self, name: str, sound: str) -> None:
        self.name = name
        self.sound = sound

    def speak(self) -> str:
        return f"{self.name} says {self.sound}!"

class Dog(Animal):
    """Dog inherits from Animal."""

    def __init__(self, name: str) -> None:
        super().__init__(name, sound="Woof")  # Call parent's __init__
        self.tricks: list[str] = []

    def learn_trick(self, trick: str) -> None:
        self.tricks.append(trick)

class Cat(Animal):
    """Cat inherits from Animal."""

    def __init__(self, name: str) -> None:
        super().__init__(name, sound="Meow")

    def speak(self) -> str:
        """Override parent's method."""
        return f"{self.name} purrs softly..."

# Polymorphism — same interface, different behavior
animals: list[Animal] = [Dog("Rex"), Cat("Whiskers"), Dog("Buddy")]
for animal in animals:
    print(animal.speak())
# Rex says Woof!
# Whiskers purrs softly...
# Buddy says Woof!

# isinstance and issubclass
print(isinstance(Dog("Rex"), Animal))  # True — Dog IS an Animal
print(issubclass(Dog, Animal))         # True
```

### 🟡 Intermediate: Why Composition Beats Inheritance

**The Inheritance Trap — the "gorilla banana" problem:**

> "You wanted a banana but what you got was a gorilla holding the banana and the entire jungle." — Joe Armstrong

```python
# ❌ BAD: Deep inheritance hierarchy
class Vehicle:
    def start(self) -> None: ...
    def stop(self) -> None: ...

class Car(Vehicle):
    def open_trunk(self) -> None: ...

class ElectricCar(Car):
    def charge(self) -> None: ...

class SelfDrivingElectricCar(ElectricCar):
    def navigate(self) -> None: ...

# Problems:
# 1. Every change to Vehicle ripples down to ALL subclasses
# 2. SelfDrivingElectricCar inherits methods it may not need
# 3. Can't create a self-driving gas car without rewriting
# 4. Testing requires mocking the entire chain
```

```python
# ✅ GOOD: Composition — "has-a" instead of "is-a"
from dataclasses import dataclass, field


@dataclass
class Engine:
    """Handles starting and stopping."""
    fuel_type: str

    def start(self) -> str:
        return f"{self.fuel_type} engine started"

    def stop(self) -> str:
        return f"{self.fuel_type} engine stopped"


@dataclass
class Battery:
    """Handles charging."""
    capacity_kwh: float
    charge_level: float = 1.0

    def charge(self) -> str:
        self.charge_level = 1.0
        return f"Battery charged to {self.capacity_kwh} kWh"


@dataclass
class Autopilot:
    """Handles self-driving."""
    version: str

    def navigate(self, destination: str) -> str:
        return f"Autopilot v{self.version}: navigating to {destination}"


@dataclass
class Storage:
    """Handles cargo storage."""
    capacity_liters: float

    def open(self) -> str:
        return f"Storage ({self.capacity_liters}L) opened"


@dataclass
class Car:
    """Composed of independent components."""
    engine: Engine
    storage: Storage
    battery: Battery | None = None
    autopilot: Autopilot | None = None

    def start(self) -> str:
        return self.engine.start()

    def navigate(self, destination: str) -> str:
        if self.autopilot is None:
            raise RuntimeError("This car doesn't have autopilot")
        return self.autopilot.navigate(destination)


# Flexible — mix and match capabilities
gas_car = Car(
    engine=Engine("gasoline"),
    storage=Storage(500),
)

electric_self_driving = Car(
    engine=Engine("electric"),
    storage=Storage(300),
    battery=Battery(75),
    autopilot=Autopilot("4.0"),
)

print(gas_car.start())                         # "gasoline engine started"
print(electric_self_driving.navigate("home"))   # "Autopilot v4.0: navigating to home"
```

**When inheritance IS appropriate:**

```python
# 1. True "is-a" relationships with shallow hierarchies (1-2 levels)
# 2. Framework extension points designed for inheritance
# 3. Abstract base classes defining interfaces

# RULE OF THUMB:
# "Is-a" → Inheritance (Dog IS an Animal)
# "Has-a" → Composition (Car HAS an Engine)
# "Can-do" → Protocol/ABC (Iterable CAN iterate)
# When in doubt → Composition
```

### 🔴 Expert: Mixins — Controlled Multiple Inheritance

```python
# Mixins are small, focused classes designed to be "mixed in"
# via multiple inheritance. They add a single capability.

import json
from typing import Any


class SerializableMixin:
    """Adds JSON serialization capability."""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "SerializableMixin":
        data: dict[str, Any] = json.loads(json_str)
        return cls(**data)


class PrintableMixin:
    """Adds pretty printing capability."""

    def pretty_print(self) -> None:
        class_name: str = type(self).__name__
        attrs: str = ", ".join(
            f"{k}={v!r}" for k, v in self.__dict__.items()
        )
        print(f"{class_name}({attrs})")


class ValidatableMixin:
    """Adds validation capability."""

    def validate(self) -> list[str]:
        errors: list[str] = []
        for attr_name, attr_type in getattr(self, "__annotations__", {}).items():
            value = getattr(self, attr_name, None)
            if value is None:
                errors.append(f"{attr_name} is required")
        return errors


# Compose mixins to build feature-rich classes
class User(SerializableMixin, PrintableMixin, ValidatableMixin):
    name: str
    email: str
    age: int

    def __init__(self, name: str, email: str, age: int) -> None:
        self.name = name
        self.email = email
        self.age = age


user = User("Alice", "alice@example.com", 30)
user.pretty_print()          # User(name='Alice', email='alice@example.com', age=30)
print(user.to_json())        # {"name": "Alice", "email": "alice@example.com", "age": 30}
print(user.validate())       # []

# Mixin rules:
# 1. Mixins should NOT have __init__ (or should call super().__init__)
# 2. Mixins should provide ONE capability
# 3. Mixins should be listed LEFT of the base class
# 4. Mixins should work with any class (no assumptions about self)
```

---

## 9.3 MRO (Method Resolution Order) and the C3 Linearization Algorithm

### 🟢 Beginner: Which Method Gets Called?

When multiple classes define the same method, Python needs to decide which one to call. The **MRO** (Method Resolution Order) defines this.

```python
class A:
    def greet(self) -> str:
        return "Hello from A"

class B(A):
    def greet(self) -> str:
        return "Hello from B"

class C(A):
    def greet(self) -> str:
        return "Hello from C"

class D(B, C):
    pass  # D doesn't define greet — which parent's version is used?

d = D()
print(d.greet())  # "Hello from B" — B comes before C in the MRO

# Inspect the MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)
# D → B → C → A → object
```

### 🟡 Intermediate: The Diamond Problem and `super()`

```
The Diamond Problem:
       A
      / \
     B   C
      \ /
       D

Which path does D.method() take?
```

```python
class A:
    def method(self) -> None:
        print("A.method")

class B(A):
    def method(self) -> None:
        print("B.method")
        super().method()   # Calls NEXT in MRO, not necessarily A!

class C(A):
    def method(self) -> None:
        print("C.method")
        super().method()

class D(B, C):
    def method(self) -> None:
        print("D.method")
        super().method()

D().method()
# D.method
# B.method
# C.method    ← B's super() calls C, NOT A!
# A.method    ← C's super() calls A

# MRO: D → B → C → A → object
# super() follows the MRO chain, ensuring each class is called ONCE
```

**`super()` is NOT "call my parent" — it's "call the NEXT in MRO":**

```python
# This distinction is critical for cooperative multiple inheritance

class Base:
    def __init__(self, **kwargs: object) -> None:
        # Absorb any remaining kwargs
        pass

class Left(Base):
    def __init__(self, left_val: int = 0, **kwargs: object) -> None:
        self.left_val = left_val
        super().__init__(**kwargs)  # Pass remaining kwargs up the MRO

class Right(Base):
    def __init__(self, right_val: int = 0, **kwargs: object) -> None:
        self.right_val = right_val
        super().__init__(**kwargs)

class Diamond(Left, Right):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # Passes kwargs through the entire chain

d = Diamond(left_val=1, right_val=2)
print(d.left_val)    # 1
print(d.right_val)   # 2
# MRO: Diamond → Left → Right → Base → object
# Left grabs left_val, passes right_val to Right via **kwargs
```

### 🔴 Expert: C3 Linearization Algorithm

```python
# The C3 linearization algorithm produces the MRO by merging
# the linearizations of parent classes while preserving:
# 1. Local precedence: if class D(B, C), B comes before C
# 2. Monotonicity: if B comes before C in D's MRO,
#    B comes before C in ALL subclasses of D

# Algorithm (simplified):
# MRO(D) = D + merge(MRO(B), MRO(C), [B, C])
#
# merge() takes the first head of any list that doesn't appear
# in the TAIL of any other list, adds it to the result, and
# removes it from all lists. Repeat until empty.

# Example:
# MRO(A) = [A, object]
# MRO(B) = [B, A, object]
# MRO(C) = [C, A, object]
# MRO(D) = D + merge([B, A, object], [C, A, object], [B, C])
#
# Step 1: B is not in tail of any list → take B
#   merge([A, object], [C, A, object], [C])
# Step 2: A is in tail of second list → skip. C is not in any tail → take C
#   merge([A, object], [A, object], [])
# Step 3: A → take A
#   merge([object], [object], [])
# Step 4: object → take object
#
# Result: [D, B, C, A, object] ✓

# If C3 cannot find a valid linearization, it raises TypeError:
# class X(B, C): ...  where B and C have conflicting orders
```

**When C3 fails (inconsistent hierarchy):**

```python
class A: pass
class B(A): pass
class C(A, B): pass  # TypeError!

# C says: A before B
# But B(A) means B before A
# Contradiction! C3 cannot resolve this.

# This prevents ambiguous hierarchies at CLASS CREATION TIME
# (not at runtime — the error is immediate)
```

---

## 9.4 Dunder Methods — `__repr__`, `__eq__`, `__hash__`, `__slots__`, and Descriptors

### 🟢 Beginner: Making Objects Behave Like Built-in Types

```python
class Vector:
    """A 2D vector with operator overloading."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        """Developer-friendly representation (unambiguous)."""
        return f"Vector({self.x!r}, {self.y!r})"

    def __str__(self) -> str:
        """User-friendly representation."""
        return f"({self.x}, {self.y})"

    def __eq__(self, other: object) -> bool:
        """Enable == comparison."""
        if not isinstance(other, Vector):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __add__(self, other: "Vector") -> "Vector":
        """Enable + operator."""
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float) -> "Vector":
        """Enable vector * scalar."""
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vector":
        """Enable scalar * vector (reverse multiplication)."""
        return self.__mul__(scalar)

    def __abs__(self) -> float:
        """Enable abs(vector) — returns magnitude."""
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def __bool__(self) -> bool:
        """Vector is truthy if non-zero."""
        return abs(self) > 0


v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(repr(v1))          # Vector(3, 4)
print(str(v1))           # (3, 4)
print(v1 + v2)           # (4, 6)
print(v1 * 3)            # (9, 12)
print(3 * v1)            # (9, 12)  — __rmul__
print(abs(v1))            # 5.0
print(v1 == Vector(3, 4)) # True
```

### 🟡 Intermediate: `__hash__`, `__eq__`, and the Hashability Contract

```python
# THE RULE: If you define __eq__, you MUST define __hash__
# (or set __hash__ = None to make instances unhashable)

# If __eq__ is defined but __hash__ is not:
# Python sets __hash__ = None → instances are UNHASHABLE
# (can't be used in sets or as dict keys)

class BadPoint:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BadPoint):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    # No __hash__ defined → __hash__ is implicitly None

p = BadPoint(1, 2)
# {p}  → TypeError: unhashable type: 'BadPoint'


class GoodPoint:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GoodPoint):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))  # Tuple of the same fields used in __eq__

p = GoodPoint(1, 2)
s = {p, GoodPoint(1, 2), GoodPoint(3, 4)}
print(len(s))  # 2 — the two (1,2) points are equal and hash the same
```

**The `@dataclass` shortcut:**

```python
from dataclasses import dataclass

@dataclass(frozen=True)  # frozen=True makes it immutable AND hashable
class Point:
    x: float
    y: float

# Automatically generates:
# __init__, __repr__, __eq__, __hash__ (when frozen), __lt__ (with order=True)

p1 = Point(1, 2)
p2 = Point(1, 2)
print(p1 == p2)            # True
print(hash(p1) == hash(p2))  # True
print({p1, p2})             # {Point(x=1, y=2)} — deduplicated
# p1.x = 99                # FrozenInstanceError — immutable!
```

**Comparison dunders — total ordering with `@functools.total_ordering`:**

```python
from functools import total_ordering

@total_ordering
class Temperature:
    """Comparable temperatures."""

    def __init__(self, celsius: float) -> None:
        self.celsius = celsius

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Temperature):
            return NotImplemented
        return self.celsius == other.celsius

    def __lt__(self, other: "Temperature") -> bool:
        if not isinstance(other, Temperature):
            return NotImplemented
        return self.celsius < other.celsius

    # @total_ordering automatically generates __le__, __gt__, __ge__
    # from __eq__ and __lt__

boiling = Temperature(100)
freezing = Temperature(0)
warm = Temperature(25)

print(freezing < warm < boiling)  # True
print(sorted([boiling, freezing, warm]))
# [Temperature(0), Temperature(25), Temperature(100)]
```

### 🔴 Expert: The Descriptor Protocol

Descriptors are objects that customize attribute access. They power `@property`, `@classmethod`, `@staticmethod`, and `__slots__`.

```python
# A descriptor is any object that defines __get__, __set__, or __delete__

class Validated:
    """A descriptor that validates assigned values."""

    def __init__(self, min_val: float = float("-inf"), max_val: float = float("inf")) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.storage_name: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        """Called when the descriptor is assigned to a class attribute."""
        self.storage_name = f"_validated_{name}"

    def __get__(self, obj: object, objtype: type | None = None) -> float:
        if obj is None:
            return self  # type: ignore[return-value]  # Access from class
        return getattr(obj, self.storage_name)

    def __set__(self, obj: object, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected number, got {type(value).__name__}")
        if not self.min_val <= value <= self.max_val:
            raise ValueError(
                f"Value {value} outside range [{self.min_val}, {self.max_val}]"
            )
        setattr(obj, self.storage_name, value)


class Sensor:
    temperature = Validated(min_val=-273.15, max_val=1000)  # Descriptor instance
    humidity = Validated(min_val=0, max_val=100)

    def __init__(self, temp: float, humidity: float) -> None:
        self.temperature = temp      # Calls Validated.__set__
        self.humidity = humidity

s = Sensor(25.0, 60.0)
print(s.temperature)  # 25.0 — calls Validated.__get__
# s.temperature = 2000  # ValueError: Value 2000 outside range [-273.15, 1000]
# s.humidity = "wet"    # TypeError: Expected number, got str
```

**How `@property` works — it's just a descriptor:**

```python
# property is a built-in descriptor class
# @property is syntactic sugar for creating one

class Circle:
    def __init__(self, radius: float) -> None:
        self._radius = radius

    @property
    def radius(self) -> float:
        """The circle's radius."""
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value

    @property
    def area(self) -> float:
        """Computed property — read-only."""
        import math
        return math.pi * self._radius ** 2

# Is equivalent to:
# radius = property(fget=get_radius, fset=set_radius, doc="The circle's radius.")
```

---

## 9.5 Abstract Base Classes, Protocols, and Structural Subtyping

### 🟢 Beginner: Abstract Base Classes (ABCs)

An ABC defines an **interface** — a contract that subclasses must fulfill.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    """Abstract base class — cannot be instantiated directly."""

    @abstractmethod
    def area(self) -> float:
        """Subclasses MUST implement this."""
        ...

    @abstractmethod
    def perimeter(self) -> float:
        """Subclasses MUST implement this."""
        ...

    def describe(self) -> str:
        """Concrete method — inherited by all subclasses."""
        return f"{type(self).__name__}: area={self.area():.2f}"

# shape = Shape()  # TypeError: Can't instantiate abstract class

class Rectangle(Shape):
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)

r = Rectangle(5, 3)
print(r.describe())      # "Rectangle: area=15.00"
print(r.area())          # 15.0
print(isinstance(r, Shape))  # True
```

### 🟡 Intermediate: Protocols — Structural Subtyping (Duck Typing Made Formal)

```python
# ABCs use NOMINAL subtyping: you must explicitly inherit
# Protocols use STRUCTURAL subtyping: you just need the right methods

from typing import Protocol, runtime_checkable

@runtime_checkable
class Drawable(Protocol):
    """Anything that has a draw() method is Drawable."""

    def draw(self, canvas: str) -> None:
        ...

class Circle:
    """Does NOT inherit from Drawable, but IS drawable."""

    def draw(self, canvas: str) -> None:
        print(f"Drawing circle on {canvas}")

class Square:
    """Also drawable without inheriting."""

    def draw(self, canvas: str) -> None:
        print(f"Drawing square on {canvas}")

class DatabaseConnection:
    """Not drawable — no draw() method."""

    def connect(self) -> None:
        ...


def render(shapes: list[Drawable]) -> None:
    """Accepts anything that has draw()."""
    for shape in shapes:
        shape.draw("main_canvas")

# This works — Circle and Square structurally match Drawable
render([Circle(), Square()])

# Runtime checking with @runtime_checkable
print(isinstance(Circle(), Drawable))           # True
print(isinstance(DatabaseConnection(), Drawable))  # False
```

**When to use ABCs vs. Protocols:**

```python
# Use ABCs when:
# - You want to share concrete method implementations
# - You need explicit registration (register())
# - The hierarchy is small and you control all implementations

# Use Protocols when:
# - You want duck typing with type checker support
# - You can't modify the classes you're checking against
# - You want minimal coupling (no inheritance required)
# - You're defining a library API that others implement

# Real-world Protocol example:
from typing import Protocol

class SupportsRead(Protocol):
    def read(self, n: int = -1) -> str: ...

def process_input(source: SupportsRead) -> str:
    """Works with files, StringIO, custom classes — anything with read()."""
    return source.read()

# All of these work:
from io import StringIO
process_input(StringIO("hello"))      # StringIO has .read()
process_input(open("file.txt"))       # File objects have .read()
```

### 🔴 Expert: ABC Registration and Virtual Subclasses

```python
from abc import ABC, abstractmethod

class MyCollection(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __contains__(self, item: object) -> bool: ...

# You can REGISTER a class as a "virtual subclass" without inheritance:
MyCollection.register(range)  # range now "is-a" MyCollection

print(isinstance(range(10), MyCollection))  # True!
print(issubclass(range, MyCollection))      # True!
# But range doesn't actually inherit anything from MyCollection.
# No methods are shared. It's purely for isinstance/issubclass checks.

# The collections.abc module uses this extensively:
from collections.abc import Sized, Iterable, Container

# dict "is" Sized, Iterable, Container, Mapping, etc.
# because these ABCs use __subclasshook__ to check for methods:
print(isinstance({}, Sized))      # True — has __len__
print(isinstance({}, Iterable))   # True — has __iter__
```

---

## 9.6 Metaclasses — When You Actually Need Them (Spoiler: Almost Never)

### 🟢 Beginner: What Is a Metaclass?

A metaclass is the "class of a class." Just as objects are instances of classes, classes are instances of metaclasses.

```python
class Dog:
    pass

# Dog is an instance of 'type' (the default metaclass)
print(type(Dog))        # <class 'type'>
print(type(Dog()))      # <class 'Dog'>
print(type(type))       # <class 'type'>  — type is its own metaclass!

# 'type' is both a class AND the metaclass of all classes:
#   object → base of all objects
#   type   → metaclass of all classes (including itself!)
```

```
The metaclass hierarchy:

    type (metaclass)
      │
      ├── creates → object (base class)
      ├── creates → int
      ├── creates → str
      ├── creates → Dog
      └── creates → type (itself!)

    When you write: class Dog: ...
    Python executes: Dog = type("Dog", (object,), {...})
```

### 🟡 Intermediate: `__init_subclass__` — The Modern Alternative

```python
# Before reaching for metaclasses, try __init_subclass__ (Python 3.6+)
# It's simpler and handles 95% of the use cases.

class Plugin:
    """Base class that auto-registers subclasses."""
    _registry: dict[str, type] = {}

    def __init_subclass__(cls, *, plugin_name: str = "", **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        name: str = plugin_name or cls.__name__.lower()
        Plugin._registry[name] = cls
        print(f"  Registered plugin: {name} → {cls.__name__}")

    @classmethod
    def get_plugin(cls, name: str) -> type:
        return cls._registry[name]


class CSVParser(Plugin, plugin_name="csv"):
    def parse(self, data: str) -> list:
        return data.split(",")

class JSONParser(Plugin, plugin_name="json"):
    def parse(self, data: str) -> dict:
        import json
        return json.loads(data)

# Registration happened automatically!
# Registered plugin: csv → CSVParser
# Registered plugin: json → JSONParser

parser_class = Plugin.get_plugin("csv")
parser = parser_class()
print(parser.parse("a,b,c"))  # ['a', 'b', 'c']
```

**Class decorators — another metaclass alternative:**

```python
import time
from typing import TypeVar

T = TypeVar("T")

def add_timestamps(cls: type[T]) -> type[T]:
    """Class decorator that adds created_at and updated_at tracking."""
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.created_at = time.time()
        self.updated_at = time.time()

    def touch(self):
        self.updated_at = time.time()

    cls.__init__ = new_init
    cls.touch = touch
    return cls

@add_timestamps
class User:
    def __init__(self, name: str) -> None:
        self.name = name

u = User("Alice")
print(u.created_at)   # Timestamp
u.touch()
print(u.updated_at)   # Updated timestamp
```

### 🔴 Expert: Writing a Real Metaclass

```python
class ValidatedMeta(type):
    """Metaclass that enforces type annotations at instance creation."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict,
        **kwargs: object,
    ) -> "ValidatedMeta":
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        return cls

    def __call__(cls, *args, **kwargs):
        """Called when creating an instance of a class using this metaclass."""
        # Create the instance normally
        instance = super().__call__(*args, **kwargs)

        # Validate all annotated attributes
        for attr_name, attr_type in getattr(cls, "__annotations__", {}).items():
            if hasattr(instance, attr_name):
                value = getattr(instance, attr_name)
                if not isinstance(value, attr_type):
                    raise TypeError(
                        f"{cls.__name__}.{attr_name} must be {attr_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
        return instance


class Config(metaclass=ValidatedMeta):
    host: str
    port: int
    debug: bool

    def __init__(self, host: str, port: int, debug: bool) -> None:
        self.host = host
        self.port = port
        self.debug = debug


c = Config("localhost", 8080, True)   # Works
# c = Config("localhost", "8080", True)  # TypeError: Config.port must be int, got str
```

**When to ACTUALLY use metaclasses:**

```python
# 1. Django ORM models — metaclass builds database schema from class attributes
# 2. SQLAlchemy declarative base — metaclass maps classes to tables
# 3. Enum — metaclass ensures unique values and prevents instantiation
# 4. ABCMeta — the metaclass behind ABC

# Decision flowchart:
# Need to customize class CREATION?
#   → Try __init_subclass__ first
#   → Try class decorator
#   → Only then consider metaclass
#
# Need to customize instance CREATION?
#   → Try __init__ and __new__
#   → Try __init_subclass__
#   → Only then consider metaclass.__call__
```

---

## 🔧 Debug This: The Broken Shape Hierarchy

Your team built a geometry library. It has bugs across inheritance, dunder methods, and composition. Find them all:

```python
import math

class Shape:
    def area(self):
        return 0

    def __eq__(self, other):
        return self.area() == other.area()

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2

    def __repr__(self):
        return f"Circle(radius={self.radius})"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

class Square(Rectangle):
    def __init__(self, side):
        super().__init__(side, side)

    def set_width(self, width):
        self.width = width
        # Should self.height update too?

# Test
c1 = Circle(5)
c2 = Circle(5)
r1 = Rectangle(3, 4)
s1 = Square(3)

print(c1 == c2)           # True — but can we put them in a set?
print({c1, c2})            # Possible TypeError?

print(s1.area())           # 9
s1.set_width(5)
print(s1.area())           # 15? But it should be a square!
print(isinstance(s1, Rectangle))  # True — but is a Square really a Rectangle?
```

### Bugs to find:

```
1. ____________________________________________________
   Hint: __eq__ is defined but __hash__ is not. Can Circle
   be used in sets and as dict keys?

2. ____________________________________________________
   Hint: __eq__ compares areas. A circle with area 12 would
   equal a rectangle with area 12. Is that correct?

3. ____________________________________________________
   Hint: Shape.area() returns 0. Should Shape be instantiable?
   Should it be an ABC?

4. ____________________________________________________
   Hint: Square(3).set_width(5) makes width=5, height=3.
   That's not a square anymore! This is the Liskov Substitution
   Principle (LSP) violation.

5. ____________________________________________________
   Hint: No type hints anywhere. No __repr__ on Rectangle or
   Square. No input validation.
```

### Solution (try first!)

```python
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import total_ordering


@total_ordering
class Shape(ABC):
    """Abstract base class for shapes. Cannot be instantiated directly."""

    # Bug 3 FIX: Make Shape abstract
    @abstractmethod
    def area(self) -> float:
        """Return the area of this shape."""
        ...

    @abstractmethod
    def perimeter(self) -> float:
        """Return the perimeter of this shape."""
        ...

    # Bug 2 FIX: Only compare shapes of the same type
    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self.area() == other.area()

    # Bug 1 FIX: Define __hash__ since we defined __eq__
    def __hash__(self) -> int:
        return hash((type(self).__name__, self.area()))

    def __lt__(self, other: "Shape") -> bool:
        if not isinstance(other, Shape):
            return NotImplemented
        return self.area() < other.area()


@dataclass(frozen=True)  # Immutable — can't change radius after creation
class Circle(Shape):
    radius: float

    def __post_init__(self) -> None:
        if self.radius < 0:
            raise ValueError(f"Radius must be non-negative, got {self.radius}")

    def area(self) -> float:
        return math.pi * self.radius ** 2

    def perimeter(self) -> float:
        return 2 * math.pi * self.radius


@dataclass(frozen=True)  # Immutable
class Rectangle(Shape):
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValueError("Dimensions must be non-negative")

    def area(self) -> float:
        return self.width * self.height

    def perimeter(self) -> float:
        return 2 * (self.width + self.height)


# Bug 4 FIX: Square is NOT a subclass of Rectangle.
# A square that can independently change width/height isn't a square.
# Either make both immutable (data classes) or use composition.
@dataclass(frozen=True)
class Square(Shape):
    side: float

    def __post_init__(self) -> None:
        if self.side < 0:
            raise ValueError("Side must be non-negative")

    def area(self) -> float:
        return self.side ** 2

    def perimeter(self) -> float:
        return 4 * self.side


# Now everything works correctly
c1 = Circle(5)
c2 = Circle(5)
print(c1 == c2)              # True
print({c1, c2})              # {Circle(radius=5)} — deduplication works!
print(c1 == Rectangle(3, 4))  # NotImplemented → False (different types)

s1 = Square(3)
print(s1.area())       # 9
# s1.side = 5          # FrozenInstanceError — can't break the invariant!

# Create a new square instead
s2 = Square(5)
print(s2.area())       # 25 — always a valid square
```

```
Bug Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Missing __hash__: __eq__ without __hash__ makes instances
   unhashable. Sets and dict keys fail with TypeError.

2. Cross-type equality: A circle shouldn't equal a rectangle
   just because they happen to have the same area. Compare
   same-type only (or return NotImplemented).

3. Concrete Shape: Shape.area() returning 0 is misleading.
   Shape should be abstract (ABC) to prevent instantiation
   and force subclasses to implement area()/perimeter().

4. LSP violation: Square inherits set_width() from Rectangle,
   but changing width without height breaks the square invariant.
   Fix: make Square independent (not a Rectangle subclass) and
   immutable (frozen dataclass).

5. Missing types: No type hints, no validation, no __repr__
   on Rectangle/Square. Using @dataclass fixes all three.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Summary: Module 9 Key Takeaways

```
┌──────────────────────────────────────────────────────────────────┐
│                       OOP MASTERY CHEAT SHEET                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  CLASSES:                                                        │
│    Class attrs: shared (beware mutable!). Instance attrs: unique.│
│    Lookup chain: instance → class → bases → __getattr__.         │
│    __slots__: 40% less memory, no arbitrary attrs, faster access.│
│                                                                   │
│  INHERITANCE vs COMPOSITION:                                     │
│    "Is-a" → Inheritance. "Has-a" → Composition.                  │
│    When in doubt → Composition.                                   │
│    Keep inheritance ≤ 2 levels deep.                             │
│    Mixins: single-purpose, no __init__, listed before base.     │
│                                                                   │
│  MRO:                                                            │
│    C3 linearization. super() follows MRO, not "parent."         │
│    D(B, C) → D, B, C, A, object. Use **kwargs for cooperative MI.│
│                                                                   │
│  DUNDER METHODS:                                                 │
│    __repr__: unambiguous (for developers). __str__: pretty.      │
│    __eq__ defined → MUST define __hash__ (or set to None).       │
│    Return NotImplemented from comparisons (not raise).           │
│    @dataclass(frozen=True): auto __init__, __repr__, __eq__,     │
│    __hash__ — the easy button for value objects.                 │
│                                                                   │
│  DESCRIPTORS:                                                    │
│    __get__/__set__/__delete__ — customize attribute access.      │
│    Powers @property, @classmethod, @staticmethod, __slots__.     │
│                                                                   │
│  ABCs vs PROTOCOLS:                                              │
│    ABC: nominal (must inherit). Share concrete methods.          │
│    Protocol: structural (duck typing). No inheritance needed.    │
│    Prefer Protocol for library APIs. ABC for internal hierarchies.│
│                                                                   │
│  METACLASSES:                                                    │
│    type is the default metaclass. Classes are instances of type. │
│    Try __init_subclass__ or class decorators FIRST.              │
│    Metaclasses only for framework-level magic (ORM, Enum, etc). │
│                                                                   │
│  Production rules:                                               │
│    Liskov Substitution: subclass instances must be valid         │
│    wherever parent instances are expected.                        │
│    Favor @dataclass for data containers.                         │
│    Favor Protocol for type-checked duck typing.                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

**Next up → Module 10: Advanced Pythonic Tools — The Professional's Toolkit**

Say "Start Module 10" when you're ready for the final module.
