# Python OOP Curriculum - Interactive HTML Explainers

This directory contains four comprehensive interactive HTML explainer pages for teaching Object-Oriented Programming in Python. All chapters are Phase V (orange #fb923c accent) with dark theme styling.

## Files

### chapter_20.html - Classes & Objects — OOP Foundations
**1,169 lines** | Covers fundamental OOP concepts

- Programming Paradigms recap (why OOP?)
- Classes as blueprints vs objects as instances
- User-defined classes (class keyword, __init__, self)
- Access conventions (public, _protected, __private name mangling)
- Class variables and methods (@staticmethod, @classmethod)
- vars() and dir() for object introspection
- Object initialization best practices
- Expert section: __dict__, __new__ vs __init__, __del__, __slots__
- 12 Socratic Q&A items with reveal buttons

### chapter_21.html - OOP Intricacies — Dunders & Overloading
**1,173 lines** | Dunders, operator overloading, type conversion

- PEP 8 naming conventions (snake_case vs CamelCase)
- Bound vs unbound methods
- Operator overloading (__add__, __sub__, __mul__, __eq__, __lt__, etc.)
- When to overload operators (Principle of Least Surprise)
- Everything is an object (int, str, function, classes all have types)
- Type conversion dunders (__int__, __float__, __str__, __repr__, __bool__)
- Classes as C-like structs
- Expert section: Descriptor protocol, data vs non-data descriptors, __slots__, rich comparison methods
- 12 Socratic Q&A items

### chapter_22.html - Inheritance & Composition
**1,353 lines** | Code reuse strategies (LARGEST)

- Composition vs inheritance ("has-a" vs "is-a")
- Containership (composition in action)
- Inheritance fundamentals (class Child(Parent), super())
- Method Resolution Order (MRO)
- isinstance() and issubclass() type checking
- The object class (root of all classes)
- Method overriding and extending
- Types of inheritance (single, multiple, multilevel, hierarchical, hybrid)
- Diamond problem and C3 linearization
- Abstract Base Classes (ABC, @abstractmethod)
- Runtime polymorphism and duck typing
- Expert section: super() mechanics, Liskov Substitution Principle, Protocols vs ABCs, mixin pattern
- 12 Socratic Q&A items

### chapter_23.html - Dataclasses, Typing & Metaclasses (BONUS CHAPTER)
**1,212 lines** | Modern Python features beyond core OOP

- @dataclass decorator (auto-generating __init__, __repr__, __eq__)
- Dataclass fields (field(), default_factory, frozen=True)
- Type hints and the typing module (List, Dict, Optional, Union, etc.)
- Generic types (TypeVar, Generic[T])
- Protocols (structural subtyping without inheritance)
- Metaclasses (type as metaclass, custom metaclasses)
- When NOT to use metaclasses (almost never)
- __init_subclass__ as lightweight alternative
- Dataclass vs attrs vs Pydantic comparison
- Expert section: ParamSpec for decorators, descriptor-based validation, __prepare__, metaclass conflicts
- 12 Socratic Q&A items

## Features

All files include:

✓ **Sticky top navigation** with chapter title and info
✓ **Hero section** with chapter title and subtitle
✓ **Mental model boxes** explaining conceptual frameworks
✓ **Progressive sections** building complexity
✓ **Toggleable deep-dive boxes** for advanced concepts (7+ per chapter)
✓ **Code examples** with Prism.js syntax highlighting (Tomorrow Night theme)
✓ **10+ Socratic Q&A** with reveal buttons for testing understanding
✓ **Dark theme** (#0a0a0f background, #fb923c orange accents)
✓ **Typography** using Inter (body), JetBrains Mono (code), Playfair Display (headings)
✓ **Responsive design** for mobile viewing
✓ **KaTeX support** for mathematical notation (if needed in future)
✓ **Inline CSS** (no external stylesheets except fonts and libraries)

## Technical Stack

- **HTML5** with semantic structure
- **CSS3** with flexbox and grid, inline for portability
- **Prism.js v1.29.0** for syntax highlighting (Python support)
- **KaTeX v0.16.0** for mathematical notation
- **Google Fonts** for custom typography
- **Vanilla JavaScript** for interactivity (no jQuery)

## Interactivity

- Click deep-dive headers to expand/collapse detailed explanations
- Click Q&A questions to reveal answers
- Smooth animations for all toggles
- Keyboard-accessible (can tab through sections)

## Content Quality

- **Comprehensive:** 4,907 total lines spanning 4 chapters
- **Pedagogical:** Socratic method with reveal-based learning
- **Progressive:** Builds from fundamentals to advanced patterns
- **Practical:** Real-world code examples throughout
- **Expert insights:** Advanced topics for intermediate learners

## Usage

Open any HTML file in a modern web browser. No build process, dependencies, or server required. All assets load from CDNs.

## Curriculum Alignment

Chapters align with Phase V (OOP) of the Python curriculum:
- Ch 20-22 cover core PDF content (Chs 18-20)
- Ch 23 extends with modern Python features (bonus material)
