# PyPI & PEP 8 Improvement Checklist

This document outlines changes to make the package more professional, pip-installable, and PEP 8 compliant.

---

## 🚨 Critical: Package Structure

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| Rename `src/` to `ehteval/` | ❌ | High | `src` is too generic; `ehteval` is short and memorable |
| Rename repo to `ehteval` | ❌ | High | Align repo name with package name |
| Move `evaluate.py` into package | ❌ | High | Currently in root; should be `ehteval/cli.py` |
| Add CLI entry points | ❌ | High | Users can't run `ehteval` after `pip install` |
| Update `pyproject.toml` package name | ❌ | High | Change `name` and `packages` to `ehteval` |

**Proposed structure:**
```
ehteval/
├── ehteval/
│   ├── __init__.py
│   ├── cli.py          # moved from evaluate.py
│   ├── chisq.py
│   ├── nxcorr.py
│   └── ...
├── tests/
├── pyproject.toml
└── README.md
```

---

## 📦 pyproject.toml Improvements

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| Add `[project.scripts]` entry points | ❌ | High | `ehteval = "ehteval.cli:main"` |
| Pin `ehtplot` to specific commit | ❌ | Medium | Current git+ dep may break; pin with `@<commit>` |
| Add `[project.optional-dependencies]` for Julia | ❌ | Low | Document `julia` extras group for optional deps |
| Add Python 3.12 classifier | ❌ | Low | Test and add if compatible |

---

## 🐍 PEP 8 Compliance

### Naming Conventions (PEP 8)

| File | Issue | Priority |
|------|-------|----------|
| All files | Function names use `mixedCase` (e.g., `calcWidth`) | Medium |
| All files | Should be `snake_case` (e.g., `calc_width`) | Medium |
| `nxcorr.py` | `jensen_shannon_distance` ✅ correct | — |
| `rex.py` | `extract_ring_quantites` → `extract_ring_quantities` (typo) | Low |

**Non-compliant function names to fix:**
- `create_parser` ✅ (correct)
- `compute_ramesh_metric` ✅ (correct)
- `quad_interp_radius` ✅ (correct)
- `calc_width` ✅ (correct)
- But inconsistent: check all files for `camelCase` vs `snake_case`

### Line Length & Formatting

| Item | Status | Notes |
|------|--------|-------|
| Lines > 79/99 chars | ❌ Many | Use `black` or `ruff format` to auto-fix |
| Inconsistent spacing | ❌ Some | `func(arg1,arg2)` vs `func(arg1, arg2)` |
| Trailing whitespace | ❌ Unknown | Run linter to check |

### Imports (PEP 8)

| Issue | Files | Notes |
|-------|-------|-------|
| Wildcard imports | `rex.py` line 35 | `from ehtim.const_def import *` should be explicit |
| Import order | All files | Should be: stdlib → third-party → local |
| Unused imports | Unknown | Run `ruff` or `flake8` to detect |

### Documentation (PEP 257)

| Item | Status | Notes |
|------|--------|-------|
| Module docstrings | ❌ Missing | Files lack `"""Module description."""` at top |
| Function docstrings | ⚠️ Partial | Some functions have docstrings, many don't |
| Type hints | ❌ Missing | No type annotations (PEP 484) |
| Docstring style | ⚠️ Mixed | Standardize on numpy or Google style |

---

## 🔧 Code Quality

### Warnings & Errors

| Item | File | Notes |
|------|------|-------|
| Blanket `warnings.filterwarnings('ignore')` | All files | Should be specific or removed |
| `print()` statements for logging | All files | Consider using `logging` module |
| Bare `except:` clauses | Check all | Should catch specific exceptions |

### Code Organization

| Item | Status | Priority |
|------|--------|----------|
| Create `utils.py` for shared code | ❌ | Medium |
| Threading env vars repeated in every file | ❌ | Medium |
| Matplotlib style config repeated | ❌ | Low |

**Example repeated code (appears in all files):**
```python
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
```

---

## 🧪 Testing

| Item | Status | Notes |
|------|--------|-------|
| Tests use `subprocess` to run scripts | ⚠️ | Works, but can't test individual functions |
| No unit tests for helper functions | ❌ | Add tests for `calc_width`, `pnxcorr`, etc. |
| Test fixtures | ⚠️ | Using real data files; consider mocking |
| CI/CD pipeline | ❌ | Add GitHub Actions workflow |
| Coverage reporting | ❌ | Add `pytest-cov` to dev deps |

---

## 📝 Documentation

| Item | Status | Notes |
|------|--------|-------|
| API docs (Sphinx/mkdocs) | ❌ | No auto-generated docs |
| CHANGELOG.md | ❌ | Missing |
| CONTRIBUTING.md | ❌ | Missing |
| Examples directory | ❌ | Could add Jupyter notebooks |

---

## 🚀 Quick Wins (Low Effort, High Impact)

1. [ ] Run `ruff format .` to auto-fix formatting
2. [ ] Run `ruff check . --fix` to fix import order and unused imports
3. [ ] Add `[project.scripts]` to `pyproject.toml`
4. [ ] Fix `extract_ring_quantites` typo
5. [ ] Add module-level docstrings to all files
6. [ ] Replace wildcard import in `rex.py`

---

## 📋 Recommended Tool Configuration

Add to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 99
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "D"]
ignore = ["D100", "D104"]  # Allow missing module/package docstrings initially

[tool.ruff.lint.pydocstyle]
convention = "numpy"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

---

## Summary

| Category | Ready | Needs Work |
|----------|-------|------------|
| Package Structure | ❌ | Rename `src/`, add entry points |
| PEP 8 Naming | ⚠️ | Mostly OK, some fixes needed |
| PEP 8 Formatting | ❌ | Run auto-formatter |
| Documentation | ❌ | Add docstrings and type hints |
| Testing | ⚠️ | Works but could be improved |
| Dependencies | ⚠️ | Pin `ehtplot` commit |

**Overall: Not ready for PyPI publication without addressing Critical items.**
