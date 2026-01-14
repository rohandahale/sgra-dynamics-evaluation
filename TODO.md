# PyPI Release Readiness Checklist

---

| Item | Status | Notes |
|------|--------|-------|
| `pyproject.toml` or `setup.py` | ❌ Missing | **Required** - No build configuration file exists |
| `setup.cfg` | ❌ Missing | Alternative config (optional if using pyproject.toml) |
| `src/__init__.py` | ❌ Missing | Package not importable as a Python module |
| `MANIFEST.in` | ❌ Missing | Needed to include non-Python files (Julia files, YAML, etc.) |

---

## Package Structure Issues

| Item | Status | Notes |
|------|--------|-------|
| Proper package layout | ❌ Not structured | Scripts are standalone files, not importable modules |
| Version number | ❌ Missing | No `__version__` defined anywhere |
| Entry points / CLI | ❌ Missing | No console script entry points defined |

---

| Item | Status | Notes |
|------|--------|-------|
| LICENSE | ✅ Present | MIT License (good) |
| README.md | ✅ Present | Needs PyPI-specific badges and installation instructions |
| Tests | ✅ Present | Good test coverage in `tests/` directory |
| .gitignore | ✅ Present | Already excludes build artifacts |
| Documentation | ✅ Present | `docs.md` exists |

---

## Dependency & Distribution

| Item | Issue |
|------|-------|
| `environment.yml` | Conda-based; pip requires `requirements.txt` or deps in `pyproject.toml` |
| Julia dependency | Complex: `julia=1.10.9`, `juliacall`, Julia packages via `Project.toml` - requires special handling |
| Git-based pip deps | `ehtplot` installed from GitHub - should pin to version or host on PyPI |
| System deps | `fftw`, `nfft`, `pynfft` - may need special build instructions |

---

## Summary of Changes Required

1. [ ] **Create `pyproject.toml`** with build system, metadata, dependencies, and entry points
2. [ ] **Add `src/__init__.py`** to make `src` an importable package (consider renaming to `sgra_dynamics_evaluation`)
3. [ ] **Add `__version__`** to the package
4. [ ] **Create `MANIFEST.in`** to include Julia files (`*.jl`, `Project.toml`, `Manifest.toml`), YAML files, and documentation
5. [ ] **Update README.md** with pip install instructions, badges, and non-conda installation notes
6. [ ] **Address Julia dependency** - document Julia installation requirement clearly or explore alternatives
7. [ ] **Pin dependency versions** in pyproject.toml for reproducibility
8. [ ] **Consider renaming** `src/` to a proper package name like `sgra_dynamics_evaluation/`
