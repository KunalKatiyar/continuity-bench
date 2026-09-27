"""Shared runner so each test file's __main__ block is one line.

Every test file here is plain asserts with no fixtures, so `python3 test_x.py` and
pytest both work on the same functions and neither needs a framework.
"""


# unittest's own, so pytest reports these as skips too rather than errors. A guard
# that silently returns counts as ok in both runners, which is how six corpus guards
# came to report green on a fresh clone with no corpus - green and vacuous, exactly
# when they are most needed.
from unittest import SkipTest as Skipped


def run(namespace):
    """Run every test_* callable in namespace, print results, return the failure count."""
    failures = skipped = 0
    for name, fn in sorted(namespace.items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"ok   {name}")
        except Skipped as exc:
            skipped += 1
            print(f"skip {name}: {exc}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    tail = f", {skipped} skipped" if skipped else ""
    print(f"\n{failures} failure(s){tail}")
    return failures
