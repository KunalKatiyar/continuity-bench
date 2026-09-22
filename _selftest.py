"""Shared runner so each test file's __main__ block is one line.

Every test file here is plain asserts with no fixtures, so `python3 test_x.py` and
pytest both work on the same functions and neither needs a framework.
"""


def run(namespace):
    """Run every test_* callable in namespace, print results, return the failure count."""
    failures = 0
    for name, fn in sorted(namespace.items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"ok   {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:
            failures += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    return failures
