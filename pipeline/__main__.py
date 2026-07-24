"""Entrypoint so the whole pipeline runs with `python -m pipeline`."""

from .main import main

if __name__ == "__main__":
    raise SystemExit(main())
