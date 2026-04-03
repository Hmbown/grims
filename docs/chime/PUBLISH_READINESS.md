# Publish Readiness Assessment

Date: 2026-04-03

## Current status

`chime` is close to GitHub-release ready from this repository, but it is not fully publish-ready yet.

## Ready now

- The package is structurally self-contained under [`chime/`](/Volumes/VIXinSSD/drbown/bown-power/chime).
- Packaging metadata exists in [`chime/pyproject.toml`](/Volumes/VIXinSSD/drbown/bown-power/chime/pyproject.toml).
- Citation metadata exists in [`chime/CITATION.cff`](/Volumes/VIXinSSD/drbown/bown-power/chime/CITATION.cff).
- Changelog and package README are present.
- The package includes tests, examples, and checked-in result artifacts.
- Author metadata now names Hunter Bown in package and citation files.

## Blocking items before public release

- The package changes still need to be committed and pushed as a coherent release unit. This workspace also contains unrelated repo changes outside `chime/`, so release scope should be chosen deliberately when creating the GitHub commit or PR.

## Recommended before release

- Decide whether `chime` will live in this monorepo long-term or move to its own repo later. If it moves, update all package URLs again at that time.
- Keep [`chime/results/`](/Volumes/VIXinSSD/drbown/bown-power/chime/results) only if you want reproducibility material in the public repo. Local run output under `chime_output/` should stay ignored rather than checked in.
- If PyPI publication is planned, verify the project name `chime-jwst` is the intended final package name.

## Verified checks

- `pytest -q chime/tests`: 50 passed
- `python3 -m compileall chime/src`: passed
- `python3 -m build --sdist --wheel chime`: passed
- Wheel install into a fresh virtualenv: passed
- CLI smoke tests from installed wheel:
  - `chime --help`: passed
  - `chime --targets`: passed

## Remaining uncertainty

- Tests were not rerun inside the fresh virtualenv after wheel installation; only install and CLI smoke checks were performed there.
