# Contributing

Working agreements for this repository. They exist so the delivery history is readable and every
change arrives with the same quality guarantees.

## Branching model (GitFlow)

| Branch | Purpose |
|---|---|
| `main` | Official release branch. This is what the challenge reviewer reads. Always green. |
| `develop` | Integration branch. Features land here first. |
| `feature/mle-00X-<semantic-feature>` | One branch per ticket. **Never deleted**, so the delivery order stays auditable. |

Flow: `feature/*` → PR → `develop` → release PR → `main`.

## Tickets

Ticket id format: `MLE-00X`, referenced in branch names, commit scopes and PR titles as
`[MLE-00X semantic-feature]`.

## Commits — Conventional Commits

```
<type>[optional scope]: <description>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`, `build`, `ci`.
Imperative mood, one logical change per commit.

## Definition of done

A change is done when all of the following pass locally and in CI:

```bash
make lint        # ruff check + format check
make typecheck   # mypy
make security    # pip-audit dependency vulnerability scan
make model-test  # model acceptance tests + coverage
make api-test    # API acceptance tests + coverage
```

Tests come first (TDD): write the failing test, then the implementation.

## Local environment

```bash
uv venv --python 3.10
uv pip install -r requirements.txt -r requirements-dev.txt -r requirements-test.txt
```

The pinned dependency set from the challenge is kept as-is; Python 3.10 is the highest interpreter
with wheels for `numpy 1.22.4` / `pandas 1.3.5`.
