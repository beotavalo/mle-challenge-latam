---
name: git-best-practices
description: Apply the team's version-control standards — Conventional Commits formatting for commit messages, and Git submodules for managing large data and model files. Use this skill whenever working with Git: writing or rewording a commit message, committing changes, setting up a repository, deciding how to store large data or model artifacts, configuring submodules, or whenever the user mentions commits, commit conventions, changelogs, .gitmodules, or version control. Apply it even when the user just says "commit this" without naming a convention.
---

# Git Best Practices

This skill encodes the team's version-control conventions. Apply them whenever generating commits or structuring a repository, and explain the reasoning so the user understands why a convention helps.

## Commit message standards — Conventional Commits

Follow the Conventional Commits specification. A well-structured history enables automated versioning, automated changelog generation, a clear change history, and an easier code review process.

Format:

```
<type>[optional scope]: <description>
```

Types:

- `feat` — new features
- `fix` — bug fixes
- `docs` — documentation changes
- `style` — formatting changes (no logic change)
- `refactor` — code refactoring
- `test` — adding, modifying, or running tests of features
- `chore` — maintenance tasks
- `perf` — performance improvements
- `build` — changes to the build system or external dependencies
- `ci` — changes to CI configuration files and scripts

Write the description in the imperative mood and keep it concise. Add a scope in parentheses when it clarifies what area changed.

**Examples:**

```
feat(preprocessing): add new data normalization pipeline
fix(model): resolve memory leak in batch processing
docs(readme): update installation instructions
test(coverage): increase unit test coverage to 70%
```

When asked to commit changes, infer the correct type and scope from the actual diff rather than guessing, and prefer one logical change per commit so the history stays meaningful.

## Data and model management — use Git submodules

Keep large data and model files out of the main repository by storing them in separate repositories referenced as Git submodules. This keeps the main repo small, speeds up CI/CD pipelines, gives better version control for large files, and allows flexible access control for sensitive data.

Approach:

1. Create separate repositories for data and for models.
2. Reference them in the main project via `.gitmodules`.

Example `.gitmodules` configuration:

```
[submodule "data"]
    path = data
    url = git@github.com:organization/project-data.git
[submodule "models"]
    path = models
    url = git@github.com:organization/project-models.git
```

Remember that submodules pin a specific commit of each referenced repository — update the pointer deliberately and commit that change so collaborators pull the intended versions.
