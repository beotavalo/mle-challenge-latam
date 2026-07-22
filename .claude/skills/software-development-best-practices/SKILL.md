---
name: software-development-best-practices
description: Apply the team's software development standards covering requirements validation, documentation, readable code, SOLID principles, testing, security (OWASP), code reviews, observability, dependency management, performance, accessibility, and responsible GenAI usage. Use this skill whenever writing, reviewing, or planning code — including starting a new ticket or feature, opening or reviewing a pull request, deciding how to structure or name code, adding tests or logging, evaluating third-party libraries, or whenever the user mentions best practices, coding standards, code quality, code review, or "the right way" to build something. Apply it even when the user doesn't explicitly ask for "standards."
---

# Software Development Best Practices

This skill encodes the team's engineering standards. Apply the relevant sections proactively during development rather than waiting to be asked. These recommendations should evolve based on client needs, industry trends, and internal feedback — treat them as a living standard, not rigid law, and explain trade-offs when a situation warrants deviation.

Use the section relevant to the current task. You don't need to recite every rule; pull in what applies and explain the reasoning so the user understands the "why."

## 1. Validate requirements before coding

A surprising amount of wasted effort comes from building the wrong thing. Before writing code:

- Read the requirement and confirm it has a clear description and acceptance criteria.
- Raise any questions that surface on first read — ambiguity is cheapest to resolve early.
- Define a step-by-step plan to accomplish the requirement.
- Check internally (teammates, Team Lead) and in the client's existing systems whether part of the solution already exists, so you can reuse code instead of reinventing it.
- Share your intended strategy in the ticket and tag the Team Lead for feedback before diving in.

## 2. Write comments and documentation

Documentation guides other developers through the logic you implemented, and it should live alongside the code and stay current.

- Comment complex objects and non-obvious logic; explain intent, not just mechanics.
- When you change code, update the associated comments and documentation in the same change — stale docs are worse than none.
- Use OpenAPI/Swagger for APIs. (Missing Swagger docs was a key gap in the UCR project; enforce API documentation consistently to avoid repeating it.)
- Use documentation generators like JSDoc or Sphinx.
- Review documentation during code reviews, not as an afterthought.

## 3. Write readable yet efficient code

Readable code is easy to follow while still using time and space sensibly. Compressing a function into one clever line usually trades clarity for nothing. Favor clarity.

Naming conventions:

- Classes: PascalCase — `class VectorImage {}`
- Methods: camelCase — `drawImage()`
- Variables: camelCase with meaningful names — `newImageName`, never `var1`.

## 4. Adhere to the SOLID principles

- **Single responsibility:** a class should have one reason to change.
- **Open–closed:** entities should be open for extension but closed for modification.
- **Liskov substitution:** subtypes must be substitutable for their base types without breaking correctness.
- **Interface segregation:** prefer many client-specific interfaces over one general-purpose interface.
- **Dependency inversion:** depend on abstractions, not concretions.

## 5. Always unit test

Unit tests validate that individual components behave as expected.

- A unit test (manual or automated) is required to ensure code quality.
- Where the project allows, practice Test-Driven Development.
- Require tests for every bug fix and new feature.

## 6. Automate standards checking

Use tools that automatically check code against verified standards:

- Frontend — Airbnb style guide / linters.
- Backend — SonarQube.

## 7. GenAI usage guidelines

Clarify when and how GenAI tools (e.g., ChatGPT, GitHub Copilot) are used.

- Never share proprietary code, PII, or client data with public GenAI tools.
- Manually validate and peer-review any GenAI-assisted code.
- Disclose GenAI use in PR descriptions and code reviews (e.g., "Partially generated with GenAI and reviewed for accuracy"), keeping an audit trail.
- Only use enterprise-grade GenAI tools with proper security controls.

GenAI **may** be used for: boilerplate or common patterns, idea generation for problem-solving, and drafting documentation or comments.

GenAI **should not** be used for: core business logic, proprietary algorithms, or directly handling/generating sensitive data.

## 8. Secure coding practices

- Align development with the OWASP Top 10.
- Run regular security training for developers.
- Integrate static analysis (e.g., SonarQube) to detect vulnerabilities.
- Apply a secure-coding checklist during code reviews.

## 9. Code review process

Peer reviews catch roughly 60% of defects, so treat reviews as a primary quality gate.

- Use review checklists focused on readability, security, and testing.
- Require at least one technical reviewer and one peer approval.
- Document reviews in GitHub/GitLab with clear feedback and explicit approvals.
- For especially complex or sensitive work, consider pair programming until you're confident — but get approval from your Team Lead or Project Manager first, since clients often view it as costly.

## 10. Testing strategy (beyond unit tests)

- Cover all critical paths with integration and end-to-end tests.
- Run automated test suites in CI/CD pipelines.
- Set code coverage targets (e.g., 80%+ for unit tests).
- Require tests for bug fixes and new features.

## 11. Observability and logging

- Build proper observability for troubleshooting and monitoring.
- Use structured logging (JSON or key-value pairs).
- Define and document log levels (DEBUG, INFO, WARN, ERROR).
- Never log sensitive data.
- Centralize logs with tools like ELK or Datadog.

## 12. Dependency management

- Audit and update dependencies regularly to prevent vulnerabilities.
- Use automated tools like Dependabot or Snyk.
- Schedule quarterly dependency reviews.
- Verify licenses align with client agreements before introducing a library.

## 13. Git workflow standardization

- Standardize branching and merging (GitFlow or trunk-based development).
- Use meaningful commit messages following the Conventional Commits standard.
- Enforce PR templates with required sections (Summary, Testing, Risks).

(For detailed commit and version-control conventions, see the companion git best practices skill.)

## 14. Performance and scalability

- Design with scalability in mind from the start.
- Profile code with tools like New Relic or Datadog.
- Apply caching strategies appropriately.
- Monitor latency and resource usage.

## 15. Internationalization (i18n) and accessibility (a11y)

- Use i18n libraries (e.g., i18next) so products are globally ready.
- Test UI with accessibility tools (e.g., Lighthouse, axe-core).
- Follow WCAG 2.1 guidelines.

## Final recommendations

- Push to the Git repository at the end of every day to avoid losing work.
- Use defined development and deployment flows.
- Before implementing new functionality, check whether something reusable already exists — "don't reinvent the wheel." Take a holistic view: if a piece of functionality will be needed in more than one place, design for that. Always double-check library licenses with the client before importing.
