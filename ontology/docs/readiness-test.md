# Documentation Readiness Test

## Goal

Determine whether documentation is sufficient for a fresh agent session to execute ontology work autonomously.

## Test Scenario

1. Start a new session.
2. Read only:
- `ontologist/AGENTS.md`
- `ontologist/STATE.md`
- `README.md`
- `docs/architecture.md`
- `docs/design-spec.md`
- `docs/source-policy.md`
- `docs/methodology.md`
3. Execute task:
- "Add 10 terms for one use case from 3 approved sources, with full provenance and module placement."

## Pass/Fail Criteria

1. Clarification questions
- Pass: `<=2`
- Fail: `>2`

2. Source compliance
- Pass: 0 policy violations
- Fail: any Tier C or unreviewed unclear source use

3. Provenance completeness
- Pass: 100% required fields populated
- Fail: any missing required provenance field

4. Design consistency
- Pass: no namespace or module-boundary violations
- Fail: any duplicate/redefined core semantics in modules

5. Validation outcome
- Pass: all defined checks pass
- Fail: any blocking validation check fails

6. Handoff quality
- Pass: `TASKS.md`, `SESSIONS.md`, `WORKLOG.md`, `STATE.md` updated correctly
- Fail: missing or inconsistent handoff updates

## Sufficiency Rule

Documentation is considered sufficient only if the test passes in two separate fresh sessions.

## Failure Handling

If test fails:

1. Record failure cause in `WORKLOG.md`.
2. Add a remediation task in `TASKS.md`.
3. Update the relevant documentation file.
4. Re-run the test.
