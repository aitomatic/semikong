# Plan: 07-WFE-06-Etch-Systems Near-100 Completion

## Objective

Bring `ontology/07-wfe/06-etch-systems/ontology.ttl` to near-100% completion under Semicont quality gates while preserving industry-general scope.

## Scope

In scope:
- `ontology/07-wfe/06-etch-systems/ontology.ttl`
- Context/runtime tracking updates in `ontologist/04-context/`

Out of scope:
- Vendor-specific proprietary semantics
- Non-TTL artifacts under `ontology/`
- Repo-wide refactors unrelated to etch systems

## Execution Plan

1. Baseline and Gap Audit
- Run mandatory startup checklist from `ontologist/AGENTS.md`.
- Run `python ontologist/05-analytics/tools/ontology_audit.py`.
- Create a facet gap matrix for all required facets from:
  - `ontologist/01-docs/templates/prompts/facets-dimensions-template.md`

2. Provenance Completion
- Ensure substantive classes/properties have class-level provenance:
  - `dc:source`
  - `prov:wasInformedBy`
- Use internet-accessible primary/authoritative sources.
- Keep copyright-safe paraphrase only.

3. Facet Hardening
- Close gaps in:
  - capabilities/constraints
  - settings/parameterization
  - lifecycle/state semantics
  - specifications/quality criteria
  - measurement/units consistency

4. SHACL Strengthening
- Add/upgrade SHACL node/property constraints for:
  - required fields
  - cardinalities
  - datatypes
  - key numeric bounds/ranges

5. External Alignment
- Add conservative external mappings where justified:
  - `skos:closeMatch` / `skos:relatedMatch`
- Keep mapping confidence conservative and documented.

6. Status and Evidence Updates
- Update during work:
  - `ontologist/04-context/TASKS.md`
  - `ontologist/04-context/STATE.md`
  - `ontologist/04-context/WORKLOG.md`
  - `ontologist/04-context/SESSIONS.md`

7. Final Verification
- Run:
  - `python ontologist/05-analytics/tools/ontology_audit.py`
  - `python ontologist/05-analytics/tools/benchmark_calibration.py`
- Summarize residual gaps as explicit "missing-to-100%" checklist.

## Quality Gates

Must pass before task close:
- Ontology boundary policy (TTL-only under `ontology/`, except `ontology/README.md`)
- Current version baseline policy (`owl:versionInfo "0.1.0"`)
- Provenance completeness expectations
- Context/runtime synchronization in `ontologist/04-context/`

## Deliverables

1. Updated `ontology/07-wfe/06-etch-systems/ontology.ttl`
2. Facet completion table (`done` / `partial` / `missing`)
3. Source packet (URL + why authoritative)
4. Explicit remaining "missing-to-100%" checklist
5. Post-edit audit + calibration summary

## Operator Prompt (for Sammy)

Run this task using Semicont’s ontologist control plane.

Start checklist (mandatory):
1. Read `ontologist/AGENTS.md`
2. Read `ontologist/02-rules/{boundary-policy.yaml,curation-gates.yaml}`
3. Read `ontologist/01-docs/subontology-quality-checklist.md`
4. Read `ontologist/01-docs/templates/prompts/facets-dimensions-template.md`
5. Read `ontologist/04-context/{TASKS.md,STATE.md,SESSIONS.md,WORKLOG.md}`
6. Run `python ontologist/05-analytics/tools/ontology_audit.py` before edits

Then execute the plan in:
- `ontologist/01-docs/templates/prompts/07-wfe-06-etch-systems-near-100-plan.md`

Hard constraints:
- Use internet-grounded, publicly accessible sources.
- No synthetic-only claims.
- Keep industry-general scope.
- Do not commit.

Required outputs:
1. Updated `ontology/07-wfe/06-etch-systems/ontology.ttl`
2. Facet completion table
3. Source packet
4. Missing-to-100% checklist
5. Post-edit audit + benchmark summary
