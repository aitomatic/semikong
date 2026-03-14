# Tasks

## Ready

1. Phase 0 (contract alignment): reconcile taxonomy/path contract in docs and context files:
   - add `01-standards-reference` to canonical layer lists where missing
   - replace stale `00-core` references with `00-shared`
   - keep `ontology/README.md`, `ontologist/AGENTS.md`, `STATE.md`, and `TASKS.md` mutually consistent
2. Phase 1 (settings semantics): define a reusable pattern for equipment/process settings in `00-shared`:
   - capability limits (what equipment can do)
   - recipe/setpoint targets (what a process intends)
   - operating windows (acceptable execution ranges)
3. Phase 2 (validation in TTL): add SHACL-in-TTL validation modules for key WFE and shared setting properties:
   - required properties, datatype checks, min/max bounds, and cardinality checks
4. Phase 3 (units model): add a controlled units/quantity pattern (QUDT-lite or equivalent local profile) and refactor setting properties to use it consistently.
5. Phase 4 (logical rigor): add missing OWL axioms in priority branches:
   - disjointness for mutually exclusive classes
   - inverse/equivalent properties where semantically justified
   - cardinality/restriction rules beyond lot genealogy
6. Phase 5 (standards typing cleanup): normalize `01-standards-reference` so standards are typed as standards/specs (not equipment/process entities), with clear relation to impacted domains.
7. Phase 6 (reference mappings): add controlled external crosswalks (SEMI/ISO/JEDEC term mappings) using conservative mapping predicates.
8. Phase 7 (lifecycle/status model): add shared lifecycle/state semantics for equipment/process/lot/test contexts and align branch-specific status properties.
9. Phase 8 (provenance quality uplift): replace weak primary-source links with authoritative sources where available and standardize module/class provenance patterns.
10. Phase 9 (audit enforcement): extend `ontologist/05-analytics/tools/ontology_audit.py` to enforce declared curation gates (placeholder debt, provenance completeness, and SHACL validation hooks).
11. Continue branch depth work by entity priority:
   - Phase 2 Tokyo Electron focus (`07-wfe`)
   - Phase 3 JSR focus (`08-materials`)
   - Phase 4 Tata focus (`06-osat-packaging-test`)
   - Phase 5 Renesas/MaxLinear focus (device/application taxonomy)
12. Cross-phase boundary check: add example mapping nodes for target companies only after core branches are complete; keep mappings non-proprietary and outside company-specific differentiation.

## In Progress

1. Phase 3 (JSR focus): Continue materials depth - curate `08-materials/02-gases/` as production ontology

## Blocked

1. (none)

## Done

1. 07-WFE-06 completion push: closed remaining lifecycle-transition, role-modeling, SHACL, and external-crosswalk gaps; module advanced to completion target for current industry-general scope (2026-03-11).
1. Execute `ontologist/01-docs/templates/prompts/07-wfe-06-etch-systems-near-100-plan.md`:
   - Updated `ontology/07-wfe/06-etch-systems/ontology.ttl` with internet-grounded provenance, range-aware parameter modeling, stronger SHACL constraints, and conservative external mappings.
   - Completed facet completion table and missing-to-100 checklist.
   - Ran post-edit `ontology_audit.py` and `benchmark_calibration.py`.
1. Phase 1 (SEMI backbone): define and publish cross-layer core model in `00-core` and `01-standards-reference` for `Actor`, `Facility`, `ProcessStep`, `Equipment`, `Material`, `Spec`, `Metric`, `Defect`, `Yield`, and `TraceabilityUnit`.
2. Reorganized `ontologist/` into ks-style numbered directories (`01-docs`..`06-learning`).
3. Rewrote `ontologist/AGENTS.md` as canonical session/runbook entrypoint.
4. Added baseline audit script for ontology boundary and legacy checks.
5. Ran `python ontologist/05-analytics/tools/ontology_audit.py` on 2026-03-09: boundary and legacy checks passed with zero findings.
6. Phase 3 (JSR focus): Curated `08-materials/01-chemicals/` as production ontology with process chemicals, specs, properties, and validation shapes - completed 2026-03-10.
7. Phase 2 (WFE focus): Curated `07-wfe/06-etch-systems/ontology.ttl` to industry-general operational depth with taxonomy/process/capability/parameter/state/SHACL facets and primary-source provenance - completed 2026-03-10.
