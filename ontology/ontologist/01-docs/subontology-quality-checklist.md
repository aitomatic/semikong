# Sub-Ontology Quality Checklist

## Purpose

Define a consistent, intentional modeling contract for every sub-ontology module before and during curation.

## Policy

Every sub-ontology MUST be explicitly specified against all checklist dimensions below.

- A dimension may be marked `N/A` only with a short rationale.
- Unspecified dimensions are treated as incomplete design.

## Required Dimensions

1. Scope Boundary
- What domain slice is included.
- What is explicitly out of scope.
- Expected depth limit before company-specific differentiation.

2. Core Classes
- Primary entity classes and hierarchy.
- Shared-class reuse from `00-shared`.

3. Object Properties
- Key relationships between classes.
- Domain/range and directionality (including inverse properties where needed).

4. Datatype Properties
- Literal attributes with datatype choices.
- Identity/labeling fields and required metadata fields.

5. Logical Axioms And Constraints
- OWL semantics as needed (disjointness, equivalence, restrictions, cardinality).
- Cross-branch consistency assumptions.

6. Units And Quantities (If Numeric)
- Quantity/value/unit pattern.
- Controlled unit vocabulary choice and normalization strategy.

7. External Mappings
- Standards or ontology crosswalks (SEMI/ISO/JEDEC/etc.) when relevant.
- Mapping predicate choices and confidence assumptions.

8. Provenance And Evidence
- Source packet summary (internet + other allowed sources).
- Module/class provenance fields and copyright/fair-use posture.

9. Validation
- SHACL or equivalent checks planned/implemented.
- What is enforced in audit tooling vs manual review.

10. Maturity Status
- Current maturity stage (`scaffold`, `baseline`, `curated`, `validated`).
- Known gaps and next remediation actions.

## Execution Requirement

For each substantive sub-ontology task:

1. Complete this checklist before major modeling edits.
2. Record checklist decisions in context artifacts (`STATE.md`, `TASKS.md`, `WORKLOG.md`) and/or linked design notes.
3. Reconcile checklist with implementation during Done Criteria review.

## Definition Of Done Add-On

A sub-ontology task is not complete unless:

1. All ten dimensions are explicitly addressed (or marked `N/A` with rationale).
2. Boundaries are consistent with `ontology/README.md`.
3. Validation and provenance expectations are documented and auditable.
