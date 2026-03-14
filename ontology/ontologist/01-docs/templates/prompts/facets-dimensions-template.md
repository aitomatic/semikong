# Prompt Template: Required Facets and Dimensions

Use this block inside any sub-ontology curation prompt so coverage is explicit and reviewable.

## Required Facets/Dimensions

For the target sub-ontology, define and populate the following facets at industry-general scope:

1. `Scope and Boundary`
- In-scope concepts and explicit out-of-scope boundary.
- Stop before company-specific competitive differentiation.

2. `Taxonomy Depth`
- Multi-level class hierarchy to practical operational depth.
- Leaves should represent broadly reusable industry concepts, not vendor internals.

3. `Entities and Roles`
- Core entities, role classes, and role-bearing actors/components.

4. `Process/Workflow Semantics`
- Process steps, stage ordering, and where the concepts are used in flow.

5. `Capabilities and Constraints`
- What a system/material/process can do.
- Operating constraints, compatibility limits, and safety/quality constraints.

6. `Settings and Parameterization`
- Explicit setting classes/properties (not free-text only).
- Parameter semantics, units, and valid value patterns.

7. `Specifications and Quality`
- Normative specs, quality criteria, and acceptance conditions.
- Links to standards families/documents where applicable.

8. `Measurement and Units`
- Quantity/value modeling with units.
- Numeric constraints/ranges where meaningful.

9. `Lifecycle and State`
- Status/state modeling (e.g., active/superseded/retired where relevant).
- Version/revision relationships when applicable.

10. `Validation Shapes`
- SHACL shapes for required fields, cardinality, and numeric/data-type checks.

11. `Provenance and Rights`
- Internet-source-grounded evidence per major class/property.
- `dc:source`, `dc:rights`, `prov:hadPrimarySource`, `prov:wasInformedBy`.
- Copyright-safe paraphrase only; no long verbatim copying.

12. `External Alignment`
- Conservative mappings/alignment stubs to external ontologies/standards terms where justified.

## Minimum Completion Bar

For each edited module, provide a short completion table (`done`, `partial`, `missing`) across all facets above, plus rationale for any `partial`/`missing`.
