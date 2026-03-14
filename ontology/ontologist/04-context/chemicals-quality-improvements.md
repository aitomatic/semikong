# Chemicals Ontology Quality Improvements

## Objective
Fix quality gaps in 08-materials/01-chemicals ontology modules per the 10-dimension checklist.

## Modules to Improve
1. `ontology/08-materials/01-chemicals/ontology.ttl` (Foundation - 10 entities)
2. `ontology/08-materials/01-chemicals/00-developers/ontology.ttl` (9 entities)
3. `ontology/08-materials/01-chemicals/01-etchants/ontology.ttl` (14 entities)
4. `ontology/08-materials/01-chemicals/02-photoresists/ontology.ttl` (9 entities)

## Quality Dimensions to Address

### 1. Scope Boundary & Maturity Status
- Add explicit scope metadata in each module
- Document maturity stage (currently curated → target validated)
- Define what's out of scope

### 2. Units/Quantities Pattern
- Replace comment-based units with structured unitized values
- Introduce reusable pattern for: concentration, pH, temperature, etch rate, viscosity
- Support both absolute values and unitized quantities

### 3. Relationship Modeling
- Replace string-based `etchesMaterial` with object properties to material classes
- Add proper domain/range specifications
- Ensure inverse properties where appropriate

### 4. OWL Axioms
- Add at least one disjointness per submodule
- Add cardinality restrictions where semantically valid
- Ensure cross-branch consistency

### 5. SHACL Validation
- Add submodule-specific shapes for required fields
- Numeric range sanity checks
- Format validation for identifiers

### 6. External Mappings
- Add rdfs:seeAlso links to SEMI standards
- Reference authoritative sources
- Conservative mapping approach

### 7. Provenance Completion
- Add missing class-level provenance
- Ensure all substantive classes have prov:wasInformedBy

## Implementation Plan
1. Start with foundation module to establish patterns
2. Apply patterns consistently across submodules
3. Add unitized value pattern
4. Improve relationship modeling
5. Add axioms and SHACL shapes
6. Add external mappings
7. Run audit and document results

## Constraints
- Preserve industry-general scope
- No company-specific content
- Keep changes within 08-materials/01-chemicals/** only
- Maintain copyright compliance