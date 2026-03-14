# Chemicals Ontology Refactor Plan

## Current State
- `01-chemicals/ontology.ttl` contains 34 classes/individuals with full curation
- Subdirectories (00-developers, 01-etchants, 02-photoresists) contain only placeholders
- The main file has grown too large and should be distributed

## Proposed Refactor Structure

### Foundation Module: `01-chemicals/ontology.ttl`
Keep core, cross-cutting concepts:
- ProcessChemical (base class)
- ChemicalSpec, PuritySpec, SEMIGradeSpec
- HazardClassification and instances
- Core object/datatype properties
- SHACL shapes
- Units namespace declaration

### 00-developers/ontology.ttl
- Developer (subclass of Base)
- TMAH and subclasses
- Developer-specific properties
- Developer purity requirements

### 01-etchants/ontology.ttl
- Etchant (subclass of ProcessChemical)
- All mineral acids: HF, H2SO4, HNO3, HCl, H3PO4
- HydrogenPeroxide
- Etchant-specific properties and specs

### 02-photoresists/ontology.ttl
- PhotoresistChemical (subclass of ProcessChemical)
- PGMEA, other photoresist solvents
- Photoresist-specific properties

### 03-solvents/ontology.ttl (new)
- Solvent (subclass of ProcessChemical)
- IPA and other cleaning solvents
- Solvent-specific properties

### 04-bases/ontology.ttl (new)
- Base (subclass of ProcessChemical)
- AmmoniumHydroxide
- Base-specific properties

### 05-cleaning-agents/ontology.ttl (new)
- CleaningAgent (subclass of ProcessChemical)
- SC-1, SC-2 formulations
- Cleaning-specific specs

## Benefits
1. Better modularity and maintainability
2. Clearer separation of concerns
3. Easier to find and extend specific chemical types
4. Follows the pattern established in WFE layer
5. Enables parallel curation of different chemical categories

## Implementation Steps
1. Create new subdirectories as needed
2. Move classes to appropriate modules
3. Update imports in each module
4. Ensure proper cross-references
5. Run audit to verify integrity
6. Update documentation

## Dependencies
Each module will import:
- `00-shared` for base classes
- `01-chemicals` for core chemical framework
- Relevant standards from `01-standards-reference`