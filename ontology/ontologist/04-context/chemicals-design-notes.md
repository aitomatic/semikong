# Chemicals Sub-Ontology Design Notes

## 1. Scope Boundary

**Included:**
- Process chemicals used in semiconductor manufacturing (acids, bases, solvents, photoresist chemicals)
- Chemical specifications and purity requirements
- Chemical properties (concentration, pH, purity levels)
- Storage and handling constraints
- Hazard classifications
- SEMI grade classifications

**Excluded:**
- Company-specific chemical formulations
- Proprietary additive packages
- Detailed supplier information
- Pricing or commercial terms

**Depth Limit:** Stop at industry-general chemical categories and SEMI specifications. No company-specific product lines.

## 2. Core Classes

**Primary Classes:**
- ProcessChemical (subClassOf semicont-shared:Material)
- Acid (subClassOf ProcessChemical)
- Base (subClassOf ProcessChemical)
- Solvent (subClassOf ProcessChemical)
- PhotoresistChemical (subClassOf ProcessChemical)
- Developer (subClassOf Base)
- Etchant (subClassOf ProcessChemical)
- CleaningAgent (subClassOf ProcessChemical)

**Specification Classes:**
- ChemicalSpec (subClassOf semicont-shared:Spec)
- PuritySpec (subClassOf ChemicalSpec)
- SEMIGradeSpec (subClassOf ChemicalSpec)

## 3. Object Properties

- isSuitableForProcess: ProcessChemical → ProcessStep
- hasPuritySpec: ProcessChemical → PuritySpec
- hasSEMIGrade: ProcessChemical → SEMIGradeSpec
- isIncompatibleWith: ProcessChemical → ProcessChemical
- requiresStorageCondition: ProcessChemical → StorageCondition

## 4. Datatype Properties

- hasConcentration: xsd:double (with unit)
- hasPH: xsd:double
- hasPurityLevel: xsd:double (ppt or ppb)
- hasCASNumber: xsd:string
- hasHazardClass: xsd:string
- hasSEMIStandard: xsd:string

## 5. Logical Axioms

- Disjointness: Acid, Base, Solvent are pairwise disjoint
- Restrictions: ProcessChemical must have at least one PuritySpec
- Cardinality: SEMIGradeSpec has exactly one SEMIStandard value

## 6. Units and Quantities

- Concentration: percent (%), molarity (M), normality (N)
- Purity: parts per trillion (ppt), parts per billion (ppb)
- pH: pH units
- Temperature: Celsius for storage conditions

## 7. External Mappings

- SEMI C1: Guide for Analysis of Liquid Chemicals
- SEMI C30: Hydrogen Peroxide Specification
- SEMI C35: Nitric Acid Specification
- CAS Registry Numbers for chemical identification

## 8. Provenance

Sources:
- SEMI.org standards documentation
- Public domain chemical reference materials
- Industry-standard chemical safety data

Fair use: All chemical information is publicly known industry knowledge

## 9. Validation

SHACL shapes for:
- Required properties (CAS number, purity spec)
- Numeric ranges (pH 0-14, concentration >0)
- SEMI grade consistency
- Hazard classification format

## 10. Maturity Status

Target: Curated (production-ready with validation)
Current: Scaffold (placeholder only)

Next Actions:
- Implement all classes and properties
- Add SHACL validation shapes
- Add provenance documentation
- Run audit validation