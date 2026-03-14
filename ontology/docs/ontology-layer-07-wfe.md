# Wafer Fab Equipment (WFE) Layer - 07

This layer contains ontologies for wafer fabrication equipment organized by process flow sequence.

## Directory Structure

The subdirectories are ordered to follow the typical semiconductor fabrication process flow:

### 00-11: Core Fabrication Process Flow
1. **00-Wafer Handling Automation** - Entry point for wafer loading/unloading
2. **01-Oxidation Systems** - Initial thermal oxidation processes
3. **02-Deposition Tools** - Film deposition (CVD, PVD, ALD, epitaxy)
4. **03-Photoresist Processing** - Coating and soft bake preparation
5. **04-Lithography Equipment** - Patterning (DUV, EUV, multi-patterning)
6. **05-Ion Implantation** - Doping and implantation processes
7. **06-Etch Systems** - Pattern transfer (plasma, wet, atomic layer etch)
8. **07-Chemical Mechanical Planarization** - Surface planarization
9. **08-Clean Strip Tools** - Resist removal and cleaning
10. **09-Doping Equipment** - Alternative doping methods
11. **10-Thermal Processing** - Annealing, RTP, spike anneal
12. **11-Annealing Furnaces** - Batch thermal processes

### 12-13: Metrology and Inspection
13. **12-Metrology Inspection** - Critical dimension and overlay measurement
14. **13-Inspection Tools** - Defect inspection and review

### 14: Support Services
15. **14-Suppliers and Maintenance** - Equipment vendors and field service

## Process Flow Relationships

Equipment modules can reference the process flow ordering through:
- `hasStepOrder` property for numeric ordering
- `precedesStep` property for sequential relationships

## Cross-References

Links to other layers:
- **05-foundry-idm**: Process technologies and fabrication flows
- **08-materials**: Materials and chemicals used by equipment
- **09-supply-chain**: Equipment procurement and maintenance logistics

## Usage

Each equipment directory contains:
- `ontology.ttl` - Domain-specific ontology
- Equipment capabilities, parameters, and specifications
- Vendor information and maintenance concepts

The ordering reflects typical wafer processing sequence but individual fabs may have variations based on their specific process flows."},