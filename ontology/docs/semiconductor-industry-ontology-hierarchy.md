# Semiconductor Industry Ontology Hierarchy

This document proposes a comprehensive multi-layer hierarchy for the semiconductor industry ontology, organized by the manufacturing flow and industry segments.

## Layer Structure Overview

```
00-shared/                           # Cross-cutting concepts
├── ontology.ttl                     # Core shared terms
├── materials/                       # Shared materials
├── processes/                       # Common processes
├── equipment/                       # Generic equipment
└── metrics/                         # Standard KPIs

01-integrators/               # System companies (Apple, Google, Tesla)
├── ontology.ttl                     # System integration concepts
├── product-requirements/
├── system-architecture/
└── supply-chain-integration/

03-fabless/                   # Design houses (NVIDIA, AMD, Qualcomm)
├── ontology.ttl                     # Design concepts
├── ip-cores/
├── chip-architecture/
├── design-verification/
└── tape-out/

04-eda/               # Design tools (Synopsys, Cadence)
├── ontology.ttl                     # EDA concepts
├── design-automation/
├── verification-tools/
├── simulation/
└── design-data-management/

05-foundry-idm/                      # Wafer fabrication (TSMC, Intel, Samsung)
├── ontology.ttl                     # Fab concepts
├── wafer-processing/
├── process-control/
├── lot-genealogy/                   # ✅ EXISTING
└── yield-management/

06-osat-packaging-test/osat/                            # Outsourced assembly/test (ASE, Amkor)
├── ontology.ttl                     # OSAT concepts
├── assembly-services/
├── test-services/
├── packaging-services/
└── logistics/

06-osat-packaging-test/                   # Packaging and testing
├── ontology.ttl                     # Root foundation
├── equipment/                       # Packaging/test equipment (missing)
├── packaging/                       # Packaging processes
│   └── ontology.ttl                 # Packaging foundation
└── test/                            # Testing domain
    ├── ontology.ttl                 # Test foundation
    ├── final-test-execution/        # ✅ EXISTING
    ├── rules.yaml                   # Test domain rules
    └── skills.md                    # Test domain skills

07-wfe/                             # Wafer fab equipment (Applied Materials, ASML)
├── ontology.ttl                     # WFE concepts
├── deposition/
├── lithography/
├── etch/
├── ion-implantation/
├── cleaning/
└── metrology/

08-materials/             # Materials suppliers (Air Liquide, JSR)
├── ontology.ttl                     # Materials concepts
├── gases/
├── chemicals/
├── photoresists/
├── wafers/
└── targets/

09-supply-chain/  # Supply chain and compliance
├── ontology.ttl                     # Supply chain concepts
├── supply-chain-visibility/
├── quality-management/
├── regulatory-compliance/
├── sustainability/
└── risk-management/
```

## Detailed Hierarchy by Domain

### 00-shared - Cross-cutting Concepts
The shared layer contains concepts used across multiple industry layers:

**Core Ontology** (`ontology.ttl`)
- Base classes: Material, Process, Equipment, Product, Measurement
- Fundamental properties: hasProcess, hasMeasurement, measuredValue

**Materials** (`materials/`)
- Material categories: conductors, semiconductors, insulators
- Material properties: purity, conductivity, dielectric constant
- Material specifications: grade, formulation, shelf-life

**Processes** (`processes/`)
- Process types: thermal, chemical, physical, optical
- Process parameters: temperature, pressure, time, flow rate
- Process control: setpoints, tolerances, monitoring

**Equipment** (`equipment/`)
- Equipment types: chambers, tools, instruments
- Equipment states: idle, processing, maintenance, down
- Equipment capabilities: throughput, precision, specifications

**Metrics** (`metrics/`)
- KPI categories: yield, throughput, efficiency, quality
- Metric definitions: formulas, units, targets
- Benchmarking: industry standards, best-in-class

### 01-integrators - System Companies
Companies that design and integrate complete systems:

**Product Requirements** (`product-requirements/`)
- Performance specifications: speed, power, functionality
- Environmental requirements: temperature, humidity, shock
- Reliability requirements: MTBF, failure rates

**System Architecture** (`system-architecture/`)
- System decomposition: subsystems, modules, components
- Interface definitions: electrical, mechanical, software
- Integration strategies: chiplet, SoC, system-in-package

**Supply Chain Integration** (`supply-chain-integration/`)
- Supplier management: qualification, auditing, scoring
- Risk assessment: single-source, geopolitical, capacity
- Logistics: inventory, lead times, delivery schedules

### 03-fabless - Design Houses
Companies that design chips but don't manufacture them:

**IP Cores** (`ip-cores/`)
- Core types: CPU, GPU, memory, I/O, analog
- IP licensing: terms, royalties, restrictions
- IP integration: verification, compatibility, performance

**Chip Architecture** (`chip-architecture/`)
- Architecture types: RISC, CISC, DSP, AI/ML
- Design patterns: pipelining, parallelism, caching
- Power management: domains, states, optimization

**Design Verification** (`design-verification/`)
- Verification methods: simulation, formal, emulation
- Test coverage: functional, structural, fault
- Bug tracking: severity, priority, resolution

**Tape-out** (`tape-out/`)
- Release process: checks, approvals, sign-offs
- Mask preparation: layers, revisions, cost
- Foundry handoff: specifications, deliverables, schedules

### 04-eda - Design Tools
Companies providing electronic design automation tools:

**Design Automation** (`design-automation/`)
- Synthesis tools: RTL to gate-level, optimization
- Place & route: floorplanning, timing closure
- Design rule checking: DRC, LVS, ERC

**Verification Tools** (`verification-tools/`)
- Static verification: formal, lint, CDC
- Dynamic verification: simulation, acceleration
- Emulation: FPGA-based, speed bridges

**Simulation** (`simulation/`)
- Circuit simulation: SPICE, FastSPICE, mixed-signal
- Timing analysis: static, statistical, variation
- Power analysis: dynamic, leakage, thermal

### 05-foundry-idm - Wafer Fabrication
Companies that manufacture semiconductor wafers:

**Wafer Processing** (`wafer-processing/`)
- Process flows: FEOL, BEOL, MOL
- Process modules: deposition, litho, etch, CMP
- Integration schemes: FinFET, GAA, 3D NAND

**Process Control** (`process-control/`)
- Statistical process control: SPC charts, Cpk
- Advanced process control: run-to-run, fault detection
- Metrology: CD, overlay, thickness, defectivity

**Lot Genealogy** (`lot-genealogy/`) ✅ EXISTING
- Lot tracking: splits, merges, holds
- Wafer history: process steps, equipment, parameters
- Genealogy queries: parent/child, ancestry, impact analysis

**Yield Management** (`yield-management/`)
- Yield analysis: systematic, random, defect pareto
- Yield prediction: models, simulations, learning
- Yield improvement: root cause, corrective actions

### 06-osat-packaging-test/osat - Outsourced Assembly & Test
Companies providing assembly and test services:

**Assembly Services** (`assembly-services/`)
- Die prep: grind, polish, singulation
- Die attach: epoxy, eutectic, flip-chip
- Wire bonding: gold, copper, ribbon

**Test Services** (`test-services/`)
- Test development: patterns, programs, correlation
- Production test: handler, prober, throughput
- Test optimization: parallel test, multi-site

**Packaging Services** (`packaging-services/`)
- Package types: QFN, BGA, WLCSP, 2.5D/3D
- Package design: substrate, routing, thermal
- Package assembly: molding, marking, plating

### 06-osat-packaging-test - Packaging and Testing
In-house packaging and testing operations:

**Root Foundation** (`ontology.ttl`)
- Core concepts spanning packaging and test
- Cross-domain relationships

**Equipment Domain** (`equipment/` - MISSING)
- Die attach equipment: accuracy, throughput, UPH
- Wire bonders: loop profile, ball size, strength
- Molding systems: transfer, compression, void control

**Packaging Domain** (`packaging/`)
- Foundation: (`ontology.ttl`)
- Flip-chip: bumps, underfill, reliability
- Wafer-level: RDL, micro-bumps, TSV
- System-in-package: chiplets, interconnect, thermal

**Test Domain** (`test/`)
- Foundation: (`ontology.ttl`)
- Domain rules: (`rules.yaml`)
- Domain skills: (`skills.md`)
- Final test execution: ✅ EXISTING
  - Test programs: DC, AC, functional
  - Test equipment: ATE, instruments, interfaces
  - Test results: pass/fail, binning, datalogs
- Retest: yield recovery, debug, validation
- Burn-in: stress conditions, early life failures
- Yield excursion: detection, analysis, containment
- Test data analysis: correlation, trends, predictions

### 07-wfe - Wafer Fab Equipment
Companies manufacturing wafer fabrication equipment:

**Deposition** (`deposition/`)
- CVD: PECVD, LPCVD, ALD
- PVD: sputtering, evaporation
- Epitaxy: CVD, MBE, selective

**Lithography** (`lithography/`)
- Scanners: NA, resolution, overlay
- Masks: OPC, ILT, phase-shift
- Resists: chemically amplified, EUV

**Etch** (`etch/`)
- Plasma etch: RIE, ICP, ALE
- Wet etch: isotropic, anisotropic
- Selectivity: material, crystallographic

### 08-materials - Materials Suppliers
Companies supplying materials to fabs:

**Gases** (`gases/`)
- Bulk gases: N2, O2, H2, Ar
- Specialty gases: SiH4, NF3, WF6
- Gas delivery: purifiers, manifolds, safety

**Chemicals** (`chemicals/`)
- Acids: HF, H2SO4, HCl, H3PO4
- Solvents: IPA, acetone, NMP
- Bases: NH4OH, KOH, TMAH

**Photoresists** (`photoresists/`)
- DUV resists: 248nm, 193nm
- EUV resists: molecular, hybrid
- Developers: TMAH, solvent-based

### 09-supply-chain - Supply Chain Management
Cross-industry supply chain and compliance:

**Supply Chain Visibility** (`supply-chain-visibility/`)
- Track & trace: lot genealogy, serialization
- Inventory management: WIP, finished goods, consignment
- Demand forecasting: models, accuracy, horizon

**Quality Management** (`quality-management/`)
- Quality systems: ISO 9001, IATF 16949
- Control plans: characteristics, methods, frequency
- Corrective actions: 8D, root cause, verification

**Regulatory Compliance** (`regulatory-compliance/`)
- Export controls: ITAR, EAR, deemed exports
- Environmental: REACH, RoHS, WEEE
- Safety: OSHA, SEMI S2, CE marking

## Implementation Recommendations

1. **Start with shared layer**: Establish core concepts before layer-specific terms
2. **Use existing modules as templates**: lot-genealogy and final-test-execution show the pattern
3. **Maintain consistency**: Each module should have ontology.ttl, rules.yaml, skills.md
4. **Add SHACL shapes early**: Validation ensures quality as the ontology grows
5. **Document relationships**: Cross-layer dependencies should be explicit
6. **Plan for evolution**: Structure allows adding new domains or use-cases within layers
7. **Complete layer 06 structure**: Add missing equipment domain and populate packaging domain

## Cross-Layer Dependencies

Key relationships between layers:
- 03-fabless → 04-eda (tool usage)
- 03-fabless → 05-foundry-idm (tape-out handoff)
- 05-foundry-idm → 07-wfe (equipment purchase)
- 05-foundry-idm → 08-materials (material supply)
- 05-foundry-idm → 06-osat-packaging-test/osat (outsourced services)
- 06-osat-packaging-test → 06-osat-packaging-test/osat (service overlap)
- All layers → 00-shared (common concepts)
- All layers → 09-supply-chain (compliance requirements)

## Subcategory Expansion Strawman

To demonstrate how quickly subcategories multiply, here's a partial expansion of just the **Lithography** domain in the fab layer:

### Lithography Expansion Example

```
lithography/
├── resists/                    # Photoresist materials
│   ├── DUV/
│   │   ├── 248nm/             # KrF resist platforms
│   │   │   ├── conventional/
│   │   │   ├── immersion/
│   │   │   └── double-patterning/
│   │   └── 193nm/             # ArF resist platforms
│   │       ├── dry/
│   │       ├── immersion/
│   │       └── multiple-patterning/
│   ├── EUV/
│   │   ├── molecular/
│   │   ├── hybrid/
│   │   ├── metal-oxide/
│   │   └── underlayers/
│   └── e-beam/
│       ├── positive-tone/
│       ├── negative-tone/
│       └── chemically-amplified/
├── masks/                      # Photomasks
│   ├── binary/
│   ├── phase-shift/
│   │   ├── attenuated/
│   │   └── alternating/
│   ├── OPC-types/
│   │   ├── rule-based/
│   │   ├── model-based/
│   │   └── ILT/
│   └── mask-blanks/
├── exposure-systems/
│   ├── DUV-scanners/
│   │   ├── 248nm/
│   │   └── 193nm/
│   ├── EUV-scanners/
│   └── e-beam-writers/
├── metrology/
│   ├── CD-measurement/
│   ├── overlay-measurement/
│   ├── defect-inspection/
│   └── profile-measurement/
└── processes/
    ├── single-patterning/
    ├── double-patterning/
    │   ├── LELE/
    │   ├── LFLE/
    │   └── SADP/
    ├── triple-patterning/
    └── quadruple-patterning/
```

This partial expansion shows ~40 subcategories for just ONE process area. The fab layer has 10+ major process areas (deposition, etch, implant, etc.), each with similar complexity:

- **Deposition**: CVD, PVD, ALD, Epitaxy → 50+ subcats
- **Etch**: Plasma, wet, cryo, ALE → 60+ subcats
- **Implant**: Species, energy, dose → 30+ subcats
- **Metrology**: CD, overlay, defects, film → 80+ subcats
- **Equipment**: Generations, vendors, models → 100+ subcats

**Conservative estimate for fab layer alone: 300-400 subcategories**

Expanding to all layers with similar detail:
- EDA tools: 150+ (simulation, verification, implementation)
- Materials: 200+ (gases, chemicals, wafers, targets)
- WFE: 250+ (deposition, litho, etch, clean, metrology)
- Testing: 100+ (final test, burn-in, reliability)

**Total realistic count: 1000+ subcategories** for a comprehensive ontology that captures the industry's full complexity.