# Physics Concepts Added to Address SemicONTO Gaps

## Overview

Successfully added fundamental semiconductor physics concepts to complement our industry-aligned materials ontology, addressing the gaps identified when comparing with SemicONTO.

## New Modules Created

### 00-physics/ (Foundation)
- **Core Classes**: 10
  - `SemiconductorMaterial` (links to shared:Material)
  - `IntrinsicSemiconductor` / `ExtrinsicSemiconductor` (with disjoint axiom)
  - `Carrier` hierarchy: `Electron` / `Hole`
  - `Dopant` hierarchy: `Donor` / `Acceptor`
  - `PhysicalProperty`

- **Key Properties**: 8
  - `hasBandgap`, `hasElectronMobility`, `hasHoleMobility`
  - `hasIntrinsicCarrierConcentration`
  - `hasDopant`, `hasDopantConcentration`, `hasDopantLevel`
  - `hasEffectiveMass`

- **OWL Axioms**: 3
  - Disjointness: Intrinsic/Extrinsic, Electron/Hole, Donor/Acceptor
  - Restriction: Extrinsic semiconductors must have ≥1 dopant

### 00-physics/01-bandstructure/
- **Core Classes**: 7
  - `BandStructure`, `EnergyBand`, `ConductionBand` / `ValenceBand`
  - `BandgapType`: `DirectBandgap` / `IndirectBandgap`
  - `DensityOfStates`

- **Key Properties**: 4
  - `hasBandgapType`, `hasConductionBandEdge`, `hasValenceBandEdge`
  - `hasEffectiveDensityOfStates`

- **Notable**: Links materials to bandgap types (direct/indirect)

### 00-physics/02-carriers/
- **Core Classes**: 9
  - `CarrierTransport`: `DriftTransport` / `DiffusionTransport`
  - `CarrierLifetime`, `Recombination` / `Generation`
  - `CarrierConcentration`: `ElectronConcentration` / `HoleConcentration`

- **Key Properties**: 7
  - `hasCarrierLifetime`, `hasElectronConcentration`, `hasHoleConcentration`
  - `hasDiffusionCoefficient`, `hasRecombinationRate`, `hasGenerationRate`
  - `isDopantType` (n-type/p-type classification)

### 01-devices/ (Foundation)
- **Core Classes**: 12
  - `SemiconductorDevice`: `Diode`, `Transistor` (MOSFET, BJT)
  - `DeviceComponent`: `Terminal`, `Source`/`Drain`/`Gate`, `Channel`, `Oxide`
  - `Junction`: `PNJunction`, `MetalSemiconductorJunction`

- **Key Properties**: 7
  - `hasComponent`, `deviceHasMaterial`, `componentHasMaterial`
  - `formsJunction`, `hasThresholdVoltage`, `hasBreakdownVoltage`
  - `hasChannelLength`/`Width`, `hasOxideThickness`

## Integration with Existing Materials

### Enhanced 08-materials/05-substrate-materials/
- Added `semicont-00-physics:IntrinsicSemiconductor` as superclass to `MonocrystallineSilicon`
- Enhanced silicon wafer example with physics properties:
  - Bandgap: 1.12 eV
  - Electron mobility: 1350 cm²/V·s
  - Hole mobility: 480 cm²/V·s
  - Carrier lifetime: 2.5 ms
  - Bandgap type: Indirect (Silicon)

## Key Design Decisions

1. **Layered Architecture**: Physics concepts in 00-physics, devices in 01-devices
2. **Industry Alignment**: Maintained SEMI standards integration
3. **Reused Patterns**: UnitizedValue, provenance metadata, SHACL validation
4. **Formal Reasoning**: Added OWL axioms for disjointness and restrictions
5. **Practical Examples**: Real-world values for silicon properties

## Verification

- ✅ Ontology audit: No boundary/legacy issues
- ✅ All modules achieve `curated` maturity level
- ✅ Consistent with existing patterns and standards
- ✅ Links physics to practical materials

## Benefits vs SemicONTO

1. **Complementary Coverage**: We cover manufacturing + physics, SemicONTO focuses on fundamental physics
2. **Industry Integration**: Physics concepts link to real materials and SEMI standards
3. **Practical Values**: Include actual mobility, bandgap, lifetime values
4. **Device Context**: Connect physics to actual device structures

## Future Enhancements

- Add more compound semiconductor physics (GaAs, GaN, SiC)
- Create device-specific physics modules (power devices, RF devices)
- Add temperature-dependent properties
- Include more advanced transport phenomena

The physics additions successfully bridge the gap between fundamental semiconductor physics and practical manufacturing, providing a more complete ontology that rivals SemicONTO's theoretical depth while maintaining industry relevance.