# Troubleshoot Removal Rate Issues

## When Removal Rate Drops

### Step 1: Quick Checks (5 minutes)
1. **Check the obvious**
   - Is slurry flowing? (visual check)
   - Any alarms on the tool?
   - When was this pad conditioned last?

2. **Look at recent data**
   - Trend chart for last 2 hours
   - Any parameter changes?
   - Correlation with pad life?

### Step 2: Systematic Diagnosis (15 minutes)

#### Check Pad Conditioning
```
IF conditioner last run > 2 hours ago:
    Run conditioner for 60 seconds
    Measure removal rate change
    IF improvement > 5%:
        Problem found - pad was glazing
        Increase conditioning frequency
    ELSE:
        Continue diagnosis
```

#### Check Slurry System
1. **Flow rate verification**
   - Compare setpoint vs actual
   - Use graduated cylinder if suspicious
   - Check for blockages in lines

2. **Chemistry check**
   - pH within spec? (±0.2)
   - Temperature stable? (±2°C)
   - Any visible contamination?

#### Check Mechanical Systems
1. **Pressure verification**
   - Calibrated this week?
   - Uniform across wafer?
   - No pressure spikes in data?

2. **Speed verification**
   - Carrier and platen speeds correct?
   - No belt slippage (linear tools)?
   - Motor current stable?

### Step 3: Advanced Diagnostics (30 minutes)

#### Thermal Analysis
- Use IR camera if available
- Check for hot spots on pad
- Verify heat exchanger operation
- Check ambient temperature trends

#### Vibration Analysis
- Listen for unusual sounds
- Check motor current patterns
- Look for rhythmic variations
- Correlate with removal rate changes

### Common Solutions

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| RR drops gradually | Pad glazing | Increase conditioning |
| RR drops suddenly | Slurry issue | Check flow/chemistry |
| RR increases | Over-conditioning | Reduce conditioning |
| Radial non-uni | Conditioner misaligned | Re-level conditioner |

### Validation

After any fix:
1. Run 3 test wafers
2. Measure removal rate
3. Check uniformity
4. Document the fix
5. Update procedures if needed

## Emergency Procedures

If removal rate drops >30%:
1. **STOP Production**
2. **Notify supervisor**
3. **Hold all wafers**
4. **Start troubleshooting above**
5. **Document everything**

## Documentation

Always record:
- Symptom observed
- Root cause found
- Fix applied
- Validation results
- Time to resolve

See also: [Emergency Procedures](emergency-break-glass.md)

---
*Based on 10 years of CMP troubleshooting experience*
*Created: 2026-03-08 by kimi-k2-0905-preview*