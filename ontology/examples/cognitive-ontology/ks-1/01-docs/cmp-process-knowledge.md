# CMP Process Knowledge

## Real-World Insights

### The Tuesday Morning Effect

After 15 years in CMP, I've noticed something peculiar - we always have more issues on Tuesday mornings. Not Monday, not Wednesday, but Tuesday. Here's why:

Monday maintenance crews often:
1. Replace pads without checking conditioner alignment
2. Rush to finish before end of shift
3. Skip the post-maintenance checklist

The effects don't show up until Tuesday because:
- Monday PM wafers are engineering/test lots
- Tuesday starts production ramp
- Removal rate drops 15-20% by 10 AM

**Solution we implemented**: Tuesday morning checklist at 8 AM sharp

See also: [Post-Maintenance Checklist](post-maintenance-checklist.md)

### The 3:00 PM Dip

Summer 2024 taught us about environmental effects. Every afternoon around 3 PM:
- Removal rate would increase 8-12%
- Uniformity would degrade
- Defectivity would spike

Root cause: HVAC energy saving mode! The cleanroom temperature rose 2-3°C, affecting:
1. Slurry viscosity (lower temp = higher removal)
2. Pad expansion (thermal effects)
3. Chemical reaction rates

**Fix**: Maintained ±1°C during production hours, cost $2000/month, saved $50K in scrap

### New Guy Syndrome

Every new engineer makes the same mistakes:
1. **Trusts the numbers too much** - Doesn't listen to the machine
2. **Over-adjusts** - Changes 3 parameters at once
3. **Ignores patterns** - Misses the obvious trends

**Training approach**: Pair with senior for 6 weeks, document their intuition

## Practical Wisdom

### Sound Clues
The Mirra makes different sounds when:
- **High pitch whine**: Slurry flow too low
- **Rhythmic thumping**: Pad conditioner misaligned
- **Grinding noise**: Hard particle contamination

### Visual Cues
Wafer appearance tells you:
- **Center bright**: Edge-fast removal (check carrier speed)
- **Radial streaks**: Pad glazing (condition more)
- **Color variations**: Slurry chemistry issues

### Timing Intuition
Experienced operators know:
- Pad life by sound changes (100 hours = tone shift)
- Endpoint by motor current pattern
- When to stop by