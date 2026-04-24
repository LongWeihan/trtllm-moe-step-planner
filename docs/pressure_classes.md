# Pressure Classes

The compact version uses three explicit pressure classes.

## `balanced`

- Default pressure class
- Used for control / no-skew workload
- Typical `pressure_score`: `1.0`

## `hot_expert`

- Represents expert concentration
- Used to model batch straggler and TPOT tail pressure
- Typical `pressure_score`: `2.2`

## `hot_rank`

- Represents rank-level concentration
- Used to model step variance and decode imbalance
- Typical `pressure_score`: `2.6`

## Why the model is simple

This is intentionally not a kernel-time predictor.  
It is a scheduling signal model:

- simple enough to ship quickly,
- expressive enough to produce a budgeted step plan,
- simple enough to audit.
