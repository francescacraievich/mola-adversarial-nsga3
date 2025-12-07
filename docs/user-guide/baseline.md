# Baseline Performance

## Overview

This document describes the expected baseline performance of MOLA SLAM on unperturbed data.

## Baseline ATE (Unperturbed Data)

**Baseline ATE: ~23cm (0.23 meters)**

This is MOLA's natural localization error when processing clean, unperturbed LiDAR data from the Isaac Sim simulation.

### Baseline Performance

MOLA achieves ~23cm ATE on our test trajectory, which represents good SLAM performance:

For comparison, typical MOLA performance:
- **Ideal conditions (real robot, optimized params)**: 10-20cm ATE
- **Good conditions (our case)**: 20-30cm ATE
- **Moderate conditions**: 30-50cm ATE
- **Poor conditions (degraded sensors, fast motion)**: 50cm+ ATE

### Baseline Characteristics

Running MOLA on unperturbed data:
- **Frames processed**: 113 frames (10Hz LiDAR)
- **Keyframes selected**: 49 keyframes (MOLA selects when robot moves)
- **Trajectory length**: ~11.3 seconds
- **Points per frame**: ~50,000 points
- **Loop closures**: 2-3 successful loop closures

The baseline provides a reference point for measuring adversarial attack effectiveness.
