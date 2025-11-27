# NSGA-II Algorithm

## Overview

NSGA-II (Non-dominated Sorting Genetic Algorithm II) is a multi-objective evolutionary algorithm used to find optimal solutions when you have conflicting objectives. In this project, we want to maximize SLAM error while minimizing the perturbation size, which are inherently conflicting goals.

Unlike single-objective optimization that finds one best solution, NSGA-II finds a set of Pareto-optimal solutions that represent different trade-offs between the objectives.

## Why Multi-Objective Optimization?

In adversarial attacks on SLAM systems, we face two competing objectives:

1. **Maximize ATE (Absolute Trajectory Error)** - We want to degrade MOLA's localization accuracy as much as possible
2. **Minimize perturbation magnitude** - We want perturbations to be small and imperceptible

You cannot optimize both simultaneously because:
- Larger perturbations cause more damage but are easier to detect
- Smaller perturbations are stealthier but cause less damage

NSGA-II finds the Pareto front: the set of solutions where improving one objective requires worsening the other.

## Key Concepts

### Pareto Dominance

Solution A **dominates** solution B if:
- A is better than B in at least one objective
- A is not worse than B in any objective

Example:
- Solution A: ATE = 1.5m, perturbation = 2cm
- Solution B: ATE = 1.2m, perturbation = 3cm
- A dominates B (higher ATE and lower perturbation)

### Pareto Front

The Pareto front is the set of non-dominated solutions. These are the "best" solutions where you cannot improve one objective without worsening another.

In our case, the Pareto front shows:
- Minimum perturbation needed to achieve a given ATE
- Maximum ATE achievable with a given perturbation budget

### Non-dominated Sorting

NSGA-II ranks solutions into fronts:
- **Front 1**: Non-dominated solutions (the Pareto front)
- **Front 2**: Solutions dominated only by Front 1
- **Front 3**: Solutions dominated only by Fronts 1 and 2
- And so on...

During selection, Front 1 solutions are preferred, then Front 2, etc.

### Crowding Distance

Within each front, solutions need diversity to explore different regions of the trade-off space. Crowding distance measures how isolated a solution is from its neighbors:

- **Large crowding distance**: Solution is in a sparse region (preferred)
- **Small crowding distance**: Solution has many neighbors (less preferred)

This maintains spread along the Pareto front and prevents convergence to a single point.

## Algorithm Steps

### 1. Initialization
Generate an initial population of random solutions. Each solution is a "genome" encoding perturbations for the point cloud.

In our implementation:
- Population size: 20 individuals
- Each genome encodes which attack strategy to apply and parameters

### 2. Fitness Evaluation
Evaluate each individual by:
1. Apply perturbations to point clouds
2. Run MOLA SLAM on perturbed data
3. Compare estimated trajectory with ground truth
4. Calculate ATE (objective 1) and perturbation magnitude (objective 2)

This is the most computationally expensive step, taking several minutes per evaluation.

### 3. Non-dominated Sorting
Rank all solutions into fronts based on dominance relationships. Solutions in Front 1 are the current best approximation of the Pareto front.

### 4. Crowding Distance Calculation
For each front, calculate crowding distance to identify which solutions maintain diversity along the front.

### 5. Tournament Selection
Select parents for reproduction using tournament selection:
1. Randomly pick two individuals
2. Select the one with better rank (lower front number)
3. If tied, select the one with larger crowding distance

This favors both quality (low rank) and diversity (high crowding distance).

### 6. Crossover and Mutation
Create offspring through genetic operators:

**Crossover**: Combine two parent genomes
- Example: Take attack strategy from parent A, parameters from parent B
- Crossover probability: 90%

**Mutation**: Randomly modify genome components
- Example: Change dropout rate from 5% to 7%
- Mutation probability: 10%

### 7. Combine Populations
Merge parent population (size N) with offspring population (size N) to create a combined pool of size 2N.

### 8. Environmental Selection
Select the best N individuals for the next generation:
1. Sort combined population into fronts
2. Add entire fronts to next generation until full
3. If the last front doesn't fit completely, use crowding distance to select the most diverse individuals

### 9. Termination
Repeat steps 2-8 for a fixed number of generations (typically 20 generations).

Final output is Front 1 from the last generation, which approximates the true Pareto front.

## Parameters

Key parameters in our implementation:

- **Population size**: 20 individuals per generation
- **Number of generations**: 20 (for 400 total evaluations)
- **Crossover probability**: 0.9 (90% of offspring created by crossover)
- **Mutation probability**: 0.1 (10% chance to mutate each gene)
- **Tournament size**: 2 (binary tournament selection)

For quick testing, use smaller values:
- Population size: 2
- Generations: 2 (4 total evaluations)

## Genome Encoding

Each individual is encoded as a genome that specifies:

1. **Attack strategy**: Which perturbation type to apply
   - Dropout: Remove random points
   - Gaussian noise: Add random displacement
   - Feature targeting: Perturb high-gradient points
   - Ghost points: Add false points

2. **Parameters**: Strategy-specific parameters
   - Dropout rate (0-100%)
   - Noise standard deviation
   - Number of ghost points
   - Perturbation magnitude

3. **Frame selection**: Which frames to perturb
   - All frames
   - Specific keyframes
   - Loop closure frames

The genome is represented as a numpy array that pymoo can manipulate with genetic operators.

## Constraint Handling

Physical constraints are enforced:
- **Maximum perturbation per point**: 50cm (points moved beyond this are unrealistic)
- **Dropout rate**: 0-100% (cannot remove more than all points)
- **Noise standard deviation**: 0-10cm (larger values create obvious artifacts)

Constraint violations are handled by:
1. Returning infinite fitness (solution is automatically excluded from Pareto front)
2. Repairing infeasible solutions to nearest feasible point

## Convergence

NSGA-II typically converges to a good approximation of the Pareto front within 20-50 generations. You can monitor convergence by tracking:

1. **Hypervolume**: Volume of objective space dominated by Pareto front (should increase)
2. **Best ATE**: Maximum ATE achieved (should increase)
3. **Best efficiency**: Highest ATE per cm of perturbation (should increase)

In practice, we observe:
- Baseline ATE: 68cm (unperturbed)
- After 20 generations: 120-150cm ATE with 5-10cm perturbations
- Convergence plateaus around generation 15-20

## Comparison with Single-Objective Optimization

Why not use single-objective optimization (e.g., maximize ATE only)?

**Single-objective approach:**
- Would find solutions with very high ATE but also very large perturbations
- No control over perturbation budget
- Misses efficient solutions with good ATE/perturbation trade-offs

**NSGA-II multi-objective approach:**
- Finds entire Pareto front with diverse trade-offs
- Allows choosing solution based on perturbation budget constraints
- Identifies most efficient attack strategies (highest ATE per cm)
- Discovers that dropout is more efficient than noise or ghost points

## Interpreting Results

After optimization, the Pareto front shows:

**Example solutions:**
1. Solution A: ATE = 0.95m, perturbation = 1cm (conservative attack)
2. Solution B: ATE = 1.30m, perturbation = 5cm (balanced attack)
3. Solution C: ATE = 1.65m, perturbation = 12cm (aggressive attack)

**Key insight:** Dropout at 5% is most efficient
- Achieves 18.9% ATE increase per cm of perturbation
- Outperforms Gaussian noise (15.2%), feature targeting (12.7%), and ghost points (8.3%)
- Removing points breaks feature correspondences and degrades loop closure

## Advantages of NSGA-II

1. **No weight tuning**: Unlike weighted sum approaches, NSGA-II doesn't require manually tuning weights for objectives
2. **Diverse solutions**: Maintains multiple solutions with different trade-offs
3. **Robust**: Works well even when objectives have different scales or units
4. **Parallelizable**: Fitness evaluations are independent and can be parallelized

## Limitations

1. **Computational cost**: Requires hundreds of fitness evaluations, each taking minutes
2. **Stochastic**: Results vary between runs due to randomness
3. **Scaling**: Performance degrades with >3 objectives (not an issue here)
4. **Local optima**: May get stuck in local Pareto fronts, though crossover helps escape

## Implementation Details

Our implementation uses [pymoo](https://pymoo.org/), a Python framework for multi-objective optimization:

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

algorithm = NSGA2(
    pop_size=20,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True
)

result = minimize(
    problem,
    algorithm,
    ('n_gen', 20),
    seed=1,
    verbose=True
)
```

The problem definition includes:
- Number of variables (genome length)
- Variable bounds (min/max values)
- Number of objectives (2: maximize ATE, minimize perturbation)
- Evaluation function (applies perturbations and runs MOLA)

## Further Reading

- Original NSGA-II paper: Deb et al. (2002)
- pymoo documentation: https://pymoo.org/
- Multi-objective optimization tutorial: https://pymoo.org/getting_started/part_1.html
