# Train Booking System

- Contributors: [Vivian Zhao](https://github.com/vivian1zhao), [Sofia Wolfel](https://github.com/sofiaWaffle)
- Course: Introduction to Artificial Intelligence ([16:198:520](https://www.cs.rutgers.edu/academics/graduate/m-s-program/course-synopses/course-details/16-198-520-introduction-to-artificial-intelligence))
- Professor: Dr. Wesley Cowan
- Semester: Fall 2025


## Description

In this project, we built an intelligent agent that must localize itself in a grid world without knowing its starting position. The robot receives noisy sensor readings about whether adjacent cells are walls, and must choose movement actions to reduce the set of possible valid locations until only one position remains. 

We implemented three strategies: 

1. **Baseline Strategy (A\*)**: Reference A* localization algorithm provided as the benchmark for evaluating our proposed approaches.
2. **Optimality Strategy**: Belief-state A* search that explores the space of possible robot locations and computes the minimum sequence of moves required to uniquely localize the robot. 
3. **Efficiency Strategy**: Greedy belief-reduction algorithm that selects actions to shrink the set of possible robot locations as quickly as possible while significantly reducing computation time compared to exhaustive search.

The system includes grid generation, movement simulation, belief-state updates, heuristics, and performance comparison across multiple grid sizes. 


## System Components
- **Grid World Generator**: Creates an *n×n* grid with random internal walls using seeded randomness for reproducibility
- **Belief State Representation**: Tracks all possible robot positions consistent with action history and sensory feedback
- **Movement & Simulation Engine**: Simulates robot steps, updates the belief state after each action, and determines valid successor states
- **Heuristic Search Framework**: Implements A* search over belief states to compute optimal localization paths
- **Greedy Belief Reduction**: Selects actions that minimize the size of the belief state to improve computational efficiency
- **Performance Evaluation**: Compares the baseline, optimality, and efficiency strategies across multiple grid sizes using metrics such as localization moves, runtime, and search effort
- **Visualization & Analytics**: Generates performance graphs and experimental results for comparing localization strategies 


## Tools

- Python 3
- NumPy
- Matplotlib
- Heap Queue (`heapq`)
