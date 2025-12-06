import numpy as np
import random
import heapq
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from collections import deque
import time
from itertools import count


DIRECTIONS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

ACTIONS = tuple(DIRECTIONS.keys())

# generate DxD grid ship
# 0 = closed cell, 1 = open cell
class Grid:
    def __init__(self, size = 101, seed=42):
        self.n = size
        self.cellsExpanded = 0  # count of visited/processed cells

        self.grid = np.zeros((self.n, self.n), dtype=int)  # zeros denote blocked cells

        # call functions
        self._generate_ship(seed)

        # self.start, self.target = self._choose_start_target(seed)
        

    def _generate_ship(self, seed=42):
        randNum = random.Random(seed) 

        # start by opening random cell in interior
        row = randNum.randrange(1, self.n - 1)
        col = randNum.randrange(1, self.n - 1)
        self.grid[row, col] = 1  # mark cell as open

        while True: 
            candidates = []
            for i in range(self.n): 
                for j in range(self.n): 
                    # check for blocked cell to proceed
                    if self.grid[i, j] == 0: 
                        # check if cell row & col are in range, and if neighbors are open
                        open_neighbors = self._count_open_neighbors(i, j)
                        if open_neighbors == 1: 
                            candidates.append((i, j))
                            
            # early break if no more candidates
            if not candidates: 
                break

            # select random candidate cell to open
            i, j = randNum.choice(candidates)
            self.grid[i, j] = 1

        # open neighbors of some 'dead ends' for more loops
        # find dead_end cells
        dead_ends = [
            (i, j) for i in range(self.n) for j in range(self.n)
            if self.grid[i, j] == 1 and self._count_open_neighbors(i, j) == 1]

        # shuffle dead end cells and select the first half
        randNum.shuffle(dead_ends)
        half = len(dead_ends) // 2
        half_dead_ends = dead_ends[:half]

        # pick closed neighbor at random and open it
        for i, j in half_dead_ends:
            closed_neighbors = [
                (i+r, j+c) for r, c in DIRECTIONS.values()
                if self._in_bounds(i+r, j+c) and (self.grid[i+r, j+c] == 0)]
            
            if closed_neighbors:
                neighbor_i, neighbor_j = randNum.choice(closed_neighbors)
                self.grid[neighbor_i, neighbor_j] = 1

        # print("Final grid:")
        # print(self.grid)

    
    def _count_open_neighbors(self, i, j): 
        count = 0
        for row_change, col_change in DIRECTIONS.values(): 
            r, c = i + row_change, j + col_change
            if self._in_bounds(r, c) and self.grid[r, c] == 1:
                count += 1
        return count
    
    def _in_bounds(self, r, c): 
        return 0 <= r < self.n and 0 <= c < self.n

    def _open_cells(self):
        return [(i, j) 
        for i in range(self.n) for j in range(self.n) 
        if self.grid[i, j] == 1] 

    def _dead_ends(self):
        return [(i, j)
        for i in range(self.n) for j in range(self.n)
        if self.grid[i, j] == 1 and self._count_open_neighbors(i, j) == 1]



# --------------------- End of Grid class ----------------------

# # some helper methods
# def in_bounds(grid, r, c): 
#     n = grid.shape[0]
#     return (0 <= r < n) and (0 <= c < n)

def is_open(grid, r, c): 
    return grid.grid[r, c] == 1

# generator for filtering open neighbors one at a time
def neighbors_open(grid, cell):
    i, j = cell
    for row_change, col_change in DIRECTIONS.values():
        r, c = i + row_change, j + col_change
        if grid._in_bounds(r, c) and is_open(grid, r, c):
            yield (r, c)

# --------------------- A* BASELINE (DR. COWAN'S STRATEGY) ----------------------

def heuristic(x, y): 
    # manhattan approach
    return (abs(x[0] - y[0]) + abs(x[1] - y[1]))

# standard A* search for shortest path from start to target
def a_star(grid, start, target):

    if not is_open(grid, *start) or not is_open(grid, *target):
        print("Start or target is blocked.") 
        return None
    
    directions = list(DIRECTIONS.values()) # use DIRECTIONS  
    closed_list = set() # already explored
    open_list = [] # all positions being considered
    came_from = {} # stores all routes taken
    
    gscore = {start: 0} # cost so far
    fscore = {start: heuristic(start, target)} # estimated total cost from start to target thru this node

    heapq.heappush(open_list, (fscore[start], start))

    while open_list:
        f, curr = heappop(open_list)
        grid.cellsExpanded += 1

        # if we reach target, recontruct path by stepping backwards
        if curr == target:
            path = []
            while curr in came_from:
                path.append(curr)
                curr = came_from[curr]
            path.append(start)
            path.reverse()
            return path # path found

        closed_list.add(curr)
        curr_x, curr_y = curr

        for direct_x, direct_y in directions:
            neighbor_x, neighbor_y = curr_x + direct_x, curr_y + direct_y

            if not grid._in_bounds(neighbor_x, neighbor_y):
                continue

            if grid.grid[neighbor_x, neighbor_y] == 0 or (neighbor_x, neighbor_y) in closed_list:
                continue # hit walls or already visited

            # cost from start to neighbor
            tentative_g = gscore[curr] + 1

            # if new path is shorter neighbor not seen yet
            if tentative_g < gscore.get((neighbor_x, neighbor_y), float('inf')):
                came_from[(neighbor_x, neighbor_y)] = curr
                gscore[(neighbor_x, neighbor_y)] = tentative_g
                fscore[(neighbor_x, neighbor_y)] = tentative_g + heuristic((neighbor_x, neighbor_y), target)

                heapq.heappush(open_list,(fscore[(neighbor_x, neighbor_y)],(neighbor_x, neighbor_y)))

    return [] # no path has been found


def baseline_astar(grid, dead_ends, seed=None):
    tot_moves = 0

    rand = random.Random(seed)
    pos_loc = grid._open_cells()

    if not dead_ends:
        print("No dead-ends available.")
        return 0

    # Pick a target dead-end (can also pick randomly each step to slow it down)
    target = rand.choice(dead_ends)

    while len(pos_loc) > 1:
        # Pick only ONE possible location to move per step
        idx = rand.randrange(len(pos_loc))
        loc = pos_loc[idx]
        r, c = loc

        # Move toward target using A* (take next step along path)
        path = a_star(grid, loc, target)
        if path and len(path) > 1:
            next_cell = path[1]
        else:
            next_cell = loc

        # Replace only that location in pos_loc
        pos_loc[idx] = next_cell

        # Optionally remove duplicates
        pos_loc = list(dict.fromkeys(pos_loc))

        tot_moves += 1
        # print(f"Step toward target {target}, possible locations left: {len(pos_loc)}, total moves: {tot_moves}")

    print(f"Blind bot localized: {len(pos_loc)} possible location(s) remain")
    print(f"Total moves: {tot_moves}")
    return tot_moves


# --------------------- OPTIMALITY ----------------------

# Tracks L, the set of possible locations the bot is in

def update_belief(L, action, grid):
    row_change, col_change = DIRECTIONS[action]
    # new_L - updated set of all pos positions after a move
    new_L = set()
    for r, c in L:
        new_r, new_c = r + row_change, c + col_change
        if grid._in_bounds(new_r, new_c) and grid.grid[new_r, new_c] == 1:  # if valid & open
            new_L.add((new_r, new_c))
        else:
            new_L.add((r, c))
    # if belief becomes empty, keep old belief
    if not new_L:
        return set(L)
    return new_L

def heuristics_belief(L):
    # applying heuristics using r and c --> diagonals
    if not L:
        return 0
    if len(L) == 1:
        return 0
    
    r_plus_c = []
    r_minus_c = []
    for (r, c) in L:
        r_plus_c.append(r + c)
        r_minus_c.append(r - c)
    # max_dist - how spread out is belief - more uncertainty - farther from goal
    max_dist = max(max(r_plus_c) - min(r_plus_c), max(r_minus_c) - min(r_minus_c))
   
    return max_dist + 3 * len(L) # weight penalty for num of pos positions left --> helps A star choose moves that shrinks beliefs faster


def show_belief(L, size=None):
    # visual rep of localization
    if size is None:
        return
    # empty cell initialized as '.'
    grid_lines = [['.' for _ in range(size)] for _ in range(size)]
    for (r, c) in L:
        if 0 <= r < size and 0 <= c < size:
            grid_lines[r][c] = 'X'
            # mark if bot could be in that cell
    for row in grid_lines:
        print(''.join(row)) # join chars into a string
    print()


def optimal_Astar(
    grid, 
    init_belief=None,
    max_states = 1_000_000, 
    max_moves = 10_000, 
    log_every = 100,
    init_L=None,
):
    # for part 5 optimal
    if init_belief is None:
        init_L = frozenset(grid._open_cells()) # make belief state hashable
    else:
        init_L = frozenset(init_belief) 
        
    if len(init_L) <= 1: # if localized
        return [], 0

    open_heap = [] # pq of states to explore
    g_cost = {init_L: 0} # stores path cost
    parent = {init_L: (None, None)} # for reconstructing path
    explored = 0
    counter = count()  # breaks ties if same f cost

    # initial state: (f, g, tie_counter, L)
    # f + g + h
    f_init = heuristics_belief(init_L)
    heapq.heappush(open_heap, (f_init, 0, next(counter), init_L))

    best_seen = (len(init_L), init_L) # tracks smallest belief set so far

    while open_heap:
        # prevents loops
        if explored >= max_states:
            print(f"Reached max_states={max_states} after {explored} expansions")
            return None, explored
        
        # removes belief with lowest f from pq
        _, g, _, L = heapq.heappop(open_heap) # f and tiebreaker unpacked

        # skip popped g is worse than recorded
        recorded_g = g_cost.get(L)
        if recorded_g is not None and g > recorded_g:
            # stale entry
            continue

        explored += 1 # count belief states expanded

        if len(L) < best_seen[0]: # if new belief is better than any seen before
            best_seen = (len(L), L) # update

        # logging 
        # if explored % log_every == 0 or len(L) == 1:
        #     # visualization
        #     try:
        #         show_belief(L, size=grid.n)
        #     except Exception as e:
        #         print("Could not show belief:", e)

        if len(L) == 1: # if curr belief == 1 pos loc left --> reached goal
            action_seq = []
            curr = L
            # walk back using parent map
            while parent[curr][0] is not None: # if reached initial belief
                prev, act = parent[curr] # unpack
                action_seq.append(act) # stores in reverse
                curr = prev
            action_seq.reverse()
            if len(action_seq) > max_moves:
                return None, explored
            return action_seq, explored # returns list of moves to fully localize and num states expanded 

        # loop prevention
        if explored >= max_states:
            return None, explored

        # expand pos next moves/actions
        # child belief state L2
        for act in ACTIONS:
            L2_set = update_belief(L, act, grid)
            L2 = frozenset(L2_set)
            new_g = g + 1
            # pruning
            # if never seen or seen but fewer moves 
            if L2 not in g_cost or new_g < g_cost[L2]:
                g_cost[L2] = new_g
                parent[L2] = (L, act) # track that L2 came from L using act
                # compute priority
                new_f = new_g + heuristics_belief(L2)
                heapq.heappush(open_heap, (new_f, new_g, next(counter), L2))

    # if no localization by the end of loop
    print("Search is finished --> No localization ")
    print("Best belief size seen:", best_seen[0])            
    return None, explored


# --------------------- EFFICIENCY ----------------------

# APPROACH: Greedy method that precomputes belief state 
# transitions, iteratively choosing the best action 
# (U,D,L,R) to shrink the belief set down to one point.


def precompute_belief_transitions(grid):
    # organize open cells so each coord (r, c) maps to int index
    coords = grid._open_cells()
    index = {}
    for i, cell in enumerate(coords):
        index[cell] = i

    # lookup table for every action 
    next_indices = {}
    for action, (row_change, col_change) in DIRECTIONS.items():
        map = [0] * len(coords)
        for i, (r, c) in enumerate(coords): 
            new_row = r + row_change
            new_col = c + col_change 

            # move if it stays in bounds & is an open cell
            if grid._in_bounds(new_row, new_col) and grid.grid[new_row, new_col] == 1: 
                next = index[(new_row, new_col)]
            else: 
                # stay at same index
                next = i
            
            map[i] = next
        next_indices[action] = map
    
    return index, coords, next_indices

def apply_action(belief_index, next_arr): 
    updated_set = set()
    for i in belief_index: 
        j = next_arr[i]
        updated_set.add(j)
    return updated_set

# start with full belief (set of all open cells)
# for each step, try all 4 actions & choose action with the smallest resulting belief set
# stop when 1) belief set shrinks to 1, or 2) hit max_moves
def efficiency_greedy(
    grid, 
    max_moves = 100_000_000,
    escape_every = 200,
    escape_attempts = 2,
    rand_seed = 0
):
    index, coords, next_indices = precompute_belief_transitions(grid)

    # full belief set (all open cells)
    belief = set(range(len(coords)))
    
    moves = 0
    seen = set()  # safeguard for detecting cycles
    seen.add(frozenset(belief))
    randNum = random.Random(rand_seed)

    # shrink belief down to a single cell
    while len(belief) > 1 and moves < max_moves:

        # added occasional escape to break long cycles (e.g. super loopy maps)
        if escape_every and moves > 0 and moves % escape_every == 0:
            escaped = False
            for _ in range(escape_attempts): 
                action = randNum.choice(ACTIONS)
                candidate = apply_action(belief, next_indices[action])
                
                if frozenset(candidate) not in seen or candidate != belief:
                    belief = candidate
                    moves += 1
                    seen.add(frozenset(belief))
                    escaped = True
                    break
    
            if escaped: 
                continue  # reevaluate with new belief

        # greedy method: choose action that gives smallest next belief size
        best_action = None
        best_belief = None
        best_size = 10**10  # bootleg Integer.MAX_VALUE

        for action in ACTIONS: 
            candidate = apply_action(belief, next_indices[action])
            size = len(candidate)
            if size < best_size: 
                best_size = size
                best_action = action
                best_belief = candidate

        # try alternative unseen belief with same size (avoid cycling through same beliefs)
        if frozenset(best_belief) in seen: 
            swap = False
            for action in ACTIONS: 
                candidate = apply_action(belief, next_indices[action])

                if len(candidate) == best_size and frozenset(candidate) not in seen: 
                    best_belief = candidate
                    swap = True
                    break
            
            # if still stuck & belief wouldn't change, do one-time escape
            if not swap and best_belief == belief: 
                action = randNum.choice(ACTIONS)
                best_belief = apply_action(belief, next_indices[action])
    
        # lock in the move
        belief = best_belief
        moves += 1
        seen.add(frozenset(belief))

    # success: return total moves & the final location (r, c)
    if len(belief) == 1:
        (loc_idx,) = belief  # unpack as a tuple
        return moves, coords[loc_idx]

    # fail: hit max_moves
    return moves, None


# --------------------- COMPARISON ----------------------


# compare baseline vs. optimality (# of moves)
def compare_optimality(sizes, trials=10, base_seed=42):

    avg_baseline_moves, avg_opt_moves = [], []

    for n in sizes: 
        baseline_moves_list, opt_moves_list = [], []

        for t in range(trials): 
            seed = base_seed + t
            g = Grid(n, seed)

            # A* baseline (Dr. Cowan's)
            baseline_moves = baseline_astar(g, g._dead_ends()) 
            baseline_moves_list.append(baseline_moves)
            
            # Efficiency strategy 
            opt_actions, _ = optimal_Astar(g)
            if opt_actions is None:
                opt_moves = np.nan
            else: 
                opt_moves = len(opt_actions)
            opt_moves_list.append(opt_moves)
            
        avg_baseline_moves.append(np.mean(baseline_moves_list))
        avg_opt_moves.append(np.mean(opt_moves_list))

    return sizes, avg_baseline_moves, avg_opt_moves

 
# compare baseline vs. efficiency (# of moves and time)
def compare_efficiency(sizes, trials=10, base_seed=42):  # measure_time=True

    avg_baseline_moves, avg_eff_moves = [], []
    avg_baseline_time, avg_eff_time = [], []

    for n in sizes: 
        baseline_moves_list, eff_moves_list = [], []
        baseline_time, eff_time = [], []

        for t in range(trials): 
            seed = base_seed + t
            g = Grid(n, seed)

            # A* baseline (Dr. Cowan's)
            t0 = time.perf_counter() 
            baseline_moves = baseline_astar(g, g._dead_ends()) 
            baseline_time.append(time.perf_counter() - t0)
            baseline_moves_list.append(baseline_moves)
            
            # Efficiency strategy 
            t0 = time.perf_counter() 
            eff_moves, _ = efficiency_greedy(g)
            eff_time.append(time.perf_counter() - t0)
            eff_moves_list.append(eff_moves)
            
            
        avg_baseline_moves.append(np.mean(baseline_moves_list))
        avg_eff_moves.append(np.mean(eff_moves_list))
        avg_baseline_time.append(np.mean(baseline_time))
        avg_eff_time.append(np.mean(eff_time))
        
    return sizes, avg_baseline_moves, avg_eff_moves, avg_baseline_time, avg_eff_time



# --------------------- PART 5 ----------------------

# FOR OPTIMALITY STRATEGY

def max_opt(grid, dead_ends):
    max_dist = -1
    max_pair = (dead_ends[0], dead_ends[1])
    # iterate over all pairs of dead-ends
    for i in range(len(dead_ends)):
        for j in range(i + 1, len(dead_ends)):
            # compute Manhattan distance
            d = abs(dead_ends[i][0] - dead_ends[j][0]) + abs(dead_ends[i][1] - dead_ends[j][1])
            if d > max_dist:
                max_dist = d
                max_pair = (dead_ends[i], dead_ends[j])

    worst_case_L = set(max_pair)
    print("Worst-case starting belief set:", worst_case_L)
    return worst_case_L

    
# FOR EFFICIENCY STRATEGY 

# same as efficiency_greedy() method, but using a random subset of coords instead of all open cells
def subset_efficiency_greedy(
    grid, 
    subset_coords, 
    max_moves = 1_000_000, 
    rand_seed=42
):
    
    # reuse original (but simpler)
    index, coords, next_indices = precompute_belief_transitions(grid)
    indices = {coord: i for i, coord in enumerate(coords)}

    # initial belief = the subset (as indices)
    belief = {indices[c] for c in subset_coords}

    moves = 0
    seen = set()  # safeguard for detecting cycles
    seen.add(frozenset(belief))
    randNum = random.Random(rand_seed)

    # shrink belief down to a single cell
    while len(belief) > 1 and moves < max_moves:

        # try all 4 actions, choose the one with fewest survivors
        best_belief = None
        best_size = 10**10

        for action in ACTIONS:
            candidate = apply_action(belief, next_indices[action])
            size = len(candidate)

            if size < best_size:
                best_size = size
                best_belief = candidate

        # try alternative unseen belief with same size (avoid cycling through same beliefs)
        if frozenset(best_belief) in seen:
            alt = apply_action(belief, next_indices[randNum.choice(ACTIONS)])

            if len(alt) <= best_size:
                best_belief = alt

        # lock in the move
        belief = best_belief
        moves += 1
        seen.add(frozenset(belief))

    # success: return total moves & the final location (r, c)
    if len(belief) == 1:
        (i,) = belief  # unpack as a tuple
        return moves, coords[i]    

    # fail: hit max_moves
    return moves, None


# simplify the search to only test pairs (i.e. k=2)
def find_eff_pair(grid, seed=42, sample_pairs=None):

    open_cells = grid._open_cells()
    randNum = random.Random(seed)

    # build list of pairs to test
    pairs = []
    if sample_pairs is None or len(open_cells) < 200:

        # if grid is small enough, test all pairs
        for i in range(len(open_cells)):
            for j in range(i+1, len(open_cells)):
                pairs.append((open_cells[i], open_cells[j]))

    else:
        # do random sampling on pairs
        for _ in range(sample_pairs):
            a, b = randNum.sample(open_cells, 2)
            if a != b:
                pairs.append((a, b))

    best_pair, best_moves = None, -1

    for p in pairs:
        t, _ = subset_efficiency_greedy(grid, p)
        # want to maximize moves to prolong runtime
        if t > best_moves:
            best_moves = t
            best_pair = p

    # pairs are minimal by definition (size=2)
    return best_pair, best_moves



# --------------------- MAIN ----------------------

if __name__ == "__main__":
    # g = Grid(n=20, seed=300)

    # print("Generated ship:\n", g.grid)

    
    print("--------------------------------------------------")
    print("------------------- A* TESTING -------------------")
    print("--------------------------------------------------")


    seed = 42 
    rand = random.Random(seed)

    grid = Grid(size=10, seed=seed)

    open_cells = grid._open_cells()
    dead_ends = grid._dead_ends()

    start = rand.choice(open_cells)
    target = rand.choice(dead_ends)

    print(f"Start: {start}")
    print(f"Target: {target}")

    total_moves = baseline_astar(grid, dead_ends, seed=seed)

    print("Grid:")
    print(grid.grid)
    print(f"Total moves taken by blind bot: {total_moves}")


    print("--------------------------------------------------")
    print("--------------- OPTIMALITY TESTING ---------------")
    print("--------------------------------------------------")


    opt_grid = Grid(size=10, seed=42) 
    print("Generated ship:\n", opt_grid.grid)

    actions, explored = optimal_Astar(opt_grid, log_every=50)

    if actions is None:
        print(f"No localization. States explored: {explored}")
    else:
        print(f"States explored: {explored}")
        print(f"Localized successfully in {len(actions)} moves.")
    

    print("--------------------------------------------------")
    print("--------------- EFFICIENCY TESTING ---------------")
    print("--------------------------------------------------")


    eff_grid = Grid(size=10, seed=42)
    print("Generated ship:\n", eff_grid.grid)
    
    moves, target = efficiency_greedy(eff_grid)
    print("\nEfficiency algorithm results:")
    print("Moves:", moves)
    print("Localized at:", target)
    


    print("--------------------------------------------------")
    print("------------ PERFORMANCE COMPARISONS -------------")
    print("--------------------------------------------------")

    # specific ship sizes for testing 
    opt_test_sizes = [5, 10, 15, 20, 25, 30]
    eff_test_sizes = [5, 10, 15, 20, 25, 30, 35, 40]


    # PART 3) Compare baseline vs. optimality
    sizes, baseline_moves, opt_moves = compare_optimality(opt_test_sizes, trials=3, base_seed=42)
    
    # Graph: Avg Moves for Baseline vs. Optimality
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, baseline_moves, marker='o', label="Baseline Strategy (A*)")
    plt.plot(sizes, opt_moves, marker='s', label="Optimality Strategy")

    plt.title("Baseline vs. Optimality: Average Moves to Localize vs. Ship Size")
    plt.xlabel("Ship Size (D×D)")
    plt.ylabel("Average Moves")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



    # PART 4) Compare baseline vs. efficiency
    sizes, baseline_moves, eff_moves, baseline_time, eff_time = compare_efficiency(eff_test_sizes, trials=3, base_seed=42)
    

    # Graph: Avg Moves for Baseline vs. Efficiency
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, baseline_moves, marker='o', label="Baseline Strategy (A*)")
    plt.plot(sizes, eff_moves, marker='s', label="Efficiency Strategy")

    plt.title("Baseline vs. Efficiency: Average Moves to Localize vs. Ship Size")
    plt.xlabel("Ship Size (D×D)")
    plt.ylabel("Average Moves")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


    # Graph: Time to Compute for Baseline vs. Efficiency
    plt.figure(figsize=(8, 6))
    plt.plot(sizes, baseline_time, marker='o', label="Baseline Strategy (A*)") 
    plt.plot(sizes, eff_time, marker='s', label="Efficiency Strategy")

    plt.title("Baseline vs. Efficiency: Time to Compute Solution vs. Ship Size")
    plt.xlabel("Ship Size (D×D)")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


    print("--------------------------------------------------")
    print("--------------------- PART 5 ---------------------")
    print("--------------------------------------------------")

    # for optimality strategy
    
    grid = Grid(size=10, seed=42)
    dead_ends = grid._dead_ends()

    # run max_opt
    worst_case_L = max_opt(grid, dead_ends)

    # run optimal_Astar from this worst-case starting belief
    actions, explored = optimal_Astar(grid, init_L=frozenset(worst_case_L))
    if actions is None:
        print(f"No localization. States explored: {explored}")
    else:
        print(f"States explored: {explored}")
        print(f"Localized successfully in {len(actions)} moves.")


    # for efficiency strategy
    sizes = [5, 10, 15, 20, 25, 30, 35, 40]
    for n in sizes:
        g = Grid(size=n, seed=42)
        (c1, c2), m = find_eff_pair(g, seed=123, sample_pairs=2000)  # for big grids, default try 2000 random pairs
        print(f"Size {n}: pair = {c1}, {c2} | moves = {m}")
