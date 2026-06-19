
import time
import math
import random
import tkinter as tk
from tkinter import ttk, messagebox
from queue import PriorityQueue
from pyamaze import maze, agent, COLOR, textLabel


#  HEURISTICS

def heuristic_manhattan(cell1, cell2):
    return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])

def heuristic_euclidean(cell1, cell2):
    return math.sqrt((cell1[0] - cell2[0])**2 + (cell1[1] - cell2[1])**2)

def heuristic_chebyshev(cell1, cell2):
    return max(abs(cell1[0] - cell2[0]), abs(cell1[1] - cell2[1]))

HEURISTICS = {
    "Manhattan": heuristic_manhattan,
    "Euclidean": heuristic_euclidean,
    "Chebyshev": heuristic_chebyshev,
}



#  WEIGHTED ZONE MAP
#  Randomly assigns terrain costs to cells

TERRAIN = {
    "Road":  {"cost": 1,  "color": "#FFFFFF"},
    "Grass": {"cost": 3,  "color": "#C8F0C8"},
    "Swamp": {"cost": 8,  "color": "#B0C8A0"},
}

def build_terrain_map(m):
    """Assign a random terrain type to every cell."""
    terrain_map = {}
    for cell in m.grid:
        roll = random.random()
        if roll < 0.60:
            terrain_map[cell] = "Road"
        elif roll < 0.85:
            terrain_map[cell] = "Grass"
        else:
            terrain_map[cell] = "Swamp"
    return terrain_map

def move_cost(terrain_map, cell):
    return TERRAIN[terrain_map.get(cell, "Road")]["cost"]



#  A* ALGORITHM  (returns path + explored set + f-score map)


def Astar(m, terrain_map, h_func):
    start = (m.rows, m.cols)
    goal  = (1, 1)

    g_score  = {cell: float('inf') for cell in m.grid}
    f_score  = {cell: float('inf') for cell in m.grid}
    g_score[start] = 0
    f_score[start] = h_func(start, goal)

    open_q   = PriorityQueue()
    open_q.put((f_score[start], h_func(start, goal), start))

    came_from = {}
    explored  = set()
    explored_order = []   # for step replay

    DIRS = {
        'E': (0, +1), 'W': (0, -1),
        'N': (-1, 0), 'S': (+1, 0),
    }

    while not open_q.empty():
        _, _, curr = open_q.get()

        if curr in explored:
            continue
        explored.add(curr)
        explored_order.append(curr)

        if curr == goal:
            break

        for d, (dr, dc) in DIRS.items():
            if m.maze_map[curr][d]:
                child = (curr[0] + dr, curr[1] + dc)
                cost  = move_cost(terrain_map, child)
                new_g = g_score[curr] + cost
                if new_g < g_score[child]:
                    g_score[child]  = new_g
                    f_score[child]  = new_g + h_func(child, goal)
                    open_q.put((f_score[child], h_func(child, goal), child))
                    came_from[child] = curr

    # Reconstruct forward path
    fwd_path = {}
    cell = goal
    while cell != start:
        if cell not in came_from:
            return {}, explored_order, f_score   # no path found
        fwd_path[came_from[cell]] = cell
        cell = came_from[cell]

    return fwd_path, explored_order, f_score



#  BFS  (unweighted — ignores terrain cost)


def BFS(m):
    start = (m.rows, m.cols)
    goal  = (1, 1)

    frontier  = [start]
    came_from = {}
    visited   = {start}
    explored_order = []

    DIRS = {
        'E': (0, +1), 'W': (0, -1),
        'N': (-1, 0), 'S': (+1, 0),
    }

    while frontier:
        curr = frontier.pop(0)
        explored_order.append(curr)

        if curr == goal:
            break

        for d, (dr, dc) in DIRS.items():
            if m.maze_map[curr][d]:
                child = (curr[0] + dr, curr[1] + dc)
                if child not in visited:
                    visited.add(child)
                    came_from[child] = curr
                    frontier.append(child)

    fwd_path = {}
    cell = goal
    while cell != start:
        if cell not in came_from:
            return {}, explored_order
        fwd_path[came_from[cell]] = cell
        cell = came_from[cell]

    return fwd_path, explored_order



#  DIJKSTRA  (weighted, no heuristic)

def Dijkstra(m, terrain_map):
    start = (m.rows, m.cols)
    goal  = (1, 1)

    dist      = {cell: float('inf') for cell in m.grid}
    dist[start] = 0
    open_q    = PriorityQueue()
    open_q.put((0, start))
    came_from = {}
    explored  = set()
    explored_order = []

    DIRS = {
        'E': (0, +1), 'W': (0, -1),
        'N': (-1, 0), 'S': (+1, 0),
    }

    while not open_q.empty():
        cost, curr = open_q.get()

        if curr in explored:
            continue
        explored.add(curr)
        explored_order.append(curr)

        if curr == goal:
            break

        for d, (dr, dc) in DIRS.items():
            if m.maze_map[curr][d]:
                child   = (curr[0] + dr, curr[1] + dc)
                new_cost = cost + move_cost(terrain_map, child)
                if new_cost < dist[child]:
                    dist[child]      = new_cost
                    came_from[child] = curr
                    open_q.put((new_cost, child))

    fwd_path = {}
    cell = goal
    while cell != start:
        if cell not in came_from:
            return {}, explored_order
        fwd_path[came_from[cell]] = cell
        cell = came_from[cell]

    return fwd_path, explored_order



#  STATS COLLECTOR

def run_all_algorithms(m, terrain_map, h_func):
    results = {}

    # A*
    t0 = time.perf_counter()
    path_astar, exp_astar, f_scores = Astar(m, terrain_map, h_func)
    t1 = time.perf_counter()
    results["A*"] = {
        "path":       path_astar,
        "explored":   exp_astar,
        "f_scores":   f_scores,
        "length":     len(path_astar) + 1 if path_astar else 0,
        "nodes":      len(exp_astar),
        "time_ms":    round((t1 - t0) * 1000, 3),
        "color":      COLOR.red,
    }

    # BFS
    t0 = time.perf_counter()
    path_bfs, exp_bfs = BFS(m)
    t1 = time.perf_counter()
    results["BFS"] = {
        "path":       path_bfs,
        "explored":   exp_bfs,
        "length":     len(path_bfs) + 1 if path_bfs else 0,
        "nodes":      len(exp_bfs),
        "time_ms":    round((t1 - t0) * 1000, 3),
        "color":      COLOR.blue,
    }

    # Dijkstra
    t0 = time.perf_counter()
    path_dijk, exp_dijk = Dijkstra(m, terrain_map)
    t1 = time.perf_counter()
    results["Dijkstra"] = {
        "path":       path_dijk,
        "explored":   exp_dijk,
        "length":     len(path_dijk) + 1 if path_dijk else 0,
        "nodes":      len(exp_dijk),
        "time_ms":    round((t1 - t0) * 1000, 3),
        "color":      COLOR.green,
    }

    return results



#  SELECTOR GUI
#  Appears before the maze — lets you pick
#  maze size, heuristic, terrain on/off

class SelectorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🧩 Maze Solver — Configuration")
        self.root.resizable(False, False)
        self.root.configure(bg="#1E1E2E")

        self.rows_var      = tk.IntVar(value=10)
        self.cols_var      = tk.IntVar(value=10)
        self.heuristic_var = tk.StringVar(value="Manhattan")
        self.terrain_var   = tk.BooleanVar(value=True)
        self.replay_var    = tk.BooleanVar(value=True)

        self._build_ui()
        self.confirmed = False
        self.root.mainloop()

    def _label(self, parent, text, **kw):
        return tk.Label(parent, text=text, bg="#1E1E2E",
                        fg="#CDD6F4", font=("Consolas", 11), **kw)

    def _build_ui(self):
        pad = dict(padx=20, pady=8)

        tk.Label(self.root,
                 text="⚡ MAZE SOLVER CONFIG",
                 bg="#1E1E2E", fg="#CBA6F7",
                 font=("Consolas", 16, "bold")).pack(pady=(20, 4))

        tk.Label(self.root,
                 text="A* · BFS · Dijkstra  |  Heuristics  |  Terrain  |  Replay",
                 bg="#1E1E2E", fg="#6C7086",
                 font=("Consolas", 9)).pack(pady=(0, 16))

        frame = tk.Frame(self.root, bg="#313244", bd=0)
        frame.pack(padx=24, pady=4, fill="x")

        # Maze size
        self._label(frame, "Maze rows:").grid(row=0, column=0, sticky="w", **pad)
        tk.Spinbox(frame, from_=5, to=25, textvariable=self.rows_var,
                   width=5, font=("Consolas", 11),
                   bg="#45475A", fg="#CDD6F4", bd=0,
                   buttonbackground="#585B70").grid(row=0, column=1, **pad)

        self._label(frame, "Maze cols:").grid(row=1, column=0, sticky="w", **pad)
        tk.Spinbox(frame, from_=5, to=25, textvariable=self.cols_var,
                   width=5, font=("Consolas", 11),
                   bg="#45475A", fg="#CDD6F4", bd=0,
                   buttonbackground="#585B70").grid(row=1, column=1, **pad)

        # Heuristic
        self._label(frame, "Heuristic:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Combobox(frame, textvariable=self.heuristic_var,
                     values=list(HEURISTICS.keys()),
                     state="readonly", width=14,
                     font=("Consolas", 11)).grid(row=2, column=1, **pad)

        # Toggles
        tk.Checkbutton(frame, text="  Weighted terrain (Road / Grass / Swamp)",
                       variable=self.terrain_var,
                       bg="#313244", fg="#A6E3A1", selectcolor="#313244",
                       activebackground="#313244", activeforeground="#A6E3A1",
                       font=("Consolas", 10)).grid(row=3, column=0,
                                                   columnspan=2, sticky="w",
                                                   padx=20, pady=4)

        tk.Checkbutton(frame, text="  Show step-by-step exploration replay",
                       variable=self.replay_var,
                       bg="#313244", fg="#89DCEB", selectcolor="#313244",
                       activebackground="#313244", activeforeground="#89DCEB",
                       font=("Consolas", 10)).grid(row=4, column=0,
                                                   columnspan=2, sticky="w",
                                                   padx=20, pady=(4, 12))

        tk.Button(self.root,
                  text="  ▶  RUN MAZE SOLVER  ",
                  bg="#CBA6F7", fg="#1E1E2E",
                  font=("Consolas", 13, "bold"),
                  relief="flat", cursor="hand2",
                  command=self._confirm).pack(pady=20, ipadx=10, ipady=6)

    def _confirm(self):
        self.confirmed = True
        self.root.destroy()



#  STATS POPUP
#  Opens after maze closes — shows a clean
#  comparison table and winner highlight

def show_stats_popup(results, heuristic_name, terrain_on):
    root = tk.Tk()
    root.title("📊 Algorithm Comparison Results")
    root.configure(bg="#1E1E2E")
    root.resizable(False, False)

    tk.Label(root, text="📊 ALGORITHM COMPARISON",
             bg="#1E1E2E", fg="#CBA6F7",
             font=("Consolas", 15, "bold")).pack(pady=(20, 2))

    info = f"Heuristic: {heuristic_name}   |   Terrain: {'ON' if terrain_on else 'OFF'}"
    tk.Label(root, text=info,
             bg="#1E1E2E", fg="#6C7086",
             font=("Consolas", 9)).pack(pady=(0, 14))

    frame = tk.Frame(root, bg="#313244")
    frame.pack(padx=24, pady=4)

    headers = ["Algorithm", "Path Length", "Nodes Explored", "Time (ms)", "Efficiency*"]
    col_colors = ["#CBA6F7", "#89DCEB", "#FAB387", "#A6E3A1", "#F5C2E7"]

    for c, (h, col) in enumerate(zip(headers, col_colors)):
        tk.Label(frame, text=h, bg="#313244", fg=col,
                 font=("Consolas", 10, "bold"),
                 width=16).grid(row=0, column=c, padx=4, pady=6)

    # Separator
    tk.Frame(frame, bg="#585B70", height=1).grid(
        row=1, column=0, columnspan=5, sticky="ew", padx=4)

    best_nodes = min(v["nodes"] for v in results.values())

    for r, (name, data) in enumerate(results.items(), start=2):
        efficiency = round(data["length"] / data["nodes"] * 100, 1) if data["nodes"] else 0
        highlight  = "#2A2A3E" if data["nodes"] != best_nodes else "#2D3B2D"
        row_fg     = "#CDD6F4" if data["nodes"] != best_nodes else "#A6E3A1"
        badge      = "  ← WINNER" if data["nodes"] == best_nodes else ""

        vals = [name + badge, data["length"], data["nodes"],
                data["time_ms"], f"{efficiency}%"]
        for c, val in enumerate(vals):
            tk.Label(frame, text=str(val),
                     bg=highlight, fg=row_fg,
                     font=("Consolas", 10),
                     width=16).grid(row=r, column=c, padx=4, pady=3)

    tk.Label(root,
             text="* Efficiency = path cells ÷ explored cells × 100  (higher = smarter search)",
             bg="#1E1E2E", fg="#585B70",
             font=("Consolas", 8)).pack(pady=(10, 2))

    tk.Button(root, text="  ✕  Close  ",
              bg="#F38BA8", fg="#1E1E2E",
              font=("Consolas", 11, "bold"),
              relief="flat", cursor="hand2",
              command=root.destroy).pack(pady=16, ipadx=8, ipady=4)

    root.mainloop()



#  MAIN

def main():
    # 1. Config GUI
    cfg = SelectorGUI()
    if not cfg.confirmed:
        return

    rows        = cfg.rows_var.get()
    cols        = cfg.cols_var.get()
    h_name      = cfg.heuristic_var.get()
    terrain_on  = cfg.terrain_var.get()
    replay_on   = cfg.replay_var.get()
    h_func      = HEURISTICS[h_name]

    # 2. Build maze
    m = maze(rows, cols)
    m.CreateMaze(loopPercent=20)   # loopPercent adds extra openings → harder maze

    # 3. Terrain
    terrain_map = build_terrain_map(m) if terrain_on else {c: "Road" for c in m.grid}

    # 4. Run all algorithms and collect stats
    print("\n⏳ Running all three algorithms...\n")
    results = run_all_algorithms(m, terrain_map, h_func)

    for name, data in results.items():
        print(f"  {name:10s} │ path={data['length']:4d}  "
              f"explored={data['nodes']:5d}  "
              f"time={data['time_ms']:6.3f}ms")
    print()

    # 5. Create agents (three coloured paths)
    a_astar = agent(m, footprints=True, color=COLOR.red,   shape="arrow", filled=True)
    a_bfs   = agent(m, footprints=True, color=COLOR.blue,  shape="arrow", filled=True)
    a_dijk  = agent(m, footprints=True, color=COLOR.green, shape="arrow", filled=True)

    # 6. Trace paths
    trace_dict = {}
    if results["A*"]["path"]:
        trace_dict[a_astar] = results["A*"]["path"]
    if results["BFS"]["path"]:
        trace_dict[a_bfs]   = results["BFS"]["path"]
    if results["Dijkstra"]["path"]:
        trace_dict[a_dijk]  = results["Dijkstra"]["path"]

    m.tracePath(trace_dict, delay=100 if replay_on else 1)

    # 7. Labels on the maze window
    textLabel(m, f"[RED]   A* path",
              f"{results['A*']['length']} steps | {results['A*']['nodes']} explored | {results['A*']['time_ms']}ms")
    textLabel(m, f"[BLUE]  BFS path",
              f"{results['BFS']['length']} steps | {results['BFS']['nodes']} explored | {results['BFS']['time_ms']}ms")
    textLabel(m, f"[GREEN] Dijkstra",
              f"{results['Dijkstra']['length']} steps | {results['Dijkstra']['nodes']} explored | {results['Dijkstra']['time_ms']}ms")
    textLabel(m, "Heuristic", h_name)
    textLabel(m, "Terrain",   "ON (Road/Grass/Swamp)" if terrain_on else "OFF")

    # 8. Run maze window
    m.run()

    # 9. Stats popup AFTER maze closes
    show_stats_popup(results, h_name, terrain_on)


if __name__ == "__main__":
    main()