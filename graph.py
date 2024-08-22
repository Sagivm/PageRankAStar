import networkx as nx
import random
from networkx import Graph
import numpy as np
import heapq
import pandas as pd

import statistics


class Graph(Graph):

    def __init__(self, num_nodes, edge_percentage):
        super().__init__()
        self.num_nodes = num_nodes
        self.weights = np.array([[1000] * num_nodes] * num_nodes)
        self.edge_percentage = edge_percentage

    @classmethod
    def generateGraph(cls, num_nodes: int, edge_percentage: float):
        g = Graph(num_nodes, edge_percentage)
        g.add_nodes_from(range(num_nodes))

        possible_edges = list(nx.non_edges(g))
        random.shuffle(possible_edges)

        num_edges = int(num_nodes * (num_nodes - 1) * edge_percentage)

        for u, v in possible_edges[:num_edges]:
            g.add_edge(u, v)
            g.weights[u, v] = g.get_weight(u, v)

        return g

    def get_weight(self, a, b):
        return abs(a - b + random.randint(-4, 4))

    def compute_page_rank_values(self) -> dict:
        return nx.pagerank(self, 0.85)


class Heuristic():
    page_rank = dict()

    def __init__(self, n_vertices):
        self.h_values_map = np.zeros((n_vertices, n_vertices))
        for i, row in enumerate(self.h_values_map):
            for j, _ in enumerate(row):
                self.h_values_map[i, j] = self.__simple_h_function(i, j)

    def __simple_h_function(self, v_i: int, v_j: int):
        return np.abs(v_i - v_j)

    def set_page_rank(self, _page_rank: dict):
        self.page_rank = _page_rank

    def noHeuristic(self, node_a: int, node_b: int):
        return 0

    def computeSimpleHeuristic(self, node_a: int, node_b: int):
        h_value = self.h_values_map[node_a, node_b]
        return h_value

    def computeAdvancedHeuristic(self, node_a: int, node_b: int):
        h_value = self.h_values_map[node_a, node_b]
        pr_value = self.page_rank[node_b]
        return h_value * pr_value


def heuristic(node, goals, heuristic_function):
    # Manhattan distance to the closest goal
    return min(heuristic_function(node, goal) for goal in goals)


def update_goals(node: int, goals: list, visited_goals: list):
    if node in goals:
        visited_goals.append(node)
        return set(visited_goals) == set(goals)
    return False


def a_star(graph, start, goals, heuristic_function):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goals, heuristic_function)}
    size_of_open_list = []
    visited_goals = []
    while open_list:
        size_of_open_list.append(len(open_list))
        current = heapq.heappop(open_list)[1]
        if update_goals(current, goals, visited_goals):
            return True, g_score, size_of_open_list
        for neighbor in graph.neighbors(current):
            tentative_g_score = g_score[current] + graph.weights[current, neighbor]

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # if update_goals(neighbor, goals, visited_goals):
                #     return True, g_score, size_of_open_list
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goals, heuristic_function)
                if neighbor not in [i[1] for i in open_list]:
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return False, g_score, size_of_open_list


def make_report(finish, g_score, open_list_sizes):
    return {
        "finish": finish,
        "g_score": g_score,
        "open_list_sizes": open_list_sizes,
        # "score": sum[g_score[key] for key in goals]

    }


def run_graph(coverage, number_of_targets):
    n_vertices = 100
    g = Graph.generateGraph(n_vertices, coverage)
    heuristicObj = Heuristic(n_vertices)
    heuristicObj.set_page_rank(g.compute_page_rank_values())

    # Start node and goal nodes
    start = 0
    vertices = list(range(0, n_vertices))
    random.shuffle(vertices)
    goals = vertices[:number_of_targets]  # Multiple goals

    # Perform A* and track the size of the open list
    NH_report = make_report(*a_star(g, start, goals, heuristicObj.noHeuristic))
    BH_report = make_report(*a_star(g, start, goals, heuristicObj.computeSimpleHeuristic))
    PRH_report = make_report(*a_star(g, start, goals, heuristicObj.computeAdvancedHeuristic))

    return [
        ("NH", NH_report),
        ("BH", BH_report),
        ("PRH", PRH_report)
    ]


if __name__ == "__main__":
    coverages = np.array(range(10, 85, 5)) / 100
    number_of_targets_list = np.array([1,2,4,6,8])
    reports = list()
    for coverage in coverages:
        for number_of_targets in number_of_targets_list:
            reports.append(((coverage, number_of_targets), run_graph(coverage, number_of_targets)))

    # construct reuslts
    result = np.ndarray((len(coverages), len(number_of_targets_list), 3))
    for report in reports:
        (row, column), settings_report = report
        row_i = np.where(coverages == row)[0][0]
        column_i = np.where(number_of_targets_list == column)[0][0]
        for i, (_, execution) in enumerate(settings_report):
            result[row_i, column_i, i] = statistics.median(execution["open_list_sizes"])
        x = 0

    algo = ["NH", "BH", "PRH"]
    for i in range(result.shape[-2]):
        df=pd.DataFrame(result[:, i, :], index=coverages.tolist(), columns=algo)
        df.to_csv(f"{number_of_targets_list[i]}.csv",index=True)
