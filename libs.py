from itertools import permutations
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

from typing import List, Tuple

matplotlib.use("Agg")

# type declarations
Coordinate = Tuple[int]
Nodes = List[Coordinate]
Length = float
Path = List[int]


def read_csv(filename: str) -> Tuple[int, Nodes]:
    with open(filename, "r") as f:
        node = []
        for line in f.readlines()[1:]:  # Ignore the first line.
            xy = line.split(",")
            node.append((float(xy[0]), float(xy[1])))
        n_node = len(node)
        return n_node, node


def save_path(filename: str, path: Path) -> None:
    with open(filename, "w") as f:
        f.write("index\n")
        for node in path:
            f.write(f"{node}\n")


def plot_path(path: Path, nodes: Nodes, save_fig_path: str) -> None:
    plt.clf()
    X = []
    Y = []
    for x, y in nodes:
        X.append(x)
        Y.append(y)
    plt.scatter(X, Y)
    for i, number in enumerate(path):
        x, y = nodes[number]
        plt.scatter(x, y, c="red")
        if i == 0:
            beforenode = number
            initnode = number
        else:
            x1, y1 = nodes[beforenode]
            x2, y2 = nodes[number]
            plt.plot([x1, x2], [y1, y2], "r")
            beforenode = number

    x1, y1 = nodes[beforenode]
    x2, y2 = nodes[initnode]
    plt.plot([x1, x2], [y1, y2], "r")

    plt.savefig(save_fig_path)


def distance(node1: Coordinate, node2: Coordinate) -> float:
    x1, y1 = node1
    x2, y2 = node2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_angle(node1: Coordinate, node2: Coordinate) -> float:
    """Calculate angle viewed from node1
    """
    x1, y1 = node1
    x2, y2 = node2
    theta = math.atan2(y2 - y1, x2 - x1)
    if theta < 0:
        theta = (2 * math.pi) + theta
    return theta


def calc_cost(node1: Coordinate, node2: Coordinate, node3: Coordinate) -> float:
    """
    Calculate (node1 -> node3) + (node3 -> node2) - (node1 -> node2)
    """
    dist13 = distance(node1, node3)
    dist32 = distance(node2, node3)
    dist12 = distance(node1, node2)
    return dist13 + dist32 - dist12


def calc_total_cost(nodes: Nodes, path: List[int]) -> float:
    total_cost = 0
    for i in range(len(nodes)):
        total_cost += distance(nodes[path[i]], nodes[path[i - 1]])
    return total_cost


def calc_cost_ratio(node1: Coordinate, node2: Coordinate, node3: Coordinate) -> float:
    """
    Calculate {(node1 -> node3) + (node3 -> node2)} / (node1 -> node2)
    """
    dist13 = distance(node1, node3)
    dist32 = distance(node2, node3)
    dist12 = distance(node1, node2)
    return (dist13 + dist32) / dist12


def search_all_path(n_node: int, nodes: Nodes) -> Tuple[float, List]:
    permutation = list(permutations(range(n_node)))
    shortest_path_length = float("inf")
    shortest_parh = []

    for path in permutation:
        path = list(path)
        path_length = calc_total_cost(nodes, path)

        if shortest_path_length > path_length:
            shortest_path_length = path_length
            shortest_parh = path

    return shortest_path_length, shortest_parh


def greedy_search(n_node: int, nodes: Nodes) -> Tuple[Length, Path]:
    dist = [[0] * n_node for i in range(n_node)]
    for i in range(n_node):
        for j in range(i, n_node):
            dist[i][j] = dist[j][i] = distance(nodes[i], nodes[j])

    current_node = 0
    unvisited_node = set(range(1, n_node))
    path = [current_node]
    path_length = 0

    while unvisited_node:
        next_node = min(unvisited_node, key=lambda node: dist[current_node][node])
        unvisited_node.remove(next_node)

        path.append(next_node)
        path_length += dist[current_node][next_node]

        current_node = next_node

    return path_length, path


def graham_scan(n_node: int, node: Nodes) -> Tuple[Length, Path]:
    min = 0
    # Find the node with the smallest y-coordinate
    for i in range(n_node):
        if node[min][1] > node[i][1]:
            min = i
        elif node[min][1] == node[i][1] and node[min][0] > node[i][0]:
            min = i

    angle = np.zeros((n_node, 2))
    for i in range(n_node):
        if i == min:
            angle[i] = [0, i]
        else:
            theta = get_angle(node[i], node[min])
            angle[i] = [theta, i]

    sorted_angle = angle[angle[:, 0].argsort(), :]
    # print(sorted_angle)

    path = []
    path.extend(
        [int(sorted_angle[0, 1]), int(sorted_angle[1, 1]), int(sorted_angle[2, 1])]
    )

    for i in range(3, n_node):
        path_top = len(path)
        while True:
            theta1 = get_angle(node[path[path_top - 2]], node[path[path_top - 1]])
            theta2 = get_angle(node[path[path_top - 1]], node[int(sorted_angle[i, 1])])

            if theta2 - theta1 < 0:
                del path[path_top - 1]
                path_top -= 1
            else:
                break
        path.append(int(sorted_angle[i, 1]))
        # print(path)
    return path


def insert(path: Path, nodes: Nodes) -> Tuple[Length, Path]:
    ex_path = [i for i in range(len(nodes))]
    for node in path:
        ex_path.remove(node)

    while True:
        min = 0
        costratio = [0 for i in range(len(path))]
        min_node = [0 for i in range(len(path))]
        for i in range(len(path)):
            for j in range(0, len(ex_path)):
                cost = calc_cost(nodes[path[i - 1]], nodes[path[i]], nodes[ex_path[j]])
                if j == 0 or min > cost:
                    min = cost
                    min_node[i] = ex_path[j]
            costratio[i] = calc_cost_ratio(
                nodes[path[i - 1]], nodes[path[i]], nodes[min_node[i]]
            )

        min_ratio = 10 ** 9
        min_ratio_place = 0

        for i in range(len(path)):
            if min_ratio > costratio[i]:
                min_ratio = costratio[i]
                min_ratio_place = i

        path.insert(min_ratio_place, min_node[min_ratio_place])
        ex_path.remove(min_node[min_ratio_place])

        if not ex_path:
            break

    cost = calc_total_cost(nodes, path)
    return cost, path


def swap_node(path: Path, i: int, j: int) -> Path:
    tmp = path[i + 1]
    path[i + 1] = path[j]
    path[j] = tmp
    return path


def calc_opt2_cost(i, i_1, j, j_1, path, nodes):
    l1 = distance(nodes[path[i]], nodes[path[i_1]])
    l2 = distance(nodes[path[j]], nodes[path[j_1]])
    l3 = distance(nodes[path[i]], nodes[path[j]])
    l4 = distance(nodes[path[i_1]], nodes[path[j_1]])
    return l3 + l4 - (l1 + l2)


def opt_2_swap(nodes: Nodes, n_node: int, path: Path) -> Tuple[Length, Path]:
    swaped_path = path.copy()
    start = time.time()
    total = 0
    while True:
        count = 0
        for i in range(n_node - 2):
            i_1 = i + 1
            for j in range(i + 2, n_node):
                j_1 = (j + 1) % n_node
                if i != 0 or j_1 != 0:
                    opt2_cost = calc_opt2_cost(i, i_1, j, j_1, swaped_path, nodes)
                    if opt2_cost < 0:
                        swaped_path = swap_node(swaped_path, i, j)
                        count += 1
        total += count
        if count == 0:
            break
        now = time.time()
        if now - start > 200:
            break

    # path = path.tolist()
    cost = calc_total_cost(nodes, swaped_path)
    return cost, swaped_path


def opt_2_reverse(nodes: Nodes, n_node: int, path: Path) -> Tuple[Length, Path]:
    reversed_path = path.copy()
    start = time.time()
    total = 0
    while True:
        count = 0
        for i in range(n_node - 2):
            i_1 = i + 1
            for j in range(i + 2, n_node):
                j_1 = (j + 1) % n_node

                if i != 0 or j_1 != 0:
                    opt2_cost = calc_opt2_cost(i, i_1, j, j_1, reversed_path, nodes)
                    if opt2_cost < 0:
                        reversed_path = np.array(reversed_path)
                        new_path = reversed_path[i_1 : j + 1]
                        reversed_path[i_1 : j + 1] = new_path[::-1]
                        count += 1
        total += count
        if count == 0:
            break
        now = time.time()
        if now - start > 200:
            break

    # path = path.tolist()
    cost = calc_total_cost(nodes, reversed_path)
    return cost, reversed_path
