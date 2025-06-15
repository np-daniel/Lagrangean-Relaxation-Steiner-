from typing import NamedTuple

class Node(NamedTuple):
    id: int

class ProblemInstance(NamedTuple):
    name:str
    num_nodes: int
    num_edges: int
    nodes: list[Node]
    edges: list[dict[tuple[Node,Node],float]]
    num_terminals:int
    terminals: list[Node]
