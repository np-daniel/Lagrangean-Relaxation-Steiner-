from .structures import Node, ProblemInstance
from natsort import natsorted
import os
class PreProcessing:
    def __init__(self, path: str):
        self.folder_path = path

    def read_instance(self, file_path: str) -> tuple[int, int, list[tuple[int, int, float]], list[int]]:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        num_nodes, num_edges = map(int, lines[0].split())

        edges = []
        for line in lines[1:num_edges + 1]:
            u, v, cost = map(int, line.split())
            edges.append((u, v, cost))

        terminals = list(map(int, lines[num_edges + 2].split()))

        return num_nodes, edges, terminals

    def create_nodes(self, num_nodes: int) -> list[Node]:
        return [Node(i) for i in range(1, num_nodes + 1)]

    def create_edges(self, edges_data: list[tuple[int, int, float]]) -> list[dict[tuple[Node, Node], float]]:
        edge_list = []
        for u, v, cost in edges_data:
            edge_list.append({(Node(u), Node(v)): cost})
            edge_list.append({(Node(v), Node(u)): cost})
        return edge_list

    def create_terminals(self,terminals:list)->list[Node]:
        terminals_list:list[Node] = [Node(terminal) for terminal in terminals]
        return terminals_list

    def create_instance(self, file_path: str) -> ProblemInstance:
        full_path = os.path.join(self.folder_path, file_path)
        num_nodes,edges_data,terminal_ids = self.read_instance(full_path)

        nodes = self.create_nodes(num_nodes)
        edges = self.create_edges(edges_data)
        terminals = self.create_terminals(terminal_ids)

        return ProblemInstance(
            name=file_path,
            num_nodes=num_nodes,
            num_edges=len(edges),
            nodes=nodes,
            edges=edges,
            num_terminals=len(terminals),
            terminals=terminals
        )

    def create_instances(self) -> list[ProblemInstance]:
        instances = []
        for file in natsorted(os.listdir(self.folder_path)):
            if file.endswith('.txt'):
                instance = self.create_instance(file)
                instances.append(instance)
        return instances