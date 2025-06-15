from ..preprocessing import *
from .abcs import LagrangeanModel 
import logging
import pulp
import networkx as nx
import random
import time
logger = logging.getLogger(__name__)
class Lagragean_Steiner(LagrangeanModel):
    def __init__(self, problem_instance):
        super().__init__(problem_instance=problem_instance)
        self.model = pulp.LpProblem("MultiCommoditySteiner", pulp.LpMinimize)
        self.V = [node.id for node in problem_instance.nodes]
        self.T = [node.id for node in problem_instance.terminals]
        self.r = self.T[0]
        self.Tr = self.T[1:]
        self.E = [(u.id, v.id) for edge in problem_instance.edges for (u, v) in edge]
        self.costs = { (u.id, v.id): w for edge in problem_instance.edges for ((u, v), w) in edge.items() }
        self.x = {}
        self.f = {}
    
    def solve_x_objective(self,u: dict[tuple[int, int, int], float])->float:
        self.x:dict[tuple[int,int],float]={}
        for (i,j) in self.E:
            lagrangean_costs = self.costs[i,j]-sum(u.get((i, j, k), 0) for k in self.Tr)
            if lagrangean_costs<0: 
                self.x[i,j] = 1
            else:
                self.x[i,j] = 0
        total_x_cost = sum([self.costs[i,j]-sum(u.get((i, j, k), 0) for k in self.Tr)* self.x[i,j] for (i,j) in self.E])
        return total_x_cost
    
    def solve_f_objective(self, u: dict[tuple[int, int, int], float]) -> float:
        total_cost = 0
        self.f = {}

        for k in self.Tr:
            prob = pulp.LpProblem(f"MinCostFlow_k {k} ", pulp.LpMinimize)
            f_vars = {
                (i, j): pulp.LpVariable(f"f_{i}_{j}_{k}", lowBound=0, upBound=1)
                for (i, j) in self.E
            }
            prob += pulp.lpSum(u.get((i, j, k), 0) * f_vars[i, j] for (i, j) in self.E)
            for v in self.V:
                inflow = pulp.lpSum(f_vars[i, v] for (i, j) in self.E if j == v)
                outflow = pulp.lpSum(f_vars[v, j] for (i, j) in self.E if i == v)

                demand = 1 if v == k else -1 if v == self.r else 0
                prob += (outflow - inflow == demand), f"flow_balance_{v}"

            solver = pulp.PULP_CBC_CMD(msg=0)
            logger.debug("Solving Problem")
            prob.solve(solver)

            for (i, j), var in f_vars.items():
                val = var.varValue or 0
                if val > 0:
                    self.f[i, j, k] = val
                    total_cost += u.get((i, j, k), 0) * val
        return total_cost
        
    def generate_u(self) -> dict[tuple[int, int, int], float]:
        u = {}
        for (i, j) in self.E:
            for k in self.Tr:
                u[i, j, k] = random.uniform(0.5, 1) 
        return u

    def process_lagrangean(self)->float:
        u = self.generate_u()
        lb = self.solve_x_objective(u) + self.solve_f_objective(u)
        return lb