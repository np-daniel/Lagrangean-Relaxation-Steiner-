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

    def create_objective(self, u: dict[tuple[int, int, int], float]):
        logger.debug("Creating Lagrangian objective w(u)")

        lagrangean_costs = {
            (i, j): self.costs[i, j] - sum(u.get((i, j, k), 0) for k in self.Tr)
            for (i, j) in self.E
        }

        x_term = pulp.lpSum(lagrangean_costs[i, j] * self.x[i, j] for (i, j) in self.E)

        f_term = pulp.lpSum(
            u.get((i, j, k), 0) * self.f[i, j, k]
            for k in self.Tr
            for (i, j) in self.E
        )

        self.model += x_term + f_term

    def create_variables(self,variable_type:str):
        logger.debug("Creating x_ij variables")
        self.x = pulp.LpVariable.dicts("x", self.E, cat=variable_type)
        logger.debug("Creating f_ijk variables")
        self.f = pulp.LpVariable.dicts("f", 
                                       [(i, j, k) for (i, j) in self.E for k in self.Tr], 
                                       lowBound=0, cat="Continuous")
        
    def create_constraints(self):
        logger.debug("Creating flow conservation constraints for intermediate nodes (Eq. 2)")

        for k in self.Tr:  
            for ii in self.V:
                if ii != self.r and ii != k:  
                    inflow = pulp.lpSum(self.f[j, i, k] for (j, i) in self.E if j == ii)
                    outflow = pulp.lpSum(self.f[i, j, k] for (i, j) in self.E if j == ii)
                    self.model += (outflow - inflow == 0), f"flow_conservation_{ii}_{k}"

        for k in self.Tr:
            outflow = pulp.lpSum(self.f[k, j, k] for (i, j) in self.E if i == k)
            inflow = pulp.lpSum(self.f[j, k, k] for (j, i) in self.E if i == k)
            
            self.model += (outflow - inflow == -1), f"sink_demand_{k}"

    def solve_model(self):
        solver = pulp.PULP_CBC_CMD(msg=0)
        logger.debug("Solving Problem")
        self.model.solve(solver)

    def export_result(self,):
        return (pulp.value(self.model.objective), self.model.solutionTime)
    
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