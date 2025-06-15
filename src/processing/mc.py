from .abcs import Model
from ..preprocessing import *
import pulp
import logging
import pprint
pp = pprint.PrettyPrinter(indent=10)
logger = logging.getLogger(__name__)

class MultiCommodity(Model):
    def __init__(self, problem_instance: ProblemInstance):
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

    def create_variables(self,variable_type:str):
        logger.debug("Creating x_ij variables")
        self.x = pulp.LpVariable.dicts("x", self.E, cat=variable_type)
        logger.debug("Creating f_ijk variables")
        self.f = pulp.LpVariable.dicts("f", 
                                       [(i, j, k) for (i, j) in self.E for k in self.Tr], 
                                       lowBound=0, cat="Continuous")
    def create_objective(self):
        logger.debug("Creating objective")
        self.model += pulp.lpSum(self.costs[i, j] * self.x[i, j] for (i, j) in self.E)

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

        logger.debug("Creating capacity/linking constraints (4)")
        for (i, j) in self.E:
            for k in self.Tr:
                self.model += self.f[i, j, k] <= self.x[i, j], f"capacity_{i}_{j}_{k}"

    def solve_model(self):
        solver = pulp.PULP_CBC_CMD(msg=0)
        logger.debug("Solving Problem")
        self.model.solve(solver)

    def export_result(self,):
        return (pulp.value(self.model.objective), self.model.solutionTime)

    def process_linear(self):
        self.reset_model()
        self.create_variables(variable_type="Continuous")
        self.create_objective()
        self.create_constraints()
        self.solve_model()
        results=self.export_result()
        return results

    def process_mip(self):
        self.reset_model()
        self.create_variables(variable_type="Binary")
        self.create_objective()
        self.create_constraints()
        self.solve_model()
        results=self.export_result()
        return results
    
    def reset_model(self):
        self.model = pulp.LpProblem("MultiCommoditySteiner", pulp.LpMinimize)
        self.x = {}
        self.f = {}