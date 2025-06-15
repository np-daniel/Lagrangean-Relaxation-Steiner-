from abc import ABC,abstractmethod

class Model(ABC):
    def __init__(self, problem_instance):
        self.instance = problem_instance

    @abstractmethod
    def create_objective(self):
        pass
    
    @abstractmethod
    def create_variables(self):
        pass
    
    @abstractmethod
    def create_constraints(self):
        pass

    @abstractmethod
    def solve_model(self):
        pass

    @abstractmethod
    def process_linear(self):
        pass
    
    @abstractmethod
    def process_mip(self):
        pass

    @abstractmethod
    def export_result(self):
        pass
    

class LagrangeanModel(ABC):
    def __init__(self, problem_instance):
        self.instance = problem_instance

    @abstractmethod
    def process_lagrangean(self):
        pass

    @abstractmethod
    def generate_u(self):
        pass
