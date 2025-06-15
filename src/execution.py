from .preprocessing import *
from .processing import *
from .exporting import *

def run_steiner(folder_path:str):
    yield "Reading data"
    preprocesser= PreProcessing(folder_path)
    yield "Creating Instances"
    steiner_instances:list[ProblemInstance]=preprocesser.create_instances()
    mcs_results:dict[str,tuple]={}

    yield "Solving MIP and Linear model"
    # for instance in steiner_instances:
    #     mc=MultiCommodity(problem_instance=instance) 
    #     logger.info(f"Solving Linear Model for instance: {instance.name}")
    #     linear_results = mc.process_linear()
    #     logger.info(f"Linear Results: {linear_results}")
    #     logger.info(f"Solving MIP Model for instance: {instance.name}")
    #     mip_results = mc.process_mip()
    #     logger.info(f"MIP Results: {mip_results}")
    #     mcs_results[instance.name]=(linear_results,mip_results) 

    yield "Solving Lagragean Relaxation"
    for instance in steiner_instances:
        ls=Lagragean_Steiner(problem_instance=instance) 
        logger.info(f"Solving Lagrangean Relaxation for instance: {instance.name}")
        lagrangean_results = ls.process_lagrangean()
        logger.info(f"Lagragean Relaxation Results: {lagrangean_results}")
        mcs_results[instance.name]=lagrangean_results 

    yield "Exporting Results"
    yield "Finishing Model"
