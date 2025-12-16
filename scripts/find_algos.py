import niapy.algorithms.basic
import niapy.algorithms.other
import inspect

def find_algo(name):
    for module in [niapy.algorithms.basic, niapy.algorithms.other]:
        if hasattr(module, name):
            print(f"Found {name} in {module.__name__}")
            return
    print(f"Could not find {name}")

find_algo("SimulatedAnnealing")
find_algo("HillClimbAlgorithm")
find_algo("ParticleSwarmAlgorithm")
find_algo("GeneticAlgorithm")
find_algo("ArtificialBeeColonyAlgorithm")
