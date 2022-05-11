# Pablo Nicolas Ramos
# linkedin: https://ar.linkedin.com/in/pablonicolasr
# McCulloch and Pitts neurons

import numpy as np
import pandas as pd
import random
import os


# The screen is cleaned
class ClearScreen():
    
    def __init__(self):
    
        self.clear = os.system('cls' if os.name=='nt' else 'clear')


class MPNeuron:
    
    def __init__(self):
        self.threshold = None
        
    def model(self, x):
        # input: [1, 0, 1, 0] [x1, x2, .., xn]
        z = sum(x)
        return (z >= self.threshold)
        
    def continue_to_binary(self, x):
        # input: [[1, 0, 1, 0], [1, 0, 1, 1]]
        x = pd.cut(x, bins=2, labels=[0, 1])
        return x        
    
    def predict(self, X):
        # input: [[1, 0, 1, 0], [1, 0, 1, 1]]
        Y = []
        for x in X:
            result = self.model(self.continue_to_binary(x))
            Y.append(result)
        return np.array(Y)


if __name__ == "__main__":
    # Size of the principal list (ten events)
    N = 10
    band = False
    while not band:
        # Define the size of the list    
        try:
            size_of_list = int(input("Enter the size of the list (conditions of particular event): "))
            band = True
        except Exception as e:
            print(str(e)+"\n")
            print("You must enter an integer\n")
            input("Press any key to continue...")
            ClearScreen().clear
    
    # McCullochPitts's neuron is instantiated
    mp_neuron = MPNeuron() 
    
    # If more than half of the conditions are met, True is returned
    mp_neuron.threshold = (size_of_list // 2) + 1 
    
    # Generate the conditions for each event
    list_of_lists = [[random.randint(-21, 21) for q in range(size_of_list)] for z in range(N)]
    
    print(list_of_lists)
    print(mp_neuron.predict(list_of_lists))
    
    
