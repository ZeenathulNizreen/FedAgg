import os
path = r'C:\Uni works\Research Implementations\FedAgg-Qlora\FedAgg\data\10'
print(os.path.exists(path))  # Should print True if the path is correct
print(os.listdir(path))    
print(f"Current working directory: {os.getcwd()}")
