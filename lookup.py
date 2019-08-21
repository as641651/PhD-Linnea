import json
import pprint

filename = "data_programs.json"

with open(filename,'r') as f:
    data = json.load(f)

algo = "linnea/Programs/Example03/Julia/generated/algorithm0.jl"

print(algo)
for i in range(len(data[algo])):
   print(i)
   pprint.pprint(data[algo][str(i)])
   print("\n")
