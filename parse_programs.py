import re



def extract_julia_data_struct(filepath):

    print(filepath)
    algo = {}
    n = 0
    operands = {}
    kernel_next = False

    with open(filepath) as f:
        lines = f.readlines()

        for l in lines:
            tmp =  l.strip()

            if "sizeArg" in tmp:
                ss = re.search('\((.*)\)',tmp.split(":")[1]).group(1).split(',')
                operands[tmp.split(":")[0].split()[-1]] = [int(s) for s in ss if s != '']
                #exit(code=-1)

            if "cost_K" in tmp:
                algo[n] = {}
                algo[n]["call"] = []
                algo[n]["opr"] = {}
                algo[n]["alpha"] = []
                algo[n]["cost"] = tmp.split("oprs_K")[0].split()[-1]
                oprs = tmp.split("oprs_K")[1].split()

                for opr in oprs:
                   if "ml" in opr:
                      algo[n]["opr"][opr] = operands[opr]
                   else:
                      try:
                         algo[n]["alpha"].append(float(opr))
                      except ValueError:
                         #check ex10 algo 36 (axpy)
                         pass
                kernel_next = True
                continue

            if kernel_next:
                kernel_next = False
                if "!" in tmp:
                    algo[n]["call"].append(tmp.split('!')[0])
                    kernel_args = re.search('\((.*)\)',tmp.split('!')[1]).group(1).split(",")
                    for ka in kernel_args:
                        if "L" in ka or "N" in ka or "T" in ka:
                            algo[n]["call"].append(ka)
                else:
                    algo[n]["call"].append("Manual")
                n = n+1

    return algo



import pprint
import os
import fnmatch

algorithms = {}

programs_path = "linnea/Programs/"
examples = os.listdir(programs_path)
for ex in examples:
    #print(ex)
    algos_files = fnmatch.filter(os.listdir(programs_path + ex + "/Julia/generated"), "*.jl")
    #print(algos)
    for al in algos_files:
        algo_path = os.path.join(programs_path + ex + "/Julia/generated", al)
        # deriv_path = os.path.join(programs_path + ex + "/Julia/generated/derivation", al.split(".jl")[0] + ".txt")
        #print(algo_path, deriv_path)
        algorithms[algo_path] = extract_julia_data_struct(algo_path)

    #pprint.pprint(algorithms)
    #print(algorithms)

    #exit(code=-1)

import json

with open("data_programs.json","w") as f:
    json.dump(algorithms,f)

# deriv_path = "linnea/Programs/Example03/Julia/generated/derivation/algorithm0.txt"
# algo_path = "linnea/Programs/Example03/Julia/generated/algorithm0.jl"
# algo = extract_julia_data_struct(algo_path)
# pprint.pprint(algo)


exit(code=-1)
