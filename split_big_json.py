import json
import sys

with open('dict9.json','r') as infile:
    o = json.load(infile)
    chunkSize = 300000
    keys = list(o.keys())
    for i in range(0, len(o), chunkSize):
        with open('dict9' + '_' + str(i//chunkSize) + '.json', 'w') as outfile:
            smaller_dict = {k:o[k] for k in keys[i:i+chunkSize]}		
            json.dump(smaller_dict, outfile)
