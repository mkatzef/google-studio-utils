import json
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

pkl_file = "rec.pkl"
if os.path.exists(pkl_file):
    with open(pkl_file, 'rb') as pkl_file:
        contents = pickle.load(pkl_file)
else:
    infile = "Records.json"
    with open(infile, "r") as infile:
        contents = json.load(infile)
    with open(pkl_file, 'wb') as pkl_file:
        pickle.dump(contents, pkl_file)

"""
demo
{'latitudeE7': -433700707, 'longitudeE7': 1726648530, 'accuracy': 20, 'source': 'WIFI', 'deviceTag': 2087736016, 'timestamp': '2015-10-12T09:55:45.293Z'}
"""

def find(min_time, max_time):
    found_coords = []
    for record in contents['locations']:
        if min_time < record['timestamp'] < max_time:
            found_coords.append((record['longitudeE7'] / 1e7, record['latitudeE7'] / 1e7))
    
    return found_coords


coords = find('2019-06-08T00:00:00.000Z', '2019-06-09T00:00:00.000Z')
print("<coordinates>")
for p in coords:
    print('\t%f,%f,0' % p)
print("</coordinates>")


arr = np.array(coords)
plt.figure()
plt.plot(*arr.T)
plt.show()