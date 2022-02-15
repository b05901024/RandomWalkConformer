import numpy as np

f = open("test.txt", 'r')
data = f.readlines()
f.close()
l = []
for s in data:
    if s[:13] == "{'test_mae': ":
        l.append(float(s[13:].split('}')[0]))
print(l)
print(len(l))
print("mean:", np.mean(l))
print("std:", np.std(l))
