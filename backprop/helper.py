import numpy as np

def pr(arr):
    ans = ""
    for i in range(len(arr)):
        ans += str(arr[i].shape) + " , "
    return ans