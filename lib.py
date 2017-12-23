#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np

def printTable(myDict, colList=None):
   """ Pretty print a list of dictionaries (myDict) as a dynamically sized table.
   If column names (colList) aren't specified, they will show in random order.
   Author: Thierry Husson - Use it as you want but don't blame me.
   """
   if not colList: colList = list(myDict[0].keys() if myDict else [])
   myList = [colList] # 1st row = header
   for item in myDict: myList.append([str(item[col] or '') for col in colList])
   colSize = [max(map(len,col)) for col in zip(*myList)]
   formatStr = ' | '.join(["{{:<{}}}".format(i) for i in colSize])
   myList.insert(1, ['-' * i for i in colSize]) # Seperating line
   for item in myList: print(formatStr.format(*item))



def normalize(npdata):
    ret = []
    for _, data in enumerate(npdata):
        new_data = data / np.linalg.norm(data)
        ret.append(new_data)
    return np.asarray(ret)



def print_table(d):
    for i, key in enumerate(sorted(d.keys())):
        print '{0} {1} | {2}'.format(i, key, d[key])
    