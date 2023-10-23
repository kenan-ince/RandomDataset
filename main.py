
import os

#from dnn import dnn
from lstm import lstm

rootDir = "./DataTest"

for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        dir = dirName.split('\\')
        lstm.fileWalker(dir[1], dirName)
        #dnn.fileWalker(dir[1], dirName)
