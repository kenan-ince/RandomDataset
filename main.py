
import os

from lstm import lstm

rootDir = "./Data"

#

for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        dir = dirName.split('\\')
        lstm.fileWalker(dir[1], dirName)
