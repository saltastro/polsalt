# dirprep puts the polsalt directory (where it resides) path text in the second line 
#   of selected tools in the script directory so that they may be run standalone

import os, sys, glob
poldir= os.path.dirname(os.path.abspath( __file__ ))+'/'    

scrdir=poldir+'scripts/'
toollist = [scrdir+'reducepoldata_sc.py',scrdir+'script.py']

for tool in toollist:
    with open(tool,'r') as f:
        get_all=f.readlines()
    with open(tool,'w') as f:
        for i,line in enumerate(get_all,1):             
            if i == 2:                             
                f.writelines("poldir = "+"'"+poldir+"'\n")
            else:
                f.writelines(line)
