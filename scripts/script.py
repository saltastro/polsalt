import os, sys, glob
print "Run toolprep first";exit()   # replaced with poldir text by toolprep.py

scrdir=poldir+'scripts/'
cmd = 'python '+scrdir+sys.argv[1]+' '+(' ').join(sys.argv[2:])

# python script.py pyscript.py args..
#   script.py runs any python module in script directory

if __name__=='__main__':
    os.system(cmd)
