import os, sys, glob
#poldir = '/d/freyr/Dropbox/software/SALT/polsaltcurrent/'
poldir = '/usr/users/khn/src/salt/polsaltcurrent/'

scrdir=poldir+'scripts/'
cmd = 'python2.7 '+scrdir+sys.argv[1]+' '+(' ').join(sys.argv[2:])

# python script.py pyscript.py args..
#   script.py runs any python module in script directory

if __name__=='__main__':
    os.system(cmd)
