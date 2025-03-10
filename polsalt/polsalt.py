import os, sys, glob
#poldir = '/d/freyr/Dropbox/software/SALT/polsaltcurrent/'
poldir = '/usr/users/khn/src/salt/polsaltcurrent/'

reddir=poldir+'polsalt/'
cmd = 'python2.7 '+reddir+sys.argv[1]+' '+(' ').join(sys.argv[2:])

# python polsalt.py pyscript.py args..
#   polsalt.py runs any python script in polsalt directory

if __name__=='__main__':
    os.system(cmd)
