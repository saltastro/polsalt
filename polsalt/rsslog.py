
"""RSS logging, adapted from saltsafelog"""

import inspect, time

def message(m, logfile, with_header=False, with_stdout=True):
    """Prints message *m* to logfile."""

    # Get current time
    curtime=time.strftime("%Y-%m-%d %H:%M:%S")

    # Define header
    header="\n%s MESSAGE ------------------------------------------\n" % curtime

    # Compose final message
    if with_header:
        log_message=header+m
    else:
        log_message=m

    # Print header+message to standard output
    if with_stdout:
        print log_message

    # Write header+message to logfile
    logFile = open(logfile, 'a')
    logFile.write(log_message+'\n')
    return

def history(logfile, level=1, wrap=True, wrapchar=80, exclude=[]):
   """log the history of the current calling procedure.  This includes return the name of the
       current program as well as the information about all the parameters
       which are passed to it. 
 
       level -- the inspect level 
       wrap  -- wraps characters if true
       wrapchar--number of characters to wrap at
       exclude--options to exclude 
   """           
   
   frame=inspect.getouterframes(inspect.currentframe())[level][0]
   args,_,_,values=inspect.getargvalues(frame)
   fname=str(inspect.getframeinfo(frame)[2])

   msg ='\n%s ' % fname.upper()
   lcount=0
   for i in args:
     if  i not in exclude: 
       instr="%s=%s," % (i, values[i])
       lcount += len(instr)
       if lcount>wrapchar and wrap: 
           msg += '\n'
           lcount = len(instr)
       if i.count('pass'):
           msg+="%s=%s " % (i, '****')
       else:
           msg+="%s=%s " % (i, values[i])
               
   message(msg.rstrip(), logfile)
   message("Starting %s" % fname, logfile, with_header=True)
                   
   return

