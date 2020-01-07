import os, sys
import tensorflow as tf
import numpy as np

def all2list(myvar,listlen=None):
  tlen = int(listlen) if (listlen!=None) else 1
  if(type(myvar)==int or type(myvar)==float):
    return [myvar] * tlen
  elif(type(myvar)==list or type(myvar)==tuple):
    myvar=list(myvar)
    if(len(myvar)==tlen):
      return myvar
    elif(listlen==None):
      return myvar
    elif(len(myvar)<tlen):
      myvar.extend([myvar[-1]]*(tlen-len(myvar)))
      return myvar
    else:
      return myvar[:tlen]
  pass
