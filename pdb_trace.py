import pdb
import rlcompleter





def trace():
    pdb.Pdb.complete=rlcompleter.Completer(locals()).complete
    pdb.set_trace()
