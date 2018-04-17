import pdb
import rlcompleter
pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
pdb.set_trace()


def trace():
    import pdb
    import rlcompleter
    pdb.Pdb.complete = rlcompleter.Completer(locals()).complete
    pdb.set_trace()
