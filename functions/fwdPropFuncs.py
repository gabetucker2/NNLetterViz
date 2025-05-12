# import scripts
import functions.axonFuncs as axonFuncs

# fwd prop
def C_fwdProp(C_pre, npre_npost_a):
    
    # set up pre/postsynaptic neuron counts
    n_pre = len(C_pre)
    n_post = len(npre_npost_a[0])

    # calculate axon potentials
    npre_npost_caxonPotential = [] # there are n_pre arrays of n_post axon potentials                    
    for i_pre in range(n_pre):
        c_pre = C_pre[i_pre]
        npre_npost_caxonPotential.append([])
        npost_caxonPotential = npre_npost_caxonPotential[i_pre] # there are n_post axon potentials
        for i_post in range(n_post):
            a_preToPost = npre_npost_a[i_pre][i_post]
            ac_axonPotential = axonFuncs.c_axonPotential(c_pre, a_preToPost)
            npost_caxonPotential.append(ac_axonPotential)
    
    # calculate postsynaptic action potential
    C_postActivation = []
    for i_post in range(n_post):
        C_postAxonPotentials = [row[i_post] for row in npre_npost_caxonPotential if len(row) > i_post]
        C_postActivation.append(axonFuncs.c_postActivation(C_postAxonPotentials))
    
    # return c_postActivation
    return C_postActivation
