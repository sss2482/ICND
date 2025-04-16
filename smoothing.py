def smoothrd(rcounts, T):
    smcounts = {
        'S':[0] * (T+1),
        'P':[0] * (T+1),
        'N':[0] * (T+1),
        'C':[0] * (T+1),
    }
    for counts in rcounts:
        tc = len(counts['S'])
        for i in range(tc):
            smcounts['S'][i] += counts['S'][i]
            smcounts['P'][i] += counts['P'][i]
            smcounts['N'][i] += counts['N'][i]
            smcounts['C'][i] += counts['C'][i]
        for i in range(tc, T+1):
            smcounts['S'][i] += counts['S'][-1]
            smcounts['P'][i] += counts['P'][-1]
            smcounts['N'][i] += counts['N'][-1]
            smcounts['C'][i] += counts['C'][-1]
    print(smcounts)
    for i in range(T+1):
        smcounts['S'][i] /= len(rcounts)
        smcounts['P'][i] /= len(rcounts)
        smcounts['N'][i] /= len(rcounts)
        smcounts['C'][i] /= len(rcounts)
    print(smcounts)
    
    

    return smcounts