import operator

def calculateEvaluationmatrixnew(hashmap, givenlabels,data):
    maxclus = []
    hasmap = {}
    ginindex = []
    MJ = []
    for key in hashmap.keys():
        temp = {}
        for datapoint in hashmap[key]:

            templabel = givenlabels[datapoint]
            if templabel in temp:
                temp[templabel] += 1
            else:
                temp[templabel] = 1
        datapercen = float(0)
        for keytemp in temp.keys():
            datapercen += ((temp[keytemp]/float(len(hashmap[key]))) ** 2)

        ginindex.append(float(1)-float(datapercen))
        MJ.append(float(len(hashmap[key])))
        max_value = max(temp.values())
        maxclus.append(max_value)
        #maxclus.append(max(temp.items(), key=operator.itemgetter(1))[1])
    
    summation = 0
    for a in maxclus:
        summation += a

    ginisum = float(0)
    for i in range(len(ginindex)):
        ginisum += ginindex[i] * MJ[i]


    print("Purity: " + str(summation/data.shape[0]))

    print("Giniindex: " + str(ginisum/data.shape[0]))

