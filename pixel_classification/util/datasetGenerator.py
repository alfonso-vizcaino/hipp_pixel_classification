import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

hippocampusAreas = {
    'CA1' : 1,
    'CA2' : 2,
    'CA3' : 3,
    'DG'  : 4,
    'Outer' : 0
}

def loadFromFile(fileName = "") :
    if fileName == "" :
        return
    
    unprocessedDataSet = pd.read_csv(fileName)

    # convert to labels to numbers
    unprocessedDataSet.loc[unprocessedDataSet['Y'] == 'CA1', ['Y']] = 1
    unprocessedDataSet.loc[unprocessedDataSet['Y'] == 'CA2', ['Y']] = 2
    unprocessedDataSet.loc[unprocessedDataSet['Y'] == 'CA3', ['Y']] = 3
    unprocessedDataSet.loc[unprocessedDataSet['Y'] == 'DG', ['Y']] = 4
    unprocessedDataSet.loc[unprocessedDataSet['Y'] == 'Outer', ['Y']] = 0

    #remove bogus records
    unprocessedDataSet = unprocessedDataSet.replace([np.inf, -np.inf], np.nan)
    unprocessedDataSet = unprocessedDataSet.dropna();

    nulls = np.where(pd.isnull(unprocessedDataSet))

    #Main Dataset
    hipAreaData = unprocessedDataSet.drop(unprocessedDataSet.index[nulls[0]])

    #Get all Non hippocampal pixels
    isOuterData = hipAreaData['Y']==0
    outerData = hipAreaData[isOuterData] 
    
    #Get all Hippocampal pixels: CA1, CA2, CA3, DG
    isCA1Data =  hipAreaData['Y']==1
    CA1Data = hipAreaData[isCA1Data]
    
    isCA2Data =  hipAreaData['Y']==2
    CA2Data = hipAreaData[isCA2Data]
    
    isCA3Data =  hipAreaData['Y']==3
    CA3Data = hipAreaData[isCA3Data]
    
    isDgData =  hipAreaData['Y']==4
    dgData = hipAreaData[isDgData]
    
    #stack all hippocampall pixels together    
    data = [CA1Data, CA2Data, CA3Data, dgData]
    xDevDataset = pd.concat(data)
    
    #For every image, get its amount of hippocampal pixels
    unique, counts = np.unique(xDevDataset.values[:,0], return_counts=True)

    #Obtain -randomly- the same amount of non-hippocampal pixels for every image
    #to create balanced dataset
    outer_values = pd.DataFrame()
    for img_name, count in zip(unique, counts) :
        img_rows = outerData['Source'] == img_name
        out_values = outerData.loc[ img_rows ]
        o_values = out_values.sample(n = count, random_state = 2)
        print(img_name+"; Total TP="+str(count)+"; total out_values="+str(out_values.shape[0])+"; Selected out_val="+str(o_values.shape[0]))
        outer_values = outer_values.append(o_values)

    print("Hippocampal pixel dataset shape, Rows={}, Columns={}".format(*xDevDataset.shape))
    print("Non-Hippocampal pixel dataset shape, Rows={}, Columns={}".format(*outer_values.shape))

    outerData = outer_values 
    
    return CA1Data, CA2Data, CA3Data, dgData, outerData
    
def splitDataSet(dataset, devSize=0.70, testSize=0.20, valSize=0.10) :
    #remove bogus records
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna();
    
    #shuffle (consider using - sklearn.utils.shuffle(nd) - it's 3x faster)
    dataset = dataset.sample(frac=1, random_state=99).reset_index(drop=True) 
    
    #Construct training dataset
    devDataset = dataset.sample(frac=devSize, random_state=99)
    
    #Split training dataset in data and label
    yDevDataset = devDataset[['Y']]
    xDevDataset = devDataset.drop('Y',axis=1)

    if (devSize + testSize < 1) :
        
        #Avoid duplicates
        remaining = dataset.loc[~dataset.index.isin(devDataset.index), :]
        
        #get test quantity
        testQuantity = int(dataset.shape[0] * testSize)

        #Create test set and Validation set
        testDataset = remaining.sample(n=testQuantity, random_state=99)
        valDataset = remaining.loc[~remaining.index.isin(testDataset.index), :]
        
        #Split Test dataset in data and label
        yTestDataset = testDataset[['Y']]
        xTestDataset = testDataset.drop('Y',axis=1)
        
        #Split Validation dataset in data and label
        yValDataset = valDataset[['Y']]
        xValDataset = valDataset.drop('Y',axis=1)
    else:
        testDataset = dataset.loc[~dataset.index.isin(devDataset.index), :]

        yValDataset = np.array([])
        xValDataset = np.array([])
    return (xDevDataset, yDevDataset), (xTestDataset, yTestDataset), (xValDataset, yValDataset)
    
    
def getDataSet(fileName = ""):
    ca1Data, ca2Data, ca3Data, dgData, outerData = loadFromFile(fileName)
    
    #Split dataset in 70%, 20%, 10% proportion
    ca1DevDataset, ca1TestDataset, ca1ValDataset = splitDataSet(ca1Data)
    ca2DevDataset, ca2TestDataset, ca2ValDataset = splitDataSet(ca2Data)
    ca3DevDataset, ca3TestDataset, ca3ValDataset = splitDataSet(ca3Data)
    dgDevDataset, dgTestDataset, dgValDataset = splitDataSet(dgData)
    outerDevDataset, outerTestDataset, outerValDataset = splitDataSet(outerData)

    #Stack Hippocampal pixel together to create the dataset properly
    frames = [ca1DevDataset[0], ca2DevDataset[0], ca3DevDataset[0], dgDevDataset[0], outerDevDataset[0]]
    xDevDataset = pd.concat(frames)
    frames = [ca1DevDataset[1], ca2DevDataset[1], ca3DevDataset[1], dgDevDataset[1], outerDevDataset[1]]
    yDevDataset = pd.concat(frames)
    
    frames = [ca1TestDataset[0], ca2TestDataset[0], ca3TestDataset[0], dgTestDataset[0], outerTestDataset[0]]
    xTestDataset = pd.concat(frames)
    frames = [ca1TestDataset[1], ca2TestDataset[1], ca3TestDataset[1], dgTestDataset[1], outerTestDataset[1]]
    yTestDataset = pd.concat(frames)
    
    frames = [ca1ValDataset[0], ca2ValDataset[0], ca3ValDataset[0], dgValDataset[0], outerValDataset[0]]
    xValDataset = pd.concat(frames)
    frames = [ca1ValDataset[1], ca2ValDataset[1], ca3ValDataset[1], dgValDataset[1], outerValDataset[1]]
    yValDataset = pd.concat(frames)

    return (xDevDataset, yDevDataset), (xTestDataset, yTestDataset),  (xValDataset, yValDataset)

def getClass(searchedValue=1):
    list_values = [ key for key,val in hippocampusAreas.items() if val== searchedValue ]
    return list_values[0]


def graph_hip_image(img_name, x_values, y_values, plt=plt) :
       
    img_x_values = x_values.loc[x_values['Source'] == img_name ] 
    img_y_values = y_values.loc[x_values['Source'] == img_name ] 
    
    tp = img_x_values.loc[img_y_values['Y'] > 0 ] 
    tn = img_x_values.loc[img_y_values['Y'] == 0 ] 
    
    plt.gca().invert_yaxis()
    plt.scatter(tp.loc[:,'PX'], tp.loc[:,'PY'], c="red", s=2, linewidth=1, label="Hip")
    plt.scatter(tn.loc[:,'PX'], tn.loc[:,'PY'], c='C0', s= 2,  linewidth=1, label="No Hip")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc="lower left")
    plt.show()
    
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=1, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')