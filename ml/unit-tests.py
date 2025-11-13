import DataFunctions, TrainingFunctions, TestingFunctions, Entry
import pytest
import pandas as pd
import numpy as np
#data validation

#test that the data is shaped and cleaned as expected
def testDataShape(monkeypatch):
    #create fake df that is shaped like our data
    np.random.seed(6)
    test_df = pd.DataFrame({
        


        "open": np.random.rand(10),
        "high": np.random.rand(10),
        "low": np.random.rand(10),
        "close": np.random.rand(10),
        "volume": np.random.randint(1,100, 10)
        
    })
    #read the test_df as if it were a csv file saved to the computer
    monkeypatch.setattr(pd, "read_csv", lambda path: test_df) #chatgpt gave me this line of code

    #this line just uses the test df to generate a df according to the one constructed above
    df=DataFunctions.getDataframe("does not exist.csv")


    assert isinstance(df, pd.DataFrame) #df is the right type of object
    assert df.shape == (10,5) #correct sturcture          
    assert not df.isnull().any().any() #no null values

#test that we scale the data correctly
def testScaleData():
    df=pd.DataFrame({
        "open": np.random.rand(10,100,20),
        "high": np.random.rand(10,100,20),
        "low": np.random.rand(10,100,20),
        "close": np.random.rand(10,100,20),
        "volume": np.random.randint(1,100, 30)

    })
    scaled_data, scaler = DataFunctions.scaleData(df)
    assert scaled_data.shape == df.shape #test that scaling doesnt change the structure of the data
    assert np.all(scaled_data>=0) and np.all(scaled_data<=1) # test that the data actually scales to [0,1]



def testBreakDataSequenceShapes():
    data= np.arange(40).reshape(10, 4)
    seq_length=6
    pred_col=1

    x, y = DataFunctions.breakDataIntoSequences(data,seq_length,pred_col)
    assert x.shape == (4,6,4)
    assert y.shape == (4)

def testBreakDataSequenceShapes0():
    data= np.arange(40).reshape(10, 4)
    seq_length=0
    pred_col=1

    x, y = DataFunctions.breakDataIntoSequences(data,seq_length,pred_col)
    assert x.shape == (10,0,4)
    assert y.shape == (10)

def testBreakDataSequenceShapesMax():
    data= np.arange(40).reshape(10, 4)
    seq_length=10
    pred_col=1

    x, y = DataFunctions.breakDataIntoSequences(data,seq_length,pred_col)
    assert x.shape == (0,10,4)
    assert y.shape == (0)

def testSplitShape():
    
#to write