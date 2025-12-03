import DataFunctions, TrainingFunctions, TestingFunctions
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

@pytest.mark.parametrize("split_frac", [0.6, 0.8, 0.9])
def testGenerateSplitShapes(split_frac):
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)

    X_train, X_test, y_train, y_test = DataFunctions.generateSplit(X, y, split_frac)

    expected_split = int(split_frac * len(X))
    assert len(X_train) == expected_split
    assert len(X_test) == len(X) - expected_split
    # ensure order is preserved
    assert np.array_equal(X_train[-1], X[expected_split - 1])
    assert np.array_equal(X_test[0], X[expected_split])
def testGetDataframeDrops0(tmp_path):
    # create a temporary CSV file
    file = tmp_path / "test.csv"
    df = pd.DataFrame({
        "open": [1, 2, np.nan],
        "high": [3, 4, 5]
    })
    df.to_csv(file, index=False)

    result = DataFunctions.getDataframe(str(file))

    assert result.shape == (2, 2)   # 1 row with NaN removed
    assert result.isnull().sum().sum() == 0
