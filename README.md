# Introduction

## WeWantQuant stock market predictor
Our website allows anyone to choose a stock, machine learning model, and time period. Using our own model and already existing data, we will return our prediction of where the stock will be after the given time period along with other statistics.

While there are many other programs that try to predict the stock market, ours is different in that it is a lot more accessible. We provide enough statistics to be helpful without overwhelming the user with unnecessary information. We also offer a variety of models to use and give an estimate of how accurate each model is.

# Technical Architecture 

<img width="958" height="671" alt="Untitled drawing" src="https://github.com/user-attachments/assets/f65dd887-346a-4cb6-bd79-3b1c84eb0dba" />


# Installation Instructions  
To download the repository run
```
git clone https://github.com/CS222-UIUC/fa25-team058-wewantquant.git
```

To install the necesarry libraries run
```
pip install flask pandas numpy scikit-learn yfinance joblib
```

To start the predictor run 
```
cd app && python backendapp.py
```

To access run 
```
Open http://localhost:5000
```

# Developers
- **Aniketh Ganta**: Backend lead. Connected our model to the website.
- **Chinmay Rawat**: Deep learning specialist. Implemented pytorch and our LSTM neural network.
- **Aarya Baid**: Machine learning engineer. Designed our random forest model.
- **Colin Edsey**: Front end developer. Made the front end visually appealing.
