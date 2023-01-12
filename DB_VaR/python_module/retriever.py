import numpy as np
import pandas as pd
import yfinance as yf


class Retriever:
    def __init__(self):
        source_list = ['bdd', 'yahoo', 'bloomberg']

    def getData(self, ticker_lst, start, end, frequency):
        raise NotImplementedError

    def get_data_from_bdd(self):
        raise NotImplementedError

    def get_data_from_yfinance(self, ticker_lst, period="4Y"):
        ohlc = yf.download(ticker_lst, period=period)
        prices = ohlc["Adj Close"]
        return prices

    def get_data_from_bloomberg(self):
        raise NotImplementedError

    def get_curve(self):
        raise NotImplementedError