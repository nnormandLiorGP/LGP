import xlwings as xw
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta


def refresh_vcv(price_filepath, vcv_filepath, lambda_factor=0.9978, end_date=None):
    success = False
    try:
        # import excel file as df
        prices = pd.read_excel(price_filepath)

        # clean column names and set dates as index
        col_names = prices.columns.to_list()
        col_names[0] = 'Date'
        prices.columns = col_names
        prices = prices.set_index('Date')

        # filter on last 4 years data from last date
        if end_date is None:
            end_date = prices.index[-1]
        start_date = end_date - relativedelta(years=4)
        prices = prices[start_date:end_date]

        # fill NaNs
        prices = prices.fillna(method='ffill')
        if prices.isna().sum().sum() > 0:
            prices = prices.fillna(method='bfill')

        # Resample to weekly if necessary
        if pd.infer_freq(prices.index)[0] != "W":
            wednesdays = [d for d in prices.index if d.weekday() == 2]
            prices = prices.loc[wednesdays]
        print("From", prices.index[0], "To", prices.index[-1])

        # compute returns
        returns = prices / prices.shift(1) - 1

        # compute vcv
        vcv = returns.ewm(lambda_factor).mean().cov()

        # export vcv to excel file
        vcv.to_excel(vcv_filepath)

        success = True

    except Exception as ex:
        # Do something such as error handling to be defined
        print(ex)

    res = {'success': success,
           'refresh_time': dt.datetime.now(),
           'price_start_date': prices.index[0],
           'price_end_date': prices.index[-1]}

    print(res)

    return res


def refresh_vcv_handler():
    wb = xw.Book.caller()

    price_path = wb.sheets['Controls']['B2'].value
    vcv_path = wb.sheets['Controls']['B3'].value

    res_dict = refresh_vcv(price_path, vcv_path)
    if res_dict['refresh_time']:
        wb.sheets['Controls']['B4'].value = res_dict['refresh_time']
        wb.sheets['Controls']['B5'].value = res_dict['price_end_date']
    else:
        wb.sheets['Controls']['B4'].value = "N/A"
        wb.sheets['Controls']['B5'].value = "N/A"


@xw.func
@xw.arg('w', np.array, ndim=2)
@xw.arg('raw_vcv', np.array, ndim=2)
@xw.ret(index=False, header=False)
def simulate_iVaR_static(w, raw_vcv):
    # inputs:
        # w: the input weight matrix ordered based on the vcv
        # raw_vcv: the static vcv with header and row index
    # return: vector of iVaRs

    weights = pd.DataFrame(w, columns=['inst_id', 'mkt_expo']).set_index('inst_id')
    weights = weights.astype(float)

    df_vcv = pd.DataFrame(raw_vcv)
    if df_vcv.columns.dtype != "object":
        df_vcv.columns = df_vcv.iloc[0]
        df_vcv = df_vcv.drop(df_vcv.index[0])
    df_vcv = df_vcv.set_index(df_vcv.columns[0])
    df_vcv = df_vcv.astype(float)
    filtered_vcv = df_vcv[weights.index].loc[weights.index]

    epsilon = 0.0001
    iVaR = np.zeros(len(weights.index))
    init_variance = weights.mkt_expo.values @ filtered_vcv @ weights.mkt_expo.values
    init_VaR = np.sqrt(init_variance) * 2.33 * np.sqrt(20)

    for i in range(len(weights.index)):
        w_tmp = weights.mkt_expo.values.copy()
        w_tmp[i] = w_tmp[i] * (1 + epsilon)
        adj_variance = w_tmp.T @ filtered_vcv @ w_tmp
        adj_VaR = np.sqrt(adj_variance) * 2.33 * np.sqrt(20)
        iVaR[i] = (adj_VaR - init_VaR) / epsilon
    return pd.DataFrame(iVaR, index=weights.index, columns=['iVaR'])


@xw.func
@xw.arg('w', np.array, ndim=2)
@xw.ret(index=False, header=False)
def simulate_iVaR_fromfile(w, vcv_fpath):
    # inputs:
        # w: the input weight matrix ordered based on the vcv
        # raw_vcv: the static vcv with header and row index
    # return: vector of iVaRs

    vcv_import = pd.read_excel(vcv_fpath)
    vcv_import = vcv_import.set_index(list(vcv_import.columns)[0])
    vcv_import = np.array(vcv_import.to_records(index=True))

    return simulate_iVaR_static(w, vcv_import)