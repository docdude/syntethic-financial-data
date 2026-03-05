def load_sp500_data(path):
    import pandas as pd
    data = pd.read_csv(path, index_col='Date', parse_dates=True)
    return data
