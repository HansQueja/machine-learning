
import pandas as pd

def excel_parser(filename):
    training_data = pd.read_excel(filename)

    return training_data