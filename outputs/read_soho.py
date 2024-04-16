import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np

# Define the file path
# file_path = "/home/mewada/Documents/Exploring-Climate-Dynamics-Unraveling-Trends-in-Global-Temperature-Fluctuations/datasets/Global_TAVG_monthly.txt"
file_path = "/home/mewada/Documents/Exploring-Climate-Dynamics-Unraveling-Trends-in-Global-Temperature-Fluctuations/datasets/SOHO.txt"
ghg_file_path = "/home/mewada/Documents/Exploring-Climate-Dynamics-Unraveling-Trends-in-Global-Temperature-Fluctuations/datasets/annual-co2-emissions-per-country.csv"

# Define lists to store data for each column
years = []
months = []
anomalies = []
uncertainties = []
EUV_flux = []
euv_avg = []
# Open the file and read its contents
with open(file_path, "r") as file:
    reader = csv.reader(file, delimiter="\t")

    # Skip the header row if it exists
    next(reader, None)

    # Iterate over each row in the file
    for line_num, row in enumerate(reader, start=1):
        # Check if the line number is less than 94
        if line_num < 10:
            continue
        if row[0].startswith("%"):
            continue

        # Extract data from each row
        try:
            row_data = row[0].split()
        except ValueError:
            # If an exception occurs, it means the row contains non-numeric data
            print("Non-numeric data detected in the row:")
            continue
        day = int(float(row_data[2]))
        euv_avg.append(float(row_data[3]))

print(len(euv_avg))
for i in range(0, len(euv_avg), 48):
    EUV_flux.append(sum(euv_avg[i::31]) / len(euv_avg[i::31]))

final = []
for i in range(0, len(EUV_flux), 30):
    final.append(sum(EUV_flux[i : i + 30]) / 30)

print(final)

greenhouse_gas_data = pd.read_csv(ghg_file_path)
print(greenhouse_gas_data.head())