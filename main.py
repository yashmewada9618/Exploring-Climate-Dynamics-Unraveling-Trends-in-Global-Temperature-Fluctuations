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
from sklearn.metrics import r2_score, mean_squared_error

# Define the file path
file_path = "/home/mewada/Documents/Exploring-Climate-Dynamics-Unraveling-Trends-in-Global-Temperature-Fluctuations/datasets/Global_TAVG_monthly.txt"
ghg_file_path = "/home/mewada/Documents/Exploring-Climate-Dynamics-Unraveling-Trends-in-Global-Temperature-Fluctuations/datasets/annual-co2-emissions-per-country.csv"
soho_file_path = "/home/mewada/Documents/Exploring-Climate-Dynamics-Unraveling-Trends-in-Global-Temperature-Fluctuations/datasets/SOHO.txt"

# Define lists to store data for each column
years = []
months = []
anomalies = []
uncertainties = []

# Open the file and read its contents
with open(file_path, "r") as file:
    reader = csv.reader(file, delimiter="\t")

    # Skip the header row if it exists
    next(reader, None)

    # Iterate over each row in the file
    for line_num, row in enumerate(reader, start=1):
        # Check if the line number is less than 94
        if line_num < 93:
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
        year = int(row_data[0])
        month = int(row_data[1])
        anomaly_1 = float(row_data[2])
        uncertainty_1 = float(row_data[3])
        anomaly_2 = float(row_data[4])
        uncertainty_2 = float(row_data[5])
        anomaly_3 = float(row_data[6])
        uncertainty_3 = float(row_data[7])
        anomaly_4 = float(row_data[8])
        uncertainty_4 = float(row_data[9])
        anomaly_5 = float(row_data[10])
        uncertainty_5 = float(row_data[11])

        # Append extracted data to respective lists
        years.append(year)
        months.append(month)
        anomalies.append([anomaly_1, anomaly_2, anomaly_3, anomaly_4, anomaly_5])
        uncertainties.append(
            [uncertainty_1, uncertainty_2, uncertainty_3, uncertainty_4, uncertainty_5]
        )

# Create a DataFrame from the extracted data
data = {
    "Year": years,
    "Month": months,
    "Anomaly 1": [anomaly[0] for anomaly in anomalies],
    "Anomaly 2": [anomaly[1] for anomaly in anomalies],
    "Anomaly 3": [anomaly[2] for anomaly in anomalies],
    "Anomaly 4": [anomaly[3] for anomaly in anomalies],
    "Anomaly 5": [anomaly[4] for anomaly in anomalies],
    "Uncertainty 1": [uncertainty[0] for uncertainty in uncertainties],
}

df = pd.DataFrame(data)
greenhouse_gas_data = pd.read_csv(ghg_file_path)

# Perform imputation for missing values
imputer = SimpleImputer(strategy="mean")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Plot the time series of temperature data
plt.figure(figsize=(12, 6))
plt.plot(df_imputed["Year"], df_imputed["Anomaly 1"], label="Anomaly 1")
plt.plot(df_imputed["Year"], df_imputed["Anomaly 2"], label="Anomaly 2")
plt.plot(df_imputed["Year"], df_imputed["Anomaly 3"], label="Anomaly 3")
plt.plot(df_imputed["Year"], df_imputed["Anomaly 4"], label="Anomaly 4")
plt.plot(df_imputed["Year"], df_imputed["Anomaly 5"], label="Anomaly 5")
plt.xlabel("Year")
plt.ylabel("Temperature Anomaly (°C)")
plt.title("Global Temperature Anomalies")
plt.legend()
plt.grid(True)
plt.savefig("temperature_anomalies.png")
plt.show()

# Step 4: Seasonal Decomposition
# Decompose the time series into trend, seasonal, and residual components
# Filter data for the desired year (e.g., 2022)
# year_to_plot = 2000
# data_for_year = df_imputed[
#     (df_imputed["Year"] == year_to_plot) | (df_imputed["Year"] == year_to_plot + 1)
# ]

# # Create a time series from the filtered data
# time_series = pd.Series(
#     data_for_year["Anomaly 1"].values,
#     index=pd.date_range(start=str(year_to_plot), periods=len(data_for_year)),
# )

# # Perform seasonal decomposition
# decomposition = seasonal_decompose(time_series, model="additive", period=12)
# print(time_series)
# x1 = np.arange(0, len(time_series))


# # Plot the decomposition components
# plt.figure(figsize=(10, 8))
# plt.subplot(411)
# plt.plot(x1, decomposition.observed, label="Observed")
# plt.legend()

# plt.subplot(412)
# plt.plot(x1, decomposition.trend, label="Trend")
# plt.legend()

# plt.subplot(413)
# plt.plot(x1, decomposition.seasonal, label="Seasonal")
# plt.legend()

# plt.subplot(414)
# plt.plot(x1, decomposition.resid, label="Residual")
# plt.legend()

# plt.tight_layout()
# plt.savefig("seasonal_decomposition.png")
# plt.show()


# Filter data for the specified range of years (2000 to 2010)
first_year = df_imputed["Year"].min()
last_year = df_imputed["Year"].max() - 2

# Filter data for the first and last years
data_first_year = df_imputed[df_imputed["Year"] == first_year]
data_last_year = df_imputed[df_imputed["Year"] == last_year]

# Assuming you have monthly data, create an array of angles representing each month
angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()

# Create arrays of temperature anomalies and uncertainties for the first and last years
anomalies_first_year = data_first_year["Anomaly 1"].tolist()
uncertainties_first_year = data_first_year["Uncertainty 1"].tolist()
anomalies_last_year = data_last_year["Anomaly 1"].tolist()
uncertainties_last_year = data_last_year["Uncertainty 1"].tolist()

# Plot the polar plot
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Plot anomalies with error bars representing uncertainties for the first year
ax.errorbar(
    angles,
    anomalies_first_year,
    yerr=uncertainties_first_year,
    fmt="o-",
    label=str(first_year),
)

# Plot anomalies with error bars representing uncertainties for the last year
ax.errorbar(
    angles,
    anomalies_last_year,
    yerr=uncertainties_last_year,
    fmt="o-",
    label=str(last_year),
)

ax.set_theta_direction(-1)  # Set the direction of the theta axis (clockwise)
ax.set_theta_zero_location("N")  # Set the zero location of the theta axis (North)
ax.set_xticks(angles)  # Set the ticks on the theta axis to represent each month
ax.set_xticklabels(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)
ax.set_title(
    "Polar Plot of Temperature Anomalies for {} and {}".format(first_year, last_year)
)
plt.legend()
plt.savefig("polar_plot.png")
plt.show()

# Create a DataFrame to store temperature anomalies for each year
max_months = (
    df_imputed.groupby("Year").size().max()
)  # Maximum number of months among all years
anomalies_by_year = {}
for year in range(int(df_imputed["Year"].min()), int(df_imputed["Year"].max() + 1)):
    data_for_year = df_imputed[df_imputed["Year"] == year]
    anomalies = data_for_year["Anomaly 1"].values
    anomalies = np.pad(
        anomalies, (0, max_months - len(anomalies)), "constant", constant_values=np.nan
    )
    anomalies_by_year[year] = anomalies

# Convert the dictionary to a DataFrame
df_anomalies = pd.DataFrame(anomalies_by_year)

# Plot the heat map of temperature anomalies across different years
plt.figure(figsize=(10, 8))
sns.heatmap(df_anomalies, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Heatmap of Temperature Anomalies Across Different Years")
plt.xlabel("Year")
plt.ylabel("Month")
plt.xticks(rotation=45)
plt.savefig("temperature_anomalies_heatmap.png")
plt.show()

# Identify potential hotspots
# Calculate the mean temperature anomaly for each year
mean_anomalies = df_anomalies.mean(axis=1)

# Plot the mean anomalies over time
plt.figure(figsize=(10, 6))
plt.plot(mean_anomalies, marker="o", linestyle="-")
plt.title("Mean Temperature Anomaly Over Time")
plt.xlabel("Year")
plt.ylabel("Mean Temperature Anomaly")
plt.grid(True)
plt.savefig("mean_temperature_anomaly.png")
plt.show()

# Identify potential hotspots based on mean temperature anomalies
hotspot_threshold = 0.5  # Define a threshold for identifying hotspots
hotspots = mean_anomalies[mean_anomalies > hotspot_threshold]

# Print identified hotspots
print("Identified Hotspots:")
for year, anomaly in hotspots.iteritems():
    print(f"Year: {year}, Mean Temperature Anomaly: {anomaly:.2f}")

euv_avg = []
with open(soho_file_path, "r") as file:
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

        euv_avg.append(float(row_data[3]))


merged_data = pd.merge(df, greenhouse_gas_data, on="Year", how="inner")
merged_data = pd.merge(
    merged_data,
    pd.DataFrame(euv_avg, columns=["EUV Flux"]),
    left_index=True,
    right_index=True,
)

correlation_matrix = merged_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.show()

X = merged_data[["EUV Flux", "Annual emissions"]]
y = merged_data["Anomaly 1"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R-squared:", r_squared)
print("Mean Squared Error:", mse)
