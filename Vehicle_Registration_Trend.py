import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('sorted_dataset.csv')

# Step 1: Filter out invalid years (exclude year '0')
df = df[df['NB_YEAR_MFC_VEH'] != 0]

# Step 2: Filter data for years between 2004 and 2024
df = df[(df['NB_YEAR_MFC_VEH'] >= 2000) & (df['NB_YEAR_MFC_VEH'] <= 2024)]

# Step 3: Filter the data to include only 'CD_CLASS_VEH' values of 1 or 2
df = df[df['CD_CLASS_VEH'].isin([1, 2])]

# Group the data by 'NB_YEAR_MFC_VEH' and count the number of vehicles for each year
vehicle_trend = df.groupby('NB_YEAR_MFC_VEH').size().reset_index(name='Registration_Count')

# Plotting the trend using Matplotlib
plt.figure(figsize=(12,6))
plt.plot(vehicle_trend['NB_YEAR_MFC_VEH'], vehicle_trend['Registration_Count'], marker='o', linestyle='-', color='b')

# Add labels and title
plt.title('Trend of Vehicle Registrations (2004-2024) for CD_CLASS_VEH 1 and 2')
plt.xlabel('Year of Manufacture')
plt.ylabel('Number of Vehicle Registrations')

# Show the plot
plt.grid(True)
plt.show()
