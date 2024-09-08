import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Load the dataset
df = pd.read_csv('Transport.csv')

# Remove commas from numeric columns, replace NaN with 0, and convert to integers
df['Metropolitan train'] = df['Metropolitan train'].str.replace(',', '').fillna('0').astype(int)
df['Metropolitan tram'] = df['Metropolitan tram'].str.replace(',', '').fillna('0').astype(int)
df['Metropolitan bus'] = df['Metropolitan bus'].str.replace(',', '').fillna('0').astype(int)
df['Regional train'] = df['Regional train'].str.replace(',', '').fillna('0').astype(int)
df['Regional coach'] = df['Regional coach'].str.replace(',', '').fillna('0').astype(int)
df['Regional bus'] = df['Regional bus'].str.replace(',', '').fillna('0').astype(int)

# Create a new column for total passengers by summing all transport-related columns
df['Total passengers'] = (df['Metropolitan train'] + df['Metropolitan tram'] +
                          df['Metropolitan bus'] + df['Regional train'] +
                          df['Regional coach'] + df['Regional bus'])

# Limit the data to rows with _id less than or equal to 78
df = df[df['_id'] <= 78]

# Plotting the trend using '_id' as the x-axis and 'Total passengers' as the y-axis
plt.figure(figsize=(10,6))
plt.plot(df['_id'], df['Total passengers'], marker='o', linestyle='-', color='b')

# Add labels and title
plt.title('Trend of Total Passengers Using Public Transport')
plt.xlabel('Year')
plt.ylabel('Total Number of Passengers (in Millions)')

# Format the y-axis to show values in millions (M)
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x*1e-6:.0f}M'))

# Set custom x-axis ticks to show years
plt.xticks(df['_id'], df['Year'], rotation=45, ha='right')

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()
