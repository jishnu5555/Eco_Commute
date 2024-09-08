# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset (replace with actual file path)
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

# Limit the data to rows with _id less than or equal to 78 (for 2018 to 2024)
df = df[df['_id'] <= 78]

# Plotting the existing trend using '_id' as the x-axis and 'Total passengers' as the y-axis
plt.figure(figsize=(10,6))
plt.plot(df['_id'], df['Total passengers'], marker='o', linestyle='-', color='b')

# Add labels and title
plt.title('Trend of Total Passengers Using Public Transport (2018 - 2024)')
plt.xlabel('Month')
plt.ylabel('Total Number of Passengers (in Millions)')

# Format the y-axis to show values in millions (M)
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x*1e-6:.0f}M'))

# Set custom x-axis ticks to show years (every 12 months)
plt.xticks(df['_id'], df['Year'], rotation=45, ha='right')

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()

# Prepare the data for training the model
# Use '_id' and 'Year' as the features (we'll treat _id as a time index)
X = df[['Year']]  # You could add more features like weather, population, etc. if available
y = df['Total passengers']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the trend for the test set
y_pred = model.predict(X_test)

# Print model performance metrics
print(f"R-squared: {model.score(X_test, y_test)}")

# Now, let's predict the future (2024-2029) without considering incentives
# First, simulate the future months
future_months = pd.DataFrame({
    '_id': np.arange(79, 79 + 60),  # Next 60 months: 2024-2029 (5 years)
    'Year': np.repeat(np.arange(2024, 2029), 12)  # Repeat each year 12 times (assuming monthly data)
})

# Predict future public transport usage without considering incentives
future_predictions = model.predict(future_months[['Year']])

# Combine the future months and predictions into a DataFrame for visualization
future_months['Predicted Passengers'] = future_predictions

# Plot both the historical and predicted future trends
plt.figure(figsize=(10,6))

# Plot historical data
plt.plot(df['_id'], df['Total passengers'], marker='o', linestyle='-', color='b', label='Actual Passengers (2018-2024)')

# Plot future predictions (2024-2029)
plt.plot(future_months['_id'], future_months['Predicted Passengers'], marker='o', linestyle='--', color='r', label='Predicted Passengers (2024-2029 without Incentive)')

# Add labels, title, and legend
plt.title('Public Transport Usage: Historical Data and Future Predictions without Incentive Program')
plt.xlabel('Month')
plt.ylabel('Total Number of Passengers (in Millions)')
plt.xticks(list(df['_id']) + list(future_months['_id'].iloc[::12]), list(df['Year']) + list(future_months['Year'].iloc[::12]), rotation=45, ha='right')

# Format the y-axis to show values in millions
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x*1e-6:.0f}M'))

# Add legend and grid
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

# Display the predicted future values
print(future_months[['Year', 'Predicted Passengers']])
