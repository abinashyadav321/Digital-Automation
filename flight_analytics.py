import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Flight Data
# Note: Update this to your local path: C:\Users\harer\University\Data Science\RA205_3f63401c.csv
file_path = 'RA205_3f63401c.csv'
df = pd.read_csv(file_path)

# 2. Data Cleaning & Feature Engineering (What Airbus looks for!)
# Convert the raw UTC string into a mathematical datetime object
df['UTC'] = pd.to_datetime(df['UTC'])

# The 'Position' column has combined data (e.g., "27.70,85.35"). We must split it!
df[['Latitude', 'Longitude']] = df['Position'].str.split(',', expand=True).astype(float)

# Calculate elapsed 'Flight Time' in minutes to make the X-axis readable
start_time = df['UTC'].iloc[0]
df['Flight_Minutes'] = (df['UTC'] - start_time).dt.total_seconds() / 60.0

# 3. Create the Dashboard Visualizations
plt.style.use('seaborn-v0_8-darkgrid')
# We create a figure with 2 subplots (2 rows, 1 column)
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8))

# --- TOP PLOT: Altitude and Speed Profile ---
ax1.set_xlabel('Flight Time (Minutes)', fontweight='bold')
ax1.set_ylabel('Altitude (ft)', color='tab:blue', fontweight='bold')
ax1.plot(df['Flight_Minutes'], df['Altitude'], color='tab:blue', label='Altitude', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Using the twinx() magic you learned earlier!
ax2 = ax1.twinx()
ax2.set_ylabel('Ground Speed (kts)', color='tab:red', fontweight='bold')
ax2.plot(df['Flight_Minutes'], df['Speed'], color='tab:red', label='Speed', linewidth=1.5, alpha=0.8)
ax2.tick_params(axis='y', labelcolor='tab:red')

ax1.set_title('Airbus Telemetry Analysis: RNA205 (KTM to DEL) Profile', fontsize=14, fontweight='bold')

# --- BOTTOM PLOT: Ground Track (Route Mapping) ---
ax3.set_xlabel('Longitude', fontweight='bold')
ax3.set_ylabel('Latitude', fontweight='bold')
ax3.plot(df['Longitude'], df['Latitude'], color='tab:green', marker='o', linestyle='-', markersize=2, alpha=0.7)
ax3.set_title('2D Ground Track Map', fontsize=14, fontweight='bold')

# Add KTM and DEL labels to the map
ax3.text(df['Longitude'].iloc[0], df['Latitude'].iloc[0], ' Kathmandu (KTM)', fontsize=10, weight='bold', color='black')
ax3.text(df['Longitude'].iloc[-1], df['Latitude'].iloc[-1], ' Delhi (DEL)', fontsize=10, weight='bold', color='black')

plt.tight_layout()
plt.show()