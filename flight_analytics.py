import pandas as pd
import matplotlib.pyplot as plt

file_path = 'RA205_3f63401c.csv'
df = pd.read_csv(file_path)

df['UTC'] = pd.to_datetime(df['UTC'])

df[['Latitude', 'Longitude']] = df['Position'].str.split(',', expand=True).astype(float)

start_time = df['UTC'].iloc[0]
df['Flight_Minutes'] = (df['UTC'] - start_time).dt.total_seconds() / 60.0

plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 8))

ax1.set_xlabel('Flight Time (Minutes)', fontweight='bold')
ax1.set_ylabel('Altitude (ft)', color='tab:blue', fontweight='bold')
ax1.plot(df['Flight_Minutes'], df['Altitude'], color='tab:blue', label='Altitude', linewidth=2)
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Ground Speed (kts)', color='tab:red', fontweight='bold')
ax2.plot(df['Flight_Minutes'], df['Speed'], color='tab:red', label='Speed', linewidth=1.5, alpha=0.8)
ax2.tick_params(axis='y', labelcolor='tab:red')

ax1.set_title('Airbus Telemetry Analysis: RNA205 (KTM to DEL) Profile', fontsize=14, fontweight='bold')

ax3.set_xlabel('Longitude', fontweight='bold')
ax3.set_ylabel('Latitude', fontweight='bold')
ax3.plot(df['Longitude'], df['Latitude'], color='tab:green', marker='o', linestyle='-', markersize=2, alpha=0.7)
ax3.set_title('2D Ground Track Map', fontsize=14, fontweight='bold')

ax3.text(df['Longitude'].iloc[0], df['Latitude'].iloc[0], ' Kathmandu (KTM)', fontsize=10, weight='bold', color='black')
ax3.text(df['Longitude'].iloc[-1], df['Latitude'].iloc[-1], ' Delhi (DEL)', fontsize=10, weight='bold', color='black')

plt.tight_layout()
plt.show()
