import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your existing data
df = pd.read_csv('RA205_3f63401c.csv')
df = df.dropna(subset=['Altitude', 'Speed'])

# 2. THEORETICAL FUEL MODEL (Airbus-style logic)
# Higher altitude = Thinner air = Lower fuel flow for same speed
# We use a base flow and adjust for altitude density
def estimate_fuel(row):
    base_flow = 2400  # kg/h (typical for A320/A330 class at cruise)
    # Decrease fuel flow by ~2% for every 1000ft of altitude
    alt_correction = 1 - (row['Altitude'] * 0.00002)
    # Increase flow if speed is higher
    speed_correction = row['Speed'] / 400 
    return base_flow * alt_correction * speed_correction

df['Est_Fuel_Flow'] = df.apply(estimate_fuel, axis=1)

# 3. ADVANCED VISUALIZATION
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Altitude
ax1.set_xlabel('Flight Time (Minutes)')
ax1.set_ylabel('Altitude (ft)', color='blue')
ax1.plot(df.index, df['Altitude'], color='blue', label='Altitude', linewidth=2)
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second axis for Speed
ax2 = ax1.twinx()
ax2.set_ylabel('Ground Speed (kts)', color='red')
ax2.plot(df.index, df['Speed'], color='red', label='Speed', alpha=0.6)
ax2.tick_params(axis='y', labelcolor='red')

# Create a THIRD axis for Fuel Flow (Offset to the right)
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
ax3.set_ylabel('Est. Fuel Flow (kg/h)', color='green')
ax3.plot(df.index, df['Est_Fuel_Flow'], color='green', linestyle='--', label='Est. Fuel')
ax3.tick_params(axis='y', labelcolor='green')

plt.title('Airbus Digital Twin: RA205 Telemetry + Theoretical Fuel Consumption')
fig.tight_layout()
plt.show()
