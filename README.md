Flight Telemetry & Fuel Consumption Analytics


✈️ Overview
This project contains Python scripts designed to process, clean, and visualize commercial flight telemetry data. It focuses on analyzing flight profiles (Altitude vs. Ground Speed) and simulating a theoretical fuel burn model based on atmospheric density and aircraft performance parameters.

The current dataset analyzes flight RNA205 routing from Kathmandu (KTM) to Delhi (DEL).

🛠️ Features
Data Engineering: Parses raw UTC strings into datetime objects and splits combined coordinate strings (Latitude/Longitude) into usable float data using pandas.

Flight Profiling (flight_analytics.py): Generates dual-axis visualizations of Altitude and Ground Speed over elapsed flight time, alongside a 2D spatial mapping of the flight's ground track.

Fuel Consumption Modeling (fuelburn.py): Implements a theoretical Airbus-style fuel logic model. It establishes a base fuel flow (typical for A320/A330 class) and dynamically applies correction factors based on real-time altitude density (decreasing flow by ~2% per 1000ft) and ground speed.

Advanced Visualizations: Utilizes matplotlib to create complex, multi-axis plots demonstrating the relationship between altitude, speed, and estimated fuel flow.

💻 Tech Stack
Python 3.x

Pandas: For data ingestion, cleaning, and feature engineering.

Matplotlib: For advanced, multi-axis telemetry visualization.

🚀 How to Run Locally
1. Clone the repository:

Bash
git clone https://github.com/yourusername/flight-telemetry-analytics.git
cd flight-telemetry-analytics



2. Install dependencies:

Bash
pip install -r requirements.txt



3. Execute the scripts:
Ensure the flight data CSV (RA205_3f63401c.csv) is in the root directory.

Bash
python flight_analytics.py
python fuelburn.py
📊 Visual Outputs

<img width="1000" height="800" alt="Figure_1" src="https://github.com/user-attachments/assets/dfb97370-4a41-4cc4-968a-8cbc770636e1" />
<img width="1200" height="600" alt="Fuel burn" src="https://github.com/user-attachments/assets/95a1300a-2ecd-4b6f-b8e6-a02268743a03" />



