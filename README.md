# FuelWise V2

Streamlit weekly travel budgeting app for India (INR) supporting petrol, diesel, non-plug-in hybrid, plug-in hybrid, and electric vehicles.

## Run on Windows

Open Command Prompt inside this directory:

```bat
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Visit http://localhost:8501 if your browser does not open automatically.

## Methods

- Petrol, diesel and non-plug-in hybrid: litres = distance / entered mileage.
- Electric: kWh = distance × entered kWh/100 km / 100.
- Plug-in hybrid: split total distance between electric and fuel propulsion according to the user-entered share, then apply the respective efficiencies.
- Budget: fuel litres × ₹/L + kWh × ₹/kWh. Monthly average = weekly × 52/12.
- City driving, traffic and AC are **logged for future data analysis**, not used as unvalidated adjustment factors.
- For plug-in hybrids, the electric-driving share is a user estimate; charging losses and battery state of charge are not modeled.
- For electric vehicles, the electricity rate represents the price per kWh; charging losses are not included unless reflected in the user's entered consumption.
- No claim of vehicle-specific accuracy is made without actual measured data.

## Data provenance and ML

`vehicle_petrol_mileage_2000.csv` is the original user-supplied formula-derived dataset. It is shown in Data audit, not used for ML training. Actual trip records are stored locally in `my_trip_records.csv`, created by the app. The ML tab requires 30 or more observations per vehicle category to train Linear Regression and Random Forest on an 80/20 random split. For rigorous evaluation, use independent time-based and vehicle-based testing, and compare models to a formula baseline. User-entered records are not independently verified.

## Privacy

Trip logs stay on your computer in the project folder. Avoid sharing your personal travel history publicly. The app does not require any API key or internet service for fuel prices; you enter local rates manually.
