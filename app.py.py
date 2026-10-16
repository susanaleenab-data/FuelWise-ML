import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="FuelWise V2 | Smart Mobility Budget",page_icon="🚗",layout="wide")
DATA=Path(__file__).parent
BASE=DATA/"vehicle_petrol_mileage_2000.csv"
VEHICLES=["Petrol","Diesel","Hybrid (non-plug-in)","Plug-in hybrid","Electric"]
st.markdown("""<style>
.block-container{padding-top:1.7rem}
[data-testid="stMetric"]{border:1px solid #334155;border-radius:14px;padding:13px}
h1,h2,h3{letter-spacing:-.02em}
</style>""",unsafe_allow_html=True)
def inr(n): return f"₹{n:,.2f}"
def predict_usage(km,kind,kmpl,kwh100,evshare):
    ev_km=km*(evshare/100 if kind=="Plug-in hybrid" else 1 if kind=="Electric" else 0)
    fuel_km=km-ev_km
    return fuel_km/kmpl if kind!="Electric" else 0.0, ev_km*kwh100/100
def load_log():
    if "trip_records" not in st.session_state:
        st.session_state["trip_records"] = []
    return pd.DataFrame(st.session_state["trip_records"])
def train_evaluate(df,features,target):
    clean=df.dropna(subset=features+[target]).copy()
    if len(clean)<30: return None
    X=clean[features]; y=clean[target]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    out={}
    for name,model in [("Linear Regression",LinearRegression()),("Random Forest",RandomForestRegressor(n_estimators=150,min_samples_leaf=2,random_state=42,n_jobs=-1))]:
        model.fit(Xtr,ytr)
        pred=model.predict(Xte)
        out[name]={"MAE":mean_absolute_error(yte,pred),"RMSE":np.sqrt(mean_squared_error(yte,pred)),"R²":r2_score(yte,pred) if len(yte)>1 else float("nan"),"pred":pred,"actual":yte.to_numpy(),"model":model}
    return out

st.title("🚗 FuelWise V2")
st.caption("Smart weekly travel budget • India (₹) • Transparent data science")
with st.sidebar:
    st.header("Your weekly journey")
    kind=st.selectbox("Vehicle type",VEHICLES)
    km=st.number_input("Weekly travel distance (km)",min_value=0.0,value=250.0,step=10.0)
    city=st.slider("City driving (%)",0,100,60)
    traffic=st.selectbox("Traffic level",["Light","Moderate","Heavy"])
    ac=st.checkbox("Air conditioning used",value=True)
    if kind!="Electric":
        kmpl=st.number_input("Expected fuel mileage (km/L)",min_value=0.1,value=20.0,step=0.5)
        fuel_price=st.number_input("Fuel price (₹/L)",min_value=0.0,value=105.0,step=1.0)
    else: kmpl,fuel_price=20.0,0.0
    if kind in ["Electric","Plug-in hybrid"]:
        kwh100=st.number_input("Electricity consumption (kWh/100 km)",min_value=0.1,value=16.0,step=0.5)
        elec_price=st.number_input("Electricity price (₹/kWh)",min_value=0.0,value=9.0,step=0.5)
        evshare=st.slider("Distance driven on electricity (%)",0,100,60) if kind=="Plug-in hybrid" else 100
    else: kwh100,elec_price,evshare=16.0,0.0,0
    budget=st.number_input("Weekly travel budget (₹)",min_value=0.0,value=1500.0,step=100.0)
fuel,energy=predict_usage(km,kind,kmpl,kwh100,evshare)
fuelcost=fuel*fuel_price; eleccost=energy*elec_price; total=fuelcost+eleccost
a,b,c,d=st.columns(4)
a.metric("Weekly fuel",f"{fuel:.2f} L")
b.metric("Weekly electricity",f"{energy:.2f} kWh")
c.metric("Weekly cost",inr(total))
d.metric("Monthly average",inr(total*52/12))
st.caption(f"Annual projection: **{inr(total*52)}** | Vehicle: **{kind}**")
if total>budget: st.warning(f"Your estimated weekly spend exceeds your budget by {inr(total-budget)}.")
else: st.success(f"Your estimate is within budget by {inr(budget-total)}.")
st.info("Baseline uses the mileage/energy-efficiency values you enter. Traffic, AC and city share are recorded for analysis, not applied as invented fuel-consumption multipliers. Actual results may vary.")
tabs=st.tabs(["📊 Budget dashboard","🧾 Trip log","🧠 Machine learning","🔎 Data audit"])
with tabs[0]:
    st.subheader("Fuel and electricity budget")
    left,right=st.columns(2)
    with left:
        dfp=pd.DataFrame({"Week":[f"Week {i}" for i in range(1,5)],"Estimated cost (₹)":[total]*4,"Budget (₹)":[budget]*4})
        fig=px.bar(dfp,x="Week",y=["Estimated cost (₹)","Budget (₹)"],barmode="group",title="Four-week spending plan")
        st.plotly_chart(fig,use_container_width=True)
    with right:
        pie=pd.DataFrame({"Category":["Fuel","Electricity"],"Cost":[fuelcost,eleccost]})
        pie=pie[pie.Cost>0]
        if len(pie): st.plotly_chart(px.pie(pie,names="Category",values="Cost",hole=.5,title="Energy spending split"),use_container_width=True)
        else: st.write("No spending for the selected distance.")
    st.subheader("What-if analysis")
    varying=np.linspace(max(0,km*.5),max(1,km*1.5),21)
    costs=[]
    for distance in varying:
        f,e=predict_usage(distance,kind,kmpl,kwh100,evshare)
        costs.append(f*fuel_price+e*elec_price)
    st.plotly_chart(px.line(pd.DataFrame({"Weekly distance (km)":varying,"Weekly cost (₹)":costs}),x="Weekly distance (km)",y="Weekly cost (₹)",title="How cost changes with distance"),use_container_width=True)
with tabs[1]:
    st.subheader("Record actual travel and energy used")
    st.info("Public demo: trip entries are kept temporarily in your own app session, not in a shared server file. Download your CSV before closing or resetting your session. Do not enter sensitive personal information.")
    with st.form("log_form"):
        p,q,r=st.columns(3)
        date=p.date_input("Date")
        dist=q.number_input("Actual distance (km)",min_value=0.1,value=100.0)
        logged_kind=r.selectbox("Vehicle category",VEHICLES)
        p,q,r=st.columns(3)
        litres=p.number_input("Actual fuel consumed (L)",min_value=0.0,value=0.0,step=0.1)
        kwh=q.number_input("Actual electricity consumed (kWh)",min_value=0.0,value=0.0,step=0.1)
        actual_spend=r.number_input("Actual expense (₹)",min_value=0.0,value=0.0,step=10.0)
        p,q=st.columns(2)
        observed_city=p.slider("City driving share (%)",0,100,city,key="logged_city")
        observed_traffic=q.selectbox("Observed traffic",["Light","Moderate","Heavy"])
        observed_ac=st.checkbox("AC used during this trip",value=ac,key="logged_ac")
        save=st.form_submit_button("Save trip record",type="primary")
    if save:
        if litres==0 and kwh==0: st.error("Enter at least one measured consumption value.")
        elif logged_kind=="Electric" and litres>0: st.error("Electric-only trips cannot include fuel litres.")
        elif logged_kind in ["Petrol","Diesel","Hybrid (non-plug-in)"] and kwh>0: st.error("Only electric or plug-in hybrid trips can include separately charged electricity.")
        else:
            record={"date":str(date),"vehicle_type":logged_kind,"distance_km":dist,"fuel_liters":litres,"electricity_kwh":kwh,"expense_inr":actual_spend,"city_pct":observed_city,"traffic":observed_traffic,"ac_used":int(observed_ac)}
            st.session_state["trip_records"].append(record)
            st.success("Trip added to this session. Download the CSV if you want to keep it.")
    logs=load_log()
    if len(logs):
        st.write(f"Saved observations: **{len(logs)}**")
        st.dataframe(logs,use_container_width=True)
        st.download_button("Download your trip history (CSV)",logs.to_csv(index=False),file_name="fuelwise_real_trip_history.csv",mime="text/csv")
        st.plotly_chart(px.line(logs.assign(date=pd.to_datetime(logs.date)).sort_values("date"),x="date",y="expense_inr",markers=True,title="Recorded expenses (₹)"),use_container_width=True)
    else: st.caption("No observations recorded in this session yet.")
with tabs[2]:
    st.subheader("Machine learning experiment")
    st.warning("Model training uses ONLY observations entered in your current session, never formula-generated synthetic data. A minimum of 30 suitable observations is required. Enter only genuine measured trips for meaningful results; small datasets may produce unstable scores.")
    logs=load_log()
    target_type=st.selectbox("Train on vehicle category",VEHICLES)
    subset=logs[logs.vehicle_type==target_type].copy() if len(logs) and "vehicle_type" in logs else pd.DataFrame()
    if len(subset):
        subset["traffic_code"]=subset.traffic.map({"Light":0,"Moderate":1,"Heavy":2})
        subset["ac_used"]=pd.to_numeric(subset.ac_used,errors="coerce")
        for k in ["distance_km","city_pct","fuel_liters","electricity_kwh"]:
            subset[k]=pd.to_numeric(subset[k],errors="coerce")
    target="electricity_kwh" if target_type=="Electric" else "fuel_liters"
    if target_type=="Plug-in hybrid":
        target=st.radio("Prediction target",["fuel_liters","electricity_kwh"],horizontal=True)
    features=["distance_km","city_pct","traffic_code","ac_used"]
    st.caption(f"Actual observations available for {target_type}: {len(subset)}. Training target: {target}.")
    result=train_evaluate(subset,features,target) if len(subset) else None
    if result:
        scores=pd.DataFrame([{"Model":name,**{k:v for k,v in obj.items() if k in ["MAE","RMSE","R²"]}} for name,obj in result.items()])
        st.dataframe(scores,use_container_width=True,hide_index=True)
        winner=min(result,key=lambda name:result[name]["MAE"])
        st.write(f"Lowest held-out MAE: **{winner}**. Metrics measure performance on a random 20% test split of recorded data.")
        preds=result[winner]
        chart=pd.DataFrame({"Actual":preds["actual"],"Predicted":preds["pred"]})
        fig=px.scatter(chart,x="Actual",y="Predicted",title="Actual vs predicted (held-out records)")
        low=min(chart.min()); high=max(chart.max())
        fig.add_trace(go.Scatter(x=[low,high],y=[low,high],mode="lines",name="Perfect prediction",line=dict(dash="dash")))
        st.plotly_chart(fig,use_container_width=True)
        if target_type=="Electric": val=energy
        elif target_type=="Plug-in hybrid": val=energy if target=="electricity_kwh" else fuel
        else: val=fuel
        st.caption(f"Formula-based weekly baseline for selected vehicle: {val:.2f} {'kWh' if target=='electricity_kwh' else 'L'}. Model scores do not establish accuracy for unobserved vehicle models or driving conditions.")
        st.info("For rigorous research, add vehicle model, road grade and weather; compare against a held-out formula baseline, and use time-based or vehicle-group splits to avoid leakage.")
    else:
        st.info("Start logging actual trips. Once you have at least 30 records for a vehicle category, this section will compare Linear Regression and Random Forest on a held-out test set.")
with tabs[3]:
    st.subheader("Original dataset audit")
    if BASE.exists():
        original=pd.read_csv(BASE)
        st.write(f"Rows: **{len(original):,}** | Columns: **{len(original.columns)}**")
        st.dataframe(original.head(15),use_container_width=True)
        st.write("Missing values by column")
        st.json(original.isna().sum().to_dict())
        if {"distance_traveled_km","vehicle_mileage_km_per_liter","petrol_used_liters"}.issubset(original.columns):
            n=original.distance_traveled_km.nunique()
            delta=(original.petrol_used_liters-original.distance_traveled_km/original.vehicle_mileage_km_per_liter).abs().max()
            st.write(f"Distinct trip distances: **{n}**")
            st.write(f"Maximum deviation from distance/mileage formula: **{delta:.8f} L**")
            st.warning("The original uploaded data contains formula-derived consumption for 1-km trips. It is preserved for auditing and is NOT used to validate real-world ML accuracy.")
    else: st.warning("Original CSV not found; put vehicle_petrol_mileage_2000.csv next to app.py.")
    st.markdown("**Data provenance:** Original supplied CSV is a formula dataset. The session-only Trip Log contains user-entered observations, which have not been independently verified. No external real-world fuel-efficiency dataset is bundled.")
