import time
import html
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import streamlit.components.v1 as components
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="AI IoT Edge–Cloud Decision Lab", page_icon="📡", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}
.main{background:linear-gradient(135deg,#061321,#10283d)}
.block-container{padding-top:1.1rem;max-width:1550px}
.hero{padding:22px 26px;border-radius:20px;background:linear-gradient(110deg,#102e4d,#14506b);border:1px solid #2a7690;color:white}
.hero h1{color:white;font-size:29px;margin:0 0 8px}.hero p{color:#d4eaf5;margin:3px 0}
.pulse{display:inline-block;width:9px;height:9px;border-radius:50%;background:#36e0b5;animation:pulse 1.4s infinite;margin-right:8px}
@keyframes pulse{0%{box-shadow:0 0 0 0 #36e0b580}70%{box-shadow:0 0 0 12px #36e0b500}100%{box-shadow:0 0 0 0 #36e0b500}}
.node{padding:13px 8px;border-radius:14px;background:linear-gradient(145deg,#123957,#0c263d);border:1px solid #286080;color:#eaf7ff;text-align:center;min-height:105px}
.node .ico{font-size:30px;display:block;margin-bottom:5px}.node small{color:#a9cfe1}
.node.selected{border:2px solid #35e0b5;box-shadow:0 0 18px #35e0b566}
.packet{height:5px;background:linear-gradient(90deg,transparent,#27d6b3,#49a9ff,transparent);border-radius:10px;background-size:200% 100%;animation:flow 1.5s linear infinite;margin:8px 0 15px}
@keyframes flow{0%{background-position:100% 0}100%{background-position:-100% 0}}
.flowline{height:3px;background:linear-gradient(90deg,#26d6b4,#55b9ff);background-size:200% 100%;animation:flow 2s linear infinite;border-radius:5px;margin:8px 4px}
.appcard{background:#102d49;border:1px solid #285575;border-radius:13px;padding:12px;text-align:center;color:#effaff;min-height:112px}
.appcard .big{font-size:34px}.appcard b{display:block;margin:3px 0}.appcard small{color:#a9cfe1}
div[data-testid="stMetric"]{background:#102d49;border:1px solid #285575;padding:12px;border-radius:12px}
div[data-testid="stMetric"] label,div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:#effaff}
div[data-testid="stTabs"] button{font-weight:700}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>📡 AI-Powered IoT–Edge–Cloud Task Decision Lab</h1>
<p><span class="pulse"></span>NEW TASK INPUT · ANIMATED IOT NETWORK · MULTI-ALGORITHM REGRESSION · INTELLIGENT EXECUTION DECISION</p>
<p>Objective 1: Predict end-to-end task completion time and compare execution locations using AI regression models.</p></div>""", unsafe_allow_html=True)

LOCATIONS = ["Local Device", "Edge Server 1", "Edge Server 2", "Edge Server 3", "Edge Server 4", "Cloud Server"]
NUM = ["task_input_kb", "cpu_cycles_giga", "return_data_kb", "deadline_ms", "bandwidth_mbps", "rtt_ms", "queue_jobs", "cpu_capacity_ghz", "cpu_utilization_pct", "effective_cpu_ghz"]
CAT = ["iot_application", "task_type", "task_priority", "location", "node_type"]
FEATURES = NUM + CAT
DATASET_PATH = Path(__file__).resolve().parent / "objective1_iot_multinode_training_dataset.csv"

APP_PRESETS = {
    "Smart Agriculture": {"icon":"🌾", "task":180, "cycles":0.8, "result":20, "deadline":100, "description":"Soil, crop, irrigation and field-sensor analytics"},
    "UAV / Drone Surveillance": {"icon":"🚁", "task":1800, "cycles":5.5, "result":120, "deadline":150, "description":"Aerial image processing, object detection and video analytics"},
    "Smart Home": {"icon":"🏠", "task":60, "cycles":0.15, "result":8, "deadline":80, "description":"Occupancy, appliance control and home automation"},
    "Healthcare Monitoring": {"icon":"🩺", "task":450, "cycles":2.2, "result":80, "deadline":100, "description":"Wearable sensor analysis and patient alert generation"},
    "Industrial IoT": {"icon":"🏭", "task":1200, "cycles":5.0, "result":100, "deadline":80, "description":"Machine condition monitoring and predictive maintenance"},
    "Smart Parking": {"icon":"🅿️", "task":90, "cycles":0.3, "result":12, "deadline":120, "description":"Vehicle detection, slot availability and guidance"},
    "Autonomous Vehicle / V2X": {"icon":"🚗", "task":900, "cycles":4.0, "result":40, "deadline":30, "description":"Road perception, hazard detection and cooperative decisions"},
    "Smart City Traffic": {"icon":"🚦", "task":700, "cycles":2.8, "result":45, "deadline":100, "description":"Traffic camera analytics and adaptive signal control"},
}

@st.cache_data(show_spinner=False)
def load_training_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Training CSV not found: {DATASET_PATH}. Keep the CSV beside app.py.")
    df = pd.read_csv(DATASET_PATH)
    required = set(FEATURES + ["completion_time_ms", "dataset_split", "task_id"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df

MODEL_BUNDLE_PATH = Path(__file__).resolve().parent / "trained_models" / "regression_models_bundle.joblib"

@st.cache_resource(show_spinner="Loading pre-trained IoT regression models...")
def load_pretrained_models():
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            "Pre-trained model bundle is missing. Keep trained_models/regression_models_bundle.joblib "
            "in the repository beside app.py. See README.md for instructions."
        )
    bundle = joblib.load(MODEL_BUNDLE_PATH)
    required = {"models", "scores", "Xtest", "ytest", "test_predictions", "train_count", "test_count", "best_model_name"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Pre-trained model bundle is incomplete: {sorted(missing)}")
    return bundle

bundle = load_pretrained_models()
models = bundle["models"]
scores = bundle["scores"]
Xtest = bundle["Xtest"]
ytest = bundle["ytest"]
test_predictions = bundle["test_predictions"]
train_count = bundle["train_count"]
test_count = bundle["test_count"]
best_model_name = bundle["best_model_name"]

# Input panel
st.sidebar.header("🧾 Enter a NEW IoT task")
st.sidebar.markdown("**A. Task identity and application**")
app=st.sidebar.selectbox("Choose IoT application / task scenario",list(APP_PRESETS.keys()))
p=APP_PRESETS[app]
st.sidebar.caption(p["description"])
all_task_types = sorted(load_training_dataset().loc[load_training_dataset()["iot_application"] == app, "task_type"].dropna().unique().tolist())
task_type = st.sidebar.selectbox("Task type", all_task_types)
task_priority = st.sidebar.selectbox("Task priority", ["Low", "Normal", "High", "Critical"], index=1)
task_name=st.sidebar.text_input("Task name / ID",value=f"{app.replace(' ','_')}_Task_001")
task=st.sidebar.number_input("Task input data (KB)",1.,50000.,float(p["task"]),10.)
cycles=st.sidebar.number_input("Required computation (billion CPU cycles)",.01,100.,float(p["cycles"]),.1)
result=st.sidebar.number_input("Output / result data (KB)",.1,10000.,float(p["result"]),5.)
deadline=st.sidebar.number_input("Task deadline (ms)",1.,100000.,float(p["deadline"]),1.)
st.sidebar.markdown("---")
st.sidebar.markdown("**B. Network and server requirements (enter values for each candidate node)**")
st.sidebar.subheader("🖥️ Network and server status")
resource_defaults={
    "Local Device":(2.5,48,1,72,2.5),
    "Edge Server 1":(10.,42,5,65,7.0),
    "Edge Server 2":(14.,55,12,110,9.0),
    "Edge Server 3":(18.,38,8,145,6.0),
    "Edge Server 4":(22.,30,4,220,8.0),
    "Cloud Server":(48.,45,25,300,65.0),
}
resources={}
for loc,default in resource_defaults.items():
    with st.sidebar.expander(loc,expanded=(loc in ["Edge Server 1", "Edge Server 4"])):
        cpu=st.slider(f"{loc} CPU capacity (GHz)",.2,100.,default[0],.2,key=loc+"cpu")
        util=st.slider(f"{loc} CPU utilization (%)",0,99,default[1],key=loc+"util")
        q=st.slider(f"{loc} queued tasks",0,250,default[2],key=loc+"q")
        bw=st.slider(f"{loc} bandwidth (Mbps)",1,1000,default[3],key=loc+"bw")
        rtt=st.slider(f"{loc} round-trip delay (ms)",.1,300.,default[4],.1,key=loc+"rtt")
        resources[loc]={"cpu":cpu,"util":util,"q":q,"bw":bw,"rtt":rtt}

st.sidebar.caption("These are manually entered demonstration values, not live sensor/router/server telemetry.")
current_task_signature = (app, task_type, task_priority, task_name, task, cycles, result, deadline,
                          tuple((loc, resources[loc]["cpu"], resources[loc]["util"], resources[loc]["q"],
                                 resources[loc]["bw"], resources[loc]["rtt"]) for loc in LOCATIONS))
if "task_submitted" not in st.session_state:
    st.session_state.task_submitted = False
if st.session_state.get("submitted_signature") != current_task_signature:
    st.session_state.task_submitted = False
if st.sidebar.button("🚀 Submit New Task to AI", type="primary", use_container_width=True):
    st.session_state.task_submitted = True
    st.session_state.submitted_signature = current_task_signature
    st.session_state.task_submission_count = st.session_state.get("task_submission_count", 0) + 1
    st.sidebar.success(f"Task received: {task_name}")
if st.session_state.task_submitted:
    st.sidebar.caption(f"Latest submitted task: {task_name}")
else:
    st.sidebar.caption("Enter the task details and press Submit New Task to start the animated decision flow.")

# Find best regression model by RMSE on held-out test split.
best_model_name=str(scores.iloc[0]["Algorithm"])
best_model=models[best_model_name]
node_type_map = {"Local Device":"Local", "Cloud Server":"Cloud"}
new_task_rows=[]
for loc in LOCATIONS:
    r=resources[loc]
    node_type=node_type_map.get(loc,"Edge")
    effective_cpu=max(r["cpu"]*(1-r["util"]/100),.01)
    new_task_rows.append({
        "task_input_kb":task, "cpu_cycles_giga":cycles, "return_data_kb":result,
        "deadline_ms":deadline, "bandwidth_mbps":r["bw"], "rtt_ms":r["rtt"],
        "queue_jobs":r["q"], "cpu_capacity_ghz":r["cpu"], "cpu_utilization_pct":r["util"],
        "effective_cpu_ghz":effective_cpu, "iot_application":app, "task_type":task_type,
        "task_priority":task_priority, "location":loc, "node_type":node_type
    })
new_task=pd.DataFrame(new_task_rows)[FEATURES]
all_model_predictions={name:mdl.predict(new_task) for name,mdl in models.items()}
selected_predictions=all_model_predictions[best_model_name]
results=pd.DataFrame({"Execution node":LOCATIONS,"Predicted completion time (ms)":selected_predictions})
results["Deadline status"] = np.where(results["Predicted completion time (ms)"]<=deadline,"Within deadline","Misses deadline")
feasible=results[results["Predicted completion time (ms)"]<=deadline]
if len(feasible):
    chosen=feasible.sort_values("Predicted completion time (ms)").iloc[0]
    decision=str(chosen["Execution node"]); decision_status="DEADLINE-FEASIBLE"
else:
    chosen=results.sort_values("Predicted completion time (ms)").iloc[0]
    decision=str(chosen["Execution node"]); decision_status="NO NODE MEETS DEADLINE"

# Hero visual application cards
st.subheader("1. Select an IoT application – task scenario")
app_cols=st.columns(4)
for col,(name,info) in zip(app_cols,list(APP_PRESETS.items())[:4]):
    with col: st.markdown(f'<div class="appcard"><div class="big">{info["icon"]}</div><b>{name}</b><small>{info["description"]}</small></div>',unsafe_allow_html=True)
app_cols2=st.columns(4)
for col,(name,info) in zip(app_cols2,list(APP_PRESETS.items())[4:]):
    with col: st.markdown(f'<div class="appcard"><div class="big">{info["icon"]}</div><b>{name}</b><small>{info["description"]}</small></div>',unsafe_allow_html=True)

st.subheader("2. Animated multi-device IoT–Edge–Cloud network")
# Continuously animated SVG network inspired by the supplied multi-device system figure.
route_map = {
    "Local Device": "M 150 255 Q 125 290 150 330",
    "Edge Server 1": "M 150 255 Q 275 120 440 155",
    "Edge Server 2": "M 150 255 Q 290 355 440 325",
    "Edge Server 3": "M 150 255 Q 390 405 590 350",
    "Edge Server 4": "M 150 255 Q 260 35 350 105",
    "Cloud Server": "M 150 255 Q 390 35 675 150",
}
active_path = route_map[decision]
source_icon = APP_PRESETS[app]["icon"]
safe_task_name = html.escape(str(task_name).strip() or "Unnamed IoT Task")
display_task_name = safe_task_name if len(safe_task_name) <= 34 else safe_task_name[:31] + "..."
submission_label = "NEW TASK RECEIVED" if st.session_state.task_submitted else "TASK READY FOR SUBMISSION"
packet_begin = "0s" if st.session_state.task_submitted else "indefinite"
return_begin = "1.6s" if st.session_state.task_submitted else "indefinite"
network_html = f'''<!DOCTYPE html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box}} body{{margin:0;background:#071827;color:#eaf6ff;font-family:Arial,sans-serif}}
.wrap{{border:1px solid #24506b;border-radius:18px;padding:8px;background:radial-gradient(ellipse at 50% 40%,#143650,#071827 75%)}}
svg{{width:100%;height:auto;display:block}} .zone{{fill:none;stroke:#9eb6c6;stroke-width:1.4;stroke-dasharray:3 5;opacity:.8}}
.zone-label{{fill:#b9d7e8;font-size:12px;font-weight:bold}} .device{{font-size:23px}} .small{{fill:#d8edf7;font-size:11px}}
.server{{fill:#123d5b;stroke:#54b9ed;stroke-width:2}} .servertext{{fill:white;font-size:12px;font-weight:bold}}
.link{{fill:none;stroke:#6b9ab8;stroke-width:1.6;stroke-dasharray:5 6;opacity:.8}}
.route{{fill:none;stroke:#2fe0ad;stroke-width:3;stroke-dasharray:8 5;filter:drop-shadow(0 0 5px #2fe0ad)}}
.packet{{fill:#35f0b8;stroke:#eafff8;stroke-width:1.5;filter:drop-shadow(0 0 7px #35f0b8)}}
.returnpacket{{fill:#ffb84d;stroke:#fff1d4;stroke-width:1.5;filter:drop-shadow(0 0 7px #ffb84d)}}
.pulse{{animation:pulse 1.6s infinite}} @keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.45}}}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;padding:4px 12px 10px;color:#cde4f2;font-size:12px}}
.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}}
</style></head><body><div class="wrap">
<svg viewBox="0 0 820 440" role="img" aria-label="Animated IoT edge cloud task movement network">
<ellipse class="zone" cx="180" cy="190" rx="165" ry="150"/><ellipse class="zone" cx="500" cy="245" rx="190" ry="155"/>
<text class="zone-label" x="42" y="55">MULTI-DEVICE IoT AREA</text><text class="zone-label" x="390" y="75">EDGE COMPUTING AREA</text>
<text class="device" x="55" y="120">📱</text><text class="device" x="105" y="185">🚗</text><text class="device" x="210" y="115">🎮</text>
<text class="device" x="65" y="300">🌾</text><text class="device" x="225" y="315">🚁</text><text class="device" x="260" y="205">📱</text>
<text class="small" x="40" y="140">Smart phone</text><text class="small" x="80" y="205">Connected car</text><text class="small" x="205" y="135">Online gaming</text>
<text class="small" x="48" y="325">Agriculture sensor</text><text class="small" x="210" y="340">UAV / drone</text>
<g transform="translate(150 225)"><path d="M0 30 L0 -15 M-13 30 L13 30 M-8 -5 Q0 -16 8 -5 M-13 -12 Q0 -31 13 -12" stroke="#b9e8ff" stroke-width="2" fill="none"/><rect x="-7" y="30" width="14" height="9" rx="2" fill="#69bfff"/><text class="small" x="-42" y="58">Base station</text></g>
<g transform="translate(110 255)"><circle r="27" fill="#153c58" stroke="#35e0b5" stroke-width="2" class="pulse"/><text x="-15" y="8" font-size="25">{source_icon}</text><text class="small" x="-48" y="43">{submission_label}</text><text x="-55" y="59" fill="#ffffff" font-size="10" font-weight="bold">{display_task_name}</text></g>
<path class="link" d="M150 225 L440 155 M150 225 L440 325 M150 225 L675 150"/>
<g transform="translate(440 155)"><rect class="server" x="-34" y="-27" width="68" height="54" rx="8"/><path d="M-22 -12 H22 M-22 0 H22 M-22 12 H22" stroke="#a9e4ff" stroke-width="4"/><text class="servertext" x="-46" y="48">EDGE SERVER 1</text><text class="small" x="-31" y="63">Queue / CPU</text></g>
<g transform="translate(440 325)"><rect class="server" x="-34" y="-27" width="68" height="54" rx="8"/><path d="M-22 -12 H22 M-22 0 H22 M-22 12 H22" stroke="#a9e4ff" stroke-width="4"/><text class="servertext" x="-46" y="48">EDGE SERVER 2</text><text class="small" x="-31" y="63">Queue / CPU</text></g>
<g transform="translate(350 105)"><rect class="server" x="-34" y="-25" width="68" height="50" rx="8"/><path d="M-22 -11 H22 M-22 0 H22 M-22 11 H22" stroke="#a9e4ff" stroke-width="4"/><text class="servertext" x="-46" y="44">EDGE SERVER 4</text><text class="small" x="-31" y="59">Queue / CPU</text></g>
<g transform="translate(590 350)"><rect class="server" x="-34" y="-25" width="68" height="50" rx="8"/><path d="M-22 -11 H22 M-22 0 H22 M-22 11 H22" stroke="#a9e4ff" stroke-width="4"/><text class="servertext" x="-46" y="44">EDGE SERVER 3</text><text class="small" x="-31" y="59">Queue / CPU</text></g>
<path class="link" d="M150 225 L350 105 M150 225 L590 350"/>
<path d="M610 62 C585 28 635 8 658 35 C680 7 725 25 720 54 C755 65 740 100 709 100 H632 C602 100 590 72 610 62Z" fill="#173f68" stroke="#8bbcff" stroke-width="2"/>
<text x="620" y="61" fill="white" font-size="13" font-weight="bold">AI LEARNING MODEL</text><text x="637" y="80" class="small">ML prediction</text>
<g transform="translate(675 175)"><rect class="server" x="-34" y="-23" width="68" height="46" rx="7"/><path d="M-22 -10 H22 M-22 0 H22 M-22 10 H22" stroke="#a9e4ff" stroke-width="4"/><text class="servertext" x="-40" y="43">CLOUD SERVER</text></g>
<path class="link" d="M675 100 L675 152 M475 155 Q565 120 640 88 M475 325 Q590 250 650 98"/>
<g transform="translate(150 355)"><rect x="-27" y="-18" width="54" height="36" rx="5" fill="#183c4d" stroke="#62d7b5" stroke-width="2"/><path d="M-17 -8 H17 M-17 0 H17 M-17 8 H17" stroke="#a7f5df" stroke-width="3"/><text class="servertext" x="-39" y="37">LOCAL EXECUTION</text></g>
<path id="chosenRoute" class="route" d="{active_path}"/>
<circle r="7" class="packet"><animateMotion dur="3.2s" begin="{packet_begin}" repeatCount="indefinite"><mpath href="#chosenRoute"/></animateMotion></circle>
<circle r="5.5" class="returnpacket"><animateMotion dur="3.2s" begin="{return_begin}" repeatCount="indefinite" keyPoints="1;0" keyTimes="0;1" calcMode="linear"><mpath href="#chosenRoute"/></animateMotion></circle>
<text x="290" y="405" fill="#35f0b8" font-size="12" font-weight="bold" class="pulse">{display_task_name} → {decision.upper()}</text>
<text x="290" y="423" fill="#ffca78" font-size="11">← AI decision / result return</text>
</svg><div class="legend"><span><i class="dot" style="background:#35f0b8"></i>New task movement</span><span><i class="dot" style="background:#ffb84d"></i>Decision / result return</span><span>Dashed links: wireless connections</span><span>Highlighted path: current AI recommendation</span></div></div></body></html>'''
components.html(network_html, height=490, scrolling=False)
st.caption(f"{'Submitted task' if st.session_state.task_submitted else 'Current task preview'}: **{task_name}** ({app} · {task_type} · {task_priority}) moves from its source device toward **{decision}**. The return packet represents the decision/result path. This is an illustrative animation, not a live network transmission.")

st.subheader("3. AI regression model selection and intelligent execution decision")
st.caption(f"Pre-trained models loaded from the supplied multi-node dataset: {train_count:,} training rows and {test_count:,} held-out test rows; six candidate locations per task. Train/test partition is provided in the CSV and grouped by task ID.")
a,b,c,d=st.columns(4)
a.metric("Best regression model",best_model_name)
b.metric("Selected execution node",decision)
c.metric("Predicted completion time",f"{float(chosen['Predicted completion time (ms)']):,.2f} ms")
d.metric("Deadline assessment",decision_status)
if decision_status=="DEADLINE-FEASIBLE":
    st.success(f"AI recommendation: Send task **{task_name}** to **{decision}**. The predicted completion time is within the {deadline:.2f} ms deadline.")
else:
    st.warning(f"No available node is predicted to meet the {deadline:.2f} ms deadline. {decision} has the lowest predicted completion time ({float(chosen['Predicted completion time (ms)']):.2f} ms), but deadline feasibility is not achieved. Consider reducing task size, increasing compute capacity or relaxing the deadline.")
st.markdown(f"**Decision explanation:** {best_model_name} achieved the lowest RMSE on the held-out synthetic benchmark. That model predicts completion time for each candidate node. The decision layer first filters nodes that satisfy the deadline, then chooses the feasible node with the lowest predicted completion time. If none meets the deadline, it reports the least-delay option without claiming it is deadline-feasible.")

# Main graphs and tabs
T1,T2,T3,T4,T5=st.tabs(["📊 Task predictions","🧠 ML algorithm comparison","⏱️ Delay components","📈 Model validation","📋 Decision details"])
with T1:
    st.markdown("**Predicted end-to-end completion time for the NEW task at every available node**")
    fig=px.bar(results,x="Execution node",y="Predicted completion time (ms)",color="Deadline status",text_auto=".2f",title=f"{best_model_name} – predicted task completion time")
    fig.add_hline(y=deadline,line_dash="dash",annotation_text="Task deadline",annotation_position="top left")
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(results.round(3),use_container_width=True,hide_index=True)
    # Predictions by every model for the candidate locations
    matrix=pd.DataFrame({"Algorithm":list(all_model_predictions.keys()),**{loc:[float(all_model_predictions[m][i]) for m in all_model_predictions] for i,loc in enumerate(LOCATIONS)}})
    st.markdown("**Cross-check: how each regression algorithm predicts the same task at each node**")
    st.dataframe(matrix.round(2),use_container_width=True,hide_index=True)
    long=matrix.melt(id_vars="Algorithm",var_name="Execution node",value_name="Predicted time (ms)")
    st.plotly_chart(px.line(long,x="Algorithm",y="Predicted time (ms)",color="Execution node",markers=True,title="Live task prediction across all ML algorithms"),use_container_width=True)
with T2:
    st.markdown("**Held-out test-set comparison – lower MAE/RMSE and higher R² are preferred.**")
    st.dataframe(scores.round(4),use_container_width=True,hide_index=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(px.bar(scores.sort_values("MAE (ms)"),x="Algorithm",y="MAE (ms)",title="Mean Absolute Error (lower is better)",text_auto=".2f"),use_container_width=True)
    with c2: st.plotly_chart(px.bar(scores.sort_values("RMSE (ms)"),x="Algorithm",y="RMSE (ms)",title="Root Mean Squared Error (lower is better)",text_auto=".2f"),use_container_width=True)
    st.plotly_chart(px.bar(scores.sort_values("R²",ascending=False),x="Algorithm",y="R²",title="R² score (higher is better)",text_auto=".4f"),use_container_width=True)
    st.plotly_chart(px.scatter(scores,x="Training (s)",y="Inference (ms/sample)",size="R²",color="Algorithm",hover_name="Algorithm",title="Training cost versus prediction latency"),use_container_width=True)
    st.info(f"Best model by the pre-declared selection rule (lowest held-out RMSE): **{best_model_name}**. Model selection is based on the benchmark test split; for a final paper, use a separate validation set for model selection and keep a final untouched test set.")
with T3:
    # Explain components using simplified physical approximation at the chosen node.
    r=resources[decision]
    upload=task*8/r["bw"]+r["rtt"]/2
    down=result*8/r["bw"]+r["rtt"]/2
    processing=cycles/max(r["cpu"]*(1-r["util"]/100),.05)*1000
    queue=r["q"]*(processing/max(1,1+r["q"]*.08))*.18
    components=pd.DataFrame({"Delay component":["Upload communication","Queue waiting (approx.)","Processing (approx.)","Result return communication"],"Estimated delay (ms)":[upload,queue,processing,down]})
    st.plotly_chart(px.bar(components,x="Delay component",y="Estimated delay (ms)",color="Delay component",text_auto=".2f",title=f"Estimated delay components – {decision}"),use_container_width=True)
    st.plotly_chart(px.pie(components,names="Delay component",values="Estimated delay (ms)",title="Share of analytical component estimate",hole=.4),use_container_width=True)
    st.dataframe(components.round(3),use_container_width=True,hide_index=True)
    st.caption("These components are a simplified analytical breakdown for interpretation. The ML model predicts total completion time; component values are not separate ML predictions and may not sum exactly to the model output.")
with T4:
    chosen_model=st.selectbox("Inspect model diagnostics",scores["Algorithm"].tolist(),index=scores["Algorithm"].tolist().index(best_model_name))
    yhat=test_predictions[chosen_model]
    diag=pd.DataFrame({"Actual completion time (ms)":ytest.values,"Predicted completion time (ms)":yhat})
    diag["Residual (ms)"]=diag["Actual completion time (ms)"]-diag["Predicted completion time (ms)"]
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(px.scatter(diag,x="Actual completion time (ms)",y="Predicted completion time (ms)",opacity=.55,title=f"Actual vs predicted – {chosen_model}"),use_container_width=True)
    with c2: st.plotly_chart(px.histogram(diag,x="Residual (ms)",nbins=40,title="Prediction residual distribution"),use_container_width=True)
    st.metric("Selected model test MAE",f"{mean_absolute_error(ytest,yhat):.2f} ms")
    st.metric("Selected model test RMSE",f"{mean_squared_error(ytest,yhat)**.5:.2f} ms")
    st.metric("Selected model test R²",f"{r2_score(ytest,yhat):.4f}")
    v= Xtest.copy();v["actual"]=ytest.values;v["predicted"]=yhat
    v["workload"]=pd.qcut(v["queue_jobs"].rank(method="first"),3,labels=["Light queue","Moderate queue","Heavy queue"])
    groups=[]
    for label,g in v.groupby("workload",observed=False):
        groups.append({"Queue workload":str(label),"Samples":len(g),"MAE (ms)":mean_absolute_error(g.actual,g.predicted),"RMSE (ms)":mean_squared_error(g.actual,g.predicted)**.5,"R²":r2_score(g.actual,g.predicted)})
    workload=pd.DataFrame(groups)
    st.dataframe(workload.round(3),use_container_width=True,hide_index=True)
    st.plotly_chart(px.bar(workload,x="Queue workload",y="RMSE (ms)",color="Queue workload",title="Prediction error across queue-load groups"),use_container_width=True)
with T5:
    st.markdown("#### Entered task specification")
    st.write({"Task ID":task_name,"Application":app,"Input size (KB)":task,"CPU cycles (billion)":cycles,"Output size (KB)":result,"Deadline (ms)":deadline})
    st.markdown("#### AI decision candidates")
    st.dataframe(results.sort_values("Predicted completion time (ms)").round(3),use_container_width=True,hide_index=True)
    st.markdown("#### Selected model benchmark metrics")
    st.dataframe(scores[scores["Algorithm"]==best_model_name].round(4),use_container_width=True,hide_index=True)
    st.markdown("#### Suggested decision sequence")
    st.markdown("1. Read new IoT task requirements.\n2. Form feature vector using task size, CPU demand, output size, bandwidth, RTT, queue length and CPU availability.\n3. Apply the best regression model selected from benchmark comparison.\n4. Predict end-to-end completion time at Local Device, Edge Servers 1–4 and Cloud Server.\n5. Check predicted time against the task deadline.\n6. Select the fastest deadline-feasible node; if none is feasible, report that condition and the least-delay candidate.")

with st.expander("Research methodology, assumptions and limitations"):
    st.markdown("""
**Training source:** objective1_iot_multinode_training_dataset.csv (72,000 rows; 12,000 tasks × six candidate nodes). The previous in-app random data generator is no longer used for model training.

**Train/test protocol:** uses the CSV-provided task-grouped dataset_split column; no new random row split is performed.

**Target variable:** numerical end-to-end task completion time in milliseconds.

**Target concept:** upload communication + queue waiting + processing + result-return communication.

**Input features:** task input size, required CPU cycles, result size, bandwidth, round-trip delay, queued tasks, CPU capacity, CPU utilization and candidate execution location.

**Regression algorithms:** Linear Regression, Ridge Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, Histogram Gradient Boosting, Voting Ensemble and Stacking Ensemble.

**Model metrics:** MAE, RMSE, R², training time and per-sample inference time.

**Decision logic:** choose the regression model with lowest RMSE on the held-out synthetic benchmark; predict time for Local Device, Edge Servers 1–4 and Cloud Server; filter by deadline; choose the lowest-time feasible node. This is an Objective 1 demonstration and a simple decision layer, not a complete Objective 2 scheduler.

**Important research limitations:**
- This prototype trains on synthetic data generated from a simplified system model. Results are demonstration results, not measured hardware evidence.
- Random row split is useful for prototyping but can produce optimistic estimates. Use time-based, device-based or scenario-held-out validation for final experiments.
- Queue delay is approximate. Replace it with independently measured queue traces or a validated queueing/simulation model.
- Manual inputs are not automatic ESP32, router, edge-server or cloud telemetry.
- The app recommends a destination; it does not actually transmit, execute or reserve a task on a remote server.
- For publication, build an independently generated dataset using NS-3/SimPy and validate against real ESP32/Wi-Fi/laptop/server measurements. Keep model selection and final test data separate.
""")
st.caption("Objective 1 live demonstration prototype · Trained on supplied synthetic multi-node dataset · Manual task/resource inputs · No real task offloading is performed")
