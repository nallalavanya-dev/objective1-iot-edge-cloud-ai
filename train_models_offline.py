"""Run locally once to train and package all nine regression models for Streamlit Cloud."""
from pathlib import Path
import time, joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, VotingRegressor, StackingRegressor

HERE=Path(__file__).resolve().parent
DATA=HERE/'objective1_iot_multinode_training_dataset.csv'
OUT=HERE/'trained_models'
OUT.mkdir(exist_ok=True)
NUM=["task_input_kb","cpu_cycles_giga","return_data_kb","deadline_ms","bandwidth_mbps","rtt_ms","queue_jobs","cpu_capacity_ghz","cpu_utilization_pct","effective_cpu_ghz"]
CAT=["iot_application","task_type","task_priority","location","node_type"]
FEATURES=NUM+CAT

def make_estimators():
    estimators=[
      ("Linear Regression",LinearRegression()),
      ("Ridge Regression",Ridge(alpha=1.0)),
      ("Decision Tree",DecisionTreeRegressor(max_depth=18,min_samples_leaf=3,random_state=42)),
      ("Random Forest",RandomForestRegressor(n_estimators=8,min_samples_leaf=2,n_jobs=1,random_state=42)),
      ("Extra Trees",ExtraTreesRegressor(n_estimators=6,min_samples_leaf=2,n_jobs=1,random_state=42)),
      ("Gradient Boosting",GradientBoostingRegressor(n_estimators=25,max_depth=3,learning_rate=.06,random_state=42)),
      ("Hist Gradient Boosting",HistGradientBoostingRegressor(max_iter=45,learning_rate=.07,max_leaf_nodes=24,random_state=42)),
    ]
    voting=VotingRegressor([("rf",RandomForestRegressor(n_estimators=4,min_samples_leaf=2,n_jobs=1,random_state=42)),("et",ExtraTreesRegressor(n_estimators=4,min_samples_leaf=2,n_jobs=1,random_state=43)),("gb",GradientBoostingRegressor(n_estimators=8,max_depth=3,learning_rate=.06,random_state=42))])
    stacking=StackingRegressor([("rf",RandomForestRegressor(n_estimators=4,min_samples_leaf=2,n_jobs=1,random_state=42)),("et",ExtraTreesRegressor(n_estimators=4,min_samples_leaf=2,n_jobs=1,random_state=43)),("gb",GradientBoostingRegressor(n_estimators=8,max_depth=3,learning_rate=.06,random_state=42))],final_estimator=Ridge(alpha=1.0),cv=2,n_jobs=1)
    return estimators+[("Voting Ensemble",voting),("Stacking Ensemble",stacking)]

def main():
    df=pd.read_csv(DATA)
    tr=df[df.dataset_split.astype(str).str.lower().eq('train')].copy()
    te=df[df.dataset_split.astype(str).str.lower().eq('test')].copy()
    if tr.empty or te.empty: raise ValueError('Dataset needs train/test values in dataset_split')
    Xtr,ytr=tr[FEATURES],tr.completion_time_ms
    Xte,yte=te[FEATURES],te.completion_time_ms
    models,preds,rows={}, {}, []
    for name,est in make_estimators():
        if name == "Stacking Ensemble":
            # Smaller fixed sample keeps the cross-validated stack practical on modest hardware.
            sample = tr.sample(n=min(12000, len(tr)), random_state=42)
            fit_X, fit_y = sample[FEATURES], sample.completion_time_ms
        else:
            fit_X, fit_y = Xtr, ytr
        prep=ColumnTransformer([
          ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),
          ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),CAT)
        ],sparse_threshold=0)
        pipe=Pipeline([('prep',prep),('regressor',est)])
        t=time.perf_counter(); pipe.fit(fit_X,fit_y); train_s=time.perf_counter()-t
        t=time.perf_counter(); yp=pipe.predict(Xte); infer=(time.perf_counter()-t)*1000/len(Xte)
        models[name]=pipe; preds[name]=np.asarray(yp,dtype=np.float32)
        rows.append({'Algorithm':name,'MAE (ms)':mean_absolute_error(yte,yp),'RMSE (ms)':mean_squared_error(yte,yp)**.5,'R²':r2_score(yte,yp),'Inference (ms/sample)':infer,'Training (s)':train_s,'Training rows':len(fit_X)})
        print(f'{name}: trained in {train_s:.1f}s; RMSE={rows[-1]["RMSE (ms)"]:.3f}')
    scores=pd.DataFrame(rows).sort_values(['RMSE (ms)','MAE (ms)']).reset_index(drop=True)
    bundle={'models':models,'scores':scores,'Xtest':Xte.reset_index(drop=True),'ytest':yte.reset_index(drop=True),'test_predictions':preds,'test_rows':te.reset_index(drop=True),'train_count':len(tr),'test_count':len(te),'best_model_name':str(scores.iloc[0]['Algorithm']),'feature_columns':FEATURES}
    joblib.dump(bundle,OUT/'regression_models_bundle.joblib',compress=3)
    scores.to_csv(OUT/'regression_model_comparison.csv',index=False)
    print('\nBest model:',bundle['best_model_name'])
    print('Saved:',OUT/'regression_models_bundle.joblib')
    print('Bundle size MB:',round((OUT/'regression_models_bundle.joblib').stat().st_size/1024/1024,2))
if __name__=='__main__': main()
