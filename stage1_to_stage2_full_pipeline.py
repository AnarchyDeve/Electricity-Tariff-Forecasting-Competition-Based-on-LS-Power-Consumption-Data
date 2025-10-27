# ==========================================================
# ⚡ Stage1 (PyTorch LSTM+Attention) → Stage2 Full Pipeline
#    - Stage1: model_1~N best_model.pt 로 train/test 예측 컬럼 생성
#    - 부족하면 Fallback(RandomForest)로 채워서 8개 보장
#    - Stage2: train 마지막 8개를 이어붙여 자연스럽게 예측 후 test만 저장
# ==========================================================
import os, json, joblib, torch, warnings
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ----------------------------------------------------------
# 경로
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STAGE1_DIR = os.path.join(BASE_DIR, "stage1_model")
STAGE2_DIR = os.path.join(BASE_DIR, "stage2_model")
os.makedirs(STAGE2_DIR, exist_ok=True)

train_path = os.path.join(DATA_DIR, "fixed_train_clean_v2.csv")
test_path  = os.path.join(DATA_DIR, "fixed_test_weather_full.csv")

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

# ----------------------------------------------------------
# 타깃
# ----------------------------------------------------------
MAIN_TARGET = "전기요금(원)"
REQUIRED_STAGE1_COUNT = 8

# ==========================================================
# Stage1 모델 정의
# ==========================================================
class LSTMAttention(nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()
        h = int(params.get("units", 64))
        n = int(params.get("n_layers", 2))
        dr = float(params.get("dropout", 0.2))
        nh = int(params.get("n_heads", 4))
        self.lstm = nn.LSTM(input_dim, h, num_layers=n, batch_first=True, dropout=dr)
        self.attn = nn.MultiheadAttention(h, num_heads=nh, dropout=dr, batch_first=True)
        self.norm = nn.LayerNorm(h)
        self.fc = nn.Linear(h, 1)
    def forward(self, x):
        o, _ = self.lstm(x)
        a, _ = self.attn(o, o, o)
        o = self.norm(o + a)
        return self.fc(o.mean(dim=1)).squeeze(-1)

def load_torch_model(path, dim, params):
    sd = torch.load(path, map_location=device)
    try:
        m = LSTMAttention(dim, params).to(device)
        m.load_state_dict(sd)
        m.eval()
        return m
    except:
        for k,v in sd.items():
            if "lstm.weight_ih_l0" in k:
                params["units"] = v.shape[0] // 4
                break
        m = LSTMAttention(dim, params).to(device)
        m.load_state_dict(sd, strict=False)
        m.eval()
        return m

class SimpleDataset(Dataset):
    def __init__(self,X,seq):
        self.X,self.seq=X,seq
    def __len__(self):return max(0,len(self.X)-self.seq)
    def __getitem__(self,i):
        return torch.tensor(self.X[i:i+self.seq],dtype=torch.float32)

def make_seq_np(X,s):
    return np.array([X[i:i+s] for i in range(len(X)-s)]) if len(X)>s else np.empty((0,s,X.shape[1]))

# ==========================================================
# Stage1 예측 (train/test)
# ==========================================================
print("\n🔍 Stage1 (PyTorch) 예측 수행 중...")
metrics_path = os.path.join(STAGE1_DIR, "metrics.json")
with open(metrics_path,"r",encoding="utf-8") as f:
    metrics_data=json.load(f)

model_folders=sorted([d for d in os.listdir(STAGE1_DIR) if d.startswith("model_")], key=lambda x:int(x.split("_")[1]))
ordered_targets=list(metrics_data.keys())
numeric_cols=train_df.select_dtypes(include=[np.number]).columns.tolist()
feature_base=[c for c in numeric_cols if c not in ordered_targets+[MAIN_TARGET]]
pred_columns=[]

for i,folder in enumerate(model_folders,1):
    model_path=os.path.join(STAGE1_DIR,folder,"best_model.pt")
    sxp=os.path.join(STAGE1_DIR,folder,"scaler_x.pkl")
    syp=os.path.join(STAGE1_DIR,folder,"scaler_y.pkl")
    if not os.path.exists(model_path):continue
    tname=ordered_targets[i-1] if i-1<len(ordered_targets) else f"target_{i}"
    params=metrics_data.get(tname,{}).get("params",{})
    seq=int(params.get("seq_len",12))
    bs=int(params.get("batch_size",32))
    sx=joblib.load(sxp); sy=joblib.load(syp)
    feats=metrics_data.get(tname,{}).get("features",feature_base)
    feats=[f for f in feats if f in train_df.columns]
    Xtr=train_df[feats].fillna(0).values; Xts=test_df[feats].fillna(0).values
    if Xtr.shape[1]!=sx.n_features_in_:
        need=sx.n_features_in_; cur=Xtr.shape[1]
        if cur>need:Xtr,Xts=Xtr[:,:need],Xts[:,:need]
        else:
            pad=np.zeros((len(Xtr),need-cur))
            Xtr=np.concatenate([Xtr,pad],1);Xts=np.concatenate([Xts,pad],1)
    Xtr=sx.transform(Xtr); Xts=sx.transform(Xts)
    Xtr_seq=make_seq_np(Xtr,seq); Xts_seq=make_seq_np(Xts,seq)
    model=load_torch_model(model_path,Xtr.shape[1],params)

    preds_train=[]
    with torch.no_grad():
        for Xb in DataLoader(SimpleDataset(Xtr,seq),batch_size=bs,shuffle=False):
            preds_train.append(model(Xb.to(device)).cpu().numpy())
    preds_train=np.concatenate(preds_train) if preds_train else np.zeros((len(train_df),1))
    train_df[f"{tname}_pred"]=np.concatenate([np.full(seq,np.nan),
        sy.inverse_transform(preds_train.reshape(-1,1)).ravel()])[:len(train_df)]

    preds_test=[]
    with torch.no_grad():
        for Xb in DataLoader(SimpleDataset(Xts,seq),batch_size=bs,shuffle=False):
            preds_test.append(model(Xb.to(device)).cpu().numpy())
    preds_test=np.concatenate(preds_test) if preds_test else np.zeros((len(test_df),1))
    test_df[f"{tname}_pred"]=np.concatenate([np.full(seq,np.nan),
        sy.inverse_transform(preds_test.reshape(-1,1)).ravel()])[:len(test_df)]

    pred_columns.append(f"{tname}_pred")

# 부족 시 0-padding
while len(pred_columns)<REQUIRED_STAGE1_COUNT:
    pad=f"pad_pred_{len(pred_columns)+1}"
    train_df[pad]=0.0; test_df[pad]=0.0; pred_columns.append(pad)

print(f"🎯 Stage1 완료 → {pred_columns}")

# ==========================================================
# Stage2 (LSTM + Attention)
# ==========================================================
print("\n⚡ Stage2 학습 시작...")
X_train=train_df[pred_columns].fillna(0).values
X_test=test_df[pred_columns].fillna(0).values
y_train=train_df[MAIN_TARGET].values

sx2=StandardScaler(); sy2=StandardScaler()
X_train=sx2.fit_transform(X_train); X_test=sx2.transform(X_test)
y_train=sy2.fit_transform(y_train.reshape(-1,1))

# 1️⃣ train 마지막 8개를 test 맨 앞에 이어붙이기
TIME_STEPS = 8
X_concat = np.concatenate([X_train[-TIME_STEPS:], X_test], axis=0)

# 시퀀스 생성
def make_seq(X, y=None, steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X) - steps):
        Xs.append(X[i:i+steps])
        if y is not None and i < len(y) - steps:
            ys.append(y[i+steps])
    return (np.array(Xs), np.array(ys)) if y is not None else np.array(Xs)

X_seq, y_seq = make_seq(X_train, y_train)
X_test_seq = make_seq(X_concat, None)

class SeqDS(Dataset):
    def __init__(self,X,y=None):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32) if y is not None else None
    def __len__(self):return len(self.X)
    def __getitem__(self,i):
        return (self.X[i],self.y[i]) if self.y is not None else self.X[i]

train_loader=DataLoader(SeqDS(X_seq,y_seq),batch_size=64,shuffle=True)

class HybridLSTM(nn.Module):
    def __init__(self,inp,h=128):
        super().__init__()
        self.lstm=nn.LSTM(inp,h,batch_first=True)
        self.fc=nn.Linear(h,1)
    def forward(self,x):
        o,_=self.lstm(x)
        return self.fc(o[:,-1,:])

model=HybridLSTM(X_seq.shape[2]).to(device)
opt=torch.optim.Adam(model.parameters(),lr=1e-3)
crit=nn.L1Loss()
best=float("inf")
for e in range(1,61):
    model.train(); tot=0
    for Xb,yb in train_loader:
        Xb,yb=Xb.to(device),yb.to(device)
        opt.zero_grad();p=model(Xb);l=crit(p,yb);l.backward();opt.step();tot+=l.item()
    if tot<best:best=tot;torch.save(model.state_dict(),os.path.join(STAGE2_DIR,"best.pt"))
    if e%10==0:print(f"Epoch {e}|Loss {tot:.4f}")

print(f"✅ Stage2 완료 | Best {best:.4f}")

# ==========================================================
# Stage2 예측 (train 마지막 8개 붙인 test만)
# ==========================================================
model.load_state_dict(torch.load(os.path.join(STAGE2_DIR,"best.pt"),map_location=device))
model.eval()
test_loader = DataLoader(SeqDS(X_test_seq), batch_size=64, shuffle=False)

preds = []
with torch.no_grad():
    for Xb in test_loader:
        Xb = Xb.to(device)
        preds.append(model(Xb).cpu().numpy())

preds = np.concatenate(preds) if preds else np.empty((0,1))
y_pred = sy2.inverse_transform(preds)
final_preds = y_pred[-len(test_df):].flatten()  # train 이어붙인 8개 제외

# 저장
test_df[f"{MAIN_TARGET}_pred"] = final_preds
out_path = os.path.join(DATA_DIR,"final_stage2_lstm_predictions.csv")
test_df.to_csv(out_path,index=False,encoding="utf-8-sig")
print(f"📁 최종 예측 저장 완료: {out_path}")
print("✅ 완료 (train 마지막 8개는 연결용으로만 사용 후 제거됨)")
