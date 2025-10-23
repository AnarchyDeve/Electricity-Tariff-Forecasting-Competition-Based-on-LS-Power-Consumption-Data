# ================================================================
# ⚡ Electricity Forecast Full Pipeline (시계열 + Optuna + TensorBoard)
# ================================================================
import os, itertools, datetime, json, math, random, threading, webbrowser
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Dropout, BatchNormalization, Bidirectional
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import optuna
from tensorboard import program

# ===================================================
# 🔥 TensorBoard 자동 실행
# ===================================================
def launch_tensorboard(logdir="logs", port=6006):
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", logdir, "--port", str(port)])
    url = tb.launch()
    print(f"\n🚀 TensorBoard 실행 중: {url}")
    webbrowser.open(url)
    return tb

tb_thread = threading.Thread(target=launch_tensorboard, kwargs={"logdir": "logs", "port": 6006})
tb_thread.daemon = True
tb_thread.start()

# ===================================================
# 0️⃣ 경로 설정
# ===================================================
train_path = "./data/fixed_train_clean.csv"
test_path  = "./data/fixed_test_weather_full.csv"

TARGETS_STEP1 = [
    "전력사용량(kWh)",
    "지상무효전력량(kVarh)",
    "진상무효전력량(kVarh)",
    "지상역률(%)",
    "진상역률(%)",
]
TARGET_STEP2 = "전기요금(원)"

TIME_FEATURES = ["sin_day","cos_day","day_of_week","is_weekend","is_holiday"]
WEATHER_FEATURES = ["기온(°C)","습도(%)","풍속(m/s)","총강수량(mm)"]
ALL_FEATURES = TIME_FEATURES + WEATHER_FEATURES

N_SPLITS = 3
N_TRIALS_STEP1 = 15
N_TRIALS_STEP2 = 15
ENSEMBLE_SEEDS = [7, 13, 29]

os.makedirs("logs", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# ===================================================
# 1️⃣ 유틸 함수
# ===================================================
def set_global_seed(seed: int):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.asarray(Xs), np.asarray(ys)

def attention_block(x):
    h = Dense(x.shape[-1], activation="tanh")(x)
    e = Dense(1)(h)
    a = tf.nn.softmax(e, axis=1)
    context = tf.reduce_sum(a * x, axis=1)
    return context
def build_model(input_shape, params, model_type="LSTM", use_attention=True, output_dim=1):
    inp = Input(shape=input_shape)
    RNN = LSTM if model_type in ["LSTM", "BiLSTM"] else GRU

    # 첫 번째 은닉층
    x = RNN(params["units1"], return_sequences=True,
            dropout=params["dropout"],
            recurrent_dropout=params.get("recurrent_dropout", 0.0))(inp)
    if params.get("use_batchnorm", False):
        x = BatchNormalization()(x)

    # 두 번째 은닉층
    x = RNN(params["units2"], return_sequences=True,
            dropout=params["dropout"],
            recurrent_dropout=params.get("recurrent_dropout", 0.0))(x)
    if params.get("use_batchnorm", False):
        x = BatchNormalization()(x)

    # 🔹 세 번째 은닉층 추가
    x = RNN(params.get("units3", 64), return_sequences=True,
            dropout=params["dropout"],
            recurrent_dropout=params.get("recurrent_dropout", 0.0))(x)
    if params.get("use_batchnorm", False):
        x = BatchNormalization()(x)

    # Attention or Global Pooling
    if use_attention:
        x = attention_block(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = Dropout(params["dropout"])(x)
    out = Dense(output_dim)(x)

    opt = tf.keras.optimizers.Adam(learning_rate=params["lr"])
    model = Model(inp, out)
    model.compile(optimizer=opt, loss="mae")
    return model

def tensorboard_dir(step_name, trial_no=None, fold=None, tag="fit"):
    parts = [step_name]
    if trial_no is not None: parts.append(f"trial_{trial_no}")
    if fold is not None: parts.append(f"fold_{fold}")
    parts.append(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    return os.path.join("logs", "_".join(parts), tag)

# ===================================================
# 2️⃣ 데이터 전처리
# ===================================================
def preprocess_dataframe(df: pd.DataFrame):
    df = df.copy()

    # 시계열 정렬
    datetime_col = None
    for col in ["측정일시_x", "측정일시_y"]:
        if col in df.columns:
            datetime_col = col
            break
    if datetime_col:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
        df = df.sort_values(by=datetime_col).reset_index(drop=True)

    # 불필요 컬럼 제거
    drop_cols = ["id", "측정일시_x", "측정일시_y"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # season → 원핫
    if "season" in df.columns:
        season_map = {"spring": 0, "summer": 1, "autumn": 2, "fall": 2, "winter": 3}
        df["season"] = df["season"].map(season_map).fillna(3).astype(int)
        df = pd.get_dummies(df, columns=["season"], prefix="season")

    # 작업유형 → 원핫
    if "작업유형" in df.columns:
        df = pd.get_dummies(df, columns=["작업유형"], prefix="작업유형")

    # 날씨코드 → Label Encoding
    if "날씨코드" in df.columns:
        le = LabelEncoder()
        df["날씨코드"] = le.fit_transform(df["날씨코드"].astype(str))

    # 나머지 object 컬럼 자동 인코딩
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    return df

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
train = preprocess_dataframe(train)
test = preprocess_dataframe(test)

# 스케일링
X_all = train.select_dtypes(include=[np.number]).drop(columns=TARGETS_STEP1 + [TARGET_STEP2], errors="ignore")
scaler_X = StandardScaler()
X_scaled_full = scaler_X.fit_transform(X_all.values)

# ===================================================
# 3️⃣ Optuna 목적 함수
# ===================================================
def objective(trial, X, y, step_name, feature_names):
    k = trial.suggest_int("n_features", max(3, len(feature_names)//2), len(feature_names))
    chosen = trial.suggest_categorical("features", random.sample(list(itertools.combinations(feature_names, k)), min(120, math.comb(len(feature_names), k))))
    feat_idx = [feature_names.index(f) for f in chosen]
    X_sel = X[:, feat_idx]

    params = {
        "model_type": trial.suggest_categorical("model_type", ["LSTM","GRU","BiLSTM","BiGRU"]),
        "units1": trial.suggest_int("units1", 32, 128, step=16),
        "units2": trial.suggest_int("units2", 16, 96, step=16),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
        "recurrent_dropout": trial.suggest_float("recurrent_dropout", 0.0, 0.3, step=0.1),
        "use_batchnorm": trial.suggest_categorical("use_batchnorm", [True, False]),
        "seq_len": trial.suggest_int("seq_len", 12, 96, step=12),
        "optimizer": trial.suggest_categorical("optimizer", ["adam","nadam","adamw"]),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16,32,64]),
        "epochs": trial.suggest_int("epochs", 30, 100, step=10),
        "lr_scheduler": trial.suggest_categorical("lr_scheduler", ["plateau","cosine"]),
        "cosine_steps": trial.suggest_int("cosine_steps", 300, 2000, step=100),
        "use_attention": trial.suggest_categorical("use_attention", [True, False]),
    }

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_mae = []

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_sel)):
        X_tr, y_tr = X_sel[tr_idx], y[tr_idx]
        X_va, y_va = X_sel[va_idx], y[va_idx]
        X_tr_seq, y_tr_seq = create_sequences(X_tr, y_tr, params["seq_len"])
        X_va_seq, y_va_seq = create_sequences(X_va, y_va, params["seq_len"])

        model = build_model((params["seq_len"], X_sel.shape[1]), params, params["model_type"], params["use_attention"])
        cbs = [TensorBoard(log_dir=tensorboard_dir(step_name, trial.number, fold)), EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]
        if params["lr_scheduler"] == "plateau":
            cbs.append(ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5))

        model.fit(X_tr_seq, y_tr_seq, validation_data=(X_va_seq, y_va_seq),
                  epochs=params["epochs"], batch_size=params["batch_size"], verbose=0, callbacks=cbs)

        y_hat = model.predict(X_va_seq, verbose=0)
        fold_mae.append(mean_absolute_error(y_va_seq, y_hat))

    trial.set_user_attr("params", params)
    trial.set_user_attr("features", chosen)
    return float(np.mean(fold_mae))
