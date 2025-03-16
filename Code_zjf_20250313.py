################################################ Data Processing and Feature Engineering ################################################
import pandas as pd
import numpy as np

# 1. 读取数据
# 基于分类油层预测的年产油和含水率已经计算完毕，分别存放在'ModelPred_OilProduction'和'ModelPred_WaterCut'变量中。
df = pd.read_csv('your_data.csv')

# 处理可能的时间序列排序（如果有年份或日期列，按时间先后排序）
df = df.sort_values(by='Year')

# 2. 缺失值处理
# 计算每列缺失值数量
missing_counts = df.isnull().sum()
print("各列缺失值数量:\n", missing_counts)
# 简单缺失值填充：数值型用中位数填充，分类型用众数填充
for col in df.columns:
    if df[col].dtype != object:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else None, inplace=True)

# 3. 异常值检测与处理
# 使用3倍标准差原则将异常值进行截断（亦可选择删除异常值或其他方法）
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if col == 'Year':  # 年份等索引字段不处理
        continue
    col_mean = df[col].mean()
    col_std = df[col].std()
    cut_off = 3 * col_std
    lower, upper = col_mean - cut_off, col_mean + cut_off
    # 将超过上下界的值截断为边界值
    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

# 4. 选择特征列和目标列
target_col = 'OilProduction'  # 此处为目标变量名，在以产油量为目标时，换成'WaterCut'即可
feature_cols = [col for col in df.columns if col != target_col]
# 从特征列表中移除时间列
feature_cols.remove('Year')
# 从特征列表中移除另外一个目标变量，当目标变量为含水率时，这里移除'OilProduction'
feature_cols.remove('WaterCut')


# 5. 特征重要性分析和特征选择（使用随机森林特征重要性 + RFE）
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

X_all = df[feature_cols]
y_all = df[target_col]

# 使用随机森林评估特征重要性
rf_for_fs = RandomForestRegressor(n_estimators=100, random_state=42)
rf_for_fs.fit(X_all, y_all)
importances = pd.Series(rf_for_fs.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("特征重要性:\n", importances)

# 应用递归特征消除(RFE)选择重要特征（此处选择前9个重要特征）
n_features_to_select = min(9, X_all.shape[1])  # 如果总特征少于9，则取全部
selector = RFE(rf_for_fs, n_features_to_select=n_features_to_select)
selector.fit(X_all, y_all)
selected_features = X_all.columns[selector.support_].tolist()
print(f"选择的特征集合: {selected_features}")

# 用选择后的特征子集更新特征集
feature_cols = selected_features

# 6. 划分训练集和测试集（按照时间顺序，保证测试集是后期样本）
train_size = int(0.8 * len(df))  # 80%作为训练
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]
X_train_full = train_df[feature_cols]
y_train_full = train_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]

# 从训练集中留出一部分作为验证集（如训练集的20%用于验证）
val_size = int(0.2 * len(train_df))
val_df = train_df.iloc[-val_size:]
train_df2 = train_df.iloc[:-val_size]
X_train = train_df2[feature_cols]
y_train = train_df2[target_col]
X_val = val_df[feature_cols]
y_val = val_df[target_col]

# 7. 特征归一化（Min-Max归一化）
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# 注意：为防止信息泄露，使用训练集数据拟合Scaler
X_train_full_scaled = X_train_full.copy()
X_val_scaled = X_val.copy()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_full_scaled[feature_cols] = scaler.fit_transform(X_train_full[feature_cols])
X_train_scaled[feature_cols] = scaler.transform(X_train[feature_cols])
X_val_scaled[feature_cols] = scaler.transform(X_val[feature_cols])
X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])


# 8. 区分单独用机器学习的特征集，以及将分类油层动态指标预测方法结果作为特征的特征集
# 将上述整理结果作为包含分类法预测结果的特征集：含水率作为目标变量时，还要删除'ModelPred_OilProduction'变量；产油量作为目标变量时，还要删除'ModelPred_WaterCut'变量。因为涉及两次计算，这里不写了。
X_train_full_ext = X_train_full_scaled.copy()
X_val_ext = X_val_scaled.copy()
X_train_ext = X_train_scaled.copy()
X_test_ext = X_test_scaled.copy()
# 单独采用机器学习建模时，删除这两个特征
X_train_full_scaled.remove('ModelPred_WaterCut')
X_train_full_scaled.remove('ModelPred_OilProduction')
X_val_scaled.remove('ModelPred_WaterCut')
X_val_scaled.remove('ModelPred_OilProduction')
X_train_scaled.remove('ModelPred_WaterCut')
X_train_scaled.remove('ModelPred_OilProduction')
X_test_scaled.remove('ModelPred_WaterCut')
X_test_scaled.remove('ModelPred_OilProduction')


################################################ Build and optimize machine learning models  ################################################
import optuna
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 为了保证Optuna实验的可重复性，设置随机种子
import random
random.seed(42)
np.random.seed(42)

#######################
# LSTM 模型超参数优化 #
#######################
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
tf.random.set_seed(42)  # 设置TensorFlow随机种子

# 首先，为LSTM/GRU创建时间序列数据格式 (样本, 时间步, 特征)
seq_length = 5  # 序列长度，可根据需要调整
def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X.iloc[i:i+seq_len].values)
        y_seq.append(y.iloc[i+seq_len])
    return np.array(X_seq), np.array(y_seq)

# 基于训练集和验证集创建序列数据
X_train_seq, y_train_seq = create_sequences(train_df2[feature_cols], train_df2[target_col], seq_length)
# 为验证集构造序列时，包括一部分训练集尾部以提供前置序列
X_val_seq, y_val_seq = create_sequences(pd.concat([train_df2[feature_cols].tail(seq_length), val_df[feature_cols]]),
                                       pd.concat([train_df2[target_col].tail(seq_length), val_df[target_col]]),
                                       seq_length)

# 定义LSTM模型的Optuna优化目标函数
def objective_lstm(trial):
    # 超参数空间定义
    units = trial.suggest_int('units', 16, 128)          # LSTM隐藏单元数
    dropout = trial.suggest_float('dropout', 0.0, 0.5)   # dropout比例
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True) # 学习率
    batch_size = trial.suggest_int('batch_size', 8, 64)  # 批次大小
    epochs = trial.suggest_int('epochs', 10, 100)        # 训练轮数

    # 构建LSTM模型
    tf.keras.backend.clear_session()  # 清除旧的模型会话
    model = Sequential()
    model.add(LSTM(units=units, dropout=dropout, recurrent_dropout=0.0, 
                   return_sequences=False, input_shape=(seq_length, len(feature_cols))))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
    # 训练模型
    model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size, 
              validation_data=(X_val_seq, y_val_seq), verbose=0)
    # 用验证集计算RMSE作为目标值
    y_val_pred = model.predict(X_val_seq, verbose=0)
    val_rmse = np.sqrt(mean_squared_error(y_val_seq, y_val_pred))
    return val_rmse

# 使用Optuna优化LSTM模型超参数
study_lstm = optuna.create_study(direction='minimize')
study_lstm.optimize(objective_lstm, n_trials=30, timeout=600)  # 试验次数可根据需要调整
print("LSTM最佳超参数:", study_lstm.best_params)

#######################
# GRU 模型超参数优化 #
#######################
# 类似于LSTM，构建GRU模型的优化目标函数
def objective_gru(trial):
    units = trial.suggest_int('units', 16, 128)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 8, 64)
    epochs = trial.suggest_int('epochs', 10, 100)

    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(GRU(units=units, dropout=dropout, recurrent_dropout=0.0, 
                  return_sequences=False, input_shape=(seq_length, len(feature_cols))))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=Adam(learning_rate=lr))
    model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size, 
              validation_data=(X_val_seq, y_val_seq), verbose=0)
    y_val_pred = model.predict(X_val_seq, verbose=0)
    val_rmse = np.sqrt(mean_squared_error(y_val_seq, y_val_pred))
    return val_rmse

study_gru = optuna.create_study(direction='minimize')
study_gru.optimize(objective_gru, n_trials=30, timeout=600)
print("GRU最佳超参数:", study_gru.best_params)

################################
# 随机森林 RF 模型超参数优化 #
################################
def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    # 构建RF模型并训练（使用训练集，验证集评估）
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                                  random_state=42)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    return val_rmse

study_rf = optuna.create_study(direction='minimize')
study_rf.optimize(objective_rf, n_trials=50)
print("RF最佳超参数:", study_rf.best_params)

######################################
# GBDT (梯度提升决策树)模型超参数优化 #
######################################
def objective_gbdt(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                                      max_depth=max_depth, subsample=subsample, random_state=42)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    return val_rmse

study_gbdt = optuna.create_study(direction='minimize')
study_gbdt.optimize(objective_gbdt, n_trials=50)
print("GBDT最佳超参数:", study_gbdt.best_params)

##################################
# 支持向量回归 SVR 模型超参数优化 #
##################################
def objective_svr(trial):
    C = trial.suggest_float('C', 0.1, 100.0, log=True)
    epsilon = trial.suggest_float('epsilon', 1e-3, 0.5, log=True)
    gamma = trial.suggest_float('gamma', 1e-4, 1.0, log=True)
    model = SVR(C=C, epsilon=epsilon, gamma=gamma)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    return val_rmse

study_svr = optuna.create_study(direction='minimize')
study_svr.optimize(objective_svr, n_trials=50)
print("SVR最佳超参数:", study_svr.best_params)

# 训练并保存每个模型在最佳参数下的最终模型
best_params_rf = study_rf.best_params
best_params_rf['random_state'] = 42
model_rf = RandomForestRegressor(**best_params_rf)
model_rf.fit(X_train_full, y_train_full)

best_params_gbdt = study_gbdt.best_params
best_params_gbdt['random_state'] = 42
model_gbdt = GradientBoostingRegressor(**best_params_gbdt)
model_gbdt.fit(X_train_full, y_train_full)

best_params_svr = study_svr.best_params
model_svr = SVR(**best_params_svr)
model_svr.fit(X_train_full, y_train_full)

# 使用最佳参数构建并训练最终LSTM模型
params_lstm = study_lstm.best_params
tf.keras.backend.clear_session()
model_lstm = Sequential()
model_lstm.add(LSTM(units=params_lstm['units'], dropout=params_lstm['dropout'], 
                    return_sequences=False, input_shape=(seq_length, len(feature_cols))))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mse', optimizer=Adam(learning_rate=params_lstm['lr']))
model_lstm.fit(X_train_seq, y_train_seq, epochs=params_lstm['epochs'], batch_size=params_lstm['batch_size'], verbose=0)

# 使用最佳参数构建并训练最终GRU模型
params_gru = study_gru.best_params
tf.keras.backend.clear_session()
model_gru = Sequential()
model_gru.add(GRU(units=params_gru['units'], dropout=params_gru['dropout'], 
                  return_sequences=False, input_shape=(seq_length, len(feature_cols))))
model_gru.add(Dense(1))
model_gru.compile(loss='mse', optimizer=Adam(learning_rate=params_gru['lr']))
model_gru.fit(X_train_seq, y_train_seq, epochs=params_gru['epochs'], batch_size=params_gru['batch_size'], verbose=0)




################################################ Model Fusion and Prediction ################################################
# 9. 模型融合：将分类油层方法预测结果作为特征加入，使用最佳模型进行融合预测
# 根据论文，对于含水率预测，RF效果最佳；对于年产油量预测，GBDT最佳。
# 这里以RF为基模型进行融合（根据具体任务选择最佳模型类型）。

# 将分类方法预测值特征加入后，用训练全集训练融合模型
model_hybrid = RandomForestRegressor(**best_params_rf)  # 使用RF最佳参数
model_hybrid.fit(X_train_full_ext, y_train_full)

# 对测试集进行预测
y_pred_lstm = model_lstm.predict(create_sequences(pd.concat([train_df[feature_cols].tail(seq_length), test_df[feature_cols]]),
                                                 pd.concat([train_df[target_col].tail(seq_length), test_df[target_col]]),
                                                 seq_length)[0])  # LSTM预测
y_pred_lstm = y_pred_lstm.reshape(-1)  # 展平为一维

y_pred_gru = model_gru.predict(create_sequences(pd.concat([train_df[feature_cols].tail(seq_length), test_df[feature_cols]]),
                                               pd.concat([train_df[target_col].tail(seq_length), test_df[target_col]]),
                                               seq_length)[0])
y_pred_gru = y_pred_gru.reshape(-1)

y_pred_rf = model_rf.predict(X_test)
y_pred_gbdt = model_gbdt.predict(X_test)
y_pred_svr = model_svr.predict(X_test)
y_pred_hybrid = model_hybrid.predict(X_test_ext)




################################################ Model Evaluation ################################################
from sklearn.metrics import r2_score, mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 避免除零错误：对于真实值为0的样本，跳过计算
    nonzero_idx = y_true != 0
    return np.mean(np.abs((y_true[nonzero_idx] - y_pred[nonzero_idx]) / y_true[nonzero_idx])) * 100

# 计算各模型在测试集上的指标
models = ["LSTM", "GRU", "RF", "GBDT", "SVR", "Hybrid"]
preds = [y_pred_lstm, y_pred_gru, y_pred_rf, y_pred_gbdt, y_pred_svr, y_pred_hybrid]
for name, y_pred in zip(models, preds):
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"{name} 模型 -> R²: {r2:.3f},  RMSE: {rmse:.3f},  MAPE: {mape:.2f}%")




################################################ Visualization Results and Error Analysis ################################################
import matplotlib.pyplot as plt

# 如果有时间轴（Year或Date），使用其作为横轴；否则使用样本索引
if 'Year' in test_df.columns:
    x_axis = test_df['Year']
elif 'Date' in test_df.columns:
    x_axis = test_df['Date']
else:
    x_axis = range(len(y_test))

# 1. 实际值 vs 预测值 对比图 (以融合模型为例，并可选对比其他模型)
plt.figure(figsize=(8,5))
plt.plot(x_axis, y_test, marker='o', label='Actual实际值')
plt.plot(x_axis, y_pred_hybrid, marker='o', label='Hybrid预测值')
# 可选：绘制其他模型的预测曲线进行比较
# plt.plot(x_axis, y_pred_rf, label='RF预测值')
plt.xlabel('Time/Index')
plt.ylabel('Target Value')
plt.title('实际值 vs 预测值 对比')
plt.legend()
plt.show()

# 2. 实际值 vs 预测值 散点图（融合模型）
plt.figure(figsize=(5,5))
plt.scatter(y_test, y_pred_hybrid, color='blue', alpha=0.7)
# 绘制y=x参考线
min_val = min(y_test.min(), y_pred_hybrid.min())
max_val = max(y_test.max(), y_pred_hybrid.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel('Actual实际值')
plt.ylabel('Predicted预测值')
plt.title('实际值 vs 预测值 (Hybrid模型)')
plt.show()

# 3. 预测误差随时间变化图（融合模型）
errors = y_test.values - y_pred_hybrid
plt.figure(figsize=(8,4))
plt.plot(x_axis, errors, marker='o', color='red')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Time/Index')
plt.ylabel('Error误差 (Actual - Predicted)')
plt.title('预测误差随时间的变化 (Hybrid模型)')
plt.show()

# 4. 误差分布直方图（融合模型）
plt.figure(figsize=(6,4))
plt.hist(errors, bins=10, color='gray', edgecolor='black')
plt.xlabel('Prediction Error误差')
plt.ylabel('Frequency频次')
plt.title('预测误差分布 (Hybrid模型)')
plt.show()

# 5. 各模型评价指标柱状图 (以R²为例)
r2_scores = [r2_score(y_test, pred) for pred in preds]
plt.figure(figsize=(6,4))
plt.bar(models, r2_scores, color=['#4c72b0','#55a868','#c44e52','#8172b3','#ccb974','#64b5cd'])
plt.ylabel('R² Score')
plt.title('各模型R²对比')
plt.show()

# （注：运行本代码块将在本地弹出图形窗口。如在Notebook中运行，确保启用matplotlib inline。）