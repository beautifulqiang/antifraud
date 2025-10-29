from math import floor, ceil
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, average_precision_score
import os
from .mcnn_model import mcnn, to_pred

# tensorboard作图
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def mcnn_main(
    train_feature_dir,
    train_label_dir,
    test_feature_dir,
    test_label_dir,
    epochs=30,
    batch_size=512,
    lr=1e-3,
    device="cpu",
    log_dir="runs/mcnn_experiment"
):
    # ========== 数据加载 ==========
    train_feature = torch.from_numpy(np.load(train_feature_dir, allow_pickle=True)).to(
        dtype=torch.float32, device=device)
    train_feature.transpose_(1, 2)
    train_label = torch.from_numpy(np.load(train_label_dir, allow_pickle=True)).to(
        dtype=torch.long, device=device)
    test_feature = torch.from_numpy(np.load(test_feature_dir, allow_pickle=True)).to(
        dtype=torch.float32, device=device)
    test_feature.transpose_(1, 2)
    test_label = torch.from_numpy(np.load(test_label_dir, allow_pickle=True)).to(
        dtype=torch.long, device=device)

    print(f"train_feature shape: {train_feature.shape}")
    print(f"train_label shape: {train_label.shape}")
    print(f"unique train labels: {torch.unique(train_label)}")

    # ========== 模型、优化器、损失函数 ==========
    model = mcnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    unique_labels, counts = torch.unique(train_label, return_counts=True)
    weights = (1 / counts) * len(train_label) / len(unique_labels)
    loss_func = torch.nn.CrossEntropyLoss(weights)

    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("current device:", device)

    # ========== TensorBoard 初始化 ==========
    # 给 log_dir 加上时间戳子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir_with_time = os.path.join(log_dir, timestamp)

    # 确保目录存在
    os.makedirs(log_dir_with_time, exist_ok=True)

    # 创建 TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir_with_time)
    print(f"TensorBoard logs will be saved to: {log_dir_with_time}")

    batch_num = ceil(len(train_label) / batch_size)

    # ========== 训练循环 ==========
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pred = []

        for batch in range(batch_num):
            optimizer.zero_grad()
            batch_mask = list(range(batch * batch_size, min((batch + 1) * batch_size, len(train_label))))
            output = model(train_feature[batch_mask])
            batch_loss = loss_func(output, train_label[batch_mask])
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
            pred.extend(to_pred(output))

        true = train_label.cpu().numpy()
        pred = np.array(pred)

        # 计算训练指标
        try:
            auc = roc_auc_score(true, pred)
        except ValueError:
            auc = np.nan
        f1 = f1_score(true, pred, average='macro')
        ap = average_precision_score(true, pred)

        avg_loss = epoch_loss / batch_num
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | AUC: {auc:.4f} | F1: {f1:.4f} | AP: {ap:.4f}")

        # 🔹 写入 TensorBoard
        writer.add_scalar("Train/Loss", avg_loss, epoch)
        writer.add_scalar("Train/AUC", auc, epoch)
        writer.add_scalar("Train/F1", f1, epoch)
        writer.add_scalar("Train/AP", ap, epoch)

    # ========== 测试集指标 ==========
    model.eval()
    pred_test = []
    with torch.no_grad():
        for batch in range(ceil(len(test_label) / batch_size)):
            batch_mask = list(range(batch * batch_size, min((batch + 1) * batch_size, len(test_label))))
            output = model(test_feature[batch_mask])
            pred_test.extend(to_pred(output))

    true_test = test_label.cpu().numpy()
    pred_test = np.array(pred_test)

    try:
        auc_test = roc_auc_score(true_test, pred_test)
    except ValueError:
        auc_test = np.nan
    f1_test = f1_score(true_test, pred_test, average='macro')
    ap_test = average_precision_score(true_test, pred_test)

    print(f"Test | AUC: {auc_test:.4f} | F1: {f1_test:.4f} | AP: {ap_test:.4f}")

    # 🔹 写入 TensorBoard（测试集）
    writer.add_scalar("Test/AUC", auc_test, epoch)
    writer.add_scalar("Test/F1", f1_test, epoch)
    writer.add_scalar("Test/AP", ap_test, epoch)

    # 关闭 TensorBoard writer
    writer.close()