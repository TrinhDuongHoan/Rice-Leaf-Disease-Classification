import torch
import torch.nn as nn
import numpy as np
import timm
from sklearn.metrics import accuracy_score, f1_score
from models.backbones.mobilenet import MobileNetV3_Small_ECA
from utils.data.loader import build_loader


BATCH_SIZE = 8
NUM_CLASSES = 10
IMG_SIZE = 224
EPOCHS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_name, num_classes = NUM_CLASSES, device = DEVICE):
    if model_name == 'mobilenetv3_eca':
        model = MobileNetV3_Small_ECA(num_classes = NUM_CLASSES).to(device)
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes, in_chans=3).to(device)
    return model

def train_model(model_name, train_loader, val_loader, device = DEVICE, epochs=EPOCHS, ckpt_path="trained_model/mobilenetv3_eca.pth"):
    
    model = build_model(model_name)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_state, best_epoch = -1.0, None, 1
    
    history = {
        "epoch": [],
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    for ep in range(1, epochs+1):
        model.train()
        tr_loss, n_train, correct = 0.0, 0, 0
        val_loss_sum, n_val = 0.0, 0
        preds, gts = [], []
    
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
    
            tr_loss += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            n_train += y.size(0)
    
            # pbar.update(1)
            cur_loss = tr_loss / max(1, n_train)
            cur_acc  = correct / max(1, n_train)

    
        tr_loss /= max(1, n_train)
        tr_acc = correct / max(1, n_train)
        scheduler.step()
    
        # ---- validate ----
        model.eval()
        with torch.inference_mode():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss_sum += loss.item() * y.size(0)
                n_val += y.size(0)
                preds.append(logits.argmax(1).cpu().numpy())
                gts.append(y.cpu().numpy())
    
                # pbar.update(1)
                cur_vloss = val_loss_sum / max(1, n_val)
    
        val_loss = val_loss_sum / max(1, n_val)
        preds = np.concatenate(preds); gts = np.concatenate(gts)
        acc = accuracy_score(gts, preds)
        f1m = f1_score(gts, preds, average="macro")

        # ---- log ----
        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(acc)

        print(f"[{model_name}] Epoch {ep:02d}/{epochs} "
              f"| train_loss={tr_loss:.4f} | train_acc={tr_acc:.4f} "
              f"| val_loss={val_loss:.4f} | val_acc={acc:.4f} ")

        # ---- save best ----
        if acc > best_acc:
            best_epoch = ep 
            best_acc = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({"model": best_state}, ckpt_path)

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, history, best_epoch
