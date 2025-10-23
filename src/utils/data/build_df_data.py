import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def build_df(root: Path, abs_path: bool) -> pd.DataFrame:
    csv = pd.read_csv(root / "train.csv")

    labels = sorted(csv["label"].unique().tolist())
    label2id = {lb: i for i, lb in enumerate(labels)}

    def mk_path(r):
        p = root / "train_images" / str(r["label"]) / str(r["image_id"])
        return str(p.resolve()) if abs_path else str(p.as_posix())

    out = pd.DataFrame({
        "image_id": csv["image_id"].astype(str),
        "image_path": csv.apply(mk_path, axis=1),
        "label": csv["label"].astype(str),
        "label_id": csv["label"].map(label2id).astype(int),
    })

    out = out[out["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
    if out.empty:
        raise ValueError("Không tìm thấy ảnh nào.")

    return out, label2id

def main(args):
    root = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df, label2id = build_df(root, abs_path=not args.relative)
    train_df, val_df = train_test_split(df, test_size=args.val_size, random_state=args.seed, stratify=df["label_id"])
    
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    print(f"Đã lưu: {out_dir / 'train.csv'} ({len(train_df)})")
    print(f"Đã lưu: {out_dir / 'val.csv'} ({len(val_df)})")
    print("Label mapping:", label2id)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="./Paddy_Dataset")
    ap.add_argument("--out_dir", default="./Paddy_Dataset/splits")
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--relative", action="store_true", help="Sử dụng đường dẫn tương đối trong CSV.")
    args = ap.parse_args()
    main(args)
