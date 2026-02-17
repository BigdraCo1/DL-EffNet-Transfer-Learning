import shutil
import random
from pathlib import Path
from collections import defaultdict

# -----------------------------
# CONFIG: ปรับตรงนี้เท่านั้น
# -----------------------------
SOURCE_DIR = Path("augmented_poker_data")
OUTPUT_DIR = Path("data_rank")

# โฟลเดอร์ชั้นบนเป็น suit
SUITS = ["spade", "heart", "diamond", "club"]

# โฟลเดอร์ชั้นล่างเป็น rank (ตามของคุณ: 2-10 และ a j q k แบบตัวเล็ก)
RANKS = ["2","3","4","5","6","7","8","9","10","a","j","q","k"]

SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

SEED = 42
random.seed(SEED)


def ensure_dirs():
    for split in SPLIT.keys():
        for r in RANKS:
            (OUTPUT_DIR / split / r).mkdir(parents=True, exist_ok=True)


def read_suit_rank_subfolders():
    """
    อ่านรูปจากโครงสร้าง:
      SOURCE_DIR/suit/rank/*.jpg
    แล้วรวมเป็น dict[rank] -> list[Path]
    """
    items = defaultdict(list)

    for suit in SUITS:
        suit_dir = SOURCE_DIR / suit
        if not suit_dir.exists():
            print(f"⚠️ missing suit folder: {suit_dir}")
            continue

        for rank in RANKS:
            rank_dir = suit_dir / rank
            if not rank_dir.exists():
                continue

            for p in rank_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    items[rank].append(p)

    return items


def copy_to_split(items_by_rank):
    ensure_dirs()

    for r in RANKS:
        paths = list(items_by_rank.get(r, []))
        random.shuffle(paths)

        n = len(paths)
        n_train = int(n * SPLIT["train"])
        n_val = int(n * SPLIT["val"])

        train_paths = paths[:n_train]
        val_paths = paths[n_train:n_train + n_val]
        test_paths = paths[n_train + n_val:]

        print(f"[rank {r}] total={n} train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")

        for split_name, split_paths in [("train", train_paths), ("val", val_paths), ("test", test_paths)]:
            for p in split_paths:
                dst = OUTPUT_DIR / split_name / r / p.name
                shutil.copy2(p, dst)


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    items = read_suit_rank_subfolders()

    # report
    for r in RANKS:
        print(f"Found {len(items.get(r, []))} images for rank '{r}'")

    copy_to_split(items)

    print("\n✅ Done! Output structure:")
    print(OUTPUT_DIR.resolve())
    print("Use data_rank/train|val|test/<rank>/... for training.")


if __name__ == "__main__":
    main()
