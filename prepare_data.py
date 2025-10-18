import pandas as pd
import zipfile
import os
import urllib.request

def download_and_prepare_movielens(output_path="dataMatrix.csv"):
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = "ml-100k.zip"
    folder_name = "ml-100k"

    # === 1. Скачать датасет ===
    if not os.path.exists(zip_path):
        print("📥 Downloading MovieLens 100K dataset...")
        urllib.request.urlretrieve(url, zip_path)
        print("✅ Download complete")

    # === 2. Распаковать ===
    if not os.path.exists(folder_name):
        print("📦 Extracting archive...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
        print("✅ Extraction complete")

    # === 3. Загрузить данные ===
    print("📊 Reading data...")
    df = pd.read_csv(
        os.path.join(folder_name, "u.data"),
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    # === 4. Преобразовать в бинарный формат ===
    # Будем считать, что рейтинг >= 4 → "нравится"
    df["rating"] = (df["rating"] >= 4).astype(int)

    # === 5. Удалить timestamp ===
    df = df.drop(columns=["timestamp"])

    # === 6. Сохранить в CSV ===
    df.to_csv(output_path, index=False)
    print(f"✅ Saved preprocessed dataset to '{output_path}'")
    print(df.head())

if __name__ == "__main__":
    download_and_prepare_movielens()
