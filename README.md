# 🍽️ Food Image Similarity Search

## 概要

Food-101データセットを用いた**画像類似検索アプリ**です。

ユーザーが料理画像をアップロードすると、CLIP（Contrastive Language-Image Pretraining）を利用して画像の特徴量を抽出し、事前に登録した画像とのコサイン類似度を計算して、最も類似した料理画像を表示します。

アプリはStreamlitで作成しており、ブラウザ上から手軽に画像検索を体験できます。

---

## 制作目的

画像分類だけでなく、

* 画像特徴量の抽出
* 類似画像検索
* Webアプリ化

まで一連の流れを実装し、画像検索システムの基本的な仕組みを学ぶことを目的として制作しました。

---

## 主な機能

* 料理画像のアップロード
* CLIPによる画像特徴量抽出
* コサイン類似度による類似画像検索
* 類似画像をTop5で表示
* 類似度スコアの表示
* StreamlitによるWebアプリ

---

## 使用技術

| カテゴリ          | 技術                  |
| ------------- | ------------------- |
| 言語            | Python              |
| Webアプリ        | Streamlit           |
| Deep Learning | PyTorch             |
| 画像特徴抽出        | OpenCLIP (ViT-B-32) |
| 画像処理          | Pillow              |
| 数値計算          | NumPy               |
| データセット        | Food-101            |

---

## システム構成

```text
アップロード画像
        │
        ▼
CLIPで特徴ベクトルへ変換
        │
        ▼
保存済み特徴量(features.npy)と比較
        │
        ▼
コサイン類似度を計算
        │
        ▼
類似度の高い画像を表示
```

---

## ディレクトリ構成

```text
food-image-similarity-search/
│
├── images/
├── streamlit_app.py
├── make_features.py
├── save_food_images.py
├── download_food.py
├── features.npy
├── image_paths.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 実行方法

### 1. リポジトリをクローン

```bash
git clone <repository-url>
cd food-image-similarity-search
```

### 2. 必要なライブラリをインストール

```bash
pip install -r requirements.txt
```

### 3. Food-101をダウンロード

```bash
python download_food.py
```

### 4. 検索対象画像を保存

```bash
python save_food_images.py
```

### 5. 特徴量を作成

```bash
python make_features.py
```

### 6. アプリを起動

```bash
streamlit run streamlit_app.py
```

---

## 工夫した点

* CLIPを利用して画像を512次元の特徴ベクトルへ変換
* 特徴量を事前計算して保存することで、検索時の処理を高速化
* コサイン類似度を利用し、意味的に近い料理画像を検索
* Streamlitを用いてブラウザ上で簡単に操作できるUIを実装

---

## 今後の改善

* 検索対象画像数の拡張
* FAISSを用いた高速検索
* テキスト検索（例：「ラーメン」）への対応
* 検索結果のフィルタリング機能
* 独自データセットによる特徴量学習（Siamese Network / Triplet Loss）
* Docker対応

---

## Author

**Shuji Taniguchi**

GitHub: https://github.com/shujitngc
