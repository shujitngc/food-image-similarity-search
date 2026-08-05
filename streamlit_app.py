import os
import pickle

import numpy as np
import streamlit as st
from PIL import Image

import torch
import open_clip


# =========================
# ページ設定
# =========================
st.set_page_config(
    page_title="料理画像類似検索",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ 料理画像類似検索アプリ")
st.write(
    "料理画像をアップロードすると、Food-101から選んだ画像の中から、"
    "見た目や意味が似ている料理画像を検索します。"
)


# =========================
# CLIPモデルの読み込み
# =========================
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )

    model = model.to(device)
    model.eval()

    return model, preprocess, device


# =========================
# 保存済み特徴量の読み込み
# =========================
@st.cache_data
def load_features():
    features = np.load("features.npy")

    with open("image_paths.pkl", "rb") as file:
        image_paths = pickle.load(file)

    return features, image_paths


# =========================
# ファイル名から料理名を取得
# =========================
def get_food_name(image_path):
    filename = os.path.basename(image_path)

    # 例：
    # ice_cream_001.jpg → ice_cream
    # pizza_020.jpg → pizza
    food_name = filename.rsplit("_", 1)[0]

    # 表示用の日本語名
    food_name_ja = {
        "pizza": "ピザ",
        "sushi": "寿司",
        "ramen": "ラーメン",
        "hamburger": "ハンバーガー",
        "ice_cream": "アイスクリーム"
    }

    return food_name_ja.get(
        food_name,
        food_name.replace("_", " ").title()
    )


# =========================
# アップロード画像を特徴量に変換
# =========================
def get_image_feature(image, model, preprocess, device):
    image = image.convert("RGB")

    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model.encode_image(image_tensor)

        # ベクトルの長さを1にそろえる
        feature = feature / feature.norm(dim=-1, keepdim=True)

    return feature.cpu().numpy()[0]


# =========================
# 類似画像検索
# =========================
def search_similar_images(
    query_feature,
    features,
    image_paths,
    top_k=5
):
    # 正規化済みベクトル同士の内積
    # ＝コサイン類似度
    similarities = features @ query_feature

    # 類似度が高い順にインデックスを取得
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append(
            {
                "path": image_paths[index],
                "score": float(similarities[index]),
                "food_name": get_food_name(image_paths[index])
            }
        )

    return results


# =========================
# モデルと特徴量を読み込む
# =========================
model, preprocess, device = load_model()
features, image_paths = load_features()


# =========================
# アップロードエリアを中央に配置
# =========================
left_space, upload_area, right_space = st.columns([1, 2, 1])

with upload_area:
    st.subheader("料理画像をアップロード")

    uploaded_file = st.file_uploader(
        "JPG、JPEG、PNG形式の画像を選択してください",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )

    top_k = st.slider(
        "表示する類似画像の枚数",
        min_value=1,
        max_value=10,
        value=5
    )


# =========================
# 検索処理
# =========================
if uploaded_file is not None:
    query_image = Image.open(uploaded_file).convert("RGB")

    st.divider()

    st.subheader("アップロードした画像")

    preview_left, preview_center, preview_right = st.columns([1, 2, 1])

    with preview_center:
        st.image(
            query_image,
            use_container_width=True
        )

    query_feature = get_image_feature(
        query_image,
        model,
        preprocess,
        device
    )

    results = search_similar_images(
        query_feature,
        features,
        image_paths,
        top_k=top_k
    )

    st.divider()
    st.subheader("検索結果")

    # 3列ずつ表示
    for start in range(0, len(results), 3):
        columns = st.columns(3)

        row_results = results[start:start + 3]

        for column, result in zip(columns, row_results):
            with column:
                result_image = Image.open(
                    result["path"]
                ).convert("RGB")

                st.image(
                    result_image,
                    use_container_width=True
                )

                st.markdown(
                    f"### {result['food_name']}"
                )

                similarity_percent = result["score"] * 100

                st.caption(
                    f"類似度：{similarity_percent:.1f}%"
                )
else:
    st.info("料理画像をアップロードすると検索を開始します。")