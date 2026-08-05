import os
import shutil

selected_classes = [
    "pizza",
    "sushi",
    "ramen",
    "hamburger",
    "ice_cream"
]

source_root = "data/food-101/images"
save_dir = "images"

os.makedirs(save_dir, exist_ok=True)

max_per_class = 100

for class_name in selected_classes:
    class_dir = os.path.join(source_root, class_name)

    image_names = os.listdir(class_dir)[:max_per_class]

    for i, image_name in enumerate(image_names):
        src_path = os.path.join(class_dir, image_name)

        new_name = f"{class_name}_{i:03d}.jpg"
        dst_path = os.path.join(save_dir, new_name)

        shutil.copy(src_path, dst_path)

print("保存完了")
print("画像枚数:", len(os.listdir(save_dir)))