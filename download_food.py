from torchvision.datasets import Food101

dataset = Food101(
    root="data",
    split="train",
    download=True
)

print("ダウンロード完了")
print("画像枚数:", len(dataset))