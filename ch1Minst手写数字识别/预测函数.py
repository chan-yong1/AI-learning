import tkinter as tk
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torchvision import transforms

# =========================================================
# 1. 模型定义（必须和训练时一模一样）
# =========================================================
class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================================================
# 2. 设备 & 加载模型
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MNIST_CNN().to(DEVICE)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=DEVICE))
model.eval()

print("Model loaded on", DEVICE)


# =========================================================
# 3. MNIST 同款 transform（非常关键）
# =========================================================
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# =========================================================
# 4. GUI 应用
# =========================================================
class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST 手写数字识别")

        self.canvas_size = 280

        self.canvas = tk.Canvas(
            root,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black"
        )
        self.canvas.pack()

        # 用 PIL 同步维护一张真正用于预测的图
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.draw_digit)

        self.result_label = tk.Label(
            root,
            text="请用鼠标写 0~9",
            font=("Arial", 20)
        )
        self.result_label.pack(pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="识别", command=self.predict).pack(
            side=tk.LEFT, padx=10
        )
        tk.Button(btn_frame, text="清空", command=self.clear).pack(
            side=tk.LEFT, padx=10
        )

    def draw_digit(self, event):
        r = 10
        x, y = event.x, event.y

        self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            fill="white", outline="white"
        )
        self.draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=255
        )

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle(
            (0, 0, self.canvas_size, self.canvas_size),
            fill=0
        )
        self.result_label.config(text="请用鼠标写 0~9")

    @torch.no_grad()
    def predict(self):
        img = self.image.resize((28, 28))

        tensor = transform(img)               # (1, 28, 28)
        x = tensor.unsqueeze(0).to(DEVICE)    # (1, 1, 28, 28)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)

        pred = prob.argmax(dim=1).item()
        conf = prob.max(dim=1).values.item()  # ✅ 正确

        self.result_label.config(
            text=f"预测结果：{pred}    置信度：{conf:.2f}"
        )



# =========================================================
# 5. 启动程序
# =========================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()
