import os
import tempfile
import cv2
import numpy as np
import torch
import torchvision
import gradio as gr

# ==========================================
# 1. 初始化与模型加载 (EF 与 分割双模型)
# ==========================================
# ==========================================
# 1. 初始化与模型加载 (修复了新版 PyTorch 的加载报错和警告)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备: {DEVICE}")


# 加载 EF 预测模型 (R2+1D)
def load_ef_model(weights_path="r2plus1d_18_32_2_pretrained.pt"):
    # 修复警告：将 pretrained=False 改为 weights=None
    model = torchvision.models.video.r2plus1d_18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    if os.path.exists(weights_path):
        # 修复报错：显式设置 weights_only=False（仅在信任该权重来源时使用）
        checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        print(f"警告：未找到 EF 权重文件 {weights_path}，将使用随机初始化模型。")
    model.to(DEVICE)
    model.eval()
    return model


# 加载分割模型 (DeepLabV3)
def load_seg_model(weights_path="deeplabv3_resnet50_random.pt"):
    # 修复警告：将 pretrained=False 改为 weights=None
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, num_classes=1)

    if os.path.exists(weights_path):
        # 修复报错：显式设置 weights_only=False
        checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        print(f"警告：未找到分割权重文件 {weights_path}，将使用随机初始化模型。")
    model.to(DEVICE)
    model.eval()
    return model


# 预先加载模型避免重复加载卡顿
EF_MODEL = load_ef_model()
SEG_MODEL = load_seg_model()


# ==========================================
# 2. 核心处理与推理流水线
# ==========================================
def process_and_infer(input_video_path):
    if input_video_path is None:
        return "请先上传视频", None

    # ImageNet 归一化标准
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # --------------------------------------
    # 步骤 A: 读取视频帧
    # --------------------------------------
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0

    orig_frames = []
    processed_frames = []  # 用于模型输入的帧

    while True:
        ret, frame = cap.read()
        if not ret: break
        orig_frames.append(frame.copy())

        # 预处理：缩放至 112x112，转为 RGB
        resized = cv2.resize(frame, (112, 112))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # 归一化
        normalized = (rgb / 255.0 - MEAN) / STD
        processed_frames.append(normalized)
    cap.release()

    if len(orig_frames) == 0:
        return "视频解析失败，未能读取到任何帧", None

    total_frames = len(processed_frames)

    # --------------------------------------
    # 步骤 B: EF 射血分数推理 (R2+1D 3D模型)
    # --------------------------------------
    # 针对 3D 卷积采样 64 帧
    length, period = 64, 2
    sampled_indices = np.arange(0, length * period, period)
    if total_frames < length * period:
        indices = np.mod(sampled_indices, total_frames)
    else:
        start = (total_frames - length * period) // 2
        indices = sampled_indices + start

    ef_input = np.stack([processed_frames[i] for i in indices])  # (64, 112, 112, 3)
    ef_tensor = torch.tensor(ef_input, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).to(
        DEVICE)  # (1, 3, 64, 112, 112)

    with torch.no_grad():
        ef_output = EF_MODEL(ef_tensor)
        predicted_ef = ef_output.item()

    # --------------------------------------
    # 步骤 C: 左心室分割推理 (DeepLabV3 2D模型，逐帧处理)
    # --------------------------------------
    temp_dir = tempfile.gettempdir()
    output_video_path = os.path.join(temp_dir, "segmented_output.mp4")

    # 获取原始视频的高宽以便输出一致大小
    h, w, _ = orig_frames[0].shape
    # 使用 mp4v 编码保证网页浏览器能原生直接播放
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    print("正在逐帧进行左心室分割并合成视频...")
    for idx, frame in enumerate(orig_frames):
        # 准备当前帧的分割输入 (1, 3, 112, 112)
        seg_input = torch.tensor(processed_frames[idx], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            seg_output = SEG_MODEL(seg_input)['out']  # DeepLabV3 返回字典，取 'out'
            # 经过 Sigmoid 激活得到概率图，大于 0.5 判定为左心室
            mask = torch.sigmoid(seg_output).squeeze().cpu().numpy() > 0.5

        # 将 112x112 的掩码放大回原始视频的分辨率大小
        mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        # 创建一个红色的半透明图层叠加在原图上（代表左心室轮廓/区域）
        vis_frame = frame.copy()
        vis_frame[mask_resized == 1] = [0, 0, 255]  # BGR 格式中的红色

        # 混合原图与红色图层，做出半透明效果 (Alpha Blending)
        fused_frame = cv2.addWeighted(frame, 0.7, vis_frame, 0.3, 0)

        # 顺便在视频左上角把预测的 EF 打印上去
        # cv2.putText(fused_frame, f"Predicted EF: {predicted_ef:.1f}%", (20, 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        out_video.write(fused_frame)

    out_video.release()

    # 返回结果：1. EF 文本数字 2. 渲染了左心室红圈的最终视频
    return f"{predicted_ef:.2f}%", output_video_path


# ==========================================
# 3. Gradio 网页端前端设计
# ==========================================
with gr.Blocks(title="EchoNet-Dynamic 综合分析系统") as demo:
    gr.Markdown("# 🩺 EchoNet-Dynamic 心脏超声全功能分析端")
    gr.Markdown("上传心脏四腔心切面（A4C）超声视频，一键获得**射血分数（EF）**并实时查看**左心室自动分割动画**。")

    with gr.Row():
        # 左侧：上传
        with gr.Column(scale=1):
            video_input = gr.Video(label="第一步：上传超声视频", sources=["upload"])
            submit_btn = gr.Button("🔥 运行多任务推理", variant="primary")

        # 右侧：输出
        with gr.Column(scale=1):
            ef_output = gr.Label(label="第二步：测定的射血分数 (EF 值)")
            video_output = gr.Video(label="第三步：左心室逐帧分割可视化结果 (LV Segmentation)")

    # 点击事件绑定
    submit_btn.click(
        fn=process_and_infer,
        inputs=video_input,
        outputs=[ef_output, video_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=5006, share=False)