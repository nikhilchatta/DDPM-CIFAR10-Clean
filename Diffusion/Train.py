import torch
import torch.nn as nn
import torch.optim as optim
from .Unet import UNet
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from Scheduler import GradualWarmupScheduler
from Diffusion.Unet import UNet  # Adjust the path if needed

def train(modelConfig):
    device = modelConfig["device"]
    print(f"Training on device: {device}")

    # ==== CIFAR-10 Dataset ====
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(
        root='./CIFAR10',
        train=True,
        transform=transform,
        download=True
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=modelConfig["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == "cuda" else False
    )

    # ==== Model ====
    net_model = UNet(
        T=modelConfig["T"],
        ch=modelConfig["channel"],
        ch_mult=modelConfig["channel_mult"],
        attn=modelConfig["attn"],
        num_res_blocks=modelConfig["num_res_blocks"],
        dropout=modelConfig["dropout"]
    ).to(device)

    # ==== Optimizer ====
    optimizer = optim.Adam(net_model.parameters(), lr=modelConfig["lr"])

    # ==== Learning Rate Scheduler ====
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=modelConfig["epoch"])
    warmup_scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=modelConfig["multiplier"],
        warm_epoch=modelConfig["warmup_epoch"],
        after_scheduler=cosine_scheduler
    )

    # ==== Loss Function ====
    criterion = nn.MSELoss()

    # ==== Training Loop ====
    net_model.train()
    for epoch in range(modelConfig["epoch"]):
        running_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)

            optimizer.zero_grad()

            # ===== Forward Pass (Replace with DDPM-specific logic if needed) =====
            outputs = net_model(images)  # <== Dummy forward; replace with actual DDPM steps
            loss = criterion(outputs, images)  # <== Placeholder loss
            # =====================================================================

            loss.backward()
            nn.utils.clip_grad_norm_(net_model.parameters(), modelConfig["grad_clip"])
            optimizer.step()

            running_loss += loss.item()

        warmup_scheduler.step()

        print(f"Epoch [{epoch + 1}/{modelConfig['epoch']}], Loss: {running_loss:.4f}")

    # ==== Save Model Weights ====
    save_path = modelConfig["save_weight_dir"] + "final_model.pt"
    torch.save(net_model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")
