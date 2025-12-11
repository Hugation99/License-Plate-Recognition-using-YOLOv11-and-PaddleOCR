from ultralytics import YOLO
import wandb


# Set the device
device=0 if torch.cuda.is_available() else 'cpu'
print(device)

wandb.login(key="2b315b0478d351c4cef814b9fba3a3dd3d7ad1a0")

model = YOLO("./models/yolo11n.pt")  # load a pretrained 

# Train the model
model.train(
    data="coco8.yaml",
    project='LP-detector', 
    name='yolo11n-lp-detector', 
    epochs=20, 
    imgsz=640,
    batch=16, 
    lr0=0.01, 
    lrf=0.0001,
    save_period=-1,
    optimizer='Adam'
    )