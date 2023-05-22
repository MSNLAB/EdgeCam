import cv2
import torch
import torchvision
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import*
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms
from torchvision.models.mobilenetv3 import mobilenet_v3_large

from retrain.detection_dataset import DetectionDataset, TrafficDataset
from retrain.detection_metric import RetrainMetric
from retrain.read_video import video_reader


# 读取MP4文件
def read_video(filename):
    cap = cv2.VideoCapture(filename)
    frames = []
    index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        index += 1
        if not ret:
            break
        if index > 10:
            break
        cv2.imwrite("./retrain_data/frame/{}.jpg".format(index), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


# 构建模型
def create_student_model():
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.roi_heads.parameters():
        param.requires_grad = True

    for module in model.roi_heads.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    return model


def process_targets(targets):
    indices = torch.where(targets[0]['scores'] > 0.8)[0]
    targets[0]['boxes'] = targets[0]['boxes'][indices]
    targets[0]['labels'] = targets[0]['labels'][indices]
    targets[0]['scores'] = targets[0]['scores'][indices]
    return targets

def _collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练的 teacher 模型
    teacher_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    teacher_model.to(device)
    teacher_model.eval()

    # 加载 student 模型
    student_model = create_student_model()
    student_model.to(device)

    # 读取视频
    video_file = 'dayroad.mp4'

    video_reader(video_file, "./retrain_data/frames", teacher_model , "./retrain_data/annotations.txt", )

    dataset = TrafficDataset(root="./retrain_data")

    data_loader = DataLoader(dataset=dataset, batch_size=2, collate_fn=_collate_fn,)

    tr_metric = RetrainMetric()

    # 训练设置
    num_epoch = 10
    roi_parameters = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(roi_parameters, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epoch):
        student_model.train()
        for images, targets in tr_metric.log_iter(epoch, num_epoch, data_loader):
            print("pre tar {}".format(targets))
            for t in targets:
                print("t {}".format(t))
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            print(targets)
            with torch.cuda.amp.autocast():
                loss_dict = student_model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            tr_metric.update(loss_dict, losses)
        # Update the learning rate
        lr_scheduler.step()

    torch.save(student_model.state_dict(), "student_model.pth")













    exit(0)
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])


    # 训练设置
    epochs = 10
    roi_parameters = [p for p in student_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(roi_parameters, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        student_model.train()


        for frame in frames:
            # 数据预处理
            input_tensor = transform(frame)
            input_tensor = input_tensor.to(device)

            # 获得 teacher 模型的输出
            with torch.no_grad():
                teacher_output = teacher_model([input_tensor])
            targets = teacher_output
            # 获得特定targets
            targets = process_targets(targets)
            # 获得 student 模型的输出
            student_output = student_model([input_tensor], targets)
            logger.debug(student_output)
            # 计算损失
            student_loss = sum(loss for loss in student_output.values())
            total_loss = student_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # Update the learning rate
        lr_scheduler.step()

    torch.save(student_model.state_dict(), "student_model.pth")

def test_student_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    teacher_model.to(device)
    teacher_model.eval()
    student_model = create_student_model()
    state_dict = torch.load("student_model.pth", map_location=device)
    student_model.load_state_dict(state_dict)
    student_model.to(device)
    student_model.eval()

    # 读取视频
    video_file = 'road.mp4'
    frames = read_video(video_file)

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    for frame in frames:
        # 数据预处理
        input_tensor = transform(frame)
        input_tensor = input_tensor.to(device)
        # 获得 teacher 模型的输出
        with torch.no_grad():
            #teacher_output = teacher_model([input_tensor])
            student_output = student_model([input_tensor])
        #logger.debug(teacher_output)
        logger.debug(student_output)





if __name__ == "__main__":
    #main()
    test_student_model()