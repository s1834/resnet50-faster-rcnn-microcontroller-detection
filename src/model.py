import torchvision

from torchvision.models.detection.faster_rcnn import FasterRCNNPredictor

def create_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = FasterRCNNPredictor(in_features, num_classes)
    
    return model
