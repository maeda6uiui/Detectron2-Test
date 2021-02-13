import argparse
import cv2
import torch

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
setup_logger()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_features_single(image_filepath:str,predictor:DefaultPredictor):
    image=cv2.imread(image_filepath)
    if image is None:
        raise RuntimeError("Failed to load image {}".format(image_filepath))
    
    with torch.no_grad():
        image_height,image_width=image.shape[:2]

        image=predictor.aug.get_transform(image).apply_image(image)
        image=torch.as_tensor(image.astype("float32").transpose(2,0,1))
        inputs=[{"image":image,"height":image_height,"width":image_width}]

        model=predictor.model

        images=model.preprocess_image(inputs)

        #Get features from the backbone
        features=model.backbone(images.tensor)
        #Get proposals
        proposals,_=model.proposal_generator(images,features)
        #Get instances
        instances,_=model.roi_heads(images,features,proposals)
        #Generate RoI features
        pred_boxes=[x.pred_boxes for x in instances]
        roi_features=model.roi_heads.box_pooler(
            [features[f] for f in features if f!="p6"],
            pred_boxes
        )
        roi_features=model.roi_heads.box_head(roi_features).cpu()

        roi_coords=torch.empty(0,4)
        for pred_box in pred_boxes:
            roi_coords=torch.cat([roi_coords,pred_box.tensor.cpu()],dim=0)

        ret={
            "instance":instances[0],
            "roi_coords":roi_coords,
            "roi_features":roi_features
        }
        return ret

def main(args):
    frcnn_model_name:str=args.frcnn_model_name
    image_filepath:str=args.image_filepath
    
    cfg=get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(frcnn_model_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.7
    cfg.MODEL.DEVICE=str(device)
    cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url(frcnn_model_name)
    predictor=DefaultPredictor(cfg)

    results=get_features_single(image_filepath,predictor)
    #print(results)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-m","--frcnn_model_name",type=str,default="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument("-i","--image_filepath",type=str,default="./image.jpg")
    args=parser.parse_args()

    main(args)
