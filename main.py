import cv2
import numpy as np
import torch
from mobilevitv2.mobilevit_v2 import MobileViTv2
from mobilevitv2.mobilevit_v2_cfg import *


def run_model(image_path='test_images/2.jpg',
              width='w2_0',
              weight_path="weights/mobilevit-w2.0.pth"):

    cfg = eval("get_mobilevit_v2_" + width)()   # noqa
    model = MobileViTv2(cfg=cfg)
    # print(model)
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(image, dtype=np.float32)
    image = image / 255.0

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    in_tensor = torch.from_numpy(image).float()
    with torch.no_grad():
        out_tensor = model(in_tensor)
    out_tensor = torch.softmax(out_tensor, dim=-1)
    print(torch.argmax(out_tensor, dim=-1))


if __name__ == "__main__":
    run_model()
