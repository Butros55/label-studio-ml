from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import torch
from PIL import Image

class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    
    def setup(self):
        self.set("model_version", "0.0.1")
        from ultralytics import YOLO
        self.my_model = YOLO(r"C:\dev\projects\boulder_ai\api\weights\best.pt")


    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        predictions = []
        class_names = {0: "black", 1: "blue", 2: "grey", 3: "orange", 4: "purple", 5: "red", 6: "turquoise", 7: "white", 8: "wood", 9: "yellow"}

        for task in tasks:
            image_url = task['data']['image_url']
            local_path = self.get_local_path(image_url, task_id=task["id"])

            # (1) Bildbreite und -höhe herausfinden
            with Image.open(local_path) as pil_img:
                w_img, h_img = pil_img.size  # z. B. 640 x 480

            # (2) YOLO-Inferenz mit conf-Threshold
            results = self.my_model(local_path, conf=0.6)
            result = results[0]

            result_items = []
            for i, box in enumerate(result.boxes):
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                label = class_names.get(cls_idx, str(cls_idx))

                # (3) Pixelkoordinaten -> Prozent umrechnen
                x_pct = (x_min / w_img) * 100
                y_pct = (y_min / h_img) * 100
                w_pct = ((x_max - x_min) / w_img) * 100
                h_pct = ((y_max - y_min) / h_img) * 100

                result_items.append({
                    "id": f"bbox-{i}",
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": x_pct,
                        "y": y_pct,
                        "width": w_pct,
                        "height": h_pct,
                        "rectanglelabels": [label]
                    }
                })

            predictions.append({
                "model_version": self.get("model_version"),
                "score": 0.6,
                "result": result_items
            })

        return ModelResponse(predictions=predictions)




    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')


def load_my_model(model_path: str):
    model = torch.load(model_path,weights_only=False, map_location=torch.device('cpu'))
    return model
