import cv2
from utils import get_datetime_str

coco_labels_name = ["unlabeled", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                    "boat",
                    "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird",
                    "cat", "dog", "horse",
                    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe",
                    "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball",
                    "kite", "baseball bat",
                    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass",
                    "cup", "fork", "knife",
                    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot_dog",
                    "pizza",
                    "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window",
                    "desk",
                    "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                    "oven",
                    "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear",
                    "hair drier",
                    "toothbrush", "hair brush"]  # len = 92

def visualizaion(predictions, src_img):
    for x in range(len(predictions)):
        pred = predictions[x]
        scores = pred["scores"]
        mask = scores > 0.7  # 只取scores值大于0.5的部分

        boxes = pred["boxes"][mask].cpu().int().detach().numpy()  # [x1, y1, x2, y2]
        labels = pred["labels"][mask].cpu()
        scores = scores[mask].cpu()
        print(f"prediction: boxes:{boxes}, labels:{labels}, scores:{scores}")

        img = src_img.copy()

        for idx in range(len(boxes)):
            cv2.rectangle(img, (boxes[idx][0], boxes[idx][1]), (boxes[idx][2], boxes[idx][3]), (255, 0, 0))
            cv2.putText(img, coco_labels_name[labels[idx]] + " " + str(scores[idx].detach().numpy()),
                        (boxes[idx][0] + 10, boxes[idx][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # cv2.imshow("image", img)
        # cv2.waitKey(100000)
        cv2.imwrite(get_datetime_str() + '.jpg', img)