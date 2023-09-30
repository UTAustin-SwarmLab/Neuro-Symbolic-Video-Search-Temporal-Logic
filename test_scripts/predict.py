from PIL import Image
from ultralytics import YOLO


def predict():
    model = YOLO("../artifacts/weights/yolov8n.pt")
    source = "data/yolo_test_image.jpg"
    print(model.names)

    results = model.predict(source, save=True, classes=[0, 2, 9])  # 0-person, 2-car, 9-trafficlight
    print(results[0])
    print(results[0].boxes.conf)
    print(results[0].boxes.cls)
    for i in range(len(results[0].boxes.conf)):
        print(float(results[0].boxes.conf[i]))
        print(int(results[0].boxes.cls[i]))

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.show()


if __name__ == "__main__":
    predict()
