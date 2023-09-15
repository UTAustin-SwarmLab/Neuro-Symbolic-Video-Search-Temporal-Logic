from PIL import Image
from ultralytics import YOLO

def predict():
    model = YOLO('../artifacts/weights/yolov8n.pt')
    source = 'pic.jpg'

    results = model.predict(source, save=True, classes=[0, 2, 9]) # 0-person, 2-car, 9-trafficlight
    print(results)

    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.show()

if __name__ == '__main__':
    predict()
