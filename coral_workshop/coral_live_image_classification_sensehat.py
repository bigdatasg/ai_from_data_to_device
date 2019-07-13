import cv2
import numpy
import argparse
import time
import re


from edgetpu.classification.engine import ClassificationEngine
from PIL import Image, ImageDraw, ImageFont

from imutils.video import FPS
from imutils.video import VideoStream

from sense_hat import SenseHat
sense = SenseHat()
sense.rotation = 180

def ReadLabelFile(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    ret = {}
    for line in lines:
        pair = re.split(r'[:\s]+', line.strip(), maxsplit=1)
        ret[int(pair[0])] = pair[1].strip()
    return ret


def draw_image(image, result):
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), result, font=ImageFont.truetype("/usr/share/fonts/truetype/piboto/Piboto-Regular.ttf", 20))
    displayImage = numpy.asarray(image)
    if(result == 'None'):
        sense.show_letter("X",text_colour=[255, 0, 0])
    else:
        sense.show_letter("D",text_colour=[0, 255, 0])
    cv2.imshow('Live Inference', displayImage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', help='File path of Tflite model.', required=True)
    parser.add_argument(
        '--label', help='File path of label file.', required=True)
    parser.add_argument(
        '--picamera', action='store_true',
        help="Use PiCamera for image capture", default=False)
    args = parser.parse_args()

    # Prepare labels.
    labels = ReadLabelFile(args.label) if args.label else None
    # Initialize engine.
    engine = ClassificationEngine(args.model)

    # Initialize video stream
    vs = VideoStream(usePiCamera=args.picamera, resolution=(640, 480)).start()
    time.sleep(1)

    fps = FPS().start()

    while True:
        try:
            # Read frame from video
            screenshot = vs.read()
            image = Image.fromarray(screenshot)

            # Perform inference
            results = engine.ClassifyWithImage(image, top_k=1)
            result = labels[results[0][0]] if results!=[] else 'None'
            draw_image(image, result)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                fps.stop()
                break

            fps.update()
        except KeyboardInterrupt:
            fps.stop()
            break

    print("Elapsed time: " + str(fps.elapsed()))
    print("Approx FPS: :" + str(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()
    time.sleep(2)


if __name__ == '__main__':
    main()
