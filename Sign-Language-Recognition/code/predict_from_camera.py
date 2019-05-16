import os
import sys
import traceback
import logging

import cv2
from sklearn.externals import joblib

from common.config import get_config
from common.image_transformation import apply_image_transformation
from common.image_transformation import resize_image


logging_format = '[%(asctime)s||%(name)s||%(levelname)s]::%(message)s'
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format=logging_format,
                    datefmt='%Y-%m-%d %H:%M:%S',)
logger = logging.getLogger(__file__)


def get_image_from_label(label):
    testing_images_dir_path = get_config('testing_images_dir_path')
    image_path = os.path.join(testing_images_dir_path, label, '001.jpg')
    image = cv2.imread(image_path)
    return image


def main():
    model_name = sys.argv[1]
    if model_name not in ['svm', 'logistic', 'knn']:
        logger.error("Invalid model-name '{}'!".format(model_name))
        return
    logger.info("Using model '{}'...".format(model_name))

    model_serialized_path = get_config(
        "model_{}_serialized_path".format(model_name))
    logger.info("Model deserialized from path '{}'".format(
                model_serialized_path))

    classifier_model = joblib.load(model_serialized_path)

    camera = cv2.VideoCapture(0)
    while True:
        ret, img = camera.read()
        img = cv2.flip(img, 1)

        if not ret:
            logger.error("Failed to capture image!")
            continue

        x1, y1, x2, y2 = 50, 50, 350, 350
        frame = img[y1:y2, x1:x2]
        frame = cv2.flip(frame, 1)

        key_pressed = cv2.waitKey(1)
        if key_pressed == 27:
            break

        try:
            frame = apply_image_transformation(frame)
            frame_flattened = frame.flatten()

            predicted_labels = classifier_model.predict([frame_flattened])
            predicted_label = predicted_labels[0]
            # logger.info("Predicted label = {}".format(predicted_label))
            
            cv2.putText(img, '%s' % (predicted_label.upper()), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
            cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255,0,0), 2)
            cv2.imshow("Webcam recording", img)

        except Exception:
            exception_traceback = traceback.format_exc()
            logger.error("Error applying image transformation")
            logger.debug(exception_traceback)
            
    cv2.destroyAllWindows()
    cv2.VideoCapture(0).release()
    logger.info("The program completed successfully !!")


if __name__ == '__main__':
    main()
