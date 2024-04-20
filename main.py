import os
import cv2
from data_utils import get_data_generators
from model import create_model

BASE_DIR = r"data\archive"
TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, "train", "images")
TRAIN_LABEL_DIR = os.path.join(BASE_DIR, "train", "labels")
TEST_IMAGE_DIR = os.path.join(BASE_DIR, "test", "images")
TEST_LABEL_DIR = os.path.join(BASE_DIR, "test", "labels")

BATCH_SIZE = 64
train_generator, val_generator, test_generator = get_data_generators(
    TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR, TEST_IMAGE_DIR, TEST_LABEL_DIR, BATCH_SIZE
)

model = create_model()
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
)

model.save("first_model.h5")

for images, labels in test_generator:
    predictions = model.predict(images)
    for image, label, prediction in zip(images, labels, predictions):
        image = (image * 255).astype("uint8")
        label = label[0]
        prediction = prediction[0]

        x1, y1, x2, y2 = [int(val) for val in label]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        x1, y1, x2, y2 = [int(val * 64) for val in prediction]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
