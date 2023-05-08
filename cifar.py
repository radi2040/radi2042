from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = load_model('cifar10_model.h5')

# Load the new image
img = load_img('new_image.jpg', target_size=(32, 32))

# Preprocess the image
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Make a prediction
pred = model.predict(img_array)

# Interpret the results
class_index = np.argmax(pred)
label_mapping = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}
pred_label = label_mapping[class_index]

# Visualize the results
plt.imshow(img)
plt.title(pred_label)
plt.show()
