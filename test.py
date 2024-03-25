from keras import models

cnn = models.load_model('model.h5')
cnn.load_weights("weights/weights")
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:/Users/Verma/Documents/ml/Projects/Covid_X-Ray/xray_dataset_covid19/predicting_cases/negative1.jfif', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result =cnn.predict(test_image)
if result[0][0] == 1:
  prediction = 'At Risk of Covid'
else:
  prediction = 'No Risk'

print(prediction)