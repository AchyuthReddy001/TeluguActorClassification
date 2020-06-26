from keras.models import load_model
from keras.preprocessing import image
import  numpy as np

class Classify:
    def __init__(self,filename):
        self.filename=filename

    def predict(self):
        model = load_model('classifermodel.h5')
        file=self.filename
        test_img = image.load_img(file, target_size=(64, 64))
        test_img = image.img_to_array(test_img)
        test_img = np.expand_dims(test_img, axis=0)
        res = model.predict(test_img)
        final = model.predict_classes(test_img)
        print(res)
        #[{ "image" : prediction}]
        if final[0] == 0:
            pred="AlluArjun"
            return [pred]
        elif final[0] == 1:
            pred="MaheshBabu"
            return [pred]
        elif final[0] == 2:
            pred="NTR"
            return [pred]
        elif final[0] == 3:
            pred="PavanKalyan"
            return [pred]
        else:
            pred="Prabhas"
            return [pred]


#Classify()