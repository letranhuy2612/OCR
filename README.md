# Retrain PaddleOCR
##  1. Data Preparation
###  1.1 DataSet Preparation
Cắt ảnh theo cell dựa theo code trên gitlab 
Sau khi cắt ta sẽ thu được 1 folder chứa ảnh các cell từ các segment
và 1 file txt chứa nhãn của cell đó nhưng cần phải soát lại  và sửa lại do có thể ocr sai 
![Screenshot_22](https://user-images.githubusercontent.com/85574548/201623043-5366dbf1-a56d-41a6-bb92-e9f099fd1c37.png)
![Screenshot_23](https://user-images.githubusercontent.com/85574548/201623077-e52a36fe-9bff-4ba9-9328-8c82a8bb91e5.png)
Ta sẽ cắt  dataset1 đầu ngẫu nhiên tất các file nagoya thu đc gần 3000 ảnh sau
dataset2 chứa các cell showwarning trong file excel và các ảnh cell này được tăng cường mỗi ảnh lên 10 lần để cho ra kết quả tốt
###  1.2 Dictionary
Finally, a dictionary ({word_dict_name}.txt) needs to be provided so that when the model is trained, all the characters that appear can be mapped to the dictionary index.

Therefore, the dictionary needs to contain all the characters that you want to be recognized correctly. 
We use `ppocr/utils/en_dict.txt` is a English dictionary with 96 characters
##  2.Training
PaddleOCR provides training scripts, evaluation scripts, and prediction scripts. In this section, the CRNN recognition model will be used as an example:
###  2.1 Start Training
Clone repo and install [requirements.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/requirements.txt) 
```bash

git https://github.com/PaddlePaddle/PaddleOCR # clone

cd PaddleOCR

pip install -r requirements.txt # install

```
First download the pretrain model, you can download the trained model to finetune on the icdar2015 data:
```

# Download the pre-trained model of en_PP-OCRv3

wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar

# Decompress model parameters

cd pretrain_models

tar -xf en_PP-OCRv3_rec_train.tar && rm -rf en_PP-OCRv3_rec_train.tar

```
We need to fix data path and parameters in file [en_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml)
Ex:epoch_num,use_gpu,save_epoch_step: Nên để bằng epoch_num để chỉ lưu mình file best_acc và latest_acc
Take `en_PP-OCRv3_rec.yml` as an example:
```

Global:

...

# Add a custom dictionary, such as modify the dictionary, please point the path to the new dictionary

character_dict_path: ppocr/utils/en_dict.txt

# Modify character type

...

# Whether to recognize spaces

use_space_char: True

Optimizer:

...

# Add learning rate decay strategy

lr:

name: Cosine

learning_rate: 0.001

...

...

Train:

dataset:

# Type of dataset，we support LMDBDataSet and SimpleDataSet

name: SimpleDataSet

# Path of dataset

data_dir: ./train_data/

# Path of train list

label_file_list: ["./train_data/train_list.txt"]

transforms:

...

- RecResizeImg:

# Modify image_shape to fit long text

image_shape: [3, 48, 320]

...

loader:

...

# Train batch_size for Single card

batch_size_per_card: 32

...

Eval:

dataset:

# Type of dataset，we support LMDBDataSet and SimpleDataSet

name: SimpleDataSet

# Path of dataset

data_dir: ./train_data

# Path of eval list

label_file_list: ["./train_data/val_list.txt"]

transforms:

...

- RecResizeImg:

# Modify image_shape to fit long text

image_shape: [3, 48, 320]

...

loader:

# Eval batch_size for Single card

batch_size_per_card: 32

...

```
Start training:
```

# GPU training Support single card and multi-card training

# Training icdar15 English data and The training log will be automatically saved as train.log under "{save_model_dir}"

python3 tools/train.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model=en_PP-OCRv3_rec_train/best_accuracy

```
PaddleOCR supports alternating training and evaluation. You can modify `eval_batch_step` in file config to set the evaluation frequency. By default, it is evaluated every 500 iter and the best acc model is saved under `output/rec_CRNN/best_accuracy` during the evaluation process.

If the evaluation set is large, the test will be time-consuming. It is recommended to reduce the number of evaluations, or evaluate after training.
###  2.2 Load Trained Model and Continue Training

If you expect to load trained model and continue the training again, you can specify the parameter `Global.checkpoints` as the model path to be loaded.

For example:

```shell

python3 tools/train.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.checkpoints=./your/trained/model

```

**Note**: The priority of `Global.checkpoints` is higher than that of `Global.pretrained_model`, that is, when two parameters are specified at the same time, the model specified by `Global.checkpoints` will be loaded first. If the model path specified by `Global.checkpoints` is wrong, the one specified by `Global.pretrained_model` will be loaded.
We train dataset2 ->pretrain_model after use it ->pretrain_model for dataset1
##  3.  Inference and load your model
###  3.1 Inference
Using the model trained by paddleocr, you can quickly get prediction through the following script.

The default prediction picture is stored in `infer_img`, and the trained weight is specified via `-o Global.checkpoints`:

According to the `save_model_dir` and `save_epoch_step` fields set in the configuration file, the following parameters will be saved:

```

output/rec/

├── best_accuracy.pdopt

├── best_accuracy.pdparams

├── best_accuracy.states

├── config.yml

├── latest.pdopt

├── latest.pdparams

├── latest.states

└── train.log

```
Among them, best_accuracy.* is the best model on the evaluation set; (iter_epoch_x.* is the model saved at intervals of `save_epoch_step`; if  save_epoch_step in (0, epoch_num))  latest.* is the model of the last epoch.
The inference model (the model saved by `paddle.jit.save`) is generally a solidified model saved after the model training is completed, and is mostly used to give prediction in deployment.

The model saved during the training process is the checkpoints model, which saves the parameters of the model and is mostly used to resume training.

Compared with the checkpoints model, the inference model will additionally save the structural information of the model. Therefore, it is easier to deploy because the model structure and model parameters are already solidified in the inference model file, and is suitable for integration with actual systems.

The recognition model is converted to the inference model in the same way as the detection, as follows:

```

# -c Set the training algorithm yml configuration file

# -o Set optional parameters

# Global.pretrained_model parameter Set the training model address to be converted without adding the file suffix .pdmodel, .pdopt or .pdparams.

# Global.save_inference_dir Set the address where the converted model will be saved.

python3 tools/export_model.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model=en_PP-OCRv3_rec_train/best_accuracy Global.save_inference_dir=./inference/en_PP-OCRv3_rec/

```

If you have a model trained on your own dataset with a different dictionary file, please make sure that you modify the `character_dict_path` in the configuration file to your dictionary file path.

After the conversion is successful, there are three files in the model save directory:

```

inference/en_PP-OCRv3_rec/

├── inference.pdiparams # The parameter file of recognition inference model

├── inference.pdiparams.info # The parameter information of recognition inference model, which can be ignored

└── inference.pdmodel # The program file of recognition model

```
###  3.2 Load your model
We load model inference 
```
from paddleocr import PaddleOCR,draw_ocr
# The path of detection and recognition model must contain model and params files
ocr = PaddleOCR( rec_model_dir='inference/en_PP-OCRv3_rec/', rec_char_dict_path='en_PP-OCRv3_rec.yml',  use_angle_cls=True)
img_path = 'PaddleOCR/doc/imgs_en/img_12.jpg'
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```
Result of  files showwarning in file excel:
```
HOK32
start = ['','','','','','','','','','07:57','07:57','07:57','07:57','07:57','','','07:57','07:57','07:57','07:57','07:57','','','07:57','07:57','07:57','07:57','07:57','','','07:57']

end = ['','','','','','','','','','17:18','17:25','17:20','17:18','17:19','','','17:18','17:18','17:42','17:33','17:19','','','17:30','17:25','17:35','17:19','17:30','','','17:26']

HOK35
start = ['','','','','','08:45','08:45','','','08:45','08:45','08:45','','','','','08:15','08:15','08:15','08:15','','','','','08:15','08:15','08:15','08:15','','','']

end = ['','','','','','17:15','17:15','','','17:15','17:15','17:15','','','','','16:45','16:45','16:45','16:45','','','','','16:45','16:45','16:45','16:45','','','']
HOK41,HOK46,HOK61,HOK62,HOK67,HOK85 đã hết showwarning
```


