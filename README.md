# Retrain PaddleOCR
##  1. Data Preparation
###  1.1 DataSet Preparation
Cắt ảnh theo cell dựa theo code trên gitlab 
```bash

git clone http://gitlab.arrow-tech.vn/AI/web_ocr_model
git checkout huy
python tool_label.py

```
Sau khi chạy ta sẽ thu được 1 folder chứa ảnh các cell từ các segment
và 1 file txt chứa nhãn của cell đó nhưng cần phải soát lại  và sửa lại do có thể ocr sai 

![Screenshot_22](https://user-images.githubusercontent.com/85574548/201623043-5366dbf1-a56d-41a6-bb92-e9f099fd1c37.png)
![Screenshot_23](https://user-images.githubusercontent.com/85574548/201623077-e52a36fe-9bff-4ba9-9328-8c82a8bb91e5.png)

Ta sẽ tạo:  
dataset1:đầu ngẫu nhiên tất các file nagoya thu đc gần 3000 ảnh sau

dataset2(data_showwarning_augument): chứa các cell showwarning trong file excel và các ảnh cell này được tăng cường mỗi ảnh lên 10 lần để cho ra kết quả tốt(Sửa lại path trong file tool_label.py thành path của các file bị showwarning và chạy
###  1.2 Dictionary
Ta cần cung cấp một từ điển ({word_dict_name}.txt) để khi mô hình được đào tạo, tất cả các ký tự xuất hiện có thể được ánh xạ tới chỉ mục từ điển
Ta sẽ dùng `ppocr/utils/en_dict.txt` là English dictionary với 96 kí tự
##  2.Training
###  2.1 Start Training
Clone repo and install [requirements.txt](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/requirements.txt) 
```bash
cd ..

git clone https://github.com/PaddlePaddle/PaddleOCR # clone

cd PaddleOCR

pip install -r requirements.txt # install

```
Trước tiên, hãy tải xuống mô hình được đào tạo trước, bạn có thể tải xuống mô hình được đào tạo để hoàn thiện dữ liệu icdar2015:
```

# Download the pre-trained model of en_PP-OCRv3

wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar

# Decompress model parameters

cd pretrain_models

tar -xf en_PP-OCRv3_rec_train.tar && rm -rf en_PP-OCRv3_rec_train.tar

```
Ta cần cài đặt lại các thông số cần thiết trong file [en_PP-OCRv3_rec.yml](../../configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml)
   - epoch_num:số epoch
   - save_epoch_step: Nên để bằng epoch_num để chỉ lưu mình file best_acc và latest_acc
   - save_model_dir: [path to your save model dir]
   - pretrained_model: [path to your pretrained_model]
   - character_dict_path: [path to your character_dict]
   - data_dir: đường dẫn đến data vừa tạo ở trên # nếu trong label có chứa đường dẫn thì chỉnh lại cho đúng. Ví dụ path là data\train\cell_4.png. mà label là         train\cell_4.png    4, thì data_dir: data
   - label_file_list: đường dẫn đến file nhãn txt tạo cùng vs ảnh ở trên
   - chỉnh batch_size phù hợp với bộ nhớ máy 

Start training:

```
python3 tools/train.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model=en_PP-OCRv3_rec_train/best_accuracy
```
   - `-c` đường dẫn đến file config vừa sửa lại các tham số như trên
   - `-o  Global.pretrained_model=` đường dẫn đến pretrained model

**Note**:Ta sẽ train model với dataset2 thu được petrained model và dùng nó làm pretrain model cho bộ dataset1
##  2.  Inference and load your model
###  2.1 Inference
Theo các trường save_model_dir và save_epoch_step được đặt trong tệp cấu hình, các tham số sau sẽ được lưu:

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
Trong số đó, best_accuracy.* là mô hình tốt nhất trong bộ đánh giá; latest.* là mô hình của kỷ nguyên cuối cùng.

Cần phải export best_accuracy.* mô hình dạng inference model: inference.*  

```

# -c đường dẫn đến file config

# -o đường đẫn đến file model best_accuracy.*

# Global.pretrained_model parameter folder chứa: .pdmodel, .pdopt or .pdparams.

# Global.save_inference_dir đường dẫn model sau khi export 

python3 tools/export_model.py -c configs/rec/PP-OCRv3/en_PP-OCRv3_rec.yml -o Global.pretrained_model=en_PP-OCRv3_rec_train/best_accuracy Global.save_inference_dir=./inference/en_PP-OCRv3_rec/

```

Sau khi chuyển đổi thành công, có ba tệp trong thư mục lưu mô hình:

```

inference/en_PP-OCRv3_rec/

├── inference.pdiparams # The parameter file of recognition inference model

├── inference.pdiparams.info # The parameter information of recognition inference model, which can be ignored

└── inference.pdmodel # The program file of recognition model

```
### 2.2 Load your model
Load model đã train lên và tiến hành dự đoán
```
from paddleocr import PaddleOCR,draw_ocr
# rec_model_dir:đường dẫn đến model vừa export ,rec_char_dict_path:đường dẫn đến file từ điển đang dùng là en_dict
ocr = PaddleOCR( rec_model_dir='inference/en_PP-OCRv3_rec/', rec_char_dict_path='ppocr/utils/en_dict.txt',  use_angle_cls=True)
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


