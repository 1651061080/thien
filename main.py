from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
# Chức năng gọi lại, chức năng đầu tiên là lưu đường dẫn mô hình, 
#chức năng thứ hai là giá trị được phát hiện, phát hiện Mất mát để giảm thiểu nó 
#và chức năng thứ ba là chỉ lưu mô hình hoạt động tốt nhất trên tập hợp xác thực
model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
#steps_per_epoch đề cập đến số lượng lô_kích thước mỗi kỷ nguyên có, là giá trị của tổng số mẫu 
#trong tập huấn luyện chia cho loạt_số
 # Dòng trên là sử dụng trình tạo để đào tạo số lượng batch_size 
 #và các mẫu và nhãn được chuyển qua myGene
# testGene = testGenerator ("dữ liệu / màng / thử nghiệm")
model.fit_generator(myGene,steps_per_epoch=1000,epochs=1,callbacks=[model_checkpoint])
testGene = testGenerator("data/membrane/test")
# 30 là bước, số bước: tổng số bước (lô mẫu) từ bộ tạo trước khi dừng.
#  Tham số tùy chọn Trình tự: Nếu không được chỉ định, len (trình tạo) sẽ được sử dụng làm số bước.
# Giá trị trả về ở trên là: Mảng lộn xộn của các giá trị dự đoán.
results = model.predict_generator(testGene,20,verbose=1)
saveResult("data/membrane/test",results)# lưu kết quả
