from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans

khong = [0,0,0]
vua = [0,0,128]
nang = [0,128,0]
ratnang = [128,128,0]
lut = [128,0,0]


COLOR_DICT = np.array([khong,vua,nang,ratnang, lut])

# Chức năng  để chuẩn hóa giá trị pixel của dữ liệu của tập huấn luyện và nhãn,
#mục đích của việc định hình lại để dự đoán nhiều lớp
def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        #câu lệnh viết tắt của if else (batch_size, wight, heigh)
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
       # np.zeros bên trong là một bộ hình dạng, mục đích này là để mở rộng độ dày
        #của dữ liệu cho lớp num_class, để đạt được cấu trúc phân loại theo hướng của lớp
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img /= 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    có thể tạo hình ảnh và mặt nạ cùng một lúc
    sử dụng cùng một hạt giống cho image_datagen và mask_datagen để đảm bảo việc chuyển đổi cho hình ảnh và mặt nạ giống nhau
    nếu bạn muốn hình dung kết quả của trình tạo, hãy đặt save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,#đường dẫn thư mục đào tạo
        classes = [image_folder],#thư mục danh mục, lớp nào cần nâng cấp
        class_mode = None,#không trả lại thẻ
        color_mode = image_color_mode,#thang độ xám, chế độ đường đơn
        target_size = target_size,#mục tiêu hình ảnh sau khi chuyển đổi
        batch_size = batch_size,#số lượng ảnh tạo ra sau mỗi lần chuyển đổi
        save_to_dir = save_to_dir,#lưu hình ảnh vào địa chủ
        save_prefix  = image_save_prefix,#Tiền tố của hình ảnh đã tạo chỉ hợp lệ khi save_to_dir được cung cấp
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)#kếp hợp thành tổng quan
    #Bởi vì lô là 2, vì vậy trả lại hai hình cùng một lúc, tức là, img là một mảng gồm 2 hình ảnh thang độ xám, [2,256,256]
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)
#Hai hình ảnh và thẻ được tạo mỗi lần, nếu bạn không hiểu lợi nhuận, vui lòng xem 
 
  # Chức năng trên chủ yếu là để tạo trình tạo ảnh tăng cường dữ liệu, rất tiện lợi khi sử dụng trình tạo này để liên tục tạo ảnh


def testGenerator(test_path,num_image = 21,target_size = (256,256),flag_multi_class = False,as_gray = True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path,"%d.png"%i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    # Tương đương với tìm kiếm tệp, tìm kiếm tệp khớp với các ký tự trong đường dẫn 
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
       # Tìm kiếm lại ảnh có ký tự mặt nạ (ảnh nhãn) trong thư mục mask_path
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr
#Chức năng này chủ yếu là để tìm kiếm các hình ảnh trong thư mục tập hợp đào tạo và thư mục thẻ, sau đó mở rộng một thứ nguyên để trả về nó ở dạng mảng, để đọc dữ liệu trong thư mục khi không sử dụng tính năng nâng cao dữ liệu

def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    # Trở thành không gian RGB, vì các màu khác chỉ có thể được hiển thị trong không gian RGB
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
        # Áp dụng các màu khác nhau cho các danh mục khác nhau, color_dict [i] 
        #là màu liên quan đến số lượng danh mục, img_out 
        #[img == i,:] là điểm của img_out ở vị trí của img bằng với danh mục i
    return img_out / 255
# Chức năng trên là cung cấp màu khác cho đầu ra sau khi 
#đưa ra kết quả sau khi kiểm tra.  Nó chỉ hoạt động trong nhiều loại trường hợp.  Hai loại đều vô dụng


def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
        # Nếu có nhiều danh mục, ảnh sẽ có màu, nếu không có nhiều danh mục (hai danh mục), ảnh sẽ có màu đen và trắng