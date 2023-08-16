#CVCNN
**ADS-B 与 GSM 的数据集分别在 get_dataseCVCNNt_ADSB 和 get_dataset_CVCNN 中获取,complexcnn定义了一个用于复数卷积的PyTorch模型.**
**ADS_B 数据集在 CVCNN 与 ResNet 两个模型上进行训练，其模型对应的.py文件名称为：CVCNN_model、ResNet_model，其模型训练对应的.py文件名称为：train_CVCNN_ADS-B、train_ResNet_ADS-B.**
**GSM 数据集在 CVCNN 与 ResNet 两个模型上进行训练，其模型对应的.py文件名称为：CV_CVCNN_Model、CV_ResNet_Model，其模型训练对应的.py文件名称为：train_GSM_CVCNN、train_GSM_ResNet.**
**利用训练好的模型权重导入到测试文件 ADS-B_test.py 与 GSM_test.py 中，导入对应的模型名称与模型训练权重，设置好相应的测试参数，即可得到测试结果与混淆矩阵，其中混淆矩阵是通过 confusion.py 文件来实现的.**
