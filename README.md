### 执行顺序
1. `trainset_construction.py` 构造初始训练集
    - 正样本全部镜像， resize
    - 负样本每张图随机采样十个patch， resize
    - 重命名后存放到`data_precessed`
2. `hog_svm_train.py` 第一次训练，模型放在`models` 目录下， `output/train_log.txt` 是训练记录
3. `hard_neg_construction.py` 构造硬负样本，并投入到`data_processed/negative`里
4. `retrain.py` 重训练，模型命名为`hog_svm_model_v2.pkl`
5. `eval_v1.py`, `eval_v2.py` 生成模型衡量
    - Miss Rate是衡量在人图中被错误判断为非人的比例
    - FPPW 是衡量在非人图中，有多少个个**窗口**被错误判断为人图的比例
    - Accuracy 是衡量所有的正确判断图片数量和总测试集数量的比值
    - Precision 是衡量所有判断为人图的样本中，真的为人图的比例
    - Recall 是衡量所有的人图中，判断正确的比例