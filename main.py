import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import random

# ================= 1. 项目配置 =================
IMAGE_BASE_PATH = r"C:\Users\Lenovo\Desktop\CV_car\VRID\image"
TRAIN_INDEX_FILE = "re_id_1000_train.txt"
TEST_INDEX_FILE = "re_id_1000_test.txt"

# ResNet50 标准输入
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # 为了稳妥，稍微调小Batch防止显存/内存溢出
NUM_CLASSES = 10

BRAND_MAPPING = {
    1: "奥迪A4", 2: "本田雅阁", 3: "别克君越", 4: "大众迈腾",
    5: "丰田花冠", 6: "丰田卡罗拉", 7: "丰田凯美瑞",
    8: "福特福克斯", 9: "日产骐达", 10: "日产轩逸"
}


# ================= 2. 数据流水线 =================
class VehicleDataset:
    def __init__(self, index_file, base_path, img_size=IMG_SIZE):
        self.img_size = img_size
        self.image_paths = []
        self.labels = []
        # 只有当提供了有效文件名时才加载，解决 dummy 报错
        if index_file and os.path.exists(index_file):
            self._load_index(index_file, base_path)

    def _load_index(self, index_file, base_path):
        print(f"正在解析索引文件: {index_file}...")
        with open(index_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line: continue
            parts = line.split('\\')
            if len(parts) >= 3:
                try:
                    brand_id = int(parts[0])
                    path1 = os.path.join(base_path, parts[0], parts[1], parts[-1])
                    path2 = os.path.join(base_path, line)

                    final_path = path1 if os.path.exists(path1) else (path2 if os.path.exists(path2) else None)

                    if final_path:
                        self.image_paths.append(final_path)
                        self.labels.append(brand_id - 1)
                except ValueError:
                    continue
        print(f"✅ 加载完成，共 {len(self.image_paths)} 张图片")

    def load_image(self, path):
        """读取图片并进行预处理"""
        try:
            # 将 byte 类型的 path 转为 string (TF 传进来的是 bytes)
            if isinstance(path, bytes):
                path = path.decode('utf-8')

            img = cv2.imread(path)
            if img is None: raise ValueError("Image invalid")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)

            # ResNet 预处理
            img = preprocess_input(img)

            # 🔴 关键修复：强制转换为 float32
            # 之前的错误是因为这里默认可能是 float64，导致 TF 崩溃
            return img.astype(np.float32)

        except Exception as e:
            # 出错返回全0矩阵，防止管道崩溃
            return np.zeros((*self.img_size, 3), dtype=np.float32)

    def get_dataset_tf(self, batch_size=32, shuffle=False, augment=False):
        if len(self.image_paths) == 0:
            print("⚠️ 警告：数据集为空，无法创建管道")
            return None

        path_ds = tf.data.Dataset.from_tensor_slices(self.image_paths)
        label_ds = tf.data.Dataset.from_tensor_slices(self.labels)
        ds = tf.data.Dataset.zip((path_ds, label_ds))

        def _process_path(path, label):
            # 告诉 TF 这个 numpy_function 肯定返回 float32
            img = tf.numpy_function(self.load_image, [path], tf.float32)
            img.set_shape([*self.img_size, 3])
            return img, label

        ds = ds.map(_process_path, num_parallel_calls=tf.data.AUTOTUNE)

        if augment:
            data_augmentation = tf.keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
                layers.RandomContrast(0.1)
            ])
            ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(buffer_size=1000)

        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds


# ================= 3. 模型构建 (本地权重优先) =================
def build_model_resnet(input_shape, num_classes):
    # 🔴 修复：优先加载本地 ResNet50 权重
    weight_filename = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weight_path = os.path.join(os.getcwd(), weight_filename)

    if os.path.exists(weight_path):
        print(f"📦 发现本地权重文件: {weight_filename}")
        weights_source = weight_path
    else:
        print(f"⚠️ 未找到本地权重 {weight_filename}，尝试联网下载...")
        weights_source = 'imagenet'

    try:
        base_model = ResNet50(weights=weights_source, include_top=False, input_shape=input_shape)
        print("✅ ResNet50 基座构建成功")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        print("⚠️ 切换到随机初始化模式 (效果可能受影响)")
        base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape)

    # 阶段一：冻结
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name="ResNet50_Vehicle_Net")
    return model, base_model


# ================= 4. 业务系统 =================
class VehicleRecognitionSystem:
    def __init__(self, model, detection_threshold=0.65):
        self.model = model
        self.detection_threshold = detection_threshold

    def predict_pipeline(self, img_array):
        img_batch = np.expand_dims(img_array, axis=0)
        probs = self.model.predict(img_batch, verbose=0)[0]
        max_conf = np.max(probs)
        pred_idx = np.argmax(probs)

        has_vehicle = max_conf >= self.detection_threshold

        return {
            "has_vehicle": has_vehicle,
            "detection_score": float(max_conf),
            "brand": BRAND_MAPPING[pred_idx + 1] if has_vehicle else "背景/未知",
        }


# ================= 5. 主程序 =================
def main():
    print("\n>>> 1. 数据集准备")
    train_indexer = VehicleDataset(TRAIN_INDEX_FILE, IMAGE_BASE_PATH)
    test_indexer = VehicleDataset(TEST_INDEX_FILE, IMAGE_BASE_PATH)

    if len(train_indexer.image_paths) == 0: return

    # 划分验证集
    X_train_paths, X_val_paths, y_train, y_val = train_test_split(
        train_indexer.image_paths, train_indexer.labels,
        test_size=0.2, stratify=train_indexer.labels, random_state=42
    )

    # 重新构建对象 (传入 None 避免 dummy 报错)
    train_ds_obj = VehicleDataset(None, None)
    train_ds_obj.image_paths, train_ds_obj.labels = X_train_paths, y_train

    val_ds_obj = VehicleDataset(None, None)
    val_ds_obj.image_paths, val_ds_obj.labels = X_val_paths, y_val

    train_ds = train_ds_obj.get_dataset_tf(batch_size=BATCH_SIZE, shuffle=True, augment=True)
    val_ds = val_ds_obj.get_dataset_tf(batch_size=BATCH_SIZE, shuffle=False)
    test_ds = test_indexer.get_dataset_tf(batch_size=BATCH_SIZE, shuffle=False)

    if train_ds is None: return

    # --- 阶段一 ---
    print("\n>>> 2. 阶段一: 冻结主干，训练分类头 (Warm-up)")
    model, base_model = build_model_resnet((224, 224, 3), NUM_CLASSES)

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 阶段一不需要跑太多轮，只要不动就行
    model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=1)

    # --- 阶段二 ---
    print("\n>>> 3. 阶段二: 解冻主干，分层微调 (Fine-tuning)")
    base_model.trainable = True

    model.compile(optimizer=keras.optimizers.Adam(1e-5),  # 极低学习率
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint('best_resnet_finetuned.h5', save_best_only=True, verbose=1)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=callbacks, verbose=1)

    # --- 评估 ---
    print("\n>>> 4. 最终评估")
    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"🔥 最终测试集准确率: {acc * 100:.2f}%")

    # --- 演示 ---
    print("\n>>> 5. 系统演示")
    system = VehicleRecognitionSystem(model)
    indices = random.sample(range(len(test_indexer.image_paths)), 3)
    for idx in indices:
        path = test_indexer.image_paths[idx]
        img = test_indexer.load_image(path)
        res = system.predict_pipeline(img)
        print(f"文件: {os.path.basename(path)} | 结果: {res['brand']} ({res['detection_score']:.2f})")


if __name__ == "__main__":
    main()