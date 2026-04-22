import math
import random
import struct


class BPNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        初始化三层BP神经网络
        input_size: 输入层神经元个数
        hidden_size: 隐藏层神经元个数
        output_size: 输出层神经元个数
        learning_rate: 学习率
        W1: 输入层到隐藏层权重矩阵
        b1: 输入层到隐藏层偏置向量
        W2: 隐藏层到输出层权重矩阵
        b2: 隐藏层到输出层偏置向量
        loss_history: 损失函数历史记录
        accuracy_history: 准确率历史记录
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 初始化权重和偏置
        self.W1 = self._initialize_weights(input_size, hidden_size)
        self.b1 = [0.0] * hidden_size
        self.W2 = self._initialize_weights(hidden_size, output_size)
        self.b2 = [0.0] * output_size

        self.loss_history = []
        self.accuracy_history = []

    def _initialize_weights(self, rows, cols):
        """
        初始化权重矩阵
        rows: 行数
        cols: 列数
        """
        weights = []
        for i in range(rows):
            row = []
            for j in range(cols):
                # 使用均匀分布初始化
                row.append(random.uniform(-0.1, 0.1))
            weights.append(row)
        return weights

    def _matrix_multiply(self, A, B):
        """
        矩阵乘法 A(m×n) × B(n×p) = C(m×p)
        """
        m = len(A)
        # 如果矩阵存在，则返回列数，否则返回0
        n = len(A[0]) if A else 0
        p = len(B[0]) if B else 0

        if n != len(B):
            raise ValueError(f"矩阵维度不匹配: A({m}x{n}) × B({len(B)}x{p}) 无法相乘!")

        result = []
        for i in range(m):
            row = []
            for j in range(p):
                sum_val = 0.0
                for k in range(n):
                    sum_val += A[i][k] * B[k][j]
                row.append(sum_val)
            result.append(row)
        return result

    def _matrix_add_vector(self, A, b):
        """矩阵加向量 A(m×n) + b(n) = C(m×n)"""
        m = len(A)
        n = len(A[0]) if A else 0

        if n != len(b):
            raise ValueError(f"维度不匹配: A({m}x{n}) + b({len(b)}) 无法相加!")

        result = []
        for i in range(m):
            row = []
            for j in range(n):
                row.append(A[i][j] + b[j])
            result.append(row)
        return result

    def _matrix_subtract(self, A, B):
        """矩阵减法"""
        m = len(A)
        n = len(A[0])

        if m != len(B) or n != len(B[0]):
            raise ValueError("矩阵维度不匹配 A({m}x{n}) - B({len(B)}x{len(B[0])}) 无法相减!")

        result = []
        for i in range(m):
            row = []
            for j in range(n):
                row.append(A[i][j] - B[i][j])
            result.append(row)
        return result

    def _matrix_transpose(self, A):
        """矩阵转置"""
        if not A:
            return []

        rows = len(A)
        cols = len(A[0])
        result = []
        for j in range(cols):
            row = []
            for i in range(rows):
                row.append(A[i][j])
            result.append(row)
        return result

    def _scalar_multiply(self, scalar, A):
        """标量乘法即向量和矩阵的运算"""
        result = []
        for i in range(len(A)):
            row = []
            for j in range(len(A[0])):
                row.append(scalar * A[i][j])
            result.append(row)
        return result

    def _elementwise_multiply(self, A, B):
        """逐元素乘法"""
        m = len(A)
        n = len(A[0])

        if m != len(B) or n != len(B[0]):
            raise ValueError("矩阵维度不匹配")

        result = []
        for i in range(m):
            row = []
            for j in range(n):
                row.append(A[i][j] * B[i][j])
            result.append(row)
        return result

    def sigmoid(self, x):
        """Sigmoid函数，用作激活函数"""
        if isinstance(x, list) and isinstance(x[0], list):
            # 处理2D列表即矩阵
            result = []
            for row in x:
                new_row = []
                for val in row:
                    # 防止数值溢出
                    if val > 100:
                        new_row.append(1.0)
                    elif val < -100:
                        new_row.append(0.0)
                    else:
                        new_row.append(1.0 / (1.0 + math.exp(-val)))
                result.append(new_row)
            return result
        elif isinstance(x, list):
            # 处理1D列表即向量
            result = []
            for val in x:
                if val > 100:
                    result.append(1.0)
                elif val < -100:
                    result.append(0.0)
                else:
                    result.append(1.0 / (1.0 + math.exp(-val)))
            return result
        else:
            # 处理单个数值
            if x > 100:
                return 1.0
            elif x < -100:
                return 0.0
            return 1.0 / (1.0 + math.exp(-x))

    def sigmoid_derivative(self, x):
        """Sigmoid函数的导数，用于反向传播"""
        if isinstance(x, list) and isinstance(x[0], list):
            #处理矩阵
            result = []
            for row in x:
                new_row = []
                for val in row:
                    new_row.append(val * (1 - val))
                result.append(new_row)
            return result
        elif isinstance(x, list):
            #处理向量
            result = []
            for val in x:
                result.append(val * (1 - val))
            return result
        else:
            #处理单个数值
            return x * (1 - x)

    def softmax(self, x):
        """Softmax函数，归一化输出"""
        result = []
        for row in x:
            # 减去最大值防止数值溢出
            max_val = max(row)
            exp_row = [math.exp(val - max_val) for val in row]
            sum_exp = sum(exp_row)
            softmax_row = [val / sum_exp for val in exp_row]
            result.append(softmax_row)
        return result

    def forward(self, X):
        """前向传播"""
        # 输入层到隐藏层: X(m×784) × W1(784×hidden) + b1(hidden) = z1(m×hidden)
        self.z1 = self._matrix_add_vector(self._matrix_multiply(X, self.W1), self.b1)
        self.a1 = self.sigmoid(self.z1)

        # 隐藏层到输出层: a1(m×hidden) × W2(hidden×10) + b2(10) = z2(m×10)
        self.z2 = self._matrix_add_vector(self._matrix_multiply(self.a1, self.W2), self.b2)
        self.a2 = self.softmax(self.z2)

        return self.a2

    def backward(self, X, y, output):
        """反向传播"""
        m = len(X)  # 样本数量

        # 输出层误差: delta2 = output - y
        delta2 = self._matrix_subtract(output, y)

        # 隐藏层误差: delta1 = (delta2 × W2^T) ⊙ sigmoid'(a1)
        W2_transpose = self._matrix_transpose(self.W2)
        delta1_product = self._matrix_multiply(delta2, W2_transpose)
        sigmoid_deriv = self.sigmoid_derivative(self.a1)
        delta1 = self._elementwise_multiply(delta1_product, sigmoid_deriv)

        # 计算梯度
        a1_transpose = self._matrix_transpose(self.a1)
        dW2 = self._scalar_multiply(1 / m, self._matrix_multiply(a1_transpose, delta2))

        # 计算偏置梯度
        db2 = [0.0] * self.output_size
        for i in range(m):
            for j in range(self.output_size):
                db2[j] += delta2[i][j]
        db2 = [x / m for x in db2]

        X_transpose = self._matrix_transpose(X)
        dW1 = self._scalar_multiply(1 / m, self._matrix_multiply(X_transpose, delta1))

        db1 = [0.0] * self.hidden_size
        for i in range(m):
            for j in range(self.hidden_size):
                db1[j] += delta1[i][j]
        db1 = [x / m for x in db1]

        # 更新权重和偏置
        self.W2 = self._matrix_subtract(
            self.W2, self._scalar_multiply(self.learning_rate, dW2)
        )
        self.b2 = [self.b2[j] - self.learning_rate * db2[j] for j in range(self.output_size)]

        self.W1 = self._matrix_subtract(
            self.W1, self._scalar_multiply(self.learning_rate, dW1)
        )
        self.b1 = [self.b1[j] - self.learning_rate * db1[j] for j in range(self.hidden_size)]

    def compute_loss(self, y_true, y_pred):
        """交叉熵损失计算，比较模型输出的预测概率与实际标签的差距"""
        m = len(y_true)
        total_loss = 0.0

        for i in range(m):
            for j in range(len(y_true[0])):
                # 添加小值防止log(0)
                if y_pred[i][j] < 1e-8:
                    y_pred[i][j] = 1e-8
                total_loss += y_true[i][j] * math.log(y_pred[i][j])

        return -total_loss / m

    def train(self, X, y, epochs, batch_size=32, validation_data=None):
        """手动实现训练过程"""
        n_samples = len(X)

        for epoch in range(epochs):
            # 手动实现数据打乱
            indices = list(range(n_samples))
            random.shuffle(indices)
            X_shuffled = [X[i] for i in indices]
            y_shuffled = [y[i] for i in indices]

            epoch_loss = 0.0
            n_batches = 0

            # 小批量训练
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X_shuffled[i:end_idx]
                y_batch = y_shuffled[i:end_idx]

                # 前向传播
                output = self.forward(X_batch)

                # 计算损失
                batch_loss = self.compute_loss(y_batch, output)
                epoch_loss += batch_loss
                n_batches += 1

                # 反向传播
                self.backward(X_batch, y_batch, output)

            # 计算平均损失
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)

            # 验证集准确率
            if validation_data is not None and epoch % 5 == 0:
                X_val, y_val = validation_data
                val_accuracy = self.evaluate(X_val, y_val)
                self.accuracy_history.append(val_accuracy)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self, X):
        """预测"""
        output = self.forward(X)
        predictions = []
        for row in output:
            max_val = max(row)
            pred = row.index(max_val)
            predictions.append(pred)
        return predictions

    def evaluate(self, X, y):
        """评估"""
        predictions = self.predict(X)
        true_labels = []
        for row in y:
            max_val = max(row)
            true_label = row.index(max_val)
            true_labels.append(true_label)

        correct = 0
        for pred, true in zip(predictions, true_labels):
            if pred == true:
                correct += 1

        return correct / len(X)


# 数据加载和预处理函数
def load_mnist_data():
    """加载MNIST数据集"""
    print("正在加载训练集图像...")
    # 加载训练集图像
    with open('train-images-idx3-ubyte', 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]

        x_train = []
        for i in range(num):
            img_data = f.read(rows * cols)
            img = [float(pixel) for pixel in img_data]
            x_train.append(img)
            if i % 10000 == 0:
                print(f"已加载 {i}/{num} 个训练图像")

    print("正在加载训练集标签...")
    # 加载训练集标签
    with open('train-labels-idx1-ubyte', 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num = struct.unpack('>I', f.read(4))[0]

        y_train = []
        for i in range(num):
            label = struct.unpack('B', f.read(1))[0]
            y_train.append(label)

    print("正在加载测试集图像...")
    # 加载测试集图像
    with open('t10k-images-idx3-ubyte', 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]

        x_test = []
        for i in range(num):
            img_data = f.read(rows * cols)
            img = [float(pixel) for pixel in img_data]
            x_test.append(img)
            if i % 2000 == 0:
                print(f"已加载 {i}/{num} 个测试图像")

    print("正在加载测试集标签...")
    # 加载测试集标签
    with open('t10k-labels-idx1-ubyte', 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num = struct.unpack('>I', f.read(4))[0]

        y_test = []
        for i in range(num):
            label = struct.unpack('B', f.read(1))[0]
            y_test.append(label)

    return (x_train, y_train), (x_test, y_test)


def preprocess_data(x, y):
    """数据预处理"""
    print("数据归一化...")
    # 归一化到0-1范围
    x_normalized = []
    for i, img in enumerate(x):
        normalized_img = [pixel / 255.0 for pixel in img]
        x_normalized.append(normalized_img)
        if i % 10000 == 0:
            print(f"已归一化 {i}/{len(x)} 个样本")

    print("One-hot编码...")
    # 手动实现one-hot编码
    y_onehot = []
    for i, label in enumerate(y):
        onehot = [0.0] * 10
        onehot[label] = 1.0
        y_onehot.append(onehot)
        if i % 10000 == 0:
            print(f"已编码 {i}/{len(y)} 个标签")

    return x_normalized, y_onehot


def create_validation_set(X, y, val_size=10000):
    """创建验证集"""
    X_val = X[:val_size]
    y_val = y[:val_size]
    X_train = X[val_size:]
    y_train = y[val_size:]
    return X_train, y_train, X_val, y_val


# 可视化函数
def print_training_progress(loss_history, accuracy_history):
    """打印训练进度"""
    print("\n训练完成!")
    print("最终损失:", f"{loss_history[-1]:.4f}")
    if accuracy_history:
        print("最终验证准确率:", f"{accuracy_history[-1]:.4f}")


def print_confusion_matrix(y_true, y_pred, num_classes=10):
    """打印混淆矩阵"""
    cm = [[0] * num_classes for _ in range(num_classes)]

    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1

    print("\n混淆矩阵:")
    print("    " + " ".join(f"{i:4d}" for i in range(num_classes)))
    print("   " + "-" * (num_classes * 5 + 1))
    for i in range(num_classes):
        print(f"{i:2d} |" + " ".join(f"{count:4d}" for count in cm[i]))


def print_sample_predictions(model, X_test, y_test, num_samples=10):
    """样本预测结果"""
    print(f"\n前{num_samples}个样本的预测结果:")
    predictions = model.predict(X_test[:num_samples])

    true_labels = []
    for row in y_test[:num_samples]:
        max_val = max(row)
        true_label = row.index(max_val)
        true_labels.append(true_label)

    correct_count = 0
    for i in range(num_samples):
        status = "✓" if predictions[i] == true_labels[i] else "✗"
        if predictions[i] == true_labels[i]:
            correct_count += 1
        print(f"样本 {i + 1}: 真实={true_labels[i]}, 预测={predictions[i]} {status}")

    print(f"样本准确率: {correct_count}/{num_samples} = {correct_count / num_samples:.2f}")


# 主程序
def main():
    print("开始三层BP神经网络...")

    # 1. 加载数据
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # 2. 数据预处理
    print("数据预处理...")
    X_train, y_train_oh = preprocess_data(x_train, y_train)
    X_test, y_test_oh = preprocess_data(x_test, y_test)

    print(f"训练集: {len(X_train)} 个样本, 每个样本 {len(X_train[0])} 个特征")
    print(f"测试集: {len(X_test)} 个样本")

    # 3. 创建验证集
    X_train_final, y_train_final, X_val, y_val = create_validation_set(
        X_train, y_train_oh, val_size=10000
    )

    # 4. 创建神经网络
    print("\n创建三层BP神经网络...")
    input_size = 784
    hidden_size = 64  # 为了计算效率，使用较小的隐藏层
    output_size = 10
    learning_rate = 0.1

    bp_net = BPNeuralNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        learning_rate=learning_rate
    )

    print(f"网络结构: {input_size} -> {hidden_size} -> {output_size}")
    print(f"学习率: {learning_rate}")
    print(f"训练样本数: {len(X_train_final)}")
    print(f"验证样本数: {len(X_val)}")
    print(f"测试样本数: {len(X_test)}")

    # 5. 训练网络（为了演示，使用较少的epoch）
    print("\n开始训练神经网络...")
    epochs = 30  # 减少epoch数量以加快训练
    batch_size = 32

    bp_net.train(
        X_train_final, y_train_final,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val)
    )

    # 6. 评估模型
    print("\n评估模型...")
    train_accuracy = bp_net.evaluate(X_train_final, y_train_final)
    test_accuracy = bp_net.evaluate(X_test, y_test_oh)

    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")

    # 7. 显示结果
    print_training_progress(bp_net.loss_history, bp_net.accuracy_history)

    # 混淆矩阵
    y_test_pred = bp_net.predict(X_test)
    y_test_true = []
    for row in y_test_oh:
        max_val = max(row)
        true_label = row.index(max_val)
        y_test_true.append(true_label)

    print_confusion_matrix(y_test_true, y_test_pred)

    # 样本预测
    print_sample_predictions(bp_net, X_test, y_test_oh, num_samples=15)

    print("\n模型训练完成!")


if __name__ == "__main__":
    main()