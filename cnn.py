import math
import random
import struct


class Conv2D:
    """2D卷积层 - 手动实现"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 初始化卷积核和偏置
        self.kernels = []
        self.bias = []
        for oc in range(out_channels):
            kernel_channel = []
            for ic in range(in_channels):
                kernel_row = []
                for i in range(kernel_size):
                    kernel_col = []
                    for j in range(kernel_size):
                        # Xavier初始化
                        scale = math.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
                        kernel_col.append(random.uniform(-1.0, 1.0) * scale)
                    kernel_row.append(kernel_col)
                kernel_channel.append(kernel_row)
            self.kernels.append(kernel_channel)
            self.bias.append(0.0)

    def forward(self, x):
        """前向传播"""
        self.input = x
        batch_size = len(x)
        in_channels = len(x[0])
        in_height = len(x[0][0])
        in_width = len(x[0][0][0])

        # 计算输出尺寸
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # 填充输入
        if self.padding > 0:
            x_padded = []
            for b in range(batch_size):
                batch = []
                for c in range(in_channels):
                    channel = []
                    # 上填充
                    for i in range(self.padding):
                        channel.append([0.0] * (in_width + 2 * self.padding))
                    # 中间行
                    for i in range(in_height):
                        row = [0.0] * self.padding + x[b][c][i] + [0.0] * self.padding
                        channel.append(row)
                    # 下填充
                    for i in range(self.padding):
                        channel.append([0.0] * (in_width + 2 * self.padding))
                    batch.append(channel)
                x_padded.append(batch)
        else:
            x_padded = x

        # 执行卷积
        output = []
        for b in range(batch_size):
            batch_output = []
            for oc in range(self.out_channels):
                channel_output = []
                for oh in range(out_height):
                    row_output = []
                    for ow in range(out_width):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        conv_sum = 0.0
                        for ic in range(self.in_channels):
                            for kh in range(self.kernel_size):
                                for kw in range(self.kernel_size):
                                    conv_sum += (x_padded[b][ic][h_start + kh][w_start + kw] *
                                                 self.kernels[oc][ic][kh][kw])
                        conv_sum += self.bias[oc]
                        row_output.append(conv_sum)
                    channel_output.append(row_output)
                batch_output.append(channel_output)
            output.append(batch_output)

        return output

    def backward(self, d_output, learning_rate):
        """反向传播"""
        batch_size = len(self.input)
        in_channels = len(self.input[0])
        in_height = len(self.input[0][0])
        in_width = len(self.input[0][0][0])

        out_channels = len(d_output[0])
        out_height = len(d_output[0][0])
        out_width = len(d_output[0][0][0])

        # 初始化梯度 - 修复这行
        d_input = []
        for b in range(batch_size):
            batch_d = []
            for c in range(in_channels):
                channel_d = []
                for h in range(in_height):
                    row_d = [0.0 for _ in range(in_width)]
                    channel_d.append(row_d)
                batch_d.append(channel_d)
            d_input.append(batch_d)

        # 初始化d_kernels
        d_kernels = []
        for oc in range(out_channels):
            oc_kernels = []
            for ic in range(in_channels):
                ic_kernels = []
                for kh in range(self.kernel_size):
                    row_kernels = [0.0 for _ in range(self.kernel_size)]
                    ic_kernels.append(row_kernels)
                oc_kernels.append(ic_kernels)
            d_kernels.append(oc_kernels)

        d_bias = [0.0] * out_channels

        # 填充输入用于梯度计算
        if self.padding > 0:
            x_padded = []
            for b in range(batch_size):
                batch = []
                for c in range(in_channels):
                    channel = []
                    for i in range(self.padding):
                        channel.append([0.0] * (in_width + 2 * self.padding))
                    for i in range(in_height):
                        row = [0.0] * self.padding + self.input[b][c][i] + [0.0] * self.padding
                        channel.append(row)
                    for i in range(self.padding):
                        channel.append([0.0] * (in_width + 2 * self.padding))
                    batch.append(channel)
                x_padded.append(batch)
        else:
            x_padded = self.input

        # 计算梯度
        for b in range(batch_size):
            for oc in range(out_channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        h_start = oh * self.stride
                        w_start = ow * self.stride

                        # 计算权重梯度
                        for ic in range(in_channels):
                            for kh in range(self.kernel_size):
                                for kw in range(self.kernel_size):
                                    d_kernels[oc][ic][kh][kw] += (
                                            x_padded[b][ic][h_start + kh][w_start + kw] *
                                            d_output[b][oc][oh][ow]
                                    )

                        # 计算偏置梯度
                        d_bias[oc] += d_output[b][oc][oh][ow]

                        # 计算输入梯度
                        for ic in range(in_channels):
                            for kh in range(self.kernel_size):
                                for kw in range(self.kernel_size):
                                    if (0 <= h_start + kh - self.padding < in_height and
                                            0 <= w_start + kw - self.padding < in_width):
                                        d_input[b][ic][h_start + kh - self.padding][w_start + kw - self.padding] += (
                                                self.kernels[oc][ic][kh][kw] * d_output[b][oc][oh][ow]
                                        )

        # 更新参数
        for oc in range(out_channels):
            for ic in range(in_channels):
                for kh in range(self.kernel_size):
                    for kw in range(self.kernel_size):
                        self.kernels[oc][ic][kh][kw] -= learning_rate * d_kernels[oc][ic][kh][kw] / batch_size
            self.bias[oc] -= learning_rate * d_bias[oc] / batch_size

        return d_input


class MaxPool2D:
    """2D最大池化层 - 手动实现"""

    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.max_indices = []

    def forward(self, x):
        """前向传播"""
        self.input = x
        batch_size = len(x)
        channels = len(x[0])
        in_height = len(x[0][0])
        in_width = len(x[0][0][0])

        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1

        output = []
        self.max_indices = []

        for b in range(batch_size):
            batch_output = []
            batch_indices = []
            for c in range(channels):
                channel_output = []
                channel_indices = []
                for oh in range(out_height):
                    row_output = []
                    row_indices = []
                    for ow in range(out_width):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        max_val = -float('inf')
                        max_h, max_w = -1, -1
                        for kh in range(self.kernel_size):
                            for kw in range(self.kernel_size):
                                val = x[b][c][h_start + kh][w_start + kw]
                                if val > max_val:
                                    max_val = val
                                    max_h, max_w = h_start + kh, w_start + kw
                        row_output.append(max_val)
                        row_indices.append((max_h, max_w))
                    channel_output.append(row_output)
                    channel_indices.append(row_indices)
                batch_output.append(channel_output)
                batch_indices.append(channel_indices)
            output.append(batch_output)
            self.max_indices.append(batch_indices)

        return output

    def backward(self, d_output):
        """反向传播"""
        batch_size = len(self.input)
        channels = len(self.input[0])
        in_height = len(self.input[0][0])
        in_width = len(self.input[0][0][0])

        d_input = []
        for b in range(batch_size):
            batch_d = []
            for c in range(channels):
                channel_d = []
                for h in range(in_height):
                    row_d = [0.0 for _ in range(in_width)]
                    channel_d.append(row_d)
                batch_d.append(channel_d)
            d_input.append(batch_d)

        for b in range(batch_size):
            for c in range(channels):
                for oh in range(len(d_output[0][0])):
                    for ow in range(len(d_output[0][0][0])):
                        max_h, max_w = self.max_indices[b][c][oh][ow]
                        d_input[b][c][max_h][max_w] += d_output[b][c][oh][ow]

        return d_input


class Flatten:
    """展平层"""

    def forward(self, x):
        """前向传播"""
        self.input_shape = (len(x), len(x[0]), len(x[0][0]), len(x[0][0][0]))
        batch_size = len(x)

        output = []
        for b in range(batch_size):
            flattened = []
            for c in range(len(x[0])):
                for h in range(len(x[0][0])):
                    for w in range(len(x[0][0][0])):
                        flattened.append(x[b][c][h][w])
            output.append(flattened)

        return output

    def backward(self, d_output):
        """反向传播"""
        batch_size, channels, height, width = self.input_shape

        d_input = []
        for b in range(batch_size):
            batch = []
            for c in range(channels):
                channel = []
                for h in range(height):
                    row = []
                    for w in range(width):
                        idx = c * height * width + h * width + w
                        row.append(d_output[b][idx])
                    channel.append(row)
                batch.append(channel)
            d_input.append(batch)

        return d_input


class ReLU:
    """ReLU激活函数"""

    def forward(self, x):
        self.input = x
        output = []
        for batch in x:
            batch_output = []
            for channel in batch:
                channel_output = []
                for row in channel:
                    row_output = [max(0.0, val) for val in row]
                    channel_output.append(row_output)
                batch_output.append(channel_output)
            output.append(batch_output)
        return output

    def backward(self, d_output):
        d_input = []
        for b in range(len(self.input)):
            batch_input = []
            for c in range(len(self.input[0])):
                channel_input = []
                for h in range(len(self.input[0][0])):
                    row_input = []
                    for w in range(len(self.input[0][0][0])):
                        grad = d_output[b][c][h][w] if self.input[b][c][h][w] > 0 else 0.0
                        row_input.append(grad)
                    channel_input.append(row_input)
                batch_input.append(channel_input)
            d_input.append(batch_input)
        return d_input


class FullyConnected:
    """全连接层"""

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        scale = math.sqrt(2.0 / (input_size + output_size))
        self.weights = []
        for i in range(input_size):
            row = []
            for j in range(output_size):
                row.append(random.uniform(-scale, scale))
            self.weights.append(row)
        self.bias = [0.0] * output_size

    def forward(self, x):
        self.input = x
        batch_size = len(x)

        output = []
        for b in range(batch_size):
            row_output = []
            for j in range(self.output_size):
                sum_val = 0.0
                for i in range(self.input_size):
                    sum_val += x[b][i] * self.weights[i][j]
                sum_val += self.bias[j]
                row_output.append(sum_val)
            output.append(row_output)

        return output

    def backward(self, d_output, learning_rate):
        batch_size = len(self.input)

        d_input = []
        for b in range(batch_size):
            row_d = [0.0 for _ in range(self.input_size)]
            d_input.append(row_d)

        d_weights = []
        for i in range(self.input_size):
            row_d = [0.0 for _ in range(self.output_size)]
            d_weights.append(row_d)

        d_bias = [0.0] * self.output_size

        for b in range(batch_size):
            for i in range(self.input_size):
                for j in range(self.output_size):
                    d_weights[i][j] += self.input[b][i] * d_output[b][j]
                    d_input[b][i] += self.weights[i][j] * d_output[b][j]
            for j in range(self.output_size):
                d_bias[j] += d_output[b][j]

        for i in range(self.input_size):
            for j in range(self.output_size):
                self.weights[i][j] -= learning_rate * d_weights[i][j] / batch_size

        for j in range(self.output_size):
            self.bias[j] -= learning_rate * d_bias[j] / batch_size

        return d_input


class CNN:
    """完整的卷积神经网络"""

    def __init__(self):
        self.conv1 = Conv2D(1, 32, 3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2, 2)

        self.conv2 = Conv2D(32, 64, 3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2, 2)

        self.conv3 = Conv2D(64, 64, 3, padding=1)
        self.relu3 = ReLU()
        self.pool3 = MaxPool2D(2, 2)

        self.flatten = Flatten()
        self.fc = FullyConnected(576, 10)

    def forward(self, x):
        if not isinstance(x[0][0][0], list):
            x = [[x_i] for x_i in x]

        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.conv3.forward(x)
        x = self.relu3.forward(x)
        x = self.pool3.forward(x)

        x = self.flatten.forward(x)
        x = self.fc.forward(x)

        return x

    def backward(self, d_output, learning_rate):
        d_output = self.fc.backward(d_output, learning_rate)
        d_output = self.flatten.backward(d_output)
        d_output = self.pool3.backward(d_output)
        d_output = self.relu3.backward(d_output)
        d_output = self.conv3.backward(d_output, learning_rate)
        d_output = self.pool2.backward(d_output)
        d_output = self.relu2.backward(d_output)
        d_output = self.conv2.backward(d_output, learning_rate)
        d_output = self.pool1.backward(d_output)
        d_output = self.relu1.backward(d_output)
        d_output = self.conv1.backward(d_output, learning_rate)
        return d_output


def softmax(x):
    """Softmax函数"""
    result = []
    for batch in x:
        max_val = max(batch)
        exp_vals = [math.exp(val - max_val) for val in batch]
        sum_exp = sum(exp_vals)
        softmax_vals = [val / sum_exp for val in exp_vals]
        result.append(softmax_vals)
    return result


def cross_entropy_loss(y_pred, y_true):
    """交叉熵损失"""
    batch_size = len(y_pred)
    total_loss = 0.0

    for i in range(batch_size):
        for j in range(len(y_pred[0])):
            if y_pred[i][j] < 1e-8:
                y_pred[i][j] = 1e-8
            total_loss += y_true[i][j] * math.log(y_pred[i][j])

    return -total_loss / batch_size


def load_mnist_data():
    """加载MNIST数据集"""
    print("正在加载训练集图像...")
    with open('train-images-idx3-ubyte', 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]

        x_train = []
        for i in range(min(1000, num)):  # 只加载1000个样本
            img_data = f.read(rows * cols)
            img = [float(pixel) for pixel in img_data]
            x_train.append(img)

    print("正在加载训练集标签...")
    with open('train-labels-idx1-ubyte', 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num = struct.unpack('>I', f.read(4))[0]

        y_train = []
        for i in range(min(1000, num)):
            label = struct.unpack('B', f.read(1))[0]
            y_train.append(label)

    print("正在加载测试集图像...")
    with open('t10k-images-idx3-ubyte', 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num = struct.unpack('>I', f.read(4))[0]
        rows = struct.unpack('>I', f.read(4))[0]
        cols = struct.unpack('>I', f.read(4))[0]

        x_test = []
        for i in range(min(200, num)):  # 只加载200个测试样本
            img_data = f.read(rows * cols)
            img = [float(pixel) for pixel in img_data]
            x_test.append(img)

    print("正在加载测试集标签...")
    with open('t10k-labels-idx1-ubyte', 'rb') as f:
        magic = struct.unpack('>I', f.read(4))[0]
        num = struct.unpack('>I', f.read(4))[0]

        y_test = []
        for i in range(min(200, num)):
            label = struct.unpack('B', f.read(1))[0]
            y_test.append(label)

    return (x_train, y_train), (x_test, y_test)


def preprocess_data(x, y):
    """数据预处理"""
    print("数据归一化...")
    x_normalized = []
    for i, img in enumerate(x):
        normalized_img = [pixel / 255.0 for pixel in img]
        x_normalized.append(normalized_img)

    print("One-hot编码...")
    y_onehot = []
    for i, label in enumerate(y):
        onehot = [0.0] * 10
        onehot[label] = 1.0
        y_onehot.append(onehot)

    return x_normalized, y_onehot


def train_cnn():
    """训练CNN"""
    print("加载MNIST数据...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    X_train, y_train_oh = preprocess_data(x_train, y_train)
    X_test, y_test_oh = preprocess_data(x_test, y_test)

    # 重塑数据为图像格式
    X_train_reshaped = []
    for img in X_train:
        image = []
        for i in range(28):
            row = img[i * 28:(i + 1) * 28]
            image.append(row)
        X_train_reshaped.append([image])

    X_test_reshaped = []
    for img in X_test:
        image = []
        for i in range(28):
            row = img[i * 28:(i + 1) * 28]
            image.append(row)
        X_test_reshaped.append([image])

    cnn = CNN()

    epochs = 2
    batch_size = 8
    learning_rate = 0.001

    print("开始训练CNN...")
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        train_size = len(X_train_reshaped)

        for i in range(0, train_size, batch_size):
            end_idx = min(i + batch_size, train_size)
            X_batch = X_train_reshaped[i:end_idx]
            y_batch = y_train_oh[i:end_idx]

            output = cnn.forward(X_batch)
            output_softmax = softmax(output)

            loss = cross_entropy_loss(output_softmax, y_batch)
            total_loss += loss
            n_batches += 1

            batch_size_curr = len(X_batch)
            d_output = []
            for b in range(batch_size_curr):
                grad_row = []
                for j in range(10):
                    grad = output_softmax[b][j] - y_batch[b][j]
                    grad_row.append(grad)
                d_output.append(grad_row)

            cnn.backward(d_output, learning_rate)

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    print("测试CNN...")
    test_size = len(X_test_reshaped)
    correct = 0
    for i in range(test_size):
        output = cnn.forward([X_test_reshaped[i]])
        pred = output[0].index(max(output[0]))
        true_label = y_test_oh[i].index(max(y_test_oh[i]))

        if pred == true_label:
            correct += 1

    accuracy = correct / test_size
    print(f"测试准确率: {accuracy:.4f}")

    return cnn


if __name__ == "__main__":
    cnn_model = train_cnn()