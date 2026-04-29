from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_NOTEBOOK = ROOT / "python_7장.ipynb"
TARGET_NOTEBOOK = ROOT / "python_7장_수정본.ipynb"


def set_source(nb: dict, index: int, source: str) -> None:
    nb["cells"][index]["source"] = source.splitlines(keepends=True)
    nb["cells"][index]["execution_count"] = None
    nb["cells"][index]["outputs"] = []


def main() -> None:
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    nb = copy.deepcopy(notebook)

    set_source(
        nb,
        0,
        "# 7.2.1 RNNCell\n"
        "# 수정본: 현재 저장소의 Python 3.10 / torchtext 환경에서 실행되도록 보정한 버전\n",
    )
    set_source(
        nb,
        6,
        "import random\n"
        "train_data, valid_data = train_data.split(random_state=random.Random(0), split_ratio=0.8)\n",
    )
    set_source(
        nb,
        11,
        "class RNNCell_Encoder(nn.Module):\n"
        "    def __init__(self, input_dim, hidden_size):\n"
        "        super(RNNCell_Encoder, self).__init__()\n"
        "        self.rnn = nn.RNNCell(input_dim, hidden_size)\n"
        "\n"
        "    def forward(self, inputs):\n"
        "        bz = inputs.shape[1]\n"
        "        ht = torch.zeros((bz, hidden_size), device=device)\n"
        "\n"
        "        for word in inputs:\n"
        "            ht = self.rnn(word, ht)\n"
        "        return ht\n"
        "\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super(Net, self).__init__()\n"
        "        self.em = nn.Embedding(len(TEXT.vocab.stoi), embeding_dim)\n"
        "        self.rnn = RNNCell_Encoder(embeding_dim, hidden_size)\n"
        "        self.fc1 = nn.Linear(hidden_size, 256)\n"
        "        self.fc2 = nn.Linear(256, 2)\n"
        "\n"
        "    def forward(self, x):\n"
        "        x = self.em(x)\n"
        "        x = self.rnn(x)\n"
        "        x = F.relu(self.fc1(x))\n"
        "        x = self.fc2(x)\n"
        "        return x\n",
    )
    set_source(
        nb,
        13,
        "def training(epoch, model, trainloader, validloader):\n"
        "    correct = 0\n"
        "    total = 0\n"
        "    running_loss = 0.0\n"
        "\n"
        "    model.train()\n"
        "    for b in trainloader:\n"
        "        x = b.text.to(device)\n"
        "        y = (b.label.to(device) - 1).long()\n"
        "        y_pred = model(x)\n"
        "        loss = loss_fn(y_pred, y)\n"
        "        optimizer.zero_grad()\n"
        "        loss.backward()\n"
        "        optimizer.step()\n"
        "\n"
        "        with torch.no_grad():\n"
        "            pred = torch.argmax(y_pred, dim=1)\n"
        "            correct += (pred == y).sum().item()\n"
        "            total += y.size(0)\n"
        "            running_loss += loss.item() * y.size(0)\n"
        "\n"
        "    epoch_loss = running_loss / len(trainloader.dataset)\n"
        "    epoch_acc = correct / total\n"
        "\n"
        "    valid_correct = 0\n"
        "    valid_total = 0\n"
        "    valid_running_loss = 0.0\n"
        "\n"
        "    model.eval()\n"
        "    with torch.no_grad():\n"
        "        for b in validloader:\n"
        "            x = b.text.to(device)\n"
        "            y = (b.label.to(device) - 1).long()\n"
        "            y_pred = model(x)\n"
        "            loss = loss_fn(y_pred, y)\n"
        "            pred = torch.argmax(y_pred, dim=1)\n"
        "            valid_correct += (pred == y).sum().item()\n"
        "            valid_total += y.size(0)\n"
        "            valid_running_loss += loss.item() * y.size(0)\n"
        "\n"
        "    epoch_valid_loss = valid_running_loss / len(validloader.dataset)\n"
        "    epoch_valid_acc = valid_correct / valid_total\n"
        "\n"
        "    print('epoch: ', epoch,\n"
        "          'loss： ', round(epoch_loss, 3),\n"
        "          'accuracy:', round(epoch_acc, 3),\n"
        "          'valid_loss： ', round(epoch_valid_loss, 3),\n"
        "          'valid_accuracy:', round(epoch_valid_acc, 3)\n"
        "          )\n"
        "    return epoch_loss, epoch_acc, epoch_valid_loss, epoch_valid_acc\n",
    )
    set_source(
        nb,
        15,
        "def evaluate(epoch, model, testloader):\n"
        "    test_correct = 0\n"
        "    test_total = 0\n"
        "    test_running_loss = 0.0\n"
        "\n"
        "    model.eval()\n"
        "    with torch.no_grad():\n"
        "        for b in testloader:\n"
        "            x = b.text.to(device)\n"
        "            y = (b.label.to(device) - 1).long()\n"
        "            y_pred = model(x)\n"
        "            loss = loss_fn(y_pred, y)\n"
        "            pred = torch.argmax(y_pred, dim=1)\n"
        "            test_correct += (pred == y).sum().item()\n"
        "            test_total += y.size(0)\n"
        "            test_running_loss += loss.item() * y.size(0)\n"
        "\n"
        "    epoch_test_loss = test_running_loss / len(testloader.dataset)\n"
        "    epoch_test_acc = test_correct / test_total\n"
        "\n"
        "    print('epoch: ', epoch,\n"
        "          'test_loss： ', round(epoch_test_loss, 3),\n"
        "          'test_accuracy:', round(epoch_test_acc, 3)\n"
        "          )\n"
        "    return epoch_test_loss, epoch_test_acc\n",
    )
    set_source(
        nb,
        22,
        "class BasicRNN(nn.Module):\n"
        "    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p = 0.2):\n"
        "        super(BasicRNN, self).__init__()\n"
        "        self.n_layers = n_layers\n"
        "        self.embed = nn.Embedding(n_vocab, embed_dim)\n"
        "        self.hidden_dim = hidden_dim\n"
        "        self.dropout = nn.Dropout(dropout_p)\n"
        "        self.rnn = nn.RNN(embed_dim, self.hidden_dim, num_layers=self.n_layers, batch_first=True)\n"
        "        self.out = nn.Linear(self.hidden_dim, n_classes)\n"
        "\n"
        "    def forward(self, x):\n"
        "        x = self.embed(x)\n"
        "        h_0 = self._init_state(batch_size=x.size(0))\n"
        "        x, _ = self.rnn(x, h_0)\n"
        "        h_t = x[:, -1, :]\n"
        "        h_t = self.dropout(h_t)\n"
        "        logit = self.out(h_t)\n"
        "        return logit\n"
        "\n"
        "    def _init_state(self, batch_size = 1):\n"
        "        weight = next(self.parameters()).data\n"
        "        return weight.new_zeros(self.n_layers, batch_size, self.hidden_dim)\n",
    )
    set_source(
        nb,
        24,
        "def train(model, optimizer, train_iter):\n"
        "    model.train()\n"
        "    for b, batch in enumerate(train_iter):\n"
        "        x = batch.text.to(device)\n"
        "        y = (batch.label.to(device) - 1).long()\n"
        "        optimizer.zero_grad()\n"
        "\n"
        "        logit = model(x)\n"
        "        loss = F.cross_entropy(logit, y)\n"
        "        loss.backward()\n"
        "        optimizer.step()\n"
        "\n"
        "        if b % 50 == 0:\n"
        "            print(\"Train Epoch: {} [{}/{} ({:.0f}%)]\\tLoss: {:.6f}\".format(e,\n"
        "                                                                           b * x.size(0),\n"
        "                                                                           len(train_iter.dataset),\n"
        "                                                                           100. * b / len(train_iter),\n"
        "                                                                           loss.item()))\n",
    )
    set_source(
        nb,
        25,
        "def evaluate(model, val_iter):\n"
        "    model.eval()\n"
        "    corrects, total, total_loss = 0, 0, 0.0\n"
        "\n"
        "    with torch.no_grad():\n"
        "        for batch in val_iter:\n"
        "            x = batch.text.to(device)\n"
        "            y = (batch.label.to(device) - 1).long()\n"
            "            logit = model(x)\n"
        "            loss = F.cross_entropy(logit, y, reduction=\"sum\")\n"
        "            total += y.size(0)\n"
        "            total_loss += loss.item()\n"
        "            corrects += (logit.argmax(dim=1) == y).sum().item()\n"
        "\n"
        "    avg_loss = total_loss / len(val_iter.dataset)\n"
        "    avg_accuracy = corrects / total\n"
        "    return avg_loss, avg_accuracy\n",
    )
    set_source(
        nb,
        28,
        "# 7.3.2 LSTMCell\n",
    )
    set_source(
        nb,
        32,
        "batch_size = 64\n"
        "train_loader = DataLoader(dataset=train_dataset,\n"
        "                         batch_size=batch_size,\n"
        "                         shuffle=True)\n"
        "valid_loader = DataLoader(dataset=valid_dataset,\n"
        "                         batch_size=batch_size,\n"
        "                         shuffle=False)\n"
        "test_loader = DataLoader(dataset=test_dataset,\n"
        "                        batch_size=batch_size,\n"
        "                        shuffle=False)\n",
    )
    set_source(
        nb,
        34,
        "class LSTMCell(nn.Module):\n"
        "    def __init__(self, input_size, hidden_size, bias=True):\n"
        "        super(LSTMCell, self).__init__()\n"
        "        self.input_size = input_size\n"
        "        self.hidden_size = hidden_size\n"
        "        self.bias = bias\n"
        "        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)\n"
        "        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)\n"
        "        self.reset_parameters()\n"
        "\n"
        "    def reset_parameters(self):\n"
        "        std = 1.0 / math.sqrt(self.hidden_size)\n"
        "        for w in self.parameters():\n"
        "            w.data.uniform_(-std, std)\n"
        "\n"
        "    def forward(self, x, hidden):\n"
        "        hx, cx = hidden\n"
        "        x = x.view(-1, x.size(1))\n"
        "\n"
        "        gates = self.x2h(x) + self.h2h(hx)\n"
        "        gates = gates.squeeze()\n"
        "        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)\n"
        "\n"
        "        ingate = torch.sigmoid(ingate)\n"
        "        forgetgate = torch.sigmoid(forgetgate)\n"
        "        cellgate = torch.tanh(cellgate)\n"
        "        outgate = torch.sigmoid(outgate)\n"
        "\n"
        "        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)\n"
        "        hy = torch.mul(outgate, torch.tanh(cy))\n"
        "        return (hy, cy)\n",
    )
    set_source(
        nb,
        35,
        "class LSTMModel(nn.Module):\n"
        "    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):\n"
        "        super(LSTMModel, self).__init__()\n"
        "        self.hidden_dim = hidden_dim\n"
        "        self.layer_dim = layer_dim\n"
        "        self.lstm = LSTMCell(input_dim, hidden_dim, bias=bias)\n"
        "        self.fc = nn.Linear(hidden_dim, output_dim)\n"
        "\n"
        "    def forward(self, x):\n"
        "        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device)\n"
        "        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device)\n"
        "\n"
        "        outs = []\n"
        "        cn = c0[0, :, :]\n"
        "        hn = h0[0, :, :]\n"
        "\n"
        "        for seq in range(x.size(1)):\n"
        "            hn, cn = self.lstm(x[:, seq, :], (hn, cn))\n"
        "            outs.append(hn)\n"
        "\n"
        "        out = outs[-1].squeeze()\n"
        "        out = self.fc(out)\n"
        "        return out\n",
    )
    set_source(
        nb,
        36,
        "input_dim = 28\n"
        "hidden_dim = 128\n"
        "layer_dim = 1\n"
        "output_dim = 10\n"
        "\n"
        "model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)\n"
        "criterion = nn.CrossEntropyLoss()\n"
        "learning_rate = 0.1\n"
        "optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)\n",
    )
    set_source(
        nb,
        37,
        "seq_dim = 28\n"
        "loss_list = []\n"
        "iter = 0\n"
        "for epoch in range(num_epochs):\n"
        "    model.train()\n"
        "    for i, (images, labels) in enumerate(train_loader):\n"
        "        images = images.view(-1, seq_dim, input_dim).to(device)\n"
        "        labels = labels.to(device)\n"
        "\n"
        "        optimizer.zero_grad()\n"
        "        outputs = model(images)\n"
        "        loss = criterion(outputs, labels)\n"
        "        loss.backward()\n"
        "        optimizer.step()\n"
        "        loss_list.append(loss.item())\n"
        "        iter += 1\n"
        "\n"
        "        if iter % 500 == 0:\n"
        "            model.eval()\n"
        "            correct = 0\n"
        "            total = 0\n"
        "            with torch.no_grad():\n"
        "                for images, labels in valid_loader:\n"
        "                    images = images.view(-1, seq_dim, input_dim).to(device)\n"
        "                    labels = labels.to(device)\n"
        "                    outputs = model(images)\n"
        "                    predicted = outputs.argmax(dim=1)\n"
        "                    total += labels.size(0)\n"
        "                    correct += (predicted == labels).sum().item()\n"
        "\n"
        "            accuracy = 100.0 * correct / total\n"
        "            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))\n",
    )
    set_source(
        nb,
        38,
        "def evaluate(model, val_iter):\n"
        "    corrects, total, total_loss = 0, 0, 0.0\n"
        "    model.eval()\n"
        "    with torch.no_grad():\n"
        "        for images, labels in val_iter:\n"
        "            images = images.view(-1, seq_dim, input_dim).to(device)\n"
        "            labels = labels.to(device)\n"
        "\n"
        "            logit = model(images)\n"
        "            loss = F.cross_entropy(logit, labels, reduction='sum')\n"
        "            predicted = logit.argmax(dim=1)\n"
        "            total += labels.size(0)\n"
        "            total_loss += loss.item()\n"
        "            corrects += (predicted == labels).sum().item()\n"
        "\n"
        "    avg_loss = total_loss / len(val_iter.dataset)\n"
        "    avg_accuracy = corrects / total\n"
        "    return avg_loss, avg_accuracy\n",
    )
    set_source(
        nb,
        48,
        "class LSTM(nn.Module):\n"
        "    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):\n"
        "        super(LSTM, self).__init__()\n"
        "        self.num_classes = num_classes\n"
        "        self.num_layers = num_layers\n"
        "        self.input_size = input_size\n"
        "        self.hidden_size = hidden_size\n"
        "        self.seq_length = seq_length\n"
        "\n"
        "        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,\n"
        "                            num_layers=num_layers, batch_first=True)\n"
        "        self.fc_1 = nn.Linear(hidden_size, 128)\n"
        "        self.fc = nn.Linear(128, num_classes)\n"
        "        self.relu = nn.ReLU()\n"
        "\n"
        "    def forward(self, x):\n"
        "        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)\n"
        "        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)\n"
        "\n"
        "        output, (hn, cn) = self.lstm(x, (h_0, c_0))\n"
        "        hn = hn.view(-1, self.hidden_size)\n"
        "        out = self.relu(hn)\n"
        "        out = self.fc_1(out)\n"
        "        out = self.relu(out)\n"
        "        out = self.fc(out)\n"
        "        return out\n",
    )
    set_source(
        nb,
        49,
        "num_epochs = 1000\n"
        "learning_rate = 0.0001\n"
        "\n"
        "input_size = 5\n"
        "hidden_size = 2\n"
        "num_layers = 1\n"
        "\n"
        "num_classes = 1\n"
        "model = LSTM(num_classes, input_size, hidden_size, num_layers, X_train_tensors_f.shape[1]).to(device)\n"
        "\n"
        "criterion = torch.nn.MSELoss()\n"
        "optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)\n",
    )
    set_source(
        nb,
        50,
        "for epoch in range(num_epochs):\n"
        "    model.train()\n"
        "    outputs = model(X_train_tensors_f.to(device))\n"
        "    optimizer.zero_grad()\n"
        "    loss = criterion(outputs, y_train_tensors.to(device))\n"
        "    loss.backward()\n"
        "    optimizer.step()\n"
        "    if epoch % 100 == 0:\n"
        "        print(\"Epoch: %d, loss: %1.5f\" % (epoch, loss.item()))\n",
    )
    set_source(
        nb,
        52,
        "model.eval()\n"
        "with torch.no_grad():\n"
        "    train_predict = model(df_x_ss.to(device))\n"
        "\n"
        "predicted = train_predict.detach().cpu().numpy()\n"
        "label_y = df_y_ms.detach().cpu().numpy()\n"
        "\n"
        "predicted = ms.inverse_transform(predicted)\n"
        "label_y = ms.inverse_transform(label_y)\n"
        "plt.figure(figsize=(10,6))\n"
        "plt.axvline(x=200, c='r', linestyle='--')\n"
        "\n"
        "plt.plot(label_y, label='Actual Data')\n"
        "plt.plot(predicted, label='Predicted Data')\n"
        "plt.title('Time-Series Prediction')\n"
        "plt.legend()\n"
        "plt.show()\n",
    )
    set_source(
        nb,
        53,
        "# 7.4.2 GRUCell\n",
    )
    set_source(
        nb,
        57,
        "batch_size = 64\n"
        "train_loader = DataLoader(dataset=train_dataset,\n"
        "                         batch_size=batch_size,\n"
        "                         shuffle=True)\n"
        "valid_loader = DataLoader(dataset=valid_dataset,\n"
        "                         batch_size=batch_size,\n"
        "                         shuffle=False)\n"
        "test_loader = DataLoader(dataset=test_dataset,\n"
        "                        batch_size=batch_size,\n"
        "                        shuffle=False)\n",
    )
    set_source(
        nb,
        59,
        "class GRUCell(nn.Module):\n"
        "    def __init__(self, input_size, hidden_size, bias=True):\n"
        "        super(GRUCell, self).__init__()\n"
        "        self.input_size = input_size\n"
        "        self.hidden_size = hidden_size\n"
        "        self.bias = bias\n"
        "        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)\n"
        "        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)\n"
        "        self.reset_parameters()\n"
        "\n"
        "    def reset_parameters(self):\n"
        "        std = 1.0 / math.sqrt(self.hidden_size)\n"
        "        for w in self.parameters():\n"
        "            w.data.uniform_(-std, std)\n"
        "\n"
        "    def forward(self, x, hidden):\n"
        "        x = x.view(-1, x.size(1))\n"
        "\n"
        "        gate_x = self.x2h(x)\n"
        "        gate_h = self.h2h(hidden)\n"
        "\n"
        "        gate_x = gate_x.squeeze()\n"
        "        gate_h = gate_h.squeeze()\n"
        "\n"
        "        i_r, i_i, i_n = gate_x.chunk(3, 1)\n"
        "        h_r, h_i, h_n = gate_h.chunk(3, 1)\n"
        "\n"
        "        resetgate = torch.sigmoid(i_r + h_r)\n"
        "        inputgate = torch.sigmoid(i_i + h_i)\n"
        "        newgate = torch.tanh(i_n + (resetgate * h_n))\n"
        "\n"
        "        hy = newgate + inputgate * (hidden - newgate)\n"
        "        return hy\n",
    )
    set_source(
        nb,
        60,
        "class GRUModel(nn.Module):\n"
        "    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):\n"
        "        super(GRUModel, self).__init__()\n"
        "        self.hidden_dim = hidden_dim\n"
        "        self.layer_dim = layer_dim\n"
        "        self.gru_cell = GRUCell(input_dim, hidden_dim, bias=bias)\n"
        "        self.fc = nn.Linear(hidden_dim, output_dim)\n"
        "\n"
        "    def forward(self, x):\n"
        "        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device)\n"
        "\n"
        "        outs = []\n"
        "        hn = h0[0, :, :]\n"
        "\n"
        "        for seq in range(x.size(1)):\n"
        "            hn = self.gru_cell(x[:, seq, :], hn)\n"
        "            outs.append(hn)\n"
        "\n"
        "        out = outs[-1].squeeze()\n"
        "        out = self.fc(out)\n"
        "        return out\n",
    )
    set_source(
        nb,
        61,
        "input_dim = 28\n"
        "hidden_dim = 128\n"
        "layer_dim = 1\n"
        "output_dim = 10\n"
        "\n"
        "model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)\n"
        "\n"
        "criterion = nn.CrossEntropyLoss()\n"
        "learning_rate = 0.1\n"
        "optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)\n",
    )
    set_source(
        nb,
        62,
        "seq_dim = 28\n"
        "loss_list = []\n"
        "iter = 0\n"
        "for epoch in range(num_epochs):\n"
        "    model.train()\n"
        "    for i, (images, labels) in enumerate(train_loader):\n"
        "        images = images.view(-1, seq_dim, input_dim).to(device)\n"
        "        labels = labels.to(device)\n"
        "\n"
        "        optimizer.zero_grad()\n"
        "        outputs = model(images)\n"
        "        loss = criterion(outputs, labels)\n"
        "        loss.backward()\n"
        "        optimizer.step()\n"
        "\n"
        "        loss_list.append(loss.item())\n"
        "        iter += 1\n"
        "\n"
        "        if iter % 500 == 0:\n"
        "            model.eval()\n"
        "            correct = 0\n"
        "            total = 0\n"
        "            with torch.no_grad():\n"
        "                for images, labels in valid_loader:\n"
        "                    images = images.view(-1, seq_dim, input_dim).to(device)\n"
        "                    labels = labels.to(device)\n"
        "                    outputs = model(images)\n"
        "                    predicted = outputs.argmax(dim=1)\n"
        "                    total += labels.size(0)\n"
        "                    correct += (predicted == labels).sum().item()\n"
        "\n"
        "            accuracy = 100.0 * correct / total\n"
        "            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))\n",
    )
    set_source(
        nb,
        63,
        "def evaluate(model, val_iter):\n"
        "    corrects, total, total_loss = 0, 0, 0.0\n"
        "    model.eval()\n"
        "    with torch.no_grad():\n"
        "        for images, labels in val_iter:\n"
        "            images = images.view(-1, seq_dim, input_dim).to(device)\n"
        "            labels = labels.to(device)\n"
        "\n"
        "            logit = model(images)\n"
        "            loss = F.cross_entropy(logit, labels, reduction='sum')\n"
        "            predicted = logit.argmax(dim=1)\n"
        "            total += labels.size(0)\n"
        "            total_loss += loss.item()\n"
        "            corrects += (predicted == labels).sum().item()\n"
        "\n"
        "    avg_loss = total_loss / len(val_iter.dataset)\n"
        "    avg_accuracy = corrects / total\n"
        "    return avg_loss, avg_accuracy\n",
    )
    set_source(
        nb,
        65,
        "# 7.4.3 GRU\n",
    )
    set_source(
        nb,
        72,
        "class GRU(nn.Module):\n"
        "    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):\n"
        "        super(GRU, self).__init__()\n"
        "        self.num_classes = num_classes\n"
        "        self.num_layers = num_layers\n"
        "        self.input_size = input_size\n"
        "        self.hidden_size = hidden_size\n"
        "        self.seq_length = seq_length\n"
        "\n"
        "        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,\n"
        "                          num_layers=num_layers, batch_first=True)\n"
        "        self.fc_1 = nn.Linear(hidden_size, 128)\n"
        "        self.fc = nn.Linear(128, num_classes)\n"
        "        self.relu = nn.ReLU()\n"
        "\n"
        "    def forward(self, x):\n"
        "        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)\n"
        "        output, hn = self.gru(x, h_0)\n"
        "        hn = hn.view(-1, self.hidden_size)\n"
        "        out = self.relu(hn)\n"
        "        out = self.fc_1(out)\n"
        "        out = self.relu(out)\n"
        "        out = self.fc(out)\n"
        "        return out\n",
    )
    set_source(
        nb,
        73,
        "num_epochs = 1000\n"
        "learning_rate = 0.0001\n"
        "\n"
        "input_size = 5\n"
        "hidden_size = 2\n"
        "num_layers = 1\n"
        "\n"
        "num_classes = 1\n"
        "model = GRU(num_classes, input_size, hidden_size, num_layers, X_train_tensors_f.shape[1]).to(device)\n"
        "\n"
        "criterion = torch.nn.MSELoss()\n"
        "optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)\n",
    )
    set_source(
        nb,
        74,
        "for epoch in range(num_epochs):\n"
        "    model.train()\n"
        "    outputs = model(X_train_tensors_f.to(device))\n"
        "    optimizer.zero_grad()\n"
        "    loss = criterion(outputs, y_train_tensors.to(device))\n"
        "    loss.backward()\n"
        "    optimizer.step()\n"
        "    if epoch % 100 == 0:\n"
        "        print(\"Epoch: %d, loss: %1.5f\" % (epoch, loss.item()))\n",
    )
    set_source(
        nb,
        76,
        "model.eval()\n"
        "with torch.no_grad():\n"
        "    train_predict = model(df_x_ss.to(device))\n"
        "\n"
        "predicted = train_predict.detach().cpu().numpy()\n"
        "label_y = df_y_ms.detach().cpu().numpy()\n"
        "\n"
        "predicted = ms.inverse_transform(predicted)\n"
        "label_y = ms.inverse_transform(label_y)\n"
        "plt.figure(figsize=(10,6))\n"
        "plt.axvline(x=200, c='r', linestyle='--')\n"
        "\n"
        "plt.plot(label_y, label='Actual Data')\n"
        "plt.plot(predicted, label='Predicted Data')\n"
        "plt.title('Time-Series Prediction')\n"
        "plt.legend()\n"
        "plt.show()\n",
    )
    set_source(
        nb,
        85,
        "class biLSTM(nn.Module):\n"
        "    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):\n"
        "        super(biLSTM, self).__init__()\n"
        "        self.num_classes = num_classes\n"
        "        self.num_layers = num_layers\n"
        "        self.input_size = input_size\n"
        "        self.hidden_size = hidden_size\n"
        "        self.seq_length = seq_length\n"
        "\n"
        "        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,\n"
        "                            num_layers=num_layers, bidirectional=True, batch_first=True)\n"
        "        self.fc = nn.Linear(hidden_size * 2, num_classes)\n"
        "        self.relu = nn.ReLU()\n"
        "\n"
        "    def forward(self, x):\n"
        "        h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)\n"
        "        c_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)\n"
        "        out, _ = self.lstm(x, (h_0, c_0))\n"
        "        out = self.fc(out[:, -1, :])\n"
        "        out = self.relu(out)\n"
        "        return out\n",
    )
    set_source(
        nb,
        86,
        "num_epochs = 1000\n"
        "learning_rate = 0.0001\n"
        "\n"
        "input_size = 5\n"
        "hidden_size = 2\n"
        "num_layers = 1\n"
        "\n"
        "num_classes = 1\n"
        "model = biLSTM(num_classes, input_size, hidden_size, num_layers, X_train_tensors_f.shape[1]).to(device)\n"
        "\n"
        "criterion = torch.nn.MSELoss()\n"
        "optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)\n",
    )
    set_source(
        nb,
        87,
        "for epoch in range(num_epochs):\n"
        "    model.train()\n"
        "    outputs = model(X_train_tensors_f.to(device))\n"
        "    optimizer.zero_grad()\n"
        "\n"
        "    loss = criterion(outputs, y_train_tensors.to(device))\n"
        "    loss.backward()\n"
        "    optimizer.step()\n"
        "    if epoch % 100 == 0:\n"
        "        print(\"Epoch: %d, loss: %1.5f\" % (epoch, loss.item()))\n",
    )
    set_source(
        nb,
        89,
        "model.eval()\n"
        "with torch.no_grad():\n"
        "    train_predict = model(df_x_ss.to(device))\n"
        "\n"
        "predicted = train_predict.detach().cpu().numpy()\n"
        "label_y = df_y_ms.detach().cpu().numpy()\n"
        "\n"
        "predicted = ms.inverse_transform(predicted)\n"
        "label_y = ms.inverse_transform(label_y)\n"
        "plt.figure(figsize=(10,6))\n"
        "plt.axvline(x=200, c='r', linestyle='--')\n"
        "\n"
        "plt.plot(label_y, label='Actual Data')\n"
        "plt.plot(predicted, label='Predicted Data')\n"
        "plt.title('Time-Series Prediction')\n"
        "plt.legend()\n"
        "plt.show()\n",
    )

    TARGET_NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(TARGET_NOTEBOOK)


if __name__ == "__main__":
    main()
