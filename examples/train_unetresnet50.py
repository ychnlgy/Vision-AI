import argparse
import torch
import tqdm
import random
import pickle
import vision_ai
from resnet34 import *
from mvordata import *


def main(args):
    device = ["cpu", "cuda"][torch.cuda.is_available()]
    DATAPATH = args.mvordata_path
    JsonFile = args.jsonfile_path

    if args.train_data and args.test_data:
        with open(args.train_data, 'rb') as f:
            train_path = pickle.load(f)
        with open(args.test_data, 'rb') as f:
            test_path = pickle.load(f)
    else:
        all_path = getImagePathFromJson(JsonFile, DATAPATH)
        train_path, test_path = splitset(all_path, args.split_percent)

    if args.save_train_data:
        with open(args.save_train_data, 'wb') as f_train:
            pickle.dump(train_path, f_train, pickle.HIGHEST_PROTOCOL)
    if args.save_test_data:
        with open(args.save_test_data, 'wb') as f_test:
            pickle.dump(test_path, f_test, pickle.HIGHEST_PROTOCOL)

    train_data = Mvordata(JsonFile, train_path)
    train_data_list, train_label_list = train_data.to_dataset()
    test_data = Mvordata(JsonFile, test_path)
    test_data_list, test_label_list = test_data.to_dataset()
    train_loader, test_loader = create_loaders(args.batch_size, args.workers, train_data_list,
                                               test_data_list, train_label_list, test_label_list)
    cpu_model = Unet_Resnet34()
    model = torch.nn.DataParallel(cpu_model).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs
    )
    print('Training Start')
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            xh = model(x)
            xh = xh.permute(0, 2, 3, 1)
            # Resizing the outputs and label to caculate pixel wise softmax loss
            outputs = xh.reshape(-1, 2)
            y = y.view(-1)
            loss = criterion(outputs, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        sched.step()

        n = 0.0
        loss = 0.0
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                xh = model(x)
                xh = xh.permute(0, 2, 3, 1)
                outputs = xh.reshape(-1, 2)
                y = y.view(-1)
                loss += criterion(outputs, y)
                n += len(x)
        print('epoch {} test loss: {}'.format(epoch, loss / n))

    save_model(cpu_model, args.save)


def save_model(model, save_path):
    print("Saving model to %s..." % save_path)
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)

    parser.add_argument("--mvordata_path", required=True)
    parser.add_argument("--jsonfile_path", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--save_train_data", required=False)
    parser.add_argument("--save_test_data", required=False)
    parser.add_argument("--train_data", required=False)
    parser.add_argument("--test_data", required=False)

    parser.add_argument("--split_percent", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    args = parser.parse_args()
    main(args)
