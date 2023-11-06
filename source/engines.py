import os, sys
from libs import *

def train_fn(
    train_loaders, num_epochs, 
    model, 
    optim, 
    save_ckp_dir, 
):
    device = torch.device("cpu")

    print("\nStart Training ...\n" + " = "*16)
    model = model.to(device)

    for epoch in range(1, num_epochs + 1):
        print("epoch {}/{}".format(epoch, num_epochs) + "\n" + " - "*16)
        model.train()
        running_corrects0, running_corrects1 = 0.0, 0.0
        for images, [labels0, labels1] in tqdm.tqdm(train_loaders["train"]):
            images, [labels0, labels1] = images.to(device), [labels0.to(device), labels1.to(device)]

            logits0, logits1 = model(images.float())
            loss0, loss1 = F.cross_entropy(logits0, labels0), F.cross_entropy(logits1, labels1)

            torch.mean(torch.stack([loss0, loss1])).backward()
            for parameter in model.parameters():
                if not hasattr(parameter, "skip_allreduce") and parameter.grad is not None:
                    parameter.grad = tutel.net.simple_all_reduce(parameter.grad)
            optim.step(), optim.zero_grad()

            running_corrects0, running_corrects1 = running_corrects0 + torch.sum(torch.max(logits0, 1)[1] == labels0.data).item(), running_corrects1 + torch.sum(torch.max(logits1, 1)[1] == labels1.data).item()
        train_accuracy0, train_accuracy1 = running_corrects0/len(train_loaders["train"].dataset), running_corrects1/len(train_loaders["train"].dataset)
        print("train_accuracy0:{:.4f}, train_accuracy1:{:.4f}".format(
            train_accuracy0, train_accuracy1
        ))

    print("\nFinish Training ...\n" + " = "*16)
    return {
        "train_accuracy0":train_accuracy0, "train_accuracy1":train_accuracy1
    }