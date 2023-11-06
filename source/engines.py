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
    best_running_loss = np.inf

    for epoch in range(1, num_epochs + 1):
        print("epoch {}/{}".format(epoch, num_epochs) + "\n" + " - "*16)
        model.train()
        running_loss = 0.0
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

            running_loss = running_loss + torch.mean(torch.stack([loss0, loss1])).item()*images.size(0)
            running_corrects0, running_corrects1 = running_corrects0 + torch.sum(torch.max(logits0, 1)[1] == labels0.data).item(), running_corrects1 + torch.sum(torch.max(logits1, 1)[1] == labels1.data).item()
        print("train_accuracy0:{:.4f}, train_accuracy1:{:.4f}".format(
            running_corrects0/len(train_loaders["train"].dataset), running_corrects1/len(train_loaders["train"].dataset)
        ))

        with torch.no_grad():
            model.eval()
            running_loss = 0.0
            running_corrects0, running_corrects1 = 0.0, 0.0
            for images, [labels0, labels1] in tqdm.tqdm(train_loaders["val"]):
                images, [labels0, labels1] = images.to(device), [labels0.to(device), labels1.to(device)]

                logits0, logits1 = model(images.float())
                loss0, loss1 = F.cross_entropy(logits0, labels0), F.cross_entropy(logits1, labels1)

                running_loss = running_loss + torch.mean(torch.stack([loss0, loss1])).item()*images.size(0)
                running_corrects0, running_corrects1 = running_corrects0 + torch.sum(torch.max(logits0, 1)[1] == labels0.data).item(), running_corrects1 + torch.sum(torch.max(logits1, 1)[1] == labels1.data).item()
        print("val_accuracy0:{:.4f}, val_accuracy1:{:.4f}".format(
            running_corrects0/len(train_loaders["val"].dataset), running_corrects1/len(train_loaders["val"].dataset)
        ))
        if running_loss/len(train_loaders["val"].dataset) < best_running_loss:
            torch.save(
                model.state_dict(), 
                "{}/best.ptl".format(save_ckp_dir), 
            )
            best_running_loss = running_loss/len(train_loaders["val"].dataset)

    print("\nFinish Training ...\n" + " = "*16)