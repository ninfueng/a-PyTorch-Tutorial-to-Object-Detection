import argparse
import time
from multiprocessing import cpu_count
from pprint import PrettyPrinter

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tqdm import tqdm

import wandb
from datasets import PascalVOCDataset
from model import SSD300, MultiBoxLoss
from quant import cvt2quant
from utils import *

parser = argparse.ArgumentParser(description="mixed-ssd300.")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--test_batch_size", type=int, default=64)
parser.add_argument("--iterations", type=int, default=120_000)
parser.add_argument("--weight-decay", type=float, default=5e-4)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--workers", type=int, default=min(cpu_count(), 20))
parser.add_argument("--project-name", type=str, default="ssd300")
# parser.add_argument("--ws", type=str, default="ttttttttttttttttttttttttttttttttttt")
# parser.add_argument("--ws", type=str, default="fffffffffffffffffffffffffffffffffff")
parser.add_argument("--ws", type=str, default="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
parser.add_argument("--wandb", dest="wandb", action="store_true")
parser.set_defaults(wandb=False)
args = parser.parse_args()

assert len(args.ws) == 35
data_folder = "./"  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = None  # path to model checkpoint, None if none
batch_size = args.batch_size  # batch size
iterations = args.iterations  # number of iterations to train
lr = args.lr  # learning rate
weight_decay = args.weight_decay  # weight decay

workers = args.workers  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9
grad_clip = 1.0
# grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation
cudnn.benchmark = True

hparams = {
    "batch_size": batch_size,
    "iterations": iterations,
    "lr": lr,
    "weight_decay": weight_decay,
}
hparams2 = {f"{ss}{idx}": ss for idx, ss in enumerate(args.ws)}
hparams.update(hparams2)
if args.wandb:
    wandb.init(
        project="mixed-ssd300",
        tags=["object-detection", "quantization"],
        config=hparams,
    )
pp = PrettyPrinter()

class DataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def main():
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at
    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        cvt2quant(model, args.ws)
#        model.base.load_pretrained_layers()
#        model.pred_convs.init_conv2d()
#        model.aux_convs.init_conv2d()
        print(model)

        if args.wandb:
            wandb.watch(model)

        model = DataParallel(model)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        # TODO: checking that model parameters problem?
#        for param_name, param in model.named_parameters():
#            if param.requires_grad:
#                if param_name.endswith(".bias"):
#                    biases.append(param)
#                else:
#                    not_biases.append(param)

#        optimizer = torch.optim.SGD(
#            params=[{"params": biases, "lr": 2 * lr}, {"params": not_biases}],
#            lr=lr,
#            momentum=momentum,
#            weight_decay=weight_decay,
#        )
       
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        print("\nLoaded checkpoint from epoch %d.\n" % start_epoch)
        model = checkpoint["model"]
        optimizer = checkpoint["optimizer"]

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)
    train_dataset = PascalVOCDataset(
        data_folder, split="train", keep_difficult=keep_difficult
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=workers,
        pin_memory=True,
    )  # note that we're passing the collate function here
    test_dataset = PascalVOCDataset(
        data_folder, split="test", keep_difficult=keep_difficult
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=workers,
        pin_memory=True,
    )
    epochs = iterations // (len(train_dataset) // 32)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    for epoch in range(start_epoch, epochs):
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        train(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
        )
        if epoch != 0 and epoch % 10 == 0:
            mAP = evaluate(test_loader, model)
            try:
                save_checkpoint(epoch, model, optimizer)
            except AttributeError:
                pass
    print(mAP)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()
    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        predicted_locs, predicted_scores = model(
            images
        )  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()
        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()
        # Print status
        if i % print_freq == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                )
            )


#    del (
#        predicted_locs,
#        predicted_scores,
#        images,
#        boxes,
#        labels,
#    )  # free some memory since their histories may be stored
#


def evaluate(test_loader, model):
    model.eval()
    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = (
        list()
    )  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        for i, (images, boxes, labels, difficulties) in enumerate(
            tqdm(test_loader, desc="Evaluating")
        ):
            images = images.to(device)  # (N, 3, 300, 300)
            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            (
                det_boxes_batch,
                det_labels_batch,
                det_scores_batch,
            ) = model.detect_objects(
                predicted_locs,
                predicted_scores,
                min_score=0.01,
                max_overlap=0.45,
                top_k=200,
            )
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

        # Calculate mAP
        APs, mAP = calculate_mAP(
            det_boxes,
            det_labels,
            det_scores,
            true_boxes,
            true_labels,
            true_difficulties,
        )
    # Print AP for each class
    pp.pprint(APs)
    print("\nMean Average Precision (mAP): %.3f" % mAP)
    # To return to hyper tunner script.
    print(mAP)
    return mAP


if __name__ == "__main__":
    import sys
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")
    main()

