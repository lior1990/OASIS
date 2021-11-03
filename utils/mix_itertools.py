def mix_dataloaders(orig_train_dataloader, val_dataloader):
    train_dataloader = iter(orig_train_dataloader)
    val_dataloader = iter(val_dataloader)

    while True:
        try:
            yield next(train_dataloader), "train"
        except StopIteration:
            train_dataloader = iter(orig_train_dataloader)
            continue

        try:
            yield next(val_dataloader), "val"
        except StopIteration:
            break
