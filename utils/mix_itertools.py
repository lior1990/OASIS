def mix_dataloaders(train_dataloader, val_dataloader):
    train_dataloader = iter(train_dataloader)
    val_dataloader = iter(val_dataloader)

    while True:
        try:
            yield next(train_dataloader), "train"
        except StopIteration:
            stop_reason = "g1"
            break

        try:
            yield next(val_dataloader), "val"
        except StopIteration:
            stop_reason = "g2"
            break

    if stop_reason == "g1":
        gen_to_exhaust = val_dataloader
        val = "val"
    else:
        gen_to_exhaust = train_dataloader
        val = "train"

    try:
        while True:
            yield next(gen_to_exhaust), val
    except StopIteration:
        pass
