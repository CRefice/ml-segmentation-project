def grid_search(params_dict, num_classes, weight_decay):
    """
    This function was used to find the best hyperparameters for the U-Net model.
    It's pretty slow and unoptimized, so don't run it if you don't have many hours of GPU time to spare.

    Arguments:
    params_dict -- a dictionary of parameters to optimize
    num_classes -- the number of classes the u-net should segment
    weight_decay -- the only non-optimized parameter
    """
    loss_list = []
    dict_list = []
    best_loss = float("inf")
    best_params = {}

    if num_classes > 2:
        criterion = combined_loss(nn.CrossEntropyLoss(weights), dice_loss)
    else:
        criterion = nn.BCEWithLogitsLoss()

    keys, values = zip(*params_dict.items())

    for v in itertools.product(*values):
        hp_dict = dict(zip(keys, v))
        print(hp_dict)

        model = unet.UNet(num_classes=NUM_CLASSES, depth=hp_dict["depth"]).to(device)
        optimizer = hp_dict["optimizer"](
            model.parameters(), lr=hp_dict["learning_rate"], weight_decay=weight_decay
        )
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=25, gamma=0.1
        )

        model, loss = train_model(
            model, optimizer, criterion, exp_lr_scheduler, num_epochs=20
        )
        loss_list.append(best_loss)
        dict_list.append(hp_dict)

        if loss < best_loss:
            ref_loss = best_loss
            best_params = hp_dict

        torch.cuda.empty_cache()

    return dict_list, loss_list
