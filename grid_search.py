def grid_search(params_dict, CLASS_NUMBER, WEIGHT_DECAY):
    loss_list = []
    dict_list = []
    best_loss = float("inf")
    best_params = {}

    if CLASS_NUMBER > 2:
        criterion = combined_loss(nn.CrossEntropyLoss(weights), dice_loss)
    else:
        criterion = nn.BCEWithLogitsLoss()

    keys, values = zip(*params_dict.items())

    for v in itertools.product(*values):
        hp_dict = dict(zip(keys, v))
        print(hp_dict)

        model = unet.UNet(num_classes=NUM_CLASSES, depth=hp_dict["depth"]).to(device)
        optimizer = hp_dict["optimizer"](
            model.parameters(), lr=hp_dict["learning_rate"], weight_decay=WEIGHT_DECAY
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
