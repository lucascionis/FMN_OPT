from src.fmn_opt import FMNOpt

# Instantiate the attack

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


fmn_opt = FMNOpt(
    model=model.eval().to(device),
    dataset=dataset,
    norm = 'inf',
    steps = steps,
    batch_size=40,
    batch_number=40,
    optimizer=optimizer,
    scheduler=scheduler,
    optimizer_config=optimizer_config,
    scheduler_config=scheduler_config,
    device=device
    )


fmn_opt.run()