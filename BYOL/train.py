import os
import sys
path = os.getcwd()
sys.path.append(path)

import hydra
import omegaconf
import pytorch_lightning as pl
import torch


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def hydra_params_to_dotdict(hparams):
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if isinstance(v, omegaconf.DictConfig):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v

        return res

    return _to_dot_dict(hparams)


@hydra.main("config/config.yaml")
def main(cfg):
    cfg_dict = hydra_params_to_dotdict(cfg)
    model = hydra.utils.instantiate(cfg.task_model, cfg_dict)

    # early_stop_callback = pl.callbacks.EarlyStopping(patience=10)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=-1,
        filepath=os.path.join(
            cfg.task_model.name, "{epoch}-{val_loss:.4f}"
        ),
        verbose=True,
        period=2
    )
    trainer = pl.Trainer(
        gpus=list(cfg.gpus),
        max_epochs=cfg.epochs,
        #early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        distributed_backend=cfg.distrib_backend,
        accumulate_grad_batches=cfg.acc_batches,
        resume_from_checkpoint=cfg.resume_ckpt
    )

    with open(os.path.join(path, "outputs", cfg.task_model.name, "cfg.txt"), "w") as file:
        file.write(str(cfg_dict))
        file.write("\n")
    print(cfg_dict)

    trainer.fit(model)


if __name__ == "__main__":
    main()
