import os
import sys
import math
import pprint

import torch

from torchdrug import core
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import dataset, layer, model, task, util
import neptune.new as neptune

run = neptune.init(
    project="R-NBFNet/Context-NBF",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzNmY3MWNhOS03YjJmLTQzZDUtYmFiMi1jMWRmMThjNDc5ZWMifQ==",
)  # your credentials

def train_and_validate(cfg, solver):
    if cfg.train.num_epoch == 0:
        return

    step = math.ceil(cfg.train.num_epoch / 10)
    best_result = float("-inf")
    best_epoch = -1

    for i in range(0, cfg.train.num_epoch, step):
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.model.split = "train"
        solver.train(**kwargs)
        solver.save("model_epoch_%d.pth" % solver.epoch)
        solver.model.split = "valid"
        metric = solver.evaluate("valid")
        result = metric[cfg.metric]
        for key in metric.keys():
            run[f"test/{key}"] = metric[key]

        if result > best_result:
            best_result = result
            best_epoch = solver.epoch

    solver.load("model_epoch_%d.pth" % best_epoch)
    return solver


def test(cfg, solver):
    solver.model.split = "valid"
    metric = solver.evaluate("valid")
    result = metric[cfg.metric]
    for key in metric.keys():
      run[f"valid/{key}"] = metric[key]
    solver.model.split = "test"
    metric = solver.evaluate("test")
    result = metric[cfg.metric]
    for key in metric.keys():
      run[f"test/{key}"] = metric[key]



if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)
    task = cfg.dataset["class"]
    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)
    # --------
    print(cfg)
    params = {}

    for key in ["dataset","engine","optimizer","task","train"]:
      for k in cfg[key].keys():
        params[f"{key}/{k}"] = cfg[key][k]
    params['dataset/class'] = task
    params['metric'] = cfg['metric']
    run["params"] = params

    # --------
    train_and_validate(cfg, solver)
    test(cfg, solver)
    run.stop()