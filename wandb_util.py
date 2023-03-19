import os
import wandb


def get_run_id(name, author='noam-fluss', project='fixmatch_debug'):
    # if project == "fixmatch_missing_classes_labels_confidence" or\
    #         project == "fixmatch_long_tail_confidence" or \
    #         project == "test" or project == "Flexmatch_first_project" or project == "Flexmatch_second_project":
    author = "noam_fluss_mail"
    api = wandb.Api()
    runs = api.runs(f"{author}/{project}",
                    {"$or": [
                        # {"config.experiment_name": "foo"},
                        {"config.name": name}]
                    })

    if len(runs) == 1:
        return runs[0].id
    return False


def init_wandb(args):
    # TODO check wandb: ERROR Failed to serialize metric: division by zero
    config = vars(args)
    name = str(args.slurm_job_id)
    print("wandb name",name)
    config['name'] = name
    config["wandb_run_id"] = None
    warmup_id = get_run_id(name, project=args.project_wandb)
    print("wandb_id",warmup_id)
    if args.project_wandb == "fixmatch_long_tail_cifar10" or args.project_wandb == "fixmatch_missing_classes_labels":
        run = wandb.init(project=args.project_wandb, entity='noam-fluss', config=config, name=name, resume=warmup_id)
    else:
        run = wandb.init(project=args.project_wandb, entity='noam_fluss_mail', config=config, name=name,
                         resume=warmup_id)
    print("wandb.run.id", wandb.run.id)
    wandb.config.update({"wandb_run_id": wandb.run.id}, allow_val_change=True)
    print("!!!")
    #wandb.env.CACHE_DIR = r"/cs/labs/daphna/noam.fluss/project/wandb_cache/"
    os.environ['WANDB_CACHE_DIR'] = r"/cs/labs/daphna/noam.fluss/project/wandb_cache/"

def finish_wandb():
    wandb.log({"finish": 1})
    wandb.finish()
