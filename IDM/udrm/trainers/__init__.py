from udrm.trainers.clam_trainer import CLAMTrainer

trainer_to_cls = {
    "clam": CLAMTrainer,
    "transformer_clam": CLAMTrainer,
    "st_vivit_clam": CLAMTrainer,
    "st_vivit_clam_stm": CLAMTrainer,
    "nsvq_clam": CLAMTrainer,
    "tssm_clam": CLAMTrainer,
}
