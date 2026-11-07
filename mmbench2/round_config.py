"""Online WAV: frozen collection dataset and target-task sampling alpha=0.75."""
from pathlib import Path
import os,json
from config import HERE,RUN as BACKGROUND
ROOT=Path(os.environ.get('MMBENCH2_ONLINE_RUN',str(HERE/'runs/online'))).expanduser().resolve()
COLLECTION=Path(os.environ.get('MMBENCH2_COLLECTION',str(HERE/'runs/collection'))).expanduser().resolve()
PART='active_wav_actioncycle_cem_v1'
VARIANTS={'wav-cem-alpha075':dict(alpha=.75)}
# Only the formal N50 collection and alpha=0.75 setting are exposed.
PHASES={'formal50':dict(episodes=50,tokenizer=50000,dynamics=30000)}

def activate(phase='formal50'):
    import config as c
    if phase not in PHASES:raise ValueError('Only the formal N50 online configuration is enabled')
    if getattr(c,'online_round_phase',None):return c
    original=c.config;spec=PHASES[phase]
    c.RUN=ROOT/phase;c.GROUP=os.environ.get('WANDB_RUN_GROUP','mmbench2-online-alpha075')
    c.SETTINGS.update({name:dict(method='wav_online_actioncycle_cem',score_horizon=32,score_d=.25) for name in VARIANTS})
    def configured(name):
        if name not in VARIANTS:raise ValueError('Online configuration must be wav-cem-alpha075')
        cfg=original(name);manifest=json.loads((c.RUN/'snapshot.json').read_text())
        cfg.update(experiment_type='online_wav_training',phase=phase,formal_paper_budget=True,
                   data_source='Real environment transitions from frozen-WM/IDM WAV action-cycle CEM',
                   pool_sources=[PART],pool_episodes_per_task=50,selected_episodes_per_task=50,
                   tokenizer_updates=50000,dynamics_updates=30000,targeted_alpha=.75,
                   snapshot_sha256=manifest['snapshot_sha256'],snapshot_path=str(c.RUN/'snapshot.json'),
                   collection_protocol_sha256=manifest['collection_protocol_sha256'],
                   initial_world_model='coverage_aware',idm_frozen=True,
                   training_data_changes_during_run=False,
                   sampling_definition='P(target task)=0.75; uniform within target and background task groups; valid-window weighting within each task.',
                   interpretation='N50 online collection, followed by tokenizer50k/dynamics30k continuation from Coverage-aware.')
        return cfg
    c.config=configured;c.online_round_phase=phase
    return c
