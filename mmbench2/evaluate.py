"""Run reconstruction/rollout/action metrics or full-episode MPC."""
import argparse,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',required=True)
    p.add_argument('--kind',choices=['offline','mpc'],required=True)
    p.add_argument('--task',default='all',help='One task name or all 20 tasks')
    p.add_argument('--online',action='store_true')
    p.add_argument('--episode-index',type=int,choices=range(3),help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.online:
        from round_config import activate
        activate()
    import config as c
    tasks=c.TARGETS if a.task=='all' else [a.task]
    from checkpoint_identity import fingerprints,bind
    modelroot=c.MODELS if a.variant in ['base','coverage_aware','combined'] else c.RUN/'models'
    identity=fingerprints(modelroot/a.variant)
    bind(c.RUN/'evaluation',a.variant,identity)
    for task in tasks:
        if task not in c.TARGETS:raise ValueError('Task must belong to the 10 seen + 10 unseen set')
        if a.kind=='mpc' and a.episode_index is None:
            from mpc_episodes import run_isolated
            out=c.RUN/'evaluation/mpc'/f'{a.variant}.{task}.json'
            command=[sys.executable,str(Path(__file__).resolve()),'--variant',a.variant,'--task',task,'--kind','mpc']
            if a.online:command+=['--online']
            run_isolated(task,a.variant,out,lambda ep:subprocess.run(command+['--episode-index',str(ep)],check=True),checkpoint_fingerprints=identity)
            continue
        if a.kind=='mpc':
            from mpc_renderer_runtime import bootstrap,install_task_overrides
            bootstrap(task);install_task_overrides(task)
        import common
        common.OUT=c.RUN/'evaluation'
        modelroot=c.MODELS if a.variant in ['base','coverage_aware','combined'] else c.RUN/'models'
        if a.kind=='offline':
            import offline_eval as ev
            ev.OUT=common.OUT;ev.MODELS=modelroot
            common.prep_eval(task)
            ev.evaluate(task,a.variant)
        else:
            import mpc_eval as ev
            ev.OUT=common.OUT;ev.MODELS=modelroot
            ev.run(task,a.variant,episodes=3,episode_indices=[a.episode_index],
                   output_name=f'mpc-episode-cache/{a.variant}.{task}.ep{a.episode_index}.json')

if __name__=='__main__':main()
