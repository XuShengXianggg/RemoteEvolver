import argparse
from .config import load

def main():
    parser=argparse.ArgumentParser(description='RemoteEvolver: GitHub tools + Skill evolution')
    parser.add_argument('--config',required=True)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('check-config')
    run=sub.add_parser('run')
    run.add_argument('--tasks',help='Overrides the tasks path in YAML')
    run.add_argument('--mode',choices=['train','test'],default=None)
    args=parser.parse_args()
    cfg=load(args.config)
    if args.command=='check-config':
        from .config import prepare_environment
        prepare_environment(cfg)
        print('Configuration and required credential variables are present. No API request made.')
        return
    from .runtime import run as execute
    tasks=args.tasks or cfg.get('tasks')
    if not tasks: parser.error('set tasks in YAML or pass --tasks')
    execute(cfg,tasks,mode=args.mode or cfg.get('mode','train'))

if __name__=='__main__': main()
