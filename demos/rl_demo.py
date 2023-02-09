from argparse import ArgumentParser
from demo_discrete import run_discrete
from demo_continuous import run_continuous

SUPPORTED_DISCRETE = ["Acrobot-v1", "CartPole-v1", "MountainCar-v0"]
SUPPORTED_CONTINUOUS = ["BipedalWalker-v3", "LunarLanderContinuous-v2", "MountainCarContinuous-v0"]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--demo_type", type=str, default='discrete',
                        help="The type of RL demo to run. Available types are \'discrete\' or \'continuous\'."
                             "Default is discrete. Note: Continuous will take a long time to converge.")
    parser.add_argument('--gym_env', type=str,
                        help=f"The gym environment to use (the problem you want to show being solved). "
                             f"Supported discrete gyms: {SUPPORTED_DISCRETE}. "
                             f"Supported continuous gyms: {SUPPORTED_CONTINUOUS}.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Running with args: {args}!")
    if args.gym_env not in SUPPORTED_DISCRETE: #or args.gym_env not in SUPPORTED_CONTINUOUS:
        raise Exception(f'{args.gym_env} is not a supported gym environment.\n'
                        f'Try: {SUPPORTED_DISCRETE} for discrete\n'
                        f'or for continuous {SUPPORTED_CONTINUOUS}')
    if args.demo_type == 'discrete':
        run_discrete(args)
    elif args.demo_type == 'continuous':
        run_continuous(args)
    else:
        raise Exception(f'Demo type: {args.demo_type} does not exists. Try \'discrete\' or \'continuous\'')
