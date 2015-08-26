"""
Run on cluster

"""

import argparse
import os
import itertools

import networkx as nx
import pandas as pd

from . import compare_cases


def generate_run(graph, iterations, epsilon_control, epsilon_damage,
                 out_dir, nodes=None, mem=6000, runtime=120, activate=''):
    """
    Generate bash scripts for an array run in qsub/bsub cluster environments

    ``graph`` (string): can be either "regular", "scalefree", or the
    path to a GraphML file.

    ``nodes`` must be given if graph is regular or scalefree.

    Other default parameters as specified in the corresponding ``run_``
    functions in compare_cases.py are used, and cannot be overriden here.

    ``activate`` (string): additional commands to execute before calling
    sandpile (e.g. activating a virtualenv)

    """
    if graph == 'regular' or graph == 'scalefree':
        assert nodes is not None

    runs = [i for i in itertools.product(epsilon_control, epsilon_damage)]

    name = out_dir.replace("/", "_")

    df_runs = pd.DataFrame(runs, columns=['epsilon_control', 'epsilon_damage'])
    df_runs.to_csv(os.path.join(out_dir, 'iterations.csv'))

    strings = ['#!/bin/sh\ncase "$1" in\n']

    for index, run in enumerate(runs):
        e1, ed = run
        if nodes:
            nodes_string = '--nodes={}'.format(nodes)
        else:
            nodes_string = ''
        run_string = ('{idx}) {act}\n'
                      'sandpile {idx} {G} {i} {e1} {ed} {nodes}\n'
                      ';;\n'.format(idx=index + 1,
                                    G=graph, i=iterations,
                                    e1=e1, ed=ed,
                                    nodes=nodes_string,
                                    act=activate))
        strings.append(run_string)

    strings.append('esac')

    bsub_run_str = ('#!/bin/sh\n'
                    '#BSUB -J {name}[1-{to}]\n'
                    '#BSUB -R "rusage[mem={mem}]"\n'
                    '#BSUB -n 1\n'
                    '#BSUB -W {runtime}\n'
                    '#BSUB -o logs/run_%I.log\n\n'.format(name=name,
                                                          to=index + 1,
                                                          mem=mem,
                                                          runtime=runtime))

    bsub_run_str += './array_run.sh ${LSB_JOBINDEX}\n'

    qsub_run_str = ('#!/bin/sh\n'
                    '#$ -t 1-{to}\n'
                    '#$ -N {name}\n'
                    '#$ -j y -o logs/run_$TASK_ID.log\n'
                    '#$ -l mem_total={mem:.1f}G\n'
                    '#$ -cwd\n'.format(name=name, to=index + 1,
                                       mem=mem / 1000))

    qsub_run_str += './array_run.sh ${SGE_TASK_ID}\n'

    with open(os.path.join(out_dir, 'array_run.sh'), 'w') as f:
        for l in strings:
            f.write(l + '\n')

    with open(os.path.join(out_dir, 'run_bsub.sh'), 'w') as f:
        f.write(bsub_run_str + '\n')

    with open(os.path.join(out_dir, 'run_qsub.sh'), 'w') as f:
        f.write(qsub_run_str + '\n')

    with open(os.path.join(out_dir, 'prep.sh'), 'w') as f:
        f.write('chmod +x *.sh\n')
        f.write('mkdir logs\n')
        f.write('mkdir results\n')


def main():
    parser = argparse.ArgumentParser(description='Run model.')
    parser.add_argument('run_id', metavar='run_id', type=int)
    parser.add_argument('graph', metavar='graph', type=str)
    parser.add_argument('iterations', metavar='iterations', type=int)
    parser.add_argument('epsilon_control', metavar='epsilon_control', type=float)
    parser.add_argument('epsilon_damage', metavar='epsilon_damage', type=float)
    parser.add_argument('--nodes', metavar='nodes', type=int)
    args = parser.parse_args()

    if args.graph == 'regular':
        runner = compare_cases.run_regular
    elif args.graph == 'scalefree':
        runner = compare_cases.run_scalefree
    else:
        runner = compare_cases.run_on_graph
        G = nx.read_graphml(args.graph)
        G = G.to_undirected()  # Force undirected

    if runner == compare_cases.run_on_graph:
        result = runner(G=G, iterations=args.iterations,
                        epsilon_control=args.epsilon_control,
                        epsilon_damage=args.epsilon_damage)
    else:
        result = runner(nodes=args.nodes, iterations=args.iterations,
                        epsilon_control=args.epsilon_control,
                        epsilon_damage=args.epsilon_damage)
    (uncontrolled, controlled, df, costs) = result
    df.to_csv('results/cascades_{:0>4d}.csv'.format(args.run_id))
    with open('results/costs_{:0>4d}.csv'.format(args.run_id), 'w') as f:
        f.write(str(costs[0]) + '\n')
        f.write(str(costs[1]) + '\n')

if __name__ == '__main__':
    main()
