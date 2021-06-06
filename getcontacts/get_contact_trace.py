#!/usr/bin/env python

"""
Takes a list of atomic contacts as input and generates a "trace-figure" showing
the presence / absence of one or more interactions. This can be useful for
evaluating correlations between interactions or check for consistent behavior
within simulation replicates.

The input is at least one contact-file (see get_dynamic_contacts.py) and a set
of interactions specified as space-separated regular expressions, for example
the following string indicates hydrogen bonds between a specific hydrogen donor
on a HIS to any acceptor on a GLU:
  "A:HIS:172:NE2 A:GLU:143:(OE.|O)"
Note that the first part of the expression matches only a single atom while
the second matches both OE1, OE2, and the O-atoms of residue 143. Using this
syntax it's possible to get residue-level interactions, e.g.:
  "A:PHE:86:C[B-Z][0-9]* A:VAL:68:C[B-Z][0-9]*"
will match any carbon-carbon interaction between side-chains in residues 68 and
86.

The output is a stack of trace-plots that specify time-points at which each
interaction is present or absent.

Example
======
The GetContacts example folder shows how to generate a trajectory from 5xnd
in which two hydrophobic SC-SC interaction can be traced with the following
command:
    get_contact_trace.py \\
        --input_contacts 5xnd_all-contacts.tsv \\
        --interactions "A:ILE:51:CD1 A:PHE:103:C[GDEZ].*" \\
                       "A:PHE:103.* A:PHE:48.*" \\
        --labels "ILE51 - PHE103" \\
                 "PHE48 - PHE103" \\
        --trace_output 5xnd_hp_trace.png
"""

from contact_calc.transformations import *
import contact_calc.argparsers as ap
import sys
import re


def main(argv=None):
    """
    Main function called once at the end of this module. Configures and parses command line arguments, parses input
    files and generates output files.
    """
    # Set up command line arguments
    import argparse
    # parser = ap.ArgumentParser(description=__doc__, formatter_class=ap.RawTextHelpFormatter)
    parser = ap.PrintUsageParser(description=__doc__)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    parser._action_groups.append(optional)  # added this line

    required.add_argument('--input_contacts',
                          required=True,
                          nargs='+',
                          type=argparse.FileType('r'),
                          help='A multi-frame contact-file generated by dynamic_contact.py')
    required.add_argument('--interactions',
                          required=True,
                          type=str,
                          nargs='+',
                          help='Interaction patterns, each a space-separated pair of regexes')

    optional.add_argument('--trace_output',
                          required=False,
                          type=str,
                          help='An image file to write the trace-plot to (png and svg supported)')
    optional.add_argument('--jaccard_output',
                          required=False,
                          type=str,
                          help='An image file to write the Jaccard-matrix to (png and svg supported)')
    optional.add_argument('--correlation_output',
                          required=False,
                          type=str,
                          help='An image file to write the Jaccard-matrix to (png and svg supported)')
    optional.add_argument('--labels',
                          required=False,
                          type=str,
                          nargs='+',
                          help='Interaction pattern labels. If not specified, the regexes will be used')

    args = parser.parse_args(argv)
    if args.trace_output is None and args.jaccard_output is None and args.correlation_output is None:
        parser.error("--trace_output or --jaccard_output must be specified")

    # Process arguments
    itypes = parse_itypes(['all'])
    print("Reading contacts")
    contact_lists = [parse_contacts(contact_file, itypes)[0] for contact_file in args.input_contacts]
    print("Parsing interaction patterns")
    interaction_patterns = parse_interaction_patterns(args.interactions, contact_lists)
    labels = parse_labels(args.labels, args.input_contacts, interaction_patterns)

    # Filter contacts and generate trace
    print("Filtering interactions")
    contact_frames = filter_contacts(contact_lists, interaction_patterns)

    if args.trace_output is not None:
        write_trace(contact_frames, labels, args.trace_output)

    if args.jaccard_output is not None:
        write_jaccard(contact_frames, labels, args.jaccard_output)

    if args.correlation_output is not None:
        write_correlation(contact_frames, labels, args.correlation_output)


def parse_interaction_patterns(ipatterns, contact_lists):
    ip_str_pairs = [ip.split() for ip in ipatterns]

    if any([len(ip) not in [1, 2] for ip in ip_str_pairs]):
        sys.stderr.write("Error: Interactions must be valid space-separated regular expressions\n")
        sys.exit(-1)

    re_pats = [list(map(re.compile, ip)) for ip in ip_str_pairs]
    ret = []
    for re_pat in re_pats:
        if len(re_pat) == 1:
            pat = re_pat[0]
            pat_partners = set()
            for contact_list in contact_lists:
                pat_partners |= set([c[3] for c in contact_list if pat.match(c[2])] + \
                                    [c[2] for c in contact_list if pat.match(c[3])])

            ret += [(pat, re.compile(p)) for p in pat_partners]
        if len(re_pat) == 2:
            ret.append((re_pat[0], re_pat[1]))

    # for re_pat in ret:
    #     print(re_pat[0].pattern, re_pat[1].pattern)
    return ret


def parse_labels(labels, input_files, interactions):
    if labels is not None:
        if len(labels) != len(interactions) * len(input_files):
            sys.stderr.write("Error: Only specified %d labels (should be %d) which doesn't match %d interaction "
                             "patterns across %d files\n" % (len(labels),
                                                             len(interactions) * len(input_files),
                                                             len(interactions),
                                                             len(input_files)))
            sys.exit(-1)
        return labels

    from itertools import product

    return [i[0].pattern + " - " + i[1].pattern for i, _ in product(interactions, input_files)]


def parse_itypes(itype_argument):
    """Parses the itype argument and returns a set of strings with all the selected interaction types """
    if "all" in itype_argument:
        return ["sb", "pc", "ps", "ts", "vdw", "hb", "lhb", "hbbb", "hbsb",
                "hbss", "wb", "wb2", "hls", "hlb", "lwb", "lwb2"]
    return set(itype_argument.split(","))


def filter_contacts(contact_lists, interaction_patterns):
    ret = []
    for ips in interaction_patterns:
        for contacts in contact_lists:
            ip0 = ips[0]
            ip1 = ips[1]

            ip_contact_frames = set()
            for c in contacts:
                frame = c[0]
                atom0 = c[2]
                atom1 = c[3]
                if (ip0.match(atom0) and ip1.match(atom1)) or (ip0.match(atom1) and ip1.match(atom0)):
                    ip_contact_frames.add(frame)

            ret.append(sorted(list(ip_contact_frames)))
    return ret


def write_correlation(contact_frames, labels, output_file):
    # Example adapted from https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    # print(contact_frames)

    sns.set(style="white")

    # Convert frames to pandas dataframe (rows are time, cols interactions)
    rows = max(map(max, contact_frames)) + 1
    cols = len(contact_frames)
    d = pd.DataFrame(data=np.zeros(shape=(rows, cols)), columns=labels)
    for i, contacts in enumerate(contact_frames):
        d[labels[i]][contacts] = 1

    # print(d)

    # Compute the correlation matrix
    dmat = d.corr()
    np.fill_diagonal(dmat.values, 0)
    # vmax = max(vmax, -vmin)
    # vmin = min(vmin, -vmax)
    vmax = 1
    vmin = -1
    # print(jac_sim)
    # print(vmin, vmax)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(dmat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    hm = sns.heatmap(dmat, mask=mask, cmap=cmap, vmax=vmax, vmin=vmin, center=0, square=True, linewidths=0)
    # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=, center=0, square=True, linewidths=0, cbar_kws={"shrink": .5})
    f.tight_layout()

    print("Writing correlation matrix to", output_file)
    f.savefig(output_file)


def write_jaccard(contact_frames, labels, output_file):
    # Example adapted from https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    # print(contact_frames)

    sns.set(style="white")

    # Convert frames to pandas dataframe (rows are time, cols interactions)
    rows = max(map(max, contact_frames)) + 1
    cols = len(contact_frames)
    d = pd.DataFrame(data=np.zeros(shape=(rows, cols)), columns=labels)
    for i, contacts in enumerate(contact_frames):
        d[labels[i]][contacts] = 1

    # print(d)

    # Compute the correlation matrix
    from sklearn.metrics.pairwise import pairwise_distances
    jac_sim = 1 - pairwise_distances(d.T, metric="hamming")
    jac_sim = pd.DataFrame(jac_sim, index=d.columns, columns=d.columns)
    np.fill_diagonal(jac_sim.values, 0)
    vmax = max(jac_sim.max())
    vmin = min(jac_sim.min())
    # vmax = max(vmax, -vmin)
    # vmin = min(vmin, -vmax)
    vmax = 1
    vmin = 0
    # print(jac_sim)
    # print(vmin, vmax)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(jac_sim, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    hm = sns.heatmap(jac_sim, mask=mask, cmap=cmap, vmax=vmax, vmin=vmin, center=0.5, square=True, linewidths=0)
    # sns.heatmap(corr, mask=mask, cmap=cmap, vmax=, center=0, square=True, linewidths=0, cbar_kws={"shrink": .5})
    f.tight_layout()

    print("Writing Jaccard similarity to", output_file)
    f.savefig(output_file)


def write_trace(contact_frames, labels, output_file):
    """
    Generates a trace-plot from the contact frames and writes a figure to an image file.

    Parameters
    ==========
    contact_frames: list of list of int
        Indicates all frame numbers for which a certain interaction is present
    labels: list of str
        The labels to write next to each trace
    output_file: str
        Path to an image file supported by matplotlib
    """
    assert len(contact_frames) == len(labels)

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    num_interactions = len(contact_frames)
    num_frames = max(map(max, contact_frames)) + 1
    f, axs = plt.subplots(num_interactions, sharex=True, sharey=True)

    # Do actual plotting
    for ax, contacts, label in zip(axs, contact_frames, labels):
        contact_set = set(contacts)
        x = range(num_frames)
        y = [1 if c in contact_set else 0 for c in range(num_frames)]
        ax.bar(x, y, width=1.0, linewidth=0, color="#76b8cb")
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, va='center', ha='left')
        ax.yaxis.set_label_coords(1.05, 0.5)

    plt.xlim((-0.5, num_frames - 0.5))
    plt.ylim((0, 1))
    # for ax in axs:
    #     ax.get_yaxis().set_visible(False)
    for ax in axs[:-1]:
        ax.get_xaxis().set_visible(False)
    plt.tight_layout()
    f.subplots_adjust(hspace=0)
    # plt.setp([a.get_xticklabels() for a in axs[:-1]], visible=False)
    axs[-1].xaxis.set_major_locator(MaxNLocator(integer=True))

    print("Writing trace-plot to", output_file)
    f.savefig(output_file)


if __name__ == "__main__":
    main()


__author__ = 'Rasmus Fonseca <fonseca.rasmus@gmail.com>, Jonas Kaindl <jkaindl@stanford.edu>'
__license__ = "Apache License 2.0"