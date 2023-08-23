
from __future__ import division, print_function
from matplotlib import pyplot as plt
plt.style.use('default')

import os
os.environ['QT_QPA_PLATFORM']='offscreen'

# receive arguments
import sys

fasta=sys.argv[1] # fasta file of positive sequences
print(fasta)
imp_score=sys.argv[2] # importance scores file
print(imp_score)
output=sys.argv[3] # output file
print(output)

from importlib import reload

###########################
#Load sequences and contribution scores
###########################

#Functions for one-hot encoding sequences
import numpy as np

def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis==0 or one_hot_axis==1
    if (one_hot_axis==0):
        assert zeros_array.shape[1] == len(sequence)
    elif (one_hot_axis==1): 
        assert zeros_array.shape[0] == len(sequence)
    #will mutate zeros_array
    for (i,char) in enumerate(sequence):
        if (char=="A" or char=="a"):
            char_idx = 0
        elif (char=="C" or char=="c"):
            char_idx = 1
        elif (char=="G" or char=="g"):
            char_idx = 2
        elif (char=="T" or char=="t"):
            char_idx = 3
        elif (char=="N" or char=="n"):
            continue #leave that pos as all 0's
        else:
            raise RuntimeError("Unsupported character: "+str(char))
        if (one_hot_axis==0):
            zeros_array[char_idx,i] = 1
        elif (one_hot_axis==1):
            zeros_array[i,char_idx] = 1

def one_hot_encode_along_channel_axis(sequence):
    to_return = np.zeros((len(sequence),4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return,
                                 sequence=sequence, one_hot_axis=1)
    return to_return

#read in the fasta files and one-hot encode
fasta_sequences = [x.rstrip() for (i,x) in enumerate(open(fasta))
              if i%2==1]
onehot_data = np.array([one_hot_encode_along_channel_axis(x)
                         for x in fasta_sequences])
print("Num onehot sequences:",len(onehot_data))


import h5py
from collections import OrderedDict

task_to_scores = OrderedDict()
task_to_hyp_scores = OrderedDict()

f = h5py.File(imp_score,"r")
tasks = f["contrib_scores"].keys()
# n = 1000 # in case this is a test run
n = len(fasta_sequences)
for task in tasks:
    #Note that the sequences can be of variable lengths;
    #in this example they all have the same length (200bp) but that is
    #not necessary.
    task_to_scores[task] = [np.array(x) for x in f['contrib_scores'][task][:n]]
    task_to_hyp_scores[task] = [np.array(x) for x in f['hyp_contrib_scores'][task][:n]]

onehot_data = [one_hot_encode_along_channel_axis(seq) for seq in fasta_sequences][:n]



###########################
#Run TF-MoDISco on the scores
###########################

print("Run TF-MoDISco on the scores")

import modisco

#### check default values
import inspect

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=10000)

tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
                        #target_seqlet_fdr controls the stringency of the threshold used.
                        # the default value is 0.2
                        target_seqlet_fdr=0.2,
                        #min_passing_windows_frac and max_passing_windows_frac can be used
                        # to manually adjust the percentile cutoffs for importance
                        # scores if you feel that the cutoff
                        # defined by the null distribution is too stringent or too
                        # lenient. The default values are 0.03 and 0.2 respectively.
                        #min_passing_windows_frac=0.03,
                        #max_passing_windows_frac=0.2
                        #The sliding window size and flanks should be adjusted according to the expected length of the core motif and its flanks.
                        #If the window size or flank sizes are too long, you risk picking up more noise.
                        sliding_window_size=15,
                        flank_size=5,
                        max_seqlets_per_metacluster=50000,
                        seqlets_to_patterns_factory=
                            modisco.tfmodisco_workflow
                                    .seqlets_to_patterns
                                    .TfModiscoSeqletsToPatternsFactory(
                                #kmer_len, num_gaps and num_mismatches are used to
                                # derive kmer embeddings for coarse-grained affinity
                                # matrix calculation. kmer_len=6, num_gaps=1
                                # and num_mismatches=0 means
                                # that kmer embeddings using 6-mers with 1 gap will be
                                # used. The default is to use longer kmers, but this
                                # can take a while to run and can lead to
                                # out-of-memory errors on some systems.
                                # Empirically, 6-mers with 1-gap
                                # seem to give good results.
                                #During the seqlet clustering, motifs are trimmed to the central trim_to_window_size bp with the highest importance
                                trim_to_window_size=15,
                                #After the trimming is done, the seqlet is expanded on either side by initial_flank_to_add
                                initial_flank_to_add=5,
                                final_min_cluster_size=30,
                        ),
                   )(
                #There is only one task, so we just call this task
                task_names=[task],
                contrib_scores=task_to_scores,
                hypothetical_contribs=task_to_hyp_scores,
                one_hot=onehot_data,
                null_per_pos_scores = null_per_pos_scores)


###########################
#Save results
###########################

import h5py
import modisco.util
reload(modisco.util)

import os
if (os.path.isfile(output + "_MoDisco_results.hdf5")):
    os.remove(str(output + "_MoDisco_results.hdf5"))

grp = h5py.File(output + "_MoDisco_results.hdf5", "w")
tfmodisco_results.save_hdf5(grp)
grp.close()


###########################
#Print results directly from hdf5
###########################

from collections import Counter
import numpy as np
import matplotlib

import sys
sys.path.append('/groups/stark/almeida/Scripts/Python/TFMoDISco/')
import bernardo_viz_sequence_looping
import matplotlib.backends.backend_pdf

from modisco.visualization import viz_sequence
reload(viz_sequence)
from matplotlib import pyplot as plt

import modisco.affinitymat.core
reload(modisco.affinitymat.core)
import modisco.cluster.phenograph.core
reload(modisco.cluster.phenograph.core)
import modisco.cluster.phenograph.cluster
reload(modisco.cluster.phenograph.cluster)
import modisco.cluster.core
reload(modisco.cluster.core)
import modisco.aggregator
reload(modisco.aggregator)

import h5py
hdf5_results = h5py.File(output + "_MoDisco_results.hdf5","r")

pp = matplotlib.backends.backend_pdf.PdfPages(output + '_MoDisco_results_clusters.pdf')

print("Metaclusters heatmap")
import seaborn as sns
activity_patterns = np.array(hdf5_results['metaclustering_results']['attribute_vectors'])[
                    np.array(
        [x[0] for x in sorted(
                enumerate(hdf5_results['metaclustering_results']['metacluster_indices']),
               key=lambda x: x[1])])]
sns.heatmap(activity_patterns, center=0)
pp.savefig()

metacluster_names = [
    x.decode("utf-8") for x in 
    list(hdf5_results["metaclustering_results"]
         ["all_metacluster_names"][:])]

all_patterns = []
background = np.array([0.25, 0.25, 0.25, 0.25]) # random

for metacluster_name in metacluster_names:
    print(metacluster_name)
    metacluster_grp = (hdf5_results["metacluster_idx_to_submetacluster_results"]
                                   [metacluster_name])
    print("activity pattern:",metacluster_grp["activity_pattern"][:])
    all_pattern_names = [x.decode("utf-8") for x in 
                         list(metacluster_grp["seqlets_to_patterns_result"]
                                             ["patterns"]["all_pattern_names"][:])]
    if (len(all_pattern_names)==0):
        print("No motifs found for this activity pattern")
    for pattern_name in all_pattern_names:
        print(metacluster_name, pattern_name)
        all_patterns.append((metacluster_name, pattern_name))
        pattern = metacluster_grp["seqlets_to_patterns_result"]["patterns"][pattern_name]
        print("total seqlets:",len(pattern["seqlets_and_alnmts"]["seqlets"]))
        print("Hypothetical scores:")
        pp.savefig(bernardo_viz_sequence_looping.pdf_weights(pattern[task+"_hypothetical_contribs"]["fwd"],
                                                            title=str(metacluster_name + " " + pattern_name + " total seqlets:" + str(len(pattern["seqlets_and_alnmts"]["seqlets"])) + " - hypothetical scores"),
                                                            subticks_frequency=5))
        print("Actual importance scores:")
        pp.savefig(bernardo_viz_sequence_looping.pdf_weights(pattern[task+"_contrib_scores"]["fwd"],
                                                            title="actual importance scores", subticks_frequency=5))
        print("onehot, fwd and rev:")
        pp.savefig(bernardo_viz_sequence_looping.pdf_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["fwd"]),
                                                        background=background),
                                                            title="onehot, fwd", subticks_frequency=5))
        pp.savefig(bernardo_viz_sequence_looping.pdf_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["rev"]),
                                                        background=background),
                                                            title="onehot, rev", subticks_frequency=5))
        #Plot the subclustering too, if available
        if ("subclusters" in pattern):
            print("PLOTTING SUBCLUSTERS")
            subclusters = np.array(pattern["subclusters"])
            twod_embedding = np.array(pattern["twod_embedding"])
            fig = plt.figure()
            plt.scatter(twod_embedding[:,0], twod_embedding[:,1], c=subclusters, cmap="tab20")
            plt.title("PLOTTING SUBCLUSTERS")
            pp.savefig()
            for subcluster_name in list(pattern["subcluster_to_subpattern"]["subcluster_names"]):
                subpattern = pattern["subcluster_to_subpattern"][subcluster_name]
                print(subcluster_name.decode("utf-8"), "size", len(subpattern["seqlets_and_alnmts"]["seqlets"]))
                subcluster = int(subcluster_name.decode("utf-8").split("_")[1])
                fig = plt.figure()
                plt.scatter(twod_embedding[:,0], twod_embedding[:,1], c=(subclusters==subcluster))
                plt.title(subcluster_name.decode("utf-8") + " size " + str(len(subpattern["seqlets_and_alnmts"]["seqlets"])))
                pp.savefig()
                pp.savefig(bernardo_viz_sequence_looping.pdf_weights(subpattern[task+"_hypothetical_contribs"]["fwd"], title="hypothetical importance scores", subticks_frequency=5))
                pp.savefig(bernardo_viz_sequence_looping.pdf_weights(subpattern[task+"_contrib_scores"]["fwd"], title="importance scores", subticks_frequency=5))
                pp.savefig(bernardo_viz_sequence_looping.pdf_weights(subpattern["sequence"]["fwd"], title="PWM", subticks_frequency=5))
        
hdf5_results.close()

pp.close()

###########################
#save TF-MoDISco motifs - PWMs
###########################

from modisco.visualization import viz_sequence
import pandas as pd

background = np.array([0.25, 0.25, 0.25, 0.25]) # random

l = []
# from original tfmodisco_results object
for i,untrimmed_motif_pattern in enumerate(tfmodisco_results
                           .metacluster_idx_to_submetacluster_results[0]
                           .seqlets_to_patterns_result.patterns):
    print(i)
    print(len(untrimmed_motif_pattern.seqlets))
    
    print("Untrimmed motif - sequence (scaled by information content)")
    viz_sequence.plot_weights(viz_sequence.ic_scale(untrimmed_motif_pattern["sequence"].fwd, background=background))

    trimmed_motif = untrimmed_motif_pattern.trim_by_ic(ppm_track_name="sequence",
                                                     background=background,
                                                     threshold=0.2)
    print("IC-trimmed motif - sequence (scaled by information content) - fwd")
    viz_sequence.plot_weights(viz_sequence.ic_scale(trimmed_motif["sequence"].fwd, background=background))
    print("IC-trimmed motif - sequence (scaled by information content) - rev")
    viz_sequence.plot_weights(viz_sequence.ic_scale(trimmed_motif["sequence"].rev, background=background))

    # append
    l.append(viz_sequence.ic_scale(trimmed_motif["sequence"].fwd, background=background))

# concatenate matrices
def func(x,i):
  PD = pd.DataFrame(x)
  PD = PD.assign(motif=pd.Series(["motif"+str(i)+"_"+str(j) for j in PD.index]))
  return PD

pd.concat([func(l[idx],idx) for idx in range(len(l))]).to_csv(output+"_MoDisco_results_new_motifs_PWMs.csv")


###########################
#save TF-MoDISco motifs - contribution scores
###########################

from modisco.visualization import viz_sequence
import pandas as pd

l_c = []
for i,pattern in enumerate(tfmodisco_results.
                           metacluster_idx_to_submetacluster_results[0].
                           seqlets_to_patterns_result.patterns):
  print(i)
  print(len(pattern.seqlets))

  print("Untrimmed motif - contrib scores")
  viz_sequence.plot_weights(pattern[task+"_contrib_scores"].fwd)

  l_c.append(pattern[task + "_contrib_scores"].fwd)

# concatenate matrices
def func(x,i):
  PD = pd.DataFrame(x)
  PD = PD.assign(motif=pd.Series(["motif"+str(i)+"_"+str(j) for j in PD.index]))
  return PD

pd.concat([func(l_c[idx],idx) for idx in range(len(l))]).to_csv(output+"_MoDisco_results_new_motifs_ContribScores.csv")

###########################
#plot TF-MoDISco motifs
###########################

print("plot TF-MoDISco motifs")

import sys
sys.path.append('/groups/stark/almeida/Scripts/Python/TFMoDISco/')
import bernardo_viz_sequence_looping
from modisco.visualization import viz_sequence

import matplotlib
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('xtick', labelsize=20)

pdf = matplotlib.backends.backend_pdf.PdfPages(output+"_MoDisco_results_motifs.pdf")

for i,untrimmed_motif_pattern in enumerate(tfmodisco_results
                           .metacluster_idx_to_submetacluster_results[0]
                           .seqlets_to_patterns_result.patterns):
    print(i)
    print(len(pattern.seqlets))
    
    pdf.savefig(bernardo_viz_sequence_looping.pdf_weights(viz_sequence.ic_scale(untrimmed_motif_pattern["sequence"].fwd, background=background),
               title=str("Motif number "+str(i+1) + "; num seqlets " + str(len(untrimmed_motif_pattern.seqlets)) + "; fwd" + "; Untrimmed motif - sequence (scaled by information content)"),
                              subticks_frequency=2))

    trimmed_motif = untrimmed_motif_pattern.trim_by_ic(ppm_track_name="sequence",
                                                     background=background,
                                                     threshold=0.2)
    pdf.savefig(bernardo_viz_sequence_looping.pdf_weights(viz_sequence.ic_scale(trimmed_motif["sequence"].fwd, background=background),
               title=str("IC-trimmed motif - sequence (scaled by information content) - fwd"),
                              subticks_frequency=2))
    pdf.savefig(bernardo_viz_sequence_looping.pdf_weights(viz_sequence.ic_scale(trimmed_motif["sequence"].rev, background=background),
               title=str("IC-trimmed motif - sequence (scaled by information content) - fwd"),
                              subticks_frequency=2))

pdf.close()
