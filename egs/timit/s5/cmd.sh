# you can change cmd.sh depending on what type of queue you are using.
# If you have no queueing system and want to run on a local machine, you
# can change all instances 'queue.pl' to run.pl (but be careful and run
# commands one by one: most recipes will exhaust the memory on your
# machine).  queue.pl works with GridEngine (qsub).  slurm.pl works
# with slurm.  Different queues are configured differently, with different
# queue names and different ways of specifying things like memory;
# to account for these differences you can create and edit the file
# conf/queue.conf to match your queue's configuration.  Search for
# conf/queue.conf in http://kaldi-asr.org/doc/queue.html for more information,
# or search for the string 'default_config' in utils/queue.pl or utils/slurm.pl.

export train_cmd="run.pl --mem 4G"
export decode_cmd="run.pl --mem 4G"
export mkgraph_cmd="run.pl --mem 8G"
# the use of cuda_cmd is deprecated but it's still sometimes used in nnet1
# example scripts.
export cuda_cmd="run.pl --gpu 1"

# the rest of this file is present for historical reasons.
# for cluster-specific configuration it's better to rely on conf/queue.conf.
if [ "$(hostname -d)" == "fit.vutbr.cz" ]; then
  #b) BUT cluster options
  queue="all.q@@blade,all.q@@speech"
  gpu_queue="long.q@@gpu"
  storage="matylda5"
  export train_cmd="run.pl -q $queue -l ram_free=1.5G,mem_free=1.5G,${storage}=0.5"
  export decode_cmd="run.pl -q $queue -l ram_free=2.5G,mem_free=2.5G,${storage}=0.1"
  export cuda_cmd="run.pl -q $gpu_queue -l gpu=1"
fi

