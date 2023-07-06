python -m cli -n=1 -e=50 -b=256 -s -w --target_file=compacted_target.npy --data_format=image -a=convo_autoencoder --embedding_size=10 --dataset_path=ptb-15lead-concatted
python -m cli -n=1 -e=50 -b=256 -s -w --target_file=compacted_target.npy --data_format=image -a=fc_autoencoder --embedding_size=10 --dataset_path=ptb-15lead-concatted

REM python -m cli -n=1 -e=50 -b=256 -s -w --target_file=compacted_target.npy --data_format=image -a=convo_autoencoder --embedding_size=10 --dataset_path=ptb-12lead-fft-plot-40p-s1k\all_samples_2_diseases
REM python -m cli -n=1 -e=50 -b=256 -s -w --target_file=compacted_target.npy --data_format=image -a=convo_autoencoder --embedding_size=10 --dataset_path=ptb-12lead-fft-plot-80p-s2k\all_samples_2_diseases
REM python -m cli -n=1 -e=50 -b=256 -s -w --target_file=compacted_target.npy --data_format=image -a=convo_autoencoder --embedding_size=10 --dataset_path=ptb-12lead-fft-plot-120p-s3k\all_samples_2_diseases
REM python -m cli -n=1 -e=50 -b=256 -s -w --target_file=compacted_target.npy --data_format=image -a=convo_autoencoder --embedding_size=10 --dataset_path=ptb-12lead-fft-plot-160p-s4k\all_samples_2_diseases
REM python -m cli -n=1 -e=50 -b=256 -s -w --target_file=compacted_target.npy --data_format=image -a=convo_autoencoder --embedding_size=10 --dataset_path=ptb-12lead-fft-plot-s30k\all_samples_2_diseases

REM python -m cli -n=1 -e=50 -b=256 -s -w --target_file=compacted_target.npy --data_format=image -a=convo_autoencoder --embedding_size=10 --dataset_path=ptb-12lead-concatted/all_samples_2_diseases
python -m cli -n=1 -e=50 -b=256 -s -w --target_file=compacted_target.npy --data_format=image -a=convo_autoencoder --embedding_size=10 --dataset_path=ptb-12lead-plot/all_samples_2_diseases

REM python -m cli -n=1 -e=120 -b=45 -s -w --data_format=image -a=resnet_autoencoder_50 --embedding_size=10 --dataset_path=ptb-12lead-concatted/all_samples_2_diseases
