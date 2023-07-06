python -m cli --data_format=matrix -a=stacked_lstm_autoencoder -w -s -n=3 -e=13 --embedding_size=10 --ae_objective=reconstruction --decoder_input=mixed_teacher_forcing --teacher_forcing_probability=20.0 --n_layers=2 --dataset_path=ptb-12lead-matrices-normalized/all_samples_2_diseases
python -m cli --data_format=matrix -a=stacked_lstm_autoencoder -w -s -n=3 -e=13 --embedding_size=10 --ae_objective=prediction --decoder_input=mixed_teacher_forcing --teacher_forcing_probability=20.0 --n_layers=2 --dataset_path=ptb-12lead-matrices-normalized/all_samples_2_diseases
python -m cli --data_format=matrix -a=stacked_lstm_autoencoder -w -s -n=3 -e=13 --embedding_size=10 --ae_objective=hybrid --decoder_input=mixed_teacher_forcing --teacher_forcing_probability=20.0 --n_layers=2 --dataset_path=ptb-12lead-matrices-normalized/all_samples_2_diseases