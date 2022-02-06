python -m cli -d=ptbmat -a=stacked_lstm_autoencoder -w -s -n=3 -e=7 -f=1 --ae_objective=hybrid --decoder_input=mixed_teacher_forcing --teacher_forcing_probability=0.0 --n_layers=4 --polar_mapping --leading_teachings=1

python -m cli -d=ptbmat -a=stacked_lstm_autoencoder -w -s -n=3 -e=7 -f=1 --ae_objective=hybrid --decoder_input=mixed_teacher_forcing --teacher_forcing_probability=0.0 --n_layers=4 --polar_mapping --leading_teachings=2

python -m cli -d=ptbmat -a=stacked_lstm_autoencoder -w -s -n=3 -e=7 -f=1 --ae_objective=hybrid --decoder_input=mixed_teacher_forcing --teacher_forcing_probability=0.0 --n_layers=4 --polar_mapping --leading_teachings=4

python -m cli -d=ptbmat -a=stacked_lstm_autoencoder -w -s -n=3 -e=7 -f=1 --ae_objective=hybrid --decoder_input=mixed_teacher_forcing --teacher_forcing_probability=0.0 --n_layers=4 --polar_mapping --leading_teachings=8
