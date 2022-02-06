::python -m cli -d=ptbmat -a=stacked_lstm_autoencoder -w -s -n=3 -e=3 -f=5 --ae_objective=hybrid --decoder_input=mixed_teacher_forcing --teacher_forcing_probability=0.20 --n_layers=4 --polar_mapping --lambda=0.033

python -m cli -d=ptbmat -a=stacked_lstm_autoencoder -w -s -n=3 -e=3 -f=5 --ae_objective=hybrid --decoder_input=mixed_teacher_forcing --teacher_forcing_probability=0.20 --n_layers=4 --polar_mapping --lambda=0.01

python -m cli -d=ptbmat -a=stacked_lstm_autoencoder -w -s -n=3 -e=3 -f=5 --ae_objective=hybrid --decoder_input=mixed_teacher_forcing --teacher_forcing_probability=0.20 --n_layers=4 --polar_mapping --lambda=0.0033

python -m cli -d=ptbmat -a=stacked_lstm_autoencoder -w -s -n=3 -e=3 -f=5 --ae_objective=hybrid --decoder_input=mixed_teacher_forcing --teacher_forcing_probability=0.20 --n_layers=4 --polar_mapping --lambda=0.001