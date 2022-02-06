python -m cli -d=ptbmat -a=stacked_lstm_autoencoder -w -s -n=3 -e=13 -f=7 --ae_objective=hybrid --decoder_input=mixed_teacher_forcing --teacher_forcing_probability=1.00 --n_layers=4 --polar_mapping

python -m cli -d=ptbmat -a=stacked_lstm_autoencoder -w -s -n=3 -e=13 -f=7 --ae_objective=hybrid --decoder_input=mixed_teacher_forcing --teacher_forcing_probability=0.80 --n_layers=4 --polar_mapping

python -m cli -d=ptbmat -a=stacked_lstm_autoencoder -w -s -n=3 -e=13 -f=7 --ae_objective=hybrid --decoder_input=mixed_teacher_forcing --teacher_forcing_probability=0.60 --n_layers=4 --polar_mapping

python -m cli -d=ptbmat -a=stacked_lstm_autoencoder -w -s -n=3 -e=13 -f=7 --ae_objective=hybrid --decoder_input=mixed_teacher_forcing --teacher_forcing_probability=0.40 --n_layers=4 --polar_mapping

python -m cli -d=ptbmat -a=stacked_lstm_autoencoder -w -s -n=3 -e=13 -f=7 --ae_objective=hybrid --decoder_input=mixed_teacher_forcing --teacher_forcing_probability=0.20 --n_layers=4 --polar_mapping

python -m cli -d=ptbmat -a=stacked_lstm_autoencoder -w -s -n=3 -e=13 -f=7 --ae_objective=hybrid --decoder_input=mixed_teacher_forcing --teacher_forcing_probability=0.0 --n_layers=4 --polar_mapping
