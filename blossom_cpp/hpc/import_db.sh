#!/bin/bash
scp tvanderkuil@snellius.surf.nl:./study.sql ~/Documents/

psql -U $(whoami) -d postgres -c "DROP DATABASE IF EXISTS optuna_study;"
psql -U $(whoami) -d postgres -c "CREATE DATABASE optuna_study;"
psql -U $(whoami)  -d optuna_study -f $HOME/Documents/study.sql
optuna-dashboard postgresql:///optuna_study