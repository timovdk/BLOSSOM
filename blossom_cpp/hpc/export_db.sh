#!/bin/bash
module load 2024
module load PostgreSQL/16.4-GCCcore-13.3.0

pg_ctl -D $HOME/pgsql/data -l $HOME/pgsql/postgres.log start

sleep 5

pg_dump -d optuna_study -F p -f study.sql

pg_ctl -D $HOME/pgsql/data stop
