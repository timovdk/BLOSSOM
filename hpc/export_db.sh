#!/bin/bash
module load 2025 PostgreSQL/17.5-GCCcore-14.2.0

pg_ctl -D $HOME/pgsql/data -l $HOME/pgsql/postgres.log start

sleep 5

pg_dump -d optuna_study -F p -f study.sql

pg_ctl -D $HOME/pgsql/data stop
