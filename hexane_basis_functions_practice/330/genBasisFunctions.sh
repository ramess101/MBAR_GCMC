#!/bin/bash
# This script performs for a single Temperature and chemical potential:
# 1) Equilibration run
# 2) Several (NSTAGES-1) short restart runs 
# 3) Outputs the configurations to a PDB file
# 4) Recomputes the energies for NBASIS sets of parameters
# 5) Compiles all restart/reruns into single set of basis functions

# Assemble the parameters required for the run and potential restart
clean() {   # Make it so that everything is killed on an interrupt
local pids=$(jobs -pr)
echo "On exit sending kill signal to: $pids"
[ -n "$pids" ] && kill $pids
exit 1
}
trap "clean" SIGINT SIGTERM EXIT SIGQUIT  # Call clean for any such signal

nproc="$1"
NSTAGES="$2"
NBASIS="$3"
scripts_path="$4"
pinoffset="$5"

for nStage in $(seq 0 $NSTAGES)

do

if [ "$nStage" -eq 0 ]
then

GOMC_CPU_GCMC +p"$nproc" in_start.conf > output.log

else

GOMC_CPU_GCMC +p"$nproc" in_restart.conf > output.log


for nRerun in $(seq 0 $NBASIS)
do

cp "$scripts_path"/Par_Mie_Alkane_Exotic_basis_function_"$nRerun".inp Par_Mie_Alkane_Exotic_basis_function.inp

GOMC_CPU_GCMC +p"$nproc" in_rerun.conf > rerun.log

cp his?rr.dat his_rr"$nStage"_basis_function_"$nRerun"

done #nRerun

fi

done #nStage

#python "$scripts_path"/create_basis_functions.py

exit 0




