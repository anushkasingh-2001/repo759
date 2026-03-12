#!/usr/bin/env zsh
#SBATCH -p instruction
#SBATCH -t 0-00:30:00
#SBATCH -J hw07_task1
#SBATCH -o output-%j.out
#SBATCH -e output-%j.err
#SBATCH --gres=gpu:1 -c 1

module load nvidia/cuda/13.0

# Compile HW05 reduction code
nvcc task2_ass_5.cu reduce.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task2_hw05

# Compile HW07 task1 codes
nvcc task1\ thrust.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1_thrust
nvcc task1\ cub.cu    -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task1_cub

# Clear old result files
> hw05_task2_results.txt
> task1_thrust_results.txt
> task1_cub_results.txt

threads=256

for p in {10..20}
do
    n=$((2**p))

    # HW05 custom CUDA reduction
    hw05_time=$(./task2_hw05 $n $threads | tail -n 1)
    echo "$n $hw05_time" >> hw05_task2_results.txt

    # HW07 Thrust reduction
    thrust_time=$(./task1_thrust $n | tail -n 1)
    echo "$n $thrust_time" >> task1_thrust_results.txt

    # HW07 CUB reduction
    cub_time=$(./task1_cub $n | tail -n 1)
    echo "$n $cub_time" >> task1_cub_results.txt

    echo "Done n=$n"
done

echo "Finished generating:"
echo "  hw05_task2_results.txt"
echo "  task1_thrust_results.txt"
echo "  task1_cub_results.txt"