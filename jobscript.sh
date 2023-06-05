 #!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1"
 #BSUB -J myJob
 #BSUB -n 4
 #BSUB -W 24:00
 #BSUB -R "span[hosts=1]"
 #BSUB -R "rusage[mem=39GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err
 echo "Running script..."

02514sh

source ./venv/bin/activate

python src/models/train_model.py 