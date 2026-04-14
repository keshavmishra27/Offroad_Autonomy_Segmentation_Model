:: Try activating the Conda environment
echo Activating the Conda environment 'EDU'...

:: Activate using your local C: drive path
call "C:\Users\hp\miniconda3\condabin\conda.bat" activate EDU

:: Install the required packages explicitly into the EDU environment
echo Installing PyTorch, Torchvision, CUDA 11.8, and Ultralytics...
conda install -n EDU -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics -y && pip install opencv-contrib-python && pip install tqdm

echo Environment setup complete. You can now run your code in the 'EDU' environment.
pause