cd ..
PROJ_HOME=`pwd`

cd $PROJ_HOME/cuda/emd/
rm -rf build/*
python setup.py install

cd $PROJ_HOME/cuda/expansion_penalty/
rm -rf build/*
python setup.py install

cd $PROJ_HOME/cuda/MDS/
rm -rf build/*
python setup.py install

cd $PROJ_HOME/cuda/cubic_feature_sampling/
rm -rf build/*
python setup.py install

cd $PROJ_HOME/cuda/gridding/
rm -rf build/*
python setup.py install

cd $PROJ_HOME/cuda/gridding_loss/
rm -rf build/*
python setup.py install
cd $PROJ_HOME