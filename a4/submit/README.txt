Markov Decision Processes

Blake Wang <bwang400@gatech.edu>
GTID#: 903379610

############# Setup #############

1. Install Python 3.8 and pip3
2. Use pip3 to install packages `pip install pymdptoolbox gym matplotlib`
3. Download the source code from https://drive.google.com/file/d/1TyMoBhsS4PJtrtQ-ukg-fPiHqZD978G5/view?usp=sharing
4. Add the mdp.py file from the newly downloaded source code into your mdptoolbox
installation directory, e.g. `cp mdp.py /Users/blake/.pyenv/versions/3.8.2/lib/python3.8/site-packages/mdptoolbox/`

############## Run ##############

Use the following scripts to run the code for the corresponding section in
bwang400-analysis.pdf.

Section 2.1
- vary_forest_size.py
- vary_lake_size.py

Section 2.2
- tune_forest.py
- tune_lake.py

Section 2.3
- converge_forest.py
- converge_lake.py

Section 3.1
- ql_tune_forest.py
- ql_tune_lake.py
