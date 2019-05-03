# ML_Project

Repository for Notre Dame Machine Learning project

## Python libraries needed to run code

All of the code for this project was run using Python3.6

- pandas
- os
- sys
- collections
- math
- seaborn
- matplotlib
- itertools
- xgboost
- sklearn
- warnings
- string
- pickle
- numpy

## Running

To run the code, simply call the following command in the terminal

```bash
./run_all.sh
```

This code runs the following commands in order

```bash
python3.6 parse_data.py
python3.6 generate_matchups.py
python3.6 get_matchup_results.py
python3.6 algorithms_seth.py
python3.6 krieg_mm_analysis.py
python3.6 krieg_plot_mm_results.py
```

If you would like to use the new parameter GridSearch results from algorithms_seth.py, they must be replaced in the model parameters in krieg_mm_analysis.py by hand.