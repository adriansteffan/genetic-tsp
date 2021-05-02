# Genetic TSP Solver
Generates approximative solutions for the Traveling Salesman Problem using genetic algorithms. This was created as part of my [coursework](https://coinse.kaist.ac.kr/teaching/2019/cs454/) at KAIST.
For further details, refer to the `report.pdf` file in the root directory.

## Prerequisites

This script uses Python3.

After cloning this repo, use this to install the required python packages:

```
pip3 install -r requirements.txt
```

## Running the script

In the root directory, run
```
python3 tsp_solver.py [PATH_TO_TSP_FILE]
```

Examples of .tsp files can be found in the `problems/` directory. More problems to test this script against are available at [TSPLIB](http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/index.html).

For details regarding command line arguments and hyperparameters, refer to the `report.pdf` file.



## Authors

* **Adrian Steffan** - [adriansteffan](https://github.com/adriansteffan)

