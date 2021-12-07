# Graph Exploration Experiments

**Attention: The simulation uses by default all threads of your machine, and hence a lot of RAM. To control the number of machine, you can set the environment variable _RAYON_NUM_THREADS_ to lower the number of used thread.**

## Preliminaries

* Unpack `resoures/sym_tsplib/set1.zip` (result should be `resources/sym_tsplib/set1/*.xml`).
* Build Blossom V: `cd blossom && make`.

## Reproduce prediction generation for cities

These commands where used to generate the predictions and evaluate all algorithms:

```bash
cargo run --release -- import resources/osm/Stockholm.graphml -w tours/stockholm.txt -o stockholm.csv --two-opt 50 preds -s 175 150 --export preds/stockholm lin --start 0.25 --step 0.25 --num 5
cargo run --release -- import resources/osm/Amsterdam.graphml -w tours/amsterdam.txt -o amsterdam.csv --two-opt 50 preds -s 150 150 --export preds/amsterdam lin --start 0.25 --step 0.25 --num 5
cargo run --release -- import resources/osm/Zagreb.graphml -w tours/zagreb.txt --two-opt 50 -o zagreb.csv preds -s 250 150 --export preds/zagreb lin --start 0.25 --step 0.25 --num 5
cargo run --release -- import resources/osm/Oslo.graphml -w tours/oslo.txt --two-opt 50 -o oslo.csv preds -s 175 150 --export preds/oslo lin --start 0.25 --step 0.25 --num 5
cargo run --release -- import resources/osm/Chisinau.graphml -w tours/chisinau.txt  --two-opt 50 -o chisinau.csv preds -s 75 150 --export preds/ lin --start 0.25 --step 0.25 --num 5
cargo run --release -- import resources/osm/Athens.graphml -w tours/athens.txt --two-opt 50 -o athens.csv preds -s 100 150 --export preds/athens lin --start 0.25 --step 0.25 --num 5
cargo run --release -- import resources/osm/Helsinki.graphml -w tours/helsinki.txt --two-opt 50 -o helsinki.csv preds -s 175 150 --export preds/helsinki lin --start 0.25 --step 0.25 --num 5
cargo run --release -- import resources/osm/Copenhagen.graphml -w tours/copenhagen.txt --two-opt 50 -o copenhagen.csv preds -s 150 150 --export preds/copenhagen lin --start 0.25 --step 0.25 --num 5
cargo run --release -- import resources/osm/Riga.graphml -w tours/riga.txt --two-opt 50 -o riga.csv preds -s 100 150 --export preds/riga lin --start 0.25 --step 0.25 --num 5
cargo run --release -- import resources/osm/Vilnius.graphml -w tours/vilnius.txt --two-opt 50 -o vilnius.csv preds -s 85 150 --export preds/vilnius lin --start 0.25 --step 0.25 --num 5
```

## Reproduce experiments

### City experiments

```bash
cargo run --release -- city Stockholm
cargo run --release -- city Amsterdam
cargo run --release -- city Zagreb
cargo run --release -- city Oslo
cargo run --release -- city Chisinau
cargo run --release -- city Athens
cargo run --release -- city Helsinki
cargo run --release -- city Copenhagen
cargo run --release -- city Riga
cargo run --release -- city Vilnius
```

View results via `python3 plots/plot_city.py [city].csv`

The actual simulation results used in the paper are located at `cities`.

View mean results via `python3 plots/plot_cities.py`

### Robustification scheme comparison

```bash
cargo run --release -- city Stockholm --theoretic
```

View results via `python3 plots/plot_city_rob_compare.py stockholm.csv`

The actual simulation results used in the paper are located at `paper_data/rob_compare.csv`.


### TSPLib experiments

```bash
cargo run --release -- tsplib
python3 plots/plot_tsplib_robust.py tsplib_robust.csv
```

The actual simulation results used in the paper are located at `paper_data/tsplib_robust.csv`.


### Rosenkrantz experiments

```bash
cargo run --release -- rosenscaled 12
python3 plots/plot_rosenkrantz_scaled.py rosenkrantz_scaled_12.csv
```

The actual simulation results used in the paper are located at `paper_data/rosenkrantz_scaled_12.csv`.


