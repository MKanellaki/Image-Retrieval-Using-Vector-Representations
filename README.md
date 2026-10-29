# Image search with vector representations

The program turns every image into a K-dimensional normalised histogram of visual words (SIFT + k-means
bag of words) and then searches for the most similar images with five methods:
**Exact**, **LSH**, **Hypercube**, **IVFFlat** and **IVFPQ**, all under the Euclidean distance.
Queries are the first image of each HPatches sequence; the database holds the other five images of every
sequence of the chosen split plus the first `D` MIRFlickr images as distractors.

Authors: 
* ΚΑΝΕΛΛΑΚΗ ΜΑΡΙΑ 1115202300060
* ΒΟΪΣΤΡΙΚΟΒ ΝΙΚΗΤΑ 1115202300019

---

## 1. Requirements

* Linux, `g++` with C++20 support, `make`
* OpenCV 4 with the `features2d` module (SIFT) and `pkg-config` able to find it (`pkg-config --cflags opencv4`)
* Data (not part of the repository):
  * HPatches: a directory with one sub-directory per sequence, containing `1.ppm … 6.ppm`
  * MIRFlickr-25K: a directory containing `im<number>.jpg`
* Memory: SIFT on the largest HPatches images (3088 × 2056) needs about 1.5 GB per worker thread. The
  program uses 4 threads (section 7), so have at least ~8 GB of free RAM for the first run.
* Disk: about 26 MB per vector cache file (`K = 256`).
* Time: the first run with a new `-vocab`, `-S` or `-seed` builds the vectors. Every later run loads the cache in a few seconds.

## 2. Build

```
make            # builds ./search
make test       # builds and runs every tests/test_*.cpp with AddressSanitizer + UBSan
make debug      # ./search_debug with -g and sanitizers
make clean
```

Warnings are enabled (`-Wall -Wextra -Wpedantic`); the build is warning-free. OpenCV headers are included
with `-isystem` so that their own warnings do not appear.

## 3. Run

```
./search -hp <hpatches_dir> -mir <mirflickr_dir> -split <split_file> \
         -vocab <K> -D <int> -set <validation|test> -o <output_file> \
         [-S <int>] [-seed <int>] \
         <method> [method options]
```

Example:

```
./search -hp hpatches -mir mirflickr -split split.txt -vocab 256 -S 500 -D 10000 \
         -set validation -o output.txt -ivfflat -kclusters 100 -nprobe 10 -seed 1
```

In our setup the data folders are `hpatches-sequences-release/` and `mirflickr/` in the repository root,
so the command uses `-hp hpatches-sequences-release -mir mirflickr`.

Options can be given in any order. Exactly one method flag is required, an option that does not belong to
the chosen method is rejected, and an option may be given only once. Any mistake prints a one-line message
and the usage text and exits with status 2. `-h` / `-help` / `--help` prints the usage. Other errors
(e.g. a missing image folder) print a message and exit with status 1.

**Run the program from the repository root.** The vector cache is stored in `./cache/` relative to the
current directory (see section 7); running from another directory silently starts a new, long vector build.

### 3.1 Shared parameters

| Option | Meaning | Default |
|---|---|---|
| `-hp <dir>` | HPatches directory | required |
| `-mir <dir>` | MIRFlickr directory | required |
| `-split <file>` | training / validation / test split | required |
| `-vocab <K>` | number of k-means centres = length of every image vector (at least 1) | required |
| `-S <int>` | max. SIFT descriptors sampled per training image for the k-means (at least 1) | 500 |
| `-D <int>` | number of MIRFlickr images in the database (0 … 25000, see the note below) | required |
| `-set <validation\|test>` | which query set (and which HPatches images form the database) | required |
| `-o <file>` | output file | required |
| `-seed <int>` | seed of every pseudo-random choice (0 … 2³² − 1) | 1 |

**Note on `-D 25000`.** In the MIRFlickr25K used during development only 24,985 of the 25,000 MIRFlickr images have at least one SIFT descriptor; the
other 15 cannot be turned into a vector (section 8.1). A larger `D` is accepted, but the program then uses
`D = 24985` and prints a warning on stderr, so the largest database has 115 + 24,985 = 25,100 images.

### 3.2 Method parameters (all optional)

| Method | Option | Meaning | Default |
|---|---|---|---|
| `-exact` | — | brute force, no parameters | |
| `-lsh` | `-k <int>` | hash functions `h_i` per `g` (at least 1) | 4 |
| | `-L <int>` | number of hash tables (at least 1) | 5 |
| | `-w <double>` | window width on the projection line (greater than 0) | 0.05 |
| `-hypercube` | `-kproj <int>` | dimension `d'` of the hypercube (1 … 24) | 10 |
| | `-w <double>` | window width (greater than 0) | 0.05 |
| | `-M <int>` | max. candidate points examined (at least 1) | 500 |
| | `-probes <int>` | max. hypercube vertices examined (at least 1) | 10 |
| `-ivfflat` | `-kclusters <int>` | number of clusters (1 … n) | ⌈√n⌉ |
| | `-nprobe <int>` | clusters examined per query (at least 1; more than `kclusters` scans every list) | 5 |
| `-ivfpq` | `-kclusters <int>` | number of coarse clusters (1 … n) | ⌈√n⌉ |
| | `-nprobe <int>` | coarse clusters examined per query (at least 1) | 5 |
| | `-M <int>` | sub-vectors of the product quantiser; `K` must be divisible by `M` | 16 |
| | `-nbits <int>` | each sub-space gets `2^nbits` centroids (1 … 16) | 8 |

`n` is the database size (115 + D). A `-kclusters` larger than `n` is an error (exit status 2). `-M` means
different things for `-hypercube` and `-ivfpq`, so it is only accepted for those two methods. `-seed` is also
used as the seed of the LSH / Hypercube / IVF indexes.

### 3.3 Parameter tuning scripts

Two scripts run the experiments of section 9. Run them from the repository root after `make`; they call
`./search` with the data folders `hpatches-sequences-release` and `mirflickr`.

**`scripts/tune.sh <method> "<params>" ["<params>" ...]`** runs `./search` once per parameter string and
appends one row per run to a CSV file: method, parameters, set, D, Recall@5, Recall@10, mAP@10, average
query time (ms), number of queries with fewer than 10 results, and wall time of the whole run (s, mostly
loading the cache and building the index; it is *not* the query time). The output file and the stderr log
of every run (with the illumination / viewpoint summaries) are kept in
`results/runs/<method>_<params>_<set>_D<D>.{txt,log}`.

```
scripts/tune.sh lsh "-k 4 -L 5 -w 0.05" "-k 4 -L 10 -w 0.1"
SET=test D=10000 OUT=results/extra.csv scripts/tune.sh ivfflat "-kclusters 159 -nprobe 10"
```

`SET` (default `validation`), `D` (default `25000`), `VOCAB` (default `256`) and `OUT` (default
`results/tuning.csv`) can be overridden. Shared options can also go in the parameter string, e.g.
`scripts/tune.sh exact "-S 300"`, but a new `-vocab`, `-S` or `-seed` first builds a new vector cache
(section 1: time and memory).

**`scripts/experiments.sh <phase>`** contains every run of the study and calls `tune.sh` for it. The
chosen parameters are set in the `*_BEST` variables at the top of the script.

| Phase | What it runs | Results |
|---|---|---|
| `exact` | Exact Search, validation, D = 25000 | `results/tune_exact.csv` |
| `lsh`, `hypercube`, `ivfflat`, `ivfpq` | the tuning grid of that method, validation, D = 25000 | `results/tune_<method>.csv` |
| `repeat` | the five chosen settings, three times in a row, validation | `results/repeat.csv` |
| `test` | the five chosen settings once on the test set, D = 1000, 5000, 10000, 25000 | `results/test.csv` |

The order we used: `exact`, then one tuning phase per method. After each phase we chose a setting from its
CSV and wrote it into the script; the IVFPQ grid reuses the chosen IVFFlat `kclusters` / `nprobe`.
`repeat` and `test` come last. A few runs outside the grids (beyond its edges) were added to the same
CSV files with `OUT=results/tune_<method>.csv scripts/tune.sh ...`.

## 4. Output

The output file has the format:

```
<METHOD NAME>
Query: <sequence_id>
Rank-1: <image_id>
...
Rank-10: <image_id>
Recall@5: <double>
Recall@10: <double>
AP@10: <double>
... (one block per query)
Mean Recall@5: <double>
Mean Recall@10: <double>
mAP@10: <double>
Average query time: <double>      (milliseconds)
```

`image_id` is `<sequence>/<number>` for HPatches (e.g. `v_bird/2`) and the file name for MIRFlickr
(e.g. `im123.jpg`). Numbers are printed with 6 decimals.

* If a method returns fewer than 10 results, the missing `Rank-k` lines are still written, with `-` as the
  image id, so every block has the same length. Missing positions count as non-relevant.
* The file is written atomically (to `<file>.tmp`, then renamed), so a failed run never destroys an earlier
  result.
* **Not in the file, printed on stderr:** the progress messages, the index build time, and a summary
  (queries, Recall@5, Recall@10, mAP@10, average time, number of queries with fewer than 10 results) for
  *all* queries, for the *illumination* sequences and for the *viewpoint* sequences. These per-category
  numbers are the ones used in the experimental study.
* Query time is measured with `std::chrono::steady_clock` around the single `query()` call, from the
  moment the query vector exists until the 10 results are returned. Image reading, SIFT, vector
  construction and index construction are not included.

Metrics: `Recall@n = |Top_n ∩ R| / 5`, `AP@10 = (1/5) Σ_j j / r_j` over the relevant images found at ranks
`r_1 < … < r_m` in the top 10, `mAP@10` is the mean of `AP@10`. Each query has exactly 5 relevant images
(the other five images of its sequence).

## 5. Files

### `include/` and `src/` (one module = one header + one source)

| Module | Description |
|---|---|
| `types.hpp` | `Matrix` (dense row-major float matrix used for descriptors, centres, image vectors), `ImageIndex`, `Split`, `Category` |
| `seed.hpp` | `derive_seed`: mixes a global seed with an index (SplitMix64) |
| `distance` | squared Euclidean distance `l2_squared` |
| `dataset` | split file, queries (image 1 of each sequence), HPatches database images (2…6), MIRFlickr files in numeric order, relevant rows of a query |
| `features` | SIFT extraction through OpenCV (the only file that includes OpenCV), uniform sampling of at most S descriptors, extractor factory (one extractor per worker thread) |
| `kmeans` | k-means (k-means++ initialisation, Lloyd iterations); used for the vocabulary, the IVF cells and the PQ codebooks |
| `bow` | `Vocabulary` (centres, encoding of descriptors into a histogram), training of the vocabulary, parallel encoding of images |
| `binary_io.hpp` | small binary reader/writer with bounds checks, used by the cache |
| `bow_pipeline` | builds all vectors once (vocabulary, HPatches validation/test, MIRFlickr), keeps them in a cache file, and assembles the database of a query set |
| `exact` | `ExactIndex`: exhaustive search |
| `lsh` | `LSHIndex` |
| `hypercube` | `HypercubeIndex` |
| `ivfflat` | `IVFFlatIndex` |
| `ivfpq` | `IVFPQIndex` |
| `metrics` | Recall@5, Recall@10, AP@10 for one query; means over any set of queries |
| `args` | command-line parsing and validation |
| `output` | writes the result file |
| `main.cpp` | wires everything together: arguments → vectors → database → index → queries → output |

All five indexes have the same interface: a constructor `(const Matrix& database, Params)` that builds the
index, and `query(const float* q, std::size_t max_results)` that returns row indices sorted by
(distance, row index). `main` wraps the chosen index in one `std::function`, so the query loop, the timing
and the evaluation are the same code for every method.

### `tests/`

One executable per module (`test_<module>.cpp`), each with its own `main`, built and run by `make test`
with sanitizers. The BoW and cache tests use a fake extractor, so they need neither images nor OpenCV
(`test_features_sift.cpp` is the one that needs OpenCV). The tests compare the fast code against simple
reference implementations, check edge cases (empty input, `k > n`, ties, corrupt cache files, bad
arguments) and check that results do not depend on the number of threads.

### Other files

| File | Description |
|---|---|
| `Makefile` | build, tests and debug build (section 2) |
| `split.txt` | the training / validation / test split of the 116 HPatches sequences (70 / 23 / 23) |
| `scripts/tune.sh`, `scripts/experiments.sh` | the tuning and evaluation scripts (section 3.3) |
| `results/tune_<method>.csv` | every tuning run on the validation set (section 9) |
| `results/repeat.csv`, `results/test.csv` | the repeated validation runs and the final test runs of the chosen settings |
| `report/report.md` | the report of the experimental study |
| `report/figures/` | the report's figures and `make_figures.py`, which draws them from `results/test.csv` |
| `README.md` | this file |

## 6. Data flow

```
split.txt ──► dataset: 70 train / 23 validation / 23 test sequences
                 │
   ┌─────────────┴─────────────────────────────┐
   │ cache/bow_….bin exists and is valid?       │── yes ──► load all vectors
   └─────────────┬─────────────────────────────┘
                 │ no
                 ▼
 1. SIFT on the 6 images of every training sequence (420 images), keep ≤ S random descriptors each
 2. k-means (K centres) on those descriptors                          → vocabulary
 3. SIFT + encoding of the 6 images of every validation/test sequence → HPatches vectors
 4. SIFT + encoding of MIRFlickr in numeric order (empty images skipped) → MIRFlickr vectors
 5. write everything to the cache
                 │
                 ▼
 database = images 2…6 of the -set sequences (115 rows) + first D MIRFlickr vectors,
            rows shuffled with -seed (section 8.3)
 queries  = image 1 of the -set sequences (23 vectors)
                 │
                 ▼
 build the chosen index on the database  (not timed)
                 │
                 ▼
 for each query: results = index.query(q, 10)   (timed)
                 Recall@5, Recall@10, AP@10 against the 5 rows of the query's sequence
                 │
                 ▼
 output file (section 4) + per-category summary on stderr
```

The training images never enter the database or the query set, so the vocabulary is learned on images
that are never searched for.

## 7. Reproducibility and the vector cache

* Every random choice (descriptor sampling, k-means++ seeding, LSH and Hypercube projections, IVF / PQ
  k-means) is derived from `-seed`. The same command line gives the same output (apart from the measured
  times). The descriptor sampling uses a different seed per image (`derive_seed(seed, image_index)`), and
  results do not depend on the number of threads.
* Building the vectors (SIFT on ~25,000 images) is by far the slowest step, so it is done once and stored in
  `cache/bow_K<K>_S<S>_seed<seed>_mir25000_split<hash>.bin`. The file name and the header contain
  `K`, `S`, `seed`, the number of MIRFlickr images requested and a hash of the split file. A different value of any of
  them uses (or builds) a different file; `-D`, `-set` and the search method never trigger a rebuild.
  The cache holds every valid MIRFlickr image in numeric order, and any `D` is a
  prefix of them.
* On loading, the cache is checked for the right size and structure, and the HPatches image ids are compared
  with the current split file. An unusable cache is reported and rebuilt. The file is written to a temporary
  name and renamed, so an interrupted run leaves no half-written cache. Delete `cache/` to force a rebuild
  (necessary if the image folders themselves change).
* The extraction uses 4 worker threads (`config.threads` in `main.cpp`). SIFT on a large HPatches image
  needs about 1.5 GB, and with one thread per core (28 on our machine) the first run used more than 8 GB
  and was killed. The thread count does not change the result, only the build time.

## 8. Design and algorithmic decisions

### 8.1 Vector representation

* **SIFT.** OpenCV's SIFT with its default parameters, on the image read as grayscale. Each descriptor has
  128 dimensions.
* **Training set for the vocabulary.** The vocabulary is trained only on the 70 training sequences, all six
  images of each (420 images). Validation and test images and MIRFlickr are never used for training. If the
  vocabulary had seen the query and database images, the evaluation would be optimistic, because the visual
  words would be fitted to exactly the images we search.
* **Descriptor sampling.** An image can have thousands of descriptors. At most `S` (default 500) are taken
  from each training image, chosen uniformly at random without replacement (partial Fisher–Yates). That gives
  at most 420 · 500 = 210,000 points for the k-means, which is enough for K = 256 centres and keeps the
  clustering fast. Every image is sampled with its own seed `derive_seed(seed, i)`, so two images with the
  same number of descriptors do not get the same row positions, and the result does not depend on which
  thread handled which image.
* **Encoding.** Every descriptor of an image (not just a sample) is assigned to its nearest centre; on a tie
  the centre with the smaller index wins. The vector is the histogram `h / n_I`, where `n_I` is the number of
  descriptors of the image, so every vector sums to 1. Without this normalisation a textured image with many
  descriptors would be far from a smooth image of the same scene only because it has more keypoints;
  dividing by `n_I` compares the *distribution* of visual words, which is what the two images share.
* **Images with no descriptors.** A vector `h / 0` is undefined. In MIRFlickr used for development 15 images had no SIFT
  descriptors; they are skipped and the next image in numeric order takes their
  place, so `D` always means "the first D *valid* images". Only 24,985 valid images exist, so `-D 25000`
  is run with D = 24,985 and a warning (section 3.1). All HPatches images have descriptors; if a validation
  or test HPatches image had none, the build would stop with the error `no SIFT descriptors in <image>`
  instead of skipping it, because a query or relevant image cannot be left out without changing the ground
  truth.

### 8.2 k-means

One implementation (`kmeans_fit`) is shared by the vocabulary, the IVF coarse clusters and the PQ codebooks.

* **Initialisation: k-means++.** The first centre is a random point; each next centre is a point chosen with
  probability proportional to its squared distance from the nearest centre chosen so far. This spreads the
  initial centres out and gives better and more stable clusterings than random initialisation.
* **Lloyd iterations.** Assignment step and update step (centre = mean of its points, summed in `double`
  to avoid rounding errors with hundreds of thousands of points). At most 25 iterations for the vocabulary
  and 15 for the IVF / PQ clusterings; the loop stops early when fewer than 0.01 % of the points change
  cluster, because after that the centres barely move.
* **Empty clusters.** If a cluster loses all its points, it takes the point that is farthest from its own
  centre among clusters with more than one member. That point is the one worst served by the current
  centres, so moving a centre to it reduces the error the most, and we never empty another cluster by doing
  so. The alternative (leaving the centre where it is) would waste one of the K centres.
* **Labels.** After the last iteration all points are assigned again against the final centres, so the
  returned (centres, labels) pair is consistent. IVFFlat and IVFPQ use these labels directly to fill their
  inverted lists.

### 8.3 Database, queries and ground truth

* Query = image 1 of each sequence of the chosen set (23 queries).
* Database = images 2…6 of the same set's sequences (23 · 5 = 115 images) and the first D valid
  MIRFlickr images. The rows are then put in a random order with `-seed` (`shuffle_database`), and the
  image ids move with them. Without the shuffle the 115 relevant images would be rows 0–114, ahead of every
  distractor, and a method that stops after a fixed number of candidates (the Hypercube's `M`) would
  always meet them first (section 10.2). Methods that look at every candidate they collect (Exact, LSH,
  IVFFlat, IVFPQ) give the same results in any order.
* The relevant images of a query are the five database rows of its own sequence. MIRFlickr images are never
  relevant; they are distractors that make the search harder as D grows.
* Training images are never in the database, and the validation and test sets are completely separate:
  parameters are chosen on validation and reported on test (section 9).
* Sequences whose name starts with `i_` change the illumination and those starting with `v_` change the
  viewpoint; the stderr summary reports the two categories separately.

### 8.4 Exact search

* Squared Euclidean distance: it gives the same ranking as the Euclidean distance and saves the square root.
* A bounded max-heap holds the best 10 (distance, row) pairs seen so far; a new point replaces the top only
  if it is better. This is O(n log 10) instead of sorting all n distances.
* Ties are broken by the smaller row index, so the output is deterministic. All other methods sort their
  candidates the same way.
* Exact gives the best mAP any method can reach with these vectors, and its query time is the time the
  approximate methods must beat.

```
query(q):
    heap = empty max-heap of (dist, row)
    for each row i of the database:
        d = ||x_i - q||²
        if |heap| < 10: push (d, i)
        else if (d, i) < top(heap): pop, push (d, i)
    return heap contents sorted by (d, i)
```

### 8.5 LSH

* **Hash functions.** Each of the `L` tables has `k` functions `h(p) = ⌊(v·p + t) / w⌋`, with
  `v ~ N(0, 1)^K` and `t ~ U[0, w)`. Points that are close have a high probability of falling in the same
  window of length `w` on the random line `v`.
* **Combining the k values.** The `k` integers are combined into one 64-bit key, `key = key · 31 + h_i`,
  and each table is an `std::unordered_map<key, list of rows>`. The lecture uses
  `ID = Σ r_i h_i mod M` and then `ID mod TableSize`; here the 64-bit key plays the role of the ID, and the
  hash map compares the full key, so only points with the same key end up in the same bucket. That is the
  lecture's "compare the IDs" refinement, without having to choose `M` and the table size by hand.
* **Query.** The union of the query's `L` buckets (duplicates removed) is the candidate set; exact distances
  are computed only for the candidates, and they are sorted by (distance, row). There is no cap on the
  number of candidates.
* **Fewer than 10 candidates.** If the buckets hold fewer than 10 images, fewer than 10 results are
  returned; the missing ranks are written as `-` and count as non-relevant. This is reported in the stderr
  summary.
* **Trade-offs.** Larger `w` makes buckets larger (more candidates: better recall, slower queries). Larger
  `k` makes the key more selective (smaller buckets: faster, lower recall). Larger `L` gives a near
  neighbour more chances to share a bucket with the query (better recall, more memory and time). The vectors
  sum to 1 and their coordinates are small, so useful values of `w` are small too.

```
build:  for t in 1..L: draw v_t1..v_tk ~ N(0,1)^K, t_t1..t_tk ~ U[0,w)
        for each row i: for t in 1..L: table_t[g_t(x_i)].append(i)
g_t(p): key = 0; for j in 1..k: key = key·31 + ⌊(v_tj·p + t_tj)/w⌋; return key
query(q):
    C = ∪_t table_t[g_t(q)]
    return the 10 rows of C with the smallest ||x - q||², ties by row
```

### 8.6 Hypercube

* **Projection to the cube.** `kproj` functions `h_i` as in LSH. Each `f_i` maps the integer `h_i(p)` to a bit.
  `f_i(h)` is the lowest bit of `derive_seed(seed ⊕ i, h)`: a fixed pseudo-random function of (seed, i, h).
  This is equivalent to flipping a fair coin the first time a value is seen and remembering it, but it
  needs no table, it gives the same bit to values that never appeared while building, and a query cannot
  change the index. (An earlier version remembered the coin flips in a mutable map; then the result of a
  query depended on which queries had run before it.)
* **Buckets.** A point goes to the vertex formed by its `kproj` bits. The index allocates all `2^kproj`
  buckets up front, even the empty ones: at `kproj = 24` that is 16.7 M empty lists (about 400 MB), and an
  earlier limit of 30 meant an allocation of about 26 GB. This is why `kproj` is limited to 24; with 25,100
  points almost all vertices are empty long before that, so a larger value has no use.
* **Query.** Probe the query's own vertex, then all vertices at Hamming distance 1, then 2, …, until
  `probes` vertices were visited, and stop as soon as `M` distinct candidates have been collected. Exact
  distances on the candidates, sorted by (distance, row).
* **Trade-offs.** Larger `kproj` makes each vertex smaller (faster, but nearby points are split across
  neighbouring vertices, so more `probes` are needed). `M` and `probes` are the two stop conditions: whichever
  is reached first ends the search.

```
build:  draw v_1..v_kproj, t_1..t_kproj
        for each row i: bucket[vertex(x_i)].append(i)
vertex(p): bits b_i = f_i(⌊(v_i·p + t_i)/w⌋), i = 1..kproj
query(q):
    V = vertex(q), then vertices at Hamming distance 1, 2, ... (at most `probes` vertices)
    C = rows in those buckets, in that order, until |C| = M
    return the 10 rows of C with the smallest ||x - q||², ties by row
```

### 8.7 IVFFlat

* **Build.** k-means on all `n` database vectors gives `kclusters` centroids; each vector is stored in the
  inverted list of its centroid (the labels returned by k-means). Default `kclusters = ⌈√n⌉` (159 for
  n = 25,100), which balances the cost of comparing with all centroids against the cost of scanning lists.
* **Query.** Rank the centroids by distance to the query (ties by centroid index), scan the lists of the
  `nprobe` nearest centroids, compute exact distances, sort by (distance, row).
* **Trade-off.** Larger `nprobe` scans more lists (better recall, slower). With `nprobe = kclusters` every
  list is scanned and the result is exactly the result of Exact Search; we used this as a correctness check.
  On the validation set with D = 24,985, IVFFlat with `-nprobe 159` (all 159 lists) returns the same 10
  results as Exact Search for every query (mAP@10 0.7457 for both).

```
build:  (C, labels) = kmeans(database, kclusters); list[labels[i]].append(i)
query(q):
    P = the nprobe centroids nearest to q
    return the 10 rows of ∪_{c∈P} list[c] with the smallest ||x - q||², ties by row
```

### 8.8 IVFPQ

* **Coarse quantiser.** k-means as in IVFFlat.
* **Product quantiser.** For every vector the residual `r = x − c(x)` with respect to its coarse centroid is
  split into `M` sub-vectors of `K/M` coordinates. Each sub-space gets its own k-means with `2^nbits`
  centroids (at most `n`), trained on the residuals of all vectors; codebook `s` uses the seed
  `seed + 13·s + 7`, so the sub-spaces get different initialisations from the one `-seed`. A vector is stored
  only as `M` codes (the indices of its nearest sub-centroids), in one flat array per inverted list: 8-bit
  codes if `nbits ≤ 8`, 16-bit otherwise. With the defaults a vector takes 16 bytes instead of
  256 · 4 = 1024.
* **Why residuals.** The residuals of all lists are centred around 0 and have a much smaller spread than the
  vectors themselves, so one set of codebooks fits every list and the quantisation error is smaller.
* **Query (asymmetric distance computation).** For each of the `nprobe` nearest coarse centroids `c`: compute
  the query residual `q − c`, fill an `M × 2^nbits` lookup table with the squared distances of each query
  sub-vector to each sub-centroid, and then the distance to every stored vector is the sum of `M` table
  entries. Only the database vectors are quantised (asymmetric), not the query.
* **No re-ranking.** The distances are approximate and the results are sorted by them (ties by row). The
  index has a `rerank` option that recomputes exact distances; the command-line program does not use it, so
  that IVFPQ is measured as the compressed method it is meant to be.
* **Why `K` must be divisible by `M`.** So that all sub-vectors have the same length `K/M` and the code of
  every sub-space means the same amount of the vector. `-M` is checked against `-vocab` when the arguments
  are parsed.
* **Trade-offs.** Larger `M` or `nbits` makes the quantisation finer (better ranking, larger codes, larger
  lookup table and longer training: `2^nbits` centroids per sub-space). `nprobe` behaves as in IVFFlat.

```
build:  (C, labels) = kmeans(database, kclusters)
        r_i = x_i - C[labels[i]]
        for s in 1..M: (B_s, code_s) = kmeans(r_i[sub-space s] for all i, 2^nbits)
        list[labels[i]].append(i, code_1(i) … code_M(i))
query(q):
    for each of the nprobe centroids c nearest to q:
        LUT[s][j] = ||(q - c)[s] - B_s[j]||²
        for each (i, codes) in list[c]: d_i = Σ_s LUT[s][codes[s]]
    return the 10 rows with the smallest d_i, ties by row
```

### 8.9 Other decisions

* **Indexes keep a pointer to the database instead of a copy.** The database (25,100 × 256 floats ≈ 25 MB)
  is built once in `main` and shared. Each index stores a `const Matrix*`, and the constructor that takes a
  temporary `Matrix&&` is deleted, so the compiler rejects code that would leave the index pointing to a
  destroyed matrix.
* **Determinism.** Everything random comes from `-seed`: the descriptor sampling (a per-image seed),
  the k-means initialisations, the LSH and Hypercube projections, and the database shuffle. The Hypercube
  bits are a pure function of (seed, i, h), so a query never changes the index. Ties in distance are broken
  by row index in every method (and by centroid index when IVFFlat / IVFPQ rank their centroids). The same
  command gives the same output file apart from the times; we checked this by running every chosen setting
  three times (identical metrics) and comparing output files of repeated runs.
* **Timing.** Only the `query()` call is timed (section 4), so the times compare the search methods and not
  the shared work of reading images and building vectors.
* **Cache.** Section 7. The expensive part (SIFT on ~25,000 images) is done once; every experiment after that
  only loads a file.
* **Output.** `Rank-k: -` for missing ranks; atomic writes for the output and cache files.

## 9. Experiments

The full study, with every table, the figures and the discussion, is in
[`report/report.md`](report/report.md). This section summarises how it was run and the main numbers.

### 9.1 Protocol

1. **Tuning** on the **validation** set with D = 25000 (24,985), the hardest database, with
   `scripts/experiments.sh <method>` (section 3.3). Every grid was run at least twice; the metrics were
   identical in every repetition, only the times varied.
2. **Selection rule:** for each method, the setting with the highest mAP@10 that is still clearly faster
   than Exact Search (Exact: mAP@10 0.746). A setting that matches Exact but is slower than it is of no use.
   The selection looks at the `avg_ms` column (time per query), not at the wall time of the run.
3. **Timing:** the five chosen settings were run three times one after the other (`repeat`), so their times
   are compared under the same machine load.
4. **Test:** the chosen settings were run once on the **test** set for D = 1000, 5000, 10000 and 25000
   (`test`). Nothing was changed after the test results were seen.

Machine: Intel Core i7-14700HX, 16 GB RAM, Ubuntu 24.04, g++ 13.3 (`-O2`), OpenCV 4.6. Queries run on one
thread; seed 1 everywhere.

### 9.2 Chosen values

| Parameter | Value | Notes |
|---|---|---|
| `K` (`-vocab`) | 256 | not tuned: a new K needs SIFT on all 25,000 images again (hours) |
| `S` | 500 | not tuned, same reason |
| LSH `k`, `L`, `w` | 8, 20, 0.1 | grid `k` 2–8, `L` 5–20, `w` 0.05–0.3 |
| Hypercube `kproj`, `w`, `M`, `probes` | 14, 0.1, 5000, 1000 | grid `kproj` 10–16, `w` 0.03–0.15, (`M`, `probes`) from (1000, 100) to (20000, 5000), on the shuffled database |
| IVFFlat `kclusters`, `nprobe` | 1000, 20 | grid `kclusters` 50–1000, `nprobe` 1–40 |
| IVFPQ `kclusters`, `nprobe`, `M`, `nbits` | 1000, 20, 128, 6 | coarse part from IVFFlat; grid `M` 8–128, `nbits` 4–10 |

### 9.3 Validation, chosen settings (D = 24,985, mean of 3 runs)

| Method | mAP@10 | R@10 | ms / query | vs Exact |
|---|---|---|---|---|
| Exact | 0.746 | 0.765 | 3.73 | 1.0× |
| LSH | 0.607 | 0.635 | 0.54 | 6.9× faster |
| Hypercube | 0.561 | 0.591 | 1.34 | 2.8× faster |
| IVFFlat | 0.732 | 0.748 | 1.04 | 3.6× faster |
| IVFPQ | 0.735 | 0.757 | 0.92 | 4.1× faster |

### 9.4 Test set (mAP@10 / ms per query)

| D | Exact | LSH | Hypercube | IVFFlat | IVFPQ |
|---|---|---|---|---|---|
| 1,000 | 0.758 / 0.15 | 0.580 / 0.03 | 0.733 / 0.18 | 0.758 / 0.18 | 0.758 / 0.51 |
| 5,000 | 0.725 / 0.70 | 0.568 / 0.09 | 0.705 / 0.81 | 0.717 / 0.34 | 0.714 / 0.59 |
| 10,000 | 0.720 / 1.32 | 0.568 / 0.17 | 0.651 / 1.14 | 0.723 / 0.58 | 0.715 / 0.71 |
| 24,985 | 0.698 / 3.52 | 0.563 / 0.38 | 0.503 / 1.33 | 0.698 / 0.99 | 0.698 / 0.90 |

* **IVFFlat and IVFPQ** follow Exact at every D and are 3.5× / 3.9× faster at D = 24,985; IVFPQ also stores
  a vector in 128 bytes instead of 1,024. At D = 1000 they are not faster than Exact: with `kclusters`
  tuned for 25k there is about one image per list.
* **LSH** is the fastest at every D, and its mAP barely depends on D, but it is about 0.13 below Exact.
* **Hypercube** drops from 0.733 to 0.503 as D grows, because its candidate limit `M = 5000` is fixed.
* Viewpoint queries are easier than illumination queries for every method (e.g. Exact at D = 24,985:
  0.720 against 0.675). Recall@5/10, the per-category numbers and the index build times are in the report.

## 10. Use of AI tools

### 10.1 Main prompts

*(List the main prompts, e.g. the ones used to review the project against the assignment, to check each
fix, and to write the tuning script and this README.)*

### 10.2 Debugging and review steps with AI

* **Out of memory on the first run.** The vector build used one thread per core (28); SIFT on the large
  HPatches images needs about 1.5 GB each, so the build used more than 8 GB and VS Code was killed. Fixed
  by setting 4 threads in `main.cpp` (section 7).
* **`-kproj` limit.** The limit was 30, which allows an allocation of about 26 GB for the hypercube
  buckets; lowered to 24 (section 8.6).
* **`-D 25000` impossible.** The cache build reported 15 MIRFlickr images without SIFT descriptors, so only
  24,985 vectors exist. `main` uses D = 24,985 and prints a warning (section 3.1).
* **Hypercube results depended on the query order.** The bits `f_i(h)` were coin flips remembered in a
  mutable map, so each query could add entries that changed later queries. Shown with a test that ran the
  same queries on two identical indexes with different query histories (38 of 50 results differed); fixed with
  the pure function of section 8.6 (0 of 50 differ).
* **Arbitrary order of equal distances.** LSH, Hypercube and IVFFlat sorted by distance only. Found when a
  test with three identical points started failing after the Hypercube fix; fixed by breaking ties by row
  index in every method.
* **Duplicated k-means.** IVFFlat and IVFPQ had their own copies of k-means without empty-cluster handling;
  replaced by the shared `kmeans_fit`. Checked by comparing IVFFlat with `nprobe = kclusters` against Exact
  Search (identical output).
* **IVFPQ 16-bit codes not tested.** The codebook size is limited to the number of points, so a test with
  4 points never reached the 16-bit path; a test with 400 points and `k_sub = 300` was added.
* **Dangling pointer risk.** Indexes store a pointer to the database; the rvalue constructors were deleted.
* **Hypercube looked better than Exact Search during tuning.** The first Hypercube grid gave settings with
  a higher mAP@10 than Exact (e.g. `-kproj 6 -w 0.2 -M 200 -probes 5`: 0.775 against 0.746, R@10 0.826
  against 0.765) while examining only 200 of 25,100 images, which should not be possible. We checked it by
  building the same index on the database rows in reverse order (a temporary copy of `main.cpp`):

  | `-kproj 6 -w 0.2` | original order | rows reversed |
  |---|---|---|
  | `-M 100 -probes 3` | 0.766 | 0.000 |
  | `-M 500 -probes 10` | 0.768 | 0.000 |
  | no `M` limit | 0.746 | 0.746 |

  Cause: with `w = 0.2` almost every image falls on the same vertex; buckets are filled in row order, and
  the 115 relevant HPatches images were rows 0–114, ahead of all distractors, so the first `M` candidates
  always contained them. Fixed by shuffling the database rows with the seed (`shuffle_database`, section
  8.3), with tests that every row keeps its image id and that `relevant_rows` still finds the same five
  images. After the fix Exact and LSH gave the same output as before, the reversed-order test gave similar
  (low) numbers in both orders, and the whole Hypercube tuning was run again on a new grid (smaller `w`,
  larger `M` and `probes`), because all earlier Hypercube results were invalid.

### 10.3 How the AI output was checked

* Every change was built with `-Wall -Wextra -Wpedantic`; the build has no warnings.
* `make test` (14 test programs, AddressSanitizer + UBSan) was run after each change; all pass.
* Full runs on the validation set with the real data were compared before and after each fix. Exact Search
  gives mAP@10 0.7457 on validation with D = 24,985, the same as a separate exact-search check written
  before `main` existed.
* IVFFlat with every list probed gives exactly the Exact Search results (section 8.7).
* Running the same LSH or Hypercube command twice gives identical output files apart from the times.

## 11. Notes

* If an approximate method finds fewer than 10 candidates, the remaining positions count as non-relevant,
  and the number of such queries is reported (stderr summary).
* Small timing differences between runs are normal: a validation or test set has only 23 queries, and the
  times depend on the load of the machine.
* `build/`, `cache/`, `results/runs/`, the `search` and `search_debug` binaries, the datasets
  (`hpatches-sequences-release`, `mirflickr`) and the scratch output files `out.txt` / `output.txt` are not
  part of the repository (`.gitignore`). The result CSVs in `results/` are kept, because the report's
  numbers come from them.
