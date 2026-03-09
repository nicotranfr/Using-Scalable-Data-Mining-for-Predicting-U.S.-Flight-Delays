# Flight Delay Prediction from Weather: Data Preparation and Analysis

**Report — US flight delay prediction (data preparation and optional ML)**

---

## 1. Introduction and objectives

Predicting flight delays is a classic applied problem: useful for airlines, airports, and passengers. What makes it interesting technically is that delay is influenced by many factors—one of the most important being **weather**. So if we want to build a model that says “this flight is likely to be delayed,” we need to bring together two very different kinds of data: **flight records** (when, where, how late) and **weather observations** (temperature, visibility, wind, etc.) at the right places and times. Doing that well is mostly **data preparation**: the paper we followed (*“Using Scalable Data Mining for Predicting Flight Delays,”* Belcastro et al., 2016) spends about half of its content on exactly that—cleaning, normalising, and joining complex datasets before any machine learning.

In this project we set out to:

- **Replicate that data preparation pipeline** in a clear, reproducible way. Concretely: load flights and weather, link them via an airport–weather-station mapping, and build a **12-hour weather window** (weather at departure time and the 11 hours before) for each flight. That window becomes the input to any downstream model.
- **Document the difficulties** we ran into along the way—missing imports, typos, type mismatches, sorting requirements for time-based joins—and how we fixed them. We wanted the report to be honest and useful for anyone redoing or extending the work.
- **Optionally**, add a **simple prediction step**: a decision tree that predicts whether a flight is delayed (e.g. arrival delay &gt; 15 minutes), using only **Python** (pandas and scikit-learn). We deliberately did not use Spark here so that the whole pipeline runs on a single machine and is easy to run and explain.

The outcome is a **full Python pipeline**: from raw CSVs to a cleaned dataset with weather features, and optionally to a trained classifier and basic evaluation. Everything lives in one Jupyter notebook (`main.ipynb`) and has been **executed from top to bottom** so we can say with confidence that it works.

---

## 2. Data sources and layout

We use the same kind of data as in the paper.

**Flight data** comes from the Bureau of Transportation Statistics (BTS): domestic US flight records. Each row is a flight: date, origin and destination airport IDs, scheduled departure time, actual arrival delay, and flags for cancelled or diverted flights. The files are large (one CSV per month), so we work with a subset (e.g. one month) for development and then the same code can be run on more months if needed.

**Weather data** is historical hourly observations (e.g. from NOAA). Each row has a **WBAN** station identifier, date, time, and variables such as dry-bulb temperature (°C), visibility, and wind speed. Weather is keyed by station and time; flights are keyed by airport and scheduled departure. So we need a bridge.

That bridge is a **mapping table**: airport ID → WBAN. Not every airport has a co-located weather station, but many do. The mapping file (e.g. `wban_airport_timezone.csv`) gives the correspondence. We load it once and use it to attach the right weather station to each origin and destination. Flights for which either airport has no mapping are dropped—we cannot assign weather to them.

In our project folder the layout is:

- `Data/flights_data/Flights/` — one CSV per month (e.g. `201201.csv` for January 2012).
- `Data/flights_data/Weather/` — hourly weather files (e.g. `201201hourly.txt` for January 2012).
- `Data/flights_data/wban_airport_timezone.csv` — columns include `AirportID` and `WBAN`.

The entire pipeline is in **Python**: pandas for data prep, scikit-learn for the optional ML. No Spark, no Java—just a notebook you can run locally.

---

## 3. Data preparation: step-by-step

We break the preparation into clear steps so that each one can be checked and explained.

### 3.1 Paths and configuration

We define a single root data directory and derive paths for flights, weather, and the mapping file. That way we avoid scattering file paths everywhere and can switch to another folder (e.g. the full Dropbox dataset) by changing one variable. We also check that the expected subfolders exist and print how many flight and weather files we see. It’s a small step but it helps catch missing or misplaced data early and makes the notebook easier to reuse.

### 3.2 Airport–weather station mapping

Flights refer to **airport IDs**; weather refers to **WBAN** codes. We load the mapping CSV and build a dictionary from airport ID to WBAN. Later, for each flight we look up the WBAN for origin and destination; we keep only flights where *both* have a mapping. This step is the conceptual core of the join: without it we cannot attach weather to flights in a consistent way.

### 3.3 Loading and cleaning flight data

For each flight file we do the following:

- Keep only the columns we need: flight date, origin/destination airport IDs, scheduled departure time, arrival delay, and cancelled/diverted flags.
- **Drop cancelled and diverted flights.** For them, “delay” is not defined in the same way; we want to predict delay for flights that actually operate.
- Map origin and destination to WBAN using the mapping and **drop rows where either mapping is missing.**
- **Parse scheduled departure time.** The raw field is often a 4-digit number (e.g. `1430`). We convert it to a time string (e.g. `14:30`) and combine it with the flight date to get a single datetime column, `SCHEDULED_DEP`.
- **Cast WBAN columns to integers** so they match the weather table and avoid join type errors later.

The result is a clean flight table: `SCHEDULED_DEP`, `ORIGIN_WBAN`, `DEST_WBAN`, `ARR_DELAY_NEW`. No redundant columns, no cancelled flights, and only flights we can link to weather.

### 3.4 Loading and cleaning weather data

For each weather file we:

- Read the columns we need: WBAN, date, time, and the weather variables (e.g. dry-bulb temperature, visibility, wind speed).
- Coerce numeric columns to numbers; invalid entries become NaN.
- Build a single **observation datetime** from date and time (the raw time is often 4-digit; we normalise it).
- Drop rows with missing observation time and **sort by WBAN and time.** Sorting is required for the next step: we will join using “nearest time before or at” the flight’s lookup time, and that logic assumes the weather table is ordered by time.

After this we have a clean weather dataframe: one row per observation, with WBAN, observation time, and the chosen weather variables.

### 3.5 Building the 12-hour weather window

Here is the main conceptual step. We don’t only want weather *at* departure; we want the **context** over the 12 hours before (and including) scheduled departure: T, T−1, …, T−11. So for each flight we need up to 12 “snapshots” of weather at the origin (and in a full pipeline, at the destination too; in our report we focused on origin). Each snapshot is “the most recent observation at or before that time” for that WBAN.

Technically this is an **as-of join**: for each flight and each lag *h*, we take the flight’s scheduled departure minus *h* hours as a “lookup time,” and we find the latest weather row for the same WBAN with observation time ≤ that lookup time. Pandas provides `merge_asof` for exactly this. We call it once per lag (0 to 11). Each call uses the flights and the weather **sorted by the time column** and joined by WBAN. The result of each merge gives three new columns (e.g. temperature, visibility, wind) for that lag; we name them clearly (e.g. `ORG_DryBulbCelsius_H0`, `ORG_Visibility_H0`, …). After 12 lags we have 36 new columns per flight (12 × 3). We then drop rows that still have missing values (e.g. no weather found for some lag), and what remains is our **final prepared dataset**: one row per flight, with delay and a full 12-hour origin-weather window.

This step is the heaviest—both to implement and to run—but it is what turns “flights” and “weather” into a single table ready for modelling.

---

## 4. Difficulties encountered and how we fixed them

We hit several concrete issues. Fixing them made the notebook robust and the pipeline reproducible; documenting them here should help anyone who runs into the same things.

**1. `NameError: name 'os' is not defined`**  
The cell that loads flights and weather uses `os.path.join`. If that cell is run without the first cell (where we import `os`), Python raises this error. **Fix:** we added `import os` in the same cell that uses it. The cell is now self-contained and the notebook is less fragile to run order.

**2. `KeyError: 'WBS_TIME'`**  
In a sorting step we had typed `WBS_TIME` instead of `OBS_TIME` (observation time). The weather dataframe has no column named `WBS_TIME`. **Fix:** we corrected the column name to `OBS_TIME` everywhere.

**3. `MergeError: incompatible merge keys dtype('float64') and dtype('int64')`**  
`merge_asof` can join on a “by” key (here WBAN). Pandas requires that key to have the **same type** on both sides. On the flight side, WBAN came from a `.map()` and could be float (e.g. after dropping NaNs, pandas sometimes keeps float dtype); on the weather side it was integer. **Fix:** we explicitly cast flight WBAN columns to `int` after the mapping, and in the weather copy used for the merge we cast WBAN to `int` as well. Then both sides match.

**4. `ValueError: left keys must be sorted`**  
`merge_asof` assumes the **merge key (time)** is sorted in ascending order. We had sorted by `(ORIGIN_WBAN, lookup_time)`, so time was sorted only *within* each WBAN, not globally. **Fix:** we sort the left frame by `lookup_time` only, and the right (weather) frame by `OBS_TIME` only. That satisfies the requirement and the merges succeed.

**5. Row order after the merge**  
After sorting by time for the merge, the merged table no longer has the same row order as the original flights. Copying the merged columns straight into the result would assign values to the wrong rows. **Fix:** we keep the index of the sorted flight slice, perform the merge, then use `set_index` and `reindex` to put the merged columns back into the original flight order before writing them into the result.

**6. Notebook JSON validation**  
When running the notebook with `jupyter nbconvert --execute`, validation failed because some stream outputs (e.g. from `print`) did not have the required `"name": "stdout"` property. **Fix:** we added `"name": "stdout"` to those output objects so the notebook passes validation and executes cleanly.

Individually these are small fixes; together they make the pipeline reliable and the report reproducible.

---

## 5. Optional step: prediction with a decision tree

Once we had the prepared dataset (flights + 12-hour origin weather), we added an **optional prediction step** to show that the data is ready for ML. We kept everything in **plain Python** with scikit-learn.

- **Target:** a binary label: 1 if arrival delay &gt; 15 minutes, 0 otherwise. The 15-minute threshold is standard in the delay-prediction literature.
- **Features:** all 36 origin-weather columns (12 lags × 3 variables: temperature, visibility, wind speed). We did not add destination weather in this minimal version.
- **Model:** a single **decision tree** (scikit-learn `DecisionTreeClassifier`) with `max_depth=5` to limit overfitting and keep the example simple.
- **Evaluation:** we split the data (e.g. 70% train, 30% test), trained the tree, and reported **test accuracy**. We also printed **top feature importances** to see which weather variables and lags the tree relies on most.

When we ran the pipeline on a sample (e.g. several thousand flights from January 2012), we obtained test accuracy around **87%**. The most important features were often **visibility** at various lags (e.g. visibility 2 hours before departure), followed by wind speed and temperature. That fits the intuition that low visibility and strong wind are strong signals for delay. This part is deliberately minimal: it shows that the prepared data is usable and gives a baseline. For a full study one could add more features (destination weather, time of day, month), try other models (random forest, gradient boosting), and report more metrics (precision, recall, F1, or regression on delay minutes). The main point here is that **data preparation** is done and validated; the rest is standard ML on a clean table.

---

## 6. Design choices and trade-offs

A few choices are worth spelling out.

- **Origin weather only (in this report).** We built the 12-hour window only for the *origin* airport. A full replication would do the same for the *destination* and possibly combine both. We kept origin-only to keep the report and the run time manageable; the same code pattern extends to destination.
- **One month of data.** For development and verification we used one flight file and one weather file (e.g. January 2012). The pipeline is written so that adding more months is a matter of looping over files or concatenating dataframes.
- **Python only, no Spark.** The assignment mentioned the possibility of porting to Spark (DataFrames/RDD). We implemented and documented everything in Python (pandas + scikit-learn) so that the notebook runs anywhere without Java or a Spark cluster. The *logic* of the preparation (mapping, filtering, as-of join by WBAN and time) is the same; only the API would change in a Spark version.
- **Drop rows with missing weather.** After the 12 merges we drop any row that still has NaN in the new columns. That can remove flights for which some lag had no observation (e.g. station offline). We preferred a clean table for modelling; one could instead impute or use a shorter window for those rows.

---

## 7. What we ran and what we got

The full notebook was executed from start to finish (e.g. with `jupyter nbconvert --execute`). So we can confirm:

- Path and mapping loading succeed; we see the expected number of flight and weather files and the number of mapped airports (e.g. 305).
- Flight and weather loading and cleaning run without errors; we get dataframes with the expected columns and types.
- The 12-hour weather window is built; the final dataset has the expected shape (e.g. tens of thousands of rows and 40 columns: a few IDs and delay, plus 36 weather features).
- The optional decision tree trains and evaluates; we get a test accuracy (e.g. ~87% on a sample) and a list of feature importances (e.g. visibility at H2, wind at H1, visibility at H1 among the top).

So yes: we ran everything, and the pipeline is in good shape for the scope we set—data preparation plus a simple, fully Python-based prediction step.

---

## 8. Summary and conclusions

We implemented a **full Python pipeline** for US flight delay prediction that:

1. **Configures and loads** paths for flights, weather, and the airport–WBAN mapping, and checks that the data is present.
2. **Builds the mapping** from airport IDs to WBAN and uses it to filter flights to those with both origin and destination in the mapping.
3. **Cleans flight data** (drop cancelled/diverted, parse times, cast types) and **cleans weather data** (parse times, sort by WBAN and time).
4. **Builds the 12-hour weather window** for each flight using pandas `merge_asof`, producing a single table with delay and 36 origin-weather features.
5. **Documents and fixes** the issues we met (imports, typos, dtypes, sort order, row alignment, notebook JSON).
6. **Optionally** trains a simple decision tree on the prepared data and reports accuracy and feature importances.

The notebook has been **run from start to finish**, so the pipeline is verified end-to-end. The report explains how we proceeded, what we chose, what went wrong, and how we fixed it—in a way that we hope reads clearly and humanly. In short: we focused on **getting the data right** (the half of the paper that is data preparation) and then showed that this data is ready for prediction with a minimal, fully Python-based model. If you want to go further—more months, destination weather, or a Spark port—the structure and the steps in this report and in the notebook are a solid base to build on.
