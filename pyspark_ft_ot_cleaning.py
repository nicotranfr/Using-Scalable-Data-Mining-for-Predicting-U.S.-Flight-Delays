from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from typing import Iterable, Optional


def first_existing(df: DataFrame, candidates: Iterable[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def hhmm_to_timestamp(date_col: str, hhmm_col: str):
    raw = F.regexp_replace(F.col(hhmm_col).cast("string"), r"[^0-9]", "")
    hhmm = F.lpad(raw, 4, "0")
    is_2400 = hhmm == F.lit("2400")
    hh = F.when(is_2400, F.lit("00")).otherwise(F.substring(hhmm, 1, 2))
    mm = F.when(is_2400, F.lit("00")).otherwise(F.substring(hhmm, 3, 2))

    base_ts = F.to_timestamp(
        F.concat_ws(" ", F.col(date_col).cast("string"), F.concat_ws(":", hh, mm, F.lit("00"))),
        "yyyy-MM-dd HH:mm:ss",
    )
    return F.when(is_2400, base_ts + F.expr("INTERVAL 1 DAY")).otherwise(base_ts)


def add_timestamp_from_candidates(df: DataFrame, out_col: str, date_candidates: Iterable[str], time_candidates: Iterable[str]) -> DataFrame:
    date_col = first_existing(df, date_candidates)
    time_col = first_existing(df, time_candidates)
    if not date_col or not time_col:
        return df
    return df.withColumn(out_col, hhmm_to_timestamp(date_col, time_col))


def build_ft(flights: DataFrame) -> DataFrame:
    ft = flights

    cancelled_col = first_existing(ft, ["CANCELLED"])
    diverted_col = first_existing(ft, ["DIVERTED"])
    if cancelled_col:
        ft = ft.filter(F.coalesce(F.col(cancelled_col).cast("double"), F.lit(0.0)) == 0.0)
    if diverted_col:
        ft = ft.filter(F.coalesce(F.col(diverted_col).cast("double"), F.lit(0.0)) == 0.0)

    origin_col = first_existing(ft, ["ORIGIN_AIRPORT_ID", "ORIGIN_AIRPORT_SEQ_ID"])
    dest_col = first_existing(ft, ["DEST_AIRPORT_ID", "DEST_AIRPORT_SEQ_ID"])
    if origin_col:
        ft = ft.withColumn("origin_airport_id", F.col(origin_col).cast("int"))
    if dest_col:
        ft = ft.withColumn("destination_airport_id", F.col(dest_col).cast("int"))

    ft = add_timestamp_from_candidates(ft, "scheduled_departure_time", ["FL_DATE", "FLIGHT_DATE"], ["CRS_DEP_TIME"])
    ft = add_timestamp_from_candidates(ft, "actual_departure_time", ["FL_DATE", "FLIGHT_DATE"], ["DEP_TIME"])
    ft = add_timestamp_from_candidates(ft, "scheduled_arrival_time", ["FL_DATE", "FLIGHT_DATE"], ["CRS_ARR_TIME"])
    ft = add_timestamp_from_candidates(ft, "actual_arrival_time", ["FL_DATE", "FLIGHT_DATE"], ["ARR_TIME"])

    arr_delay_col = first_existing(ft, ["ARR_DELAY_NEW", "ARR_DELAY"])
    if arr_delay_col:
        ft = ft.withColumn("arrival_delay_minutes", F.col(arr_delay_col).cast("double"))

    arr_del15_col = first_existing(ft, ["ARR_DEL15"])
    if arr_del15_col:
        ft = ft.withColumn("delay_label", F.col(arr_del15_col).cast("int"))
    elif "arrival_delay_minutes" in ft.columns:
        ft = ft.withColumn("delay_label", F.when(F.col("arrival_delay_minutes") >= 15, F.lit(1)).otherwise(F.lit(0)))

    return ft


def build_ot(weather: DataFrame, wban_airport_timezone: DataFrame) -> DataFrame:
    wban_col = first_existing(weather, ["WBAN"])
    map_wban_col = first_existing(wban_airport_timezone, ["WBAN"])
    map_airport_col = first_existing(wban_airport_timezone, ["AirportID", "AIRPORT_ID"])

    if not (wban_col and map_wban_col and map_airport_col):
        return weather

    mapping = wban_airport_timezone.select(
        F.col(map_wban_col).alias("map_wban"),
        F.col(map_airport_col).cast("int").alias("airport_id"),
        *([F.col("TimeZone").alias("time_zone")] if "TimeZone" in wban_airport_timezone.columns else []),
    ).dropna(subset=["map_wban", "airport_id"]).dropDuplicates(["map_wban", "airport_id"])

    ot = weather.join(mapping, F.col(wban_col) == F.col("map_wban"), "inner").drop("map_wban")

    ot = add_timestamp_from_candidates(
        ot,
        "observation_time",
        ["DATE", "YEARMODA", "OBS_DATE"],
        ["TIME", "OBS_TIME"],
    )

    return ot


def main():
    spark = SparkSession.builder.appName("ft-ot-cleaning").getOrCreate()

    flights = spark.read.csv("Data/Flights/*.csv", header=True, inferSchema=True)
    weather = spark.read.csv("Data/Weather/*.txt", header=True, inferSchema=True)
    wban_airport_timezone = spark.read.csv("Data/wban_airport_timezone.csv", header=True, inferSchema=True)

    FT = build_ft(flights)
    OT = build_ot(weather, wban_airport_timezone)

    print("Flights columns:", flights.columns)
    print("Weather columns:", weather.columns)
    print("Mapping columns:", wban_airport_timezone.columns)
    print("FT row count:", FT.count())
    print("OT row count:", OT.count())

    FT.printSchema()
    OT.printSchema()


if __name__ == "__main__":
    main()
