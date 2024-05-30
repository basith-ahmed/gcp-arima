create or replace table `taxi.taxi300k` as
WITH taxi_preproc AS (
SELECT 
  ABS(MOD(FARM_FINGERPRINT(STRING(pickup_datetime)), 10000)) AS dataset,
  (tolls_amount + fare_amount) AS fare_amount,
  pickup_datetime,
  EXTRACT(DAYOFWEEK FROM pickup_datetime) AS dayofweek,
  EXTRACT(HOUR FROM pickup_datetime) AS hourofday,
  pickup_longitude AS pickuplon,
  pickup_latitude AS pickuplat,
  dropoff_longitude AS dropofflon,
  dropoff_latitude AS dropofflat,
  passenger_count
FROM
  `nyc-tlc.yellow.trips` 
WHERE
  trip_distance > 0
  AND fare_amount >= 2.5
  AND fare_amount < 200
  AND pickup_longitude > -78
  AND pickup_longitude < -70
  AND dropoff_longitude > -78
  AND dropoff_longitude < -70
  AND pickup_latitude > 37
  AND pickup_latitude < 45
  AND dropoff_latitude > 37
  AND dropoff_latitude < 45
  AND passenger_count > 0
  AND ABS(MOD(FARM_FINGERPRINT(STRING(pickup_datetime)), 10000)) < 3
  )
  SELECT 
  dataset, 
  fare_amount,
  pickup_datetime,
  hourofday, 
  dayofweek,
  CAST(dayofweek * 24 + hourofday AS STRING) AS dayhour,
  pickuplon,
  pickuplat,
  dropofflon,
  dropofflat,
  SQRT(POW((pickuplon - dropofflon),2) + POW(( pickuplat - dropofflat), 2)) AS dist,
  #Euclidean distance between pickup and drop off
  pickuplon - dropofflon AS londiff,
  pickuplat - dropofflat AS latdiff,
  passenger_count
  FROM taxi_preproc
  WHERE dataset < 3

CREATE OR REPLACE MODEL
  taxi.taxifare_dnn OPTIONS (model_type='dnn_regressor',
    hidden_units=[144, 89, 55],
    labels=['fare_amount']) AS
SELECT
    fare_amount,
    hourofday,
    dayofweek,
    pickuplon,
    pickuplat,
    dropofflon,
    dropofflat,
    passenger_count
  FROM
    `taxi.taxi300k`
  WHERE
    dataset = 0;

SELECT
  SQRT(mean_squared_error) AS rmse
FROM
  ML.EVALUATE(MODEL taxi.taxifare_dnn, (SELECT
    fare_amount,
    hourofday,
    dayofweek,
    pickuplon,
    pickuplat,
    dropofflon,
    dropofflat,
    passenger_count
  FROM
    `taxi.taxi300k`
  WHERE
    dataset = 1
    ))

SELECT
    fare_amount,
    predicted_fare_amount,
    hourofday,
    dayofweek,
    pickuplon,
    pickuplat,
    dropofflon,
    dropofflat,
    passenger_count
FROM
  ML.PREDICT(MODEL taxi.taxifare_dnn, (SELECT
    fare_amount,
    hourofday,
    dayofweek,
    pickuplon,
    pickuplat,
    dropofflon,
    dropofflat,
    passenger_count
  FROM
    `taxi.taxi300k`
  WHERE
    dataset = 2
    ))
    LIMIT 10
