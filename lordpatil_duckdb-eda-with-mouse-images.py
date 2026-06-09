import numpy as np
import pandas as pd
import duckdb


con = duckdb.connect(database = ":memory:")


con.sql("CREATE OR REPLACE TABLE train AS SELECT * FROM read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')")


# Number of rows
con.sql("SELECT count(*) FROM train")


con.sql("SUMMARIZE train; ").df()


# lab_id
con.sql("SELECT DISTINCT lab_id FROM train")


con.sql("SELECT lab_id, COUNT(*) as values FROM train GROUP BY lab_id ORDER BY values DESC;")


con.sql("SELECT COUNT(video_id) FROM train;")


print(con.sql("SELECT DISTINCT mouse1_strain from train ORDER BY mouse1_strain"))
print(con.sql("SELECT DISTINCT mouse2_strain from train ORDER BY mouse2_strain"))
print(con.sql("SELECT DISTINCT mouse3_strain from train ORDER BY mouse3_strain"))
print(con.sql("SELECT DISTINCT mouse4_strain from train ORDER BY mouse4_strain"))


con.sql("SELECT mouse1_strain, count(*) AS counts FROM train GROUP BY mouse1_strain ORDER BY counts DESC")


con.sql("SELECT frames_per_second, COUNT(*) AS counts FROM train GROUP BY frames_per_second ORDER BY counts DESC")


con.sql("SELECT video_duration_sec, COUNT(*) AS counts FROM train GROUP BY video_duration_sec ORDER BY counts DESC")


con.sql("SELECT arena_shape, COUNT(*) AS counts  FROM train GROUP BY arena_shape ORDER BY counts DESC")


con.sql("SELECT arena_type, COUNT(*) AS counts  FROM train GROUP BY arena_type ORDER BY counts DESC")


con.sql("SELECT body_parts_tracked, COUNT(*) AS counts  FROM train GROUP BY body_parts_tracked ORDER BY counts DESC").df()


con.sql("SELECT tracking_method, COUNT(*) AS counts  FROM train GROUP BY tracking_method ORDER BY counts DESC")


con.sql("CREATE OR REPLACE TABLE test AS SELECT * FROM read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')")


con.sql("SUMMARIZE test").df()


con.sql("""
CREATE OR REPLACE VIEW train_tracking AS 
SELECT split_part(filename, '/', 6) AS lab_id,
       replace(regexp_extract(filename, '([^/]+)$', 1), '.parquet', '') AS video_id,
       * EXCLUDE filename,
FROM read_parquet(
    '/kaggle/input/MABe-mouse-behavior-detection/train_tracking/**/*.parquet',
    filename=true,
    hive_partitioning=false
);
""")


con.sql("DESCRIBE train_tracking;").df()


con.sql("SELECT * FROM train_tracking LIMIT 5;").df()


con.sql("SELECT MAX(video_frame) AS total_frames FROM train_tracking WHERE video_id=1212811043;").df()


con.sql("""
SELECT  frames_per_second, 
        video_duration_sec, 
        frames_per_second*video_duration_sec AS total_frames
FROM train
WHERE video_id=1212811043;
""").df()


con.sql("""
CREATE OR REPLACE VIEW train_annotation AS 
SELECT split_part(filename, '/', 6) AS lab_id,
       replace(regexp_extract(filename, '([^/]+)$', 1), '.parquet', '') AS video_id,
       * EXCLUDE filename,
FROM read_parquet(
    '/kaggle/input/MABe-mouse-behavior-detection/train_annotation/**/*.parquet',
    filename=true,
    hive_partitioning=false
);
""")


con.sql("DESCRIBE train_annotation;")


con.sql("SELECT * FROM train_annotation LIMIT 5;").df()




