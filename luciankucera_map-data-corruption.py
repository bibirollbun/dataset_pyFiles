import polars as pl


class Config:
    train_path = "/kaggle/input/map-charting-student-math-misunderstandings/train.csv"


df = pl.read_csv(Config.train_path)


df.filter(
    (pl.col("QuestionText") == '\\( \\frac{A}{10}=\\frac{9}{15} \\) What is the value of \\( A \\) ?') & (pl.col("Category").str.split("_").list[0] == "True")
)

