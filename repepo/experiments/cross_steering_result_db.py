""" Create a SQLite database with the results of the cross steering experiments. """ 

# NOTE: One row = 1 steering vector on 1 example

# NOTE: Database schema: 
# train_dataset_name: str
# train_dataset_variant: str
# test_dataset_name: str
# test_dataset_variant: str
# test_example.positive.text: str
# test_example.negative.text: str
# test_example.logit_diff: float
# test_example.prob: float

import sqlite3
import pathlib
from contextlib import closing
from contextlib import contextmanager

def with_cursor(func):
    def wrapped(self, *args, **kwargs):
        with self.cursor() as cur:
            return func(self, cur, *args, **kwargs)
    return wrapped

class CrossSteeringResultDatabase:
    def __init__(
        self,
        name="CrossSteeringResult",
        db_path: pathlib.Path | str = "cross_steering_result.sqlite",
    ):
        self.name = name
        if isinstance(db_path, str):
            db_path = pathlib.Path(db_path)
        self.db_path = db_path
        self.con = sqlite3.connect(self.db_path)
        self.con.execute("PRAGMA journal_mode=WAL;")

        # If table doesn't exist, create it
        if not self.table_exists():
            self.create_table()

    def __del__(self):
        self.con.close()

    def __len__(self):
        with closing(self.con.cursor()) as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.name}")
            return cur.fetchone()[0]

    def table_exists(self):
        with closing(self.con.cursor()) as cur:
            cur.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self.name}'"
            )
            return bool(cur.fetchall())

    def create_table(self):
        with closing(self.con.cursor()) as cur:
            create_db_command = (
                f"CREATE TABLE {self.name}"
                + """ (
                steering_vector_dataset_name VARCHAR(255) NOT NULL,
                steering_vector_dataset_variant VARCHAR(255) NOT NULL,
                steering_vector_multiplier FLOAT NOT NULL,
                test_dataset_name VARCHAR(255) NOT NULL,
                test_dataset_variant VARCHAR(255) NOT NULL,
                test_example_id INT NOT NULL,
                test_example_positive_text TEXT NOT NULL,
                test_example_negative_text TEXT NOT NULL,
                test_example_logit_diff FLOAT NOT NULL,
                test_example_pos_prob FLOAT NOT NULL
                ) """
            )
            cur.execute(create_db_command)
            assert self.table_exists(), "Table creation failed"

    @contextmanager
    def cursor(self):
        # Context manager that opens and closes a database cursor
        cur = self.con.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def delete_table(self):
        with closing(self.con.cursor()) as cur:
            cur.execute(f"DROP TABLE {self.name}")
        self.con.commit()
        self.db_path.unlink()

    def add(
        self, 
        steering_vector_dataset_name, 
        steering_vector_dataset_variant, 
        steering_vector_multiplier,
        test_dataset_name, 
        test_dataset_variant, 
        test_example_id, 
        test_example_positive_text, 
        test_example_negative_text, 
        test_example_logit_diff,
        test_example_pos_prob
    ):
        with closing(self.con.cursor()) as cur:
            cur.execute(
                f"INSERT INTO {self.name} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    steering_vector_dataset_name,
                    steering_vector_dataset_variant,
                    steering_vector_multiplier,
                    test_dataset_name,
                    test_dataset_variant,
                    test_example_id,
                    test_example_positive_text,
                    test_example_negative_text,
                    test_example_logit_diff,
                    test_example_pos_prob,
                ),
            )
        self.con.commit()